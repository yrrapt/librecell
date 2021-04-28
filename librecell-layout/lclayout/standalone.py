#
# Copyright 2019-2020 Thomas Kramer.
#
# This source describes Open Hardware and is licensed under the CERN-OHL-S v2.
#
# You may redistribute and modify this documentation and make products using it
# under the terms of the CERN-OHL-S v2 (https:/cern.ch/cern-ohl).
# This documentation is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY,
# INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
# Please see the CERN-OHL-S v2 for applicable conditions.
#
# Source location: https://codeberg.org/tok/librecell
#

import json
import os
from typing import Tuple, Optional, Dict, List

from lccommon import net_util
from lccommon.net_util import load_transistor_netlist, is_ground_net, is_supply_net
from lclayout.data_types import Cell

from .place.place import TransistorPlacer
from .place.euler_placer import EulerPlacer, HierarchicalPlacer
from .place.smt_placer import SMTPlacer
from .place import meta_placer

from .graphrouter.graphrouter import GraphRouter
from .graphrouter.hv_router import HVGraphRouter
from .graphrouter.pathfinder import PathFinderGraphRouter
from .graphrouter.signal_router import DijkstraRouter, ApproxSteinerTreeRouter

from .router import DefaultRouter

from .layout.transistor import TransistorLayout, DefaultTransistorLayout, ChannelType
from .layout.layers import *
from .layout import cell_template
from .layout.notch_removal import fill_notches
from . import tech_util
from collections import defaultdict

from .drc_cleaner import drc_cleaner
from .lvs import lvs
import logging

# klayout.db should not be imported if script is run from KLayout GUI.
try:
    import pya
except ImportError as e:
    import klayout.db as pya

logger = logging.getLogger(__name__)


def _merge_all_layers(shapes):
    # TODO: Move into LcLayout class.
    # Merge all polygons on all layers.
    for layer_name, s in shapes.items():
        if '_label' not in layer_name:
            r = pya.Region(s)
            r.merge()
            s.clear()
            s.insert(r)


def _draw_label(shapes, layer, pos: Tuple[int, int], text: str) -> None:
    """
    Insert a pya.Text object into `shapes`.
    :param shapes:
    :param layer:
    :param pos: Position of the text as a (x,y)-tuple.
    :param text: Text.
    :return: None
    """
    x, y = pos
    # shapes[layer].insert(pya.Text.new(text, pya.Trans(x, y), 0.1, 2))
    shapes[layer].insert(pya.Text.new(text, x, y))


class LcLayout:

    def __init__(self,
                 tech,
                 layout: pya.Layout,
                 placer: TransistorPlacer,
                 router: GraphRouter,
                 debug_routing_graph: bool = False,
                 debug_smt_solver: bool = False
                 ):
        assert isinstance(layout, pya.Layout)
        assert isinstance(placer, TransistorPlacer)
        assert isinstance(router, GraphRouter)

        self.tech = tech
        self.layout = layout
        self.placer = placer
        self.router = router
        self.debug_routing_graph = debug_routing_graph
        self.debug_smt_solver = debug_smt_solver

        self.cell_name = None
        self.io_pins = None
        self.SUPPLY_VOLTAGE_NET = None
        self.GND_NET = None

        # Top layout cell.
        self.top_cell: pya.Cell = None

        self._transistors_abstract: List[Transistor] = None

        self._transistor_layouts: Dict[Transistor, TransistorLayout] = None

        self.shapes: Dict[str, db.Shapes] = dict()

        self._routing_terminal_debug_layers = None

        self._abstract_cell: Cell = None
        self._cell_width = None
        self._cell_height = None

        self._spacing_graph = None

        # Routing graph.
        self._routing_graph: nx.Graph = None

        self._routing_trees = None

        # Pin definitions.
        self._pin_shapes = defaultdict(list)

    def get_min_spacing(self, layer1, layer2, default=0):
        return self._spacing_graph[layer1][layer2]['min_spacing']

    def _00_00_check_tech(self):
        # Assert that the layers in the keys of multi_via are ordered.
        for l1, l2 in self.tech.multi_via.keys():
            assert l1 <= l2, Exception('Layers must be ordered alphabetically. (%s <= %s)' % (l1, l2))

    def _00_01_prepare_tech(self):
        # Load spacing rules in form of a graph.
        self._spacing_graph = tech_util.spacing_graph(self.tech.min_spacing)

    def _01_load_netlist(self, netlist_path: str, cell_name: str):
        # Load netlist of cell.

        logger.info(f'Load netlist: {netlist_path}')

        self.cell_name = cell_name

        self._transistors_abstract, cell_pins = load_transistor_netlist(netlist_path, cell_name)
        self.io_pins = net_util.get_io_pins(cell_pins)

        ground_nets = {p for p in cell_pins if is_ground_net(p)}
        supply_nets = {p for p in cell_pins if is_supply_net(p)}

        assert len(ground_nets) > 0, "Could not find net name of ground."
        assert len(supply_nets) > 0, "Could not find net name of supply voltage."
        assert len(ground_nets) == 1, "Multiple ground net names: {}".format(ground_nets)
        assert len(supply_nets) == 1, "Multiple supply net names: {}".format(supply_nets)

        self.SUPPLY_VOLTAGE_NET = supply_nets.pop()
        self.GND_NET = ground_nets.pop()

        logger.info("Supply net: {}".format(self.SUPPLY_VOLTAGE_NET))
        logger.info("Ground net: {}".format(self.GND_NET))

        # Convert transistor dimensions into data base units.
        for t in self._transistors_abstract:
            t.channel_width = t.channel_width / self.tech.db_unit

        # Size transistor widths.
        logging.debug('Rescale transistors.')
        for t in self._transistors_abstract:
            t.channel_width = t.channel_width * self.tech.transistor_channel_width_sizing

            min_size = self.tech.minimum_gate_width_nfet if t.channel_type == ChannelType.NMOS else self.tech.minimum_gate_width_pfet

            if t.channel_width + 1e-12 < min_size:
                logger.warning("Channel width too small. Changing it to minimal size: %.2e < %.2e", t.channel_width,
                               min_size)

            t.channel_width = max(min_size, t.channel_width)

    def _02_setup_layout(self):
        logger.debug("Setup layout.")
        self.top_cell = self.layout.create_cell(self.cell_name)

        # Setup layers.
        self.shapes = dict()
        for name, (num, purpose) in layermap.items():
            layer = self.layout.layer(num, purpose)
            self.shapes[name] = self.top_cell.shapes(layer)

        if self.debug_routing_graph:
            # Layers for displaying routing terminals.
            self._routing_terminal_debug_layers = {
                l: self.layout.layer(idx, 200) for l, (idx, _) in layermap.items()
            }

    def _03_place_transistors(self):
        # Place transistors
        logging.info('Find transistor placement')

        abstract_cell = self.placer.place(self._transistors_abstract)
        logger.info(f"Cell placement:\n\n{abstract_cell}\n")

        self._abstract_cell = abstract_cell

    def _04_draw_transistors(self):
        logger.debug("Draw transistors.")
        # Get the locations of the transistors.
        transistor_locations = self._abstract_cell.get_transistor_locations()

        # Create the layouts of the single transistors. Layouts are already translated to the absolute position.
        self._transistor_layouts = {t: DefaultTransistorLayout(t, (x, y), self.tech)
                                    for t, (x, y) in transistor_locations}

        # Draw the transistors
        for l in self._transistor_layouts.values():
            assert isinstance(l, TransistorLayout)
            l.draw(self.shapes)

    def _05_draw_cell_template(self):
        logger.debug("Draw cell template.")

        tech = self.tech

        # Calculate dimensions of cell.
        num_unit_cells = self._abstract_cell.width
        self._cell_width = (num_unit_cells + 1) * tech.unit_cell_width
        self._cell_height = tech.unit_cell_height

        # Draw cell template.
        cell_template.draw_cell_template(self.shapes,
                                         cell_shape=(self._cell_width, self._cell_height),
                                         nwell_pwell_spacing=self._spacing_graph[l_nwell][l_pwell]['min_spacing']
                                         )

        # Draw power rails.
        vdd_rail = pya.Path([pya.Point(0, tech.unit_cell_height), pya.Point(self._cell_width, tech.unit_cell_height)],
                            tech.power_rail_width)
        vss_rail = pya.Path([pya.Point(0, 0), pya.Point(self._cell_width, 0)], tech.power_rail_width)

        # Insert power rails into layout.
        self.shapes[tech.power_layer].insert(vdd_rail).set_property('net', self.SUPPLY_VOLTAGE_NET)
        self.shapes[tech.power_layer].insert(vss_rail).set_property('net', self.GND_NET)

        # Add pin shapes for power rails.
        self.shapes[tech.pin_layer + '_pin'].insert(vdd_rail)
        self.shapes[tech.pin_layer + '_pin'].insert(vss_rail)

        # Register Pins/Ports for LEF file.
        self._pin_shapes[self.SUPPLY_VOLTAGE_NET].append((tech.power_layer, vdd_rail))
        self._pin_shapes[self.GND_NET].append((tech.power_layer, vss_rail))

    def _06_route(self):
        router = DefaultRouter(
            graph_router=self.router,
            debug_routing_graph=self.debug_routing_graph,
            tech=self.tech
        )

        self._routing_trees = router.route(self.shapes, io_pins=self.io_pins,
                                           transistor_layouts=self._transistor_layouts,
                                           routing_terminal_debug_layers=self._routing_terminal_debug_layers,
                                           top_cell=self.top_cell)

    def _08_draw_routes(self):
        pass
        # # Draw the layout of the routes.
        # for signal_name, rt in self._routing_trees.items():
        #     _draw_routing_tree(self.shapes, self._routing_graph, rt, self.tech, self.debug_routing_graph)
        #
        # # Merge the polygons on all layers.
        # _merge_all_layers(self.shapes)

    def _08_2_insert_well_taps(self, vdd_net, gnd_net):
        logger.debug("Insert well-taps.")
        spacing_graph = self._spacing_graph

        ntap_size = (100, 100)
        ntap_keepout_layers = [l_pdiffusion, l_poly, l_metal1]

        ptap_size = (100, 100)
        ptap_keepout_layers = [l_ndiffusion, l_poly, l_metal1]

        def find_tap_locations(tap_layer, well_layer, keepout_layers, tap_size) -> db.Region:
            """
            Find potential locations for well-taps.
            Returns a region object which marks all possible locations of the center of the tap.
            :param tap_layer: The layer which will hold the tap (nplus or pplus).
            :param well_layer: The well layer where the tap should be placed.
            :param keepout_layers: Layers that block taps.
            :param tap_size: (w, h) size of the tap.
            :return:
            """
            tap_locations = db.Region(self.shapes[well_layer])

            min_enc = self.tech.minimum_enclosure.get((well_layer, tap_layer), 0)
            tap_locations.size(-min_enc)

            # Cannot place the well-tap under poly or metal1 nor inside the diffusion area.
            for l in keepout_layers:
                r = db.Region(self.shapes[l])
                tap_locations -= r

            if tap_layer in spacing_graph:
                for other_layer in spacing_graph[tap_layer]:
                    min_spacing = spacing_graph[tap_layer][other_layer]['min_spacing']
                    r = db.Region(self.shapes[other_layer])
                    r.size(min_spacing)
                    tap_locations -= r
            #
            # for (outer, inner), min_enc in self.tech.minimum_enclosure.items():
            #     if inner == tap_layer:
            #         o = db.Region(self.shapes[outer])
            #         o.size(-min_enc)
            #         tap_locations &= o

            tap_locations.size(-tap_size[0], -tap_size[1])

            return tap_locations

        # Find regions where it is feasible to place well-taps.
        ntap_locations = find_tap_locations(l_nplus, l_nwell, ntap_keepout_layers, ntap_size)
        ptap_locations = find_tap_locations(l_pplus, l_pwell, ptap_keepout_layers, ptap_size)

        # Visualize the regions in the layout.
        # self.shapes[l_nplus].insert(ntap_locations)
        for p in ntap_locations.each_merged():
            ntap = self.shapes[l_nplus].insert(p)
            ntap.set_property('net', vdd_net)
        # self.shapes[l_pplus].insert(ptap_locations)
        for p in ptap_locations.each_merged():
            ptap = self.shapes[l_pplus].insert(p)
            ptap.set_property('net', gnd_net)

    def _08_03_connect_well_taps(self):
        logger.debug("Connect well-taps.")
        router = DefaultRouter(
            graph_router=self.router,
            debug_routing_graph=self.debug_routing_graph,
            tech=self.tech
        )

        routing_trees = router.route(self.shapes, io_pins=[],
                                     transistor_layouts=dict(),
                                     routing_terminal_debug_layers=self._routing_terminal_debug_layers,
                                     top_cell=self.top_cell)

    def _09_post_process(self):
        tech = self.tech
        # Register Pins/Ports for LEF file.

        if not self.debug_routing_graph:

            # Clean DRC violations that are not handled above.

            def fill_all_notches():
                # Remove notches on all layers.
                for layer, s in self.shapes.items():
                    if layer in tech.minimum_notch:

                        if layer in tech.connectable_layers:
                            r = pya.Region(s)
                            filled = fill_notches(r, tech.minimum_notch[layer])
                            s.insert(filled)
                        else:
                            # Remove notches per polygon to avoid connecting independent shapes.
                            s_filled = pya.Shapes()
                            for shape in s.each():
                                r = pya.Region(shape.polygon)

                                filled = fill_notches(r, tech.minimum_notch[layer])
                                s_filled.insert(filled)

                            s.insert(s_filled)

                    _merge_all_layers(self.shapes)

            # Fill notches that violate a notch rule.
            fill_all_notches()
            # Do a second time because first iteration could have introduced new notch violations.
            fill_all_notches()

            # Fix minimum area violations.
            fix_min_area(tech, self.shapes, debug=self.debug_smt_solver)

            # Draw pins
            # Get shapes of pins.
            pin_locations_by_net = {}
            pin_shapes_by_net = {}
            for net_name, rt in self._routing_trees.items():
                # Get virtual pin nodes.
                virtual_pins = [n for n in rt.nodes if n[0] == 'virtual_pin']
                for vp in virtual_pins:
                    # Get routing nodes adjacent to virtual pin nodes. They contain the location of the pin.
                    locations = [l for _, l in rt.edges(vp)]
                    _, net_name, _, _ = vp
                    for layer, (x, y) in locations:
                        w = tech.minimum_pin_width
                        s = self.shapes[layer]

                        # Find shape at (x,y).
                        ball = pya.Box(pya.Point(x - 1, y - 1), pya.Point(x + 1, y + 1))
                        pin_shapes = pya.Region(s).interacting(pya.Region(ball))

                        # Remember pin location and shape.
                        pin_locations_by_net[net_name] = x, y
                        pin_shapes_by_net[net_name] = pin_shapes

                        # Register pin shapes for LEF file.
                        self._pin_shapes[net_name].append((layer, pin_shapes))

            # Add pin labels
            for net_name, (x, y) in pin_locations_by_net.items():
                logger.debug('Add pin label: %s, (%d, %d)', net_name, x, y)
                _draw_label(self.shapes, tech.pin_layer + '_label', (x, y), net_name)

            # Add label for power rails
            _draw_label(self.shapes, tech.power_layer + '_label', (self._cell_width // 2, 0), self.GND_NET)
            _draw_label(self.shapes, tech.power_layer + '_label', (self._cell_width // 2, self._cell_height),
                        self.SUPPLY_VOLTAGE_NET)

            # Add pin shapes.
            for net_name, pin_shapes in pin_shapes_by_net.items():
                self.shapes[tech.pin_layer + '_pin'].insert(pin_shapes)

    def store_placement(self, placement_path: str):
        """
        Dump the transistor placement to a file such that it can be loaded in a later run.
        :param placement_path: Path to the output file.
        """
        logger.info(f"Store transistor placement: {placement_path}")
        assert self._abstract_cell is not None, "No placement known yet."

        transistors = [t for t in self._abstract_cell.lower + self._abstract_cell.upper if t is not None]
        locations = self._abstract_cell.get_transistor_locations()

        transistor_locations = {
            t.name: (x, y) for t, (x, y) in locations
        }

        transistor_nets = {
            t.name: (t.source_net, t.gate_net, t.drain_net) for t in transistors
        }

        data = {
            "placement_file_version": "0.0",
            "description": f"lclayout transistor placement of cell '{self.cell_name}'",
            "cell_name": self.cell_name,
            "transistor_locations": transistor_locations,
            "transistor_nets": transistor_nets
        }

        with open(placement_path, "w") as f:
            json.dump(data, f, indent=None, sort_keys=True)

    def load_placement(self, placement_path: str):
        """
        Load the transistor placement from a file and write it into the `self._abstract_cell` variable.
        :param placement_path: Path to the placement file.
        """
        logger.info(f"Load transistor placement: {placement_path}")

        assert self._transistors_abstract is not None, "Netlist is not loaded yet."

        data = None
        with open(placement_path, "r") as f:
            data = json.load(f)

        cell_name = data["cell_name"]
        if cell_name != self.cell_name:
            logger.error(f"Placement file is for wrong cell: '{cell_name}' instead of '{self.cell_name}'")
            exit(1)

        transistor_locations = data["transistor_locations"]
        assert isinstance(transistor_locations, dict), "'transistor_locations' must be a dictionary."

        transistor_nets = data["transistor_nets"]
        assert isinstance(transistor_nets, dict), "'transistor_nets' must be a dictionary."

        transistors_by_name = {
            t.name: t for t in self._transistors_abstract
        }

        # Do sanity checks.
        present_transistor_names = set(transistor_locations.keys())
        expected_transistor_names = set(transistors_by_name.keys())

        missing_transistors = expected_transistor_names - present_transistor_names
        excess_transistors = present_transistor_names - expected_transistor_names

        # All required transistors should be given a placement.
        if missing_transistors:
            logger.error("Placement for some transistors is not defined: {}".format(", ".join(missing_transistors)))
            exit(1)

        if excess_transistors:
            logger.error("Unknown transistor names in placement file: {}".format(", ".join(excess_transistors)))
            exit(1)

        # Find the width of the cell.
        most_right_location = max((x for x, y in transistor_locations.values()))
        max_y = max((y for x, y in transistor_locations.values()))
        assert max_y <= 1

        # Create Cell object that holds the placement of the transistors.
        cell = Cell(width=most_right_location + 1)

        # Assign transistor positions.
        matrix = [cell.lower, cell.upper]
        for transistor_name, (x, y) in transistor_locations.items():
            # Get transistor.
            t = transistors_by_name[transistor_name]

            # Load the orientation of the transistor.
            source, gate, drain = transistor_nets[transistor_name]
            # Check that the nets are correct.
            assert gate == t.gate_net, f"Gate net mismatch in transistor {transistor_name}."

            # Check that the nets are correct.
            assert {source, drain} == {t.source_net, t.drain_net}, \
                f"Source/drain net mismatch in transistor {transistor_name}."

            # Flip the transistor if necessary.
            if source == t.source_net:
                # No flipping necessary.
                assert drain == t.drain_net
            else:
                t = t.flipped()

            assert gate == t.gate_net
            assert source == t.source_net
            assert drain == t.drain_net

            assert matrix[y][x] is None, f"Transistor position is used multiple times: {(x, y)}"
            matrix[y][x] = t

        # Store the cell with the placement.
        self._abstract_cell = cell

    def create_cell_layout(self,
                           cell_name: str,
                           netlist_path: str,
                           placement_path: Optional[str]) \
            -> Tuple[pya.Cell, Dict[str, List[Tuple[str, pya.Shape]]]]:
        """

        :param cell_name: Name of the cell to be placed.
        :param netlist_path: Path to the SPICE netlist containing the cell.
        :param placement_path: Optional path to the JSON file containing the transistor placement.
            If the file exists, it will be used as source for the placement.
            If the file does not exist, the placement will be written into it.
        :return: Layout cell, pin shapes
        """

        self._00_00_check_tech()
        self._00_01_prepare_tech()

        self._01_load_netlist(netlist_path, cell_name)

        self._02_setup_layout()

        # Load or compute transistor placement.
        # The placement is either computed or loaded from a file.
        if placement_path is not None:
            if os.path.exists(placement_path):
                # Load placement from file.
                self.load_placement(placement_path)
                logger.info(f"Cell placement:\n\n{self._abstract_cell}\n")
            else:
                # Compute placement.
                self._03_place_transistors()
                # Store placement.
                self.store_placement(placement_path)
        else:
            # Compute placement.
            self._03_place_transistors()

        self._04_draw_transistors()
        self._05_draw_cell_template()
        self._06_route()
        self._08_draw_routes()
        # self._08_2_insert_well_taps(vdd_net='vdd', gnd_net='gnd')  # TODO: No hardcoded nets.
        # self._08_03_connect_well_taps()
        self._09_post_process()

        return self.top_cell, self._pin_shapes


def main():
    """
    Entry function for standalone command line tool.
    :return:
    """
    import argparse
    import datetime
    import time

    # List of available placer engines.
    placers = {
        'meta': meta_placer.MetaTransistorPlacer,
        'flat': EulerPlacer,
        'hierarchical': HierarchicalPlacer,
        'smt': SMTPlacer
    }

    signal_routers = {
        'dijkstra': DijkstraRouter,  # Fast but not stable.
        'steiner': ApproxSteinerTreeRouter,  # Slow but best results.
        # 'lp': LPSignalRouter
    }

    # Define commandline arguments.
    parser = argparse.ArgumentParser(description='Generate GDS layout from SPICE netlist.')
    parser.add_argument('--cell', required=True, metavar='NAME', type=str, help='cell name')
    parser.add_argument('--netlist', required=True, metavar='FILE', type=str, help='path to SPICE netlist')
    parser.add_argument('--output-dir', default='.', metavar='DIR', type=str, help='output directory for layouts')
    parser.add_argument('--tech', required=True, metavar='FILE', type=str, help='technology file')

    parser.add_argument('--debug-routing-graph', action='store_true',
                        help='write full routing graph to the layout instead of wires')
    parser.add_argument('--debug-smt-solver', action='store_true',
                        help='enable debug mode: display routing nodes in layout, \
                        show unsatisfiable core if SMT DRC cleaning fails.')

    parser.add_argument('--placer', default='meta', metavar='PLACER', type=str, choices=placers.keys(),
                        help='placement algorithm ({})'.format(', '.join(sorted(placers.keys()))))

    parser.add_argument('--placement-file', metavar='PLACEMENTFILE', type=str,
                        help='Use this file to store the placement such that it can be reused between runs. JSON format is used.')

    parser.add_argument('--signal-router', default='dijkstra', metavar='SIGNAL_ROUTER', type=str,
                        choices=signal_routers.keys(),
                        help='routing algorithm for single signals ({})'.format(
                            ', '.join(sorted(signal_routers.keys()))))

    # parser.add_argument('--profile', action='store_true', help='enable profiler')
    parser.add_argument('-v', '--verbose', action='store_true', help='show more information')
    parser.add_argument('--ignore-lvs', action='store_true', help='Write the layout file even if the LVS check failed.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="don't show any information except fatal events (overwrites --verbose)")
    parser.add_argument('--log', required=False, metavar='LOG_FILE', type=str,
                        help='write log to this file instead of stdout')

    # Parse arguments
    args = parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    if args.quiet:
        log_level = logging.FATAL

    # Setup logging
    logging.basicConfig(format='%(asctime)s %(module)16s %(levelname)8s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=log_level,
                        filename=args.log)

    # Load netlist of cell
    cell_name = args.cell
    netlist_path = args.netlist

    tech_file = args.tech

    tech = tech_util.load_tech_file(tech_file)

    # Create empty layout
    layout = pya.Layout()

    # Setup placer algorithm

    placer = placers[args.placer]()
    logger.info("Placement algorithm: {}".format(type(placer).__name__))

    # Setup routing algorithm

    signal_router = signal_routers[args.signal_router]()
    logger.info("Signal routing algorithm: {}".format(type(signal_router).__name__))

    router = PathFinderGraphRouter(
        signal_router
    )
    # router = LPGraphRouter()
    router = HVGraphRouter(router,
                           orientation_change_penalty=tech.orientation_change_penalty
                           )

    layouter = LcLayout(tech=tech,
                        layout=layout,
                        placer=placer,
                        router=router,
                        debug_routing_graph=args.debug_routing_graph,
                        debug_smt_solver=args.debug_smt_solver)

    # Run layout synthesis
    time_start = time.process_time()
    cell, pin_geometries = layouter.create_cell_layout(cell_name, netlist_path, args.placement_file)

    # LVS check
    logger.info("Running LVS check")
    reference_netlist = lvs.read_netlist_mos4_to_mos3(netlist_path)

    # Remove all unused circuits.
    # The reference netlist must contain only the circuit of the cell to be checked.
    # Copying a circuit into a new netlist makes `combine_devices` fail.
    circuits_to_delete = {c for c in reference_netlist.each_circuit() if c.name != cell_name}
    for c in circuits_to_delete:
        reference_netlist.remove(c)

    # Extract netlist from layout.
    extracted_netlist = lvs.extract_netlist(layout, cell)

    # Run LVS comparison of the two netlists.
    lvs_success = lvs.compare_netlist(extracted_netlist, reference_netlist)

    logger.info("LVS result: {}".format('SUCCESS' if lvs_success else 'FAILED'))

    if not lvs_success:
        logger.error("LVS check failed!")
        if not args.ignore_lvs and not args.debug_routing_graph:
            exit(1)

    # Output using defined output writers.
    from .writer.writer import Writer
    for writer in tech.output_writers:
        assert isinstance(writer, Writer)
        logger.debug("Call output writer: {}".format(type(writer).__name__))
        writer.write_layout(
            layout=layout,
            pin_geometries=pin_geometries,
            top_cell=cell,
            output_dir=args.output_dir
        )

    time_end = time.process_time()
    duration = datetime.timedelta(seconds=time_end - time_start)
    logger.info("Done (Total duration: {})".format(duration))


def fix_min_area(tech, shapes: Dict[str, pya.Shapes], debug=False):
    """
    Fix minimum area violations.
    This is a wrapper around the drc_cleaner module.
    :param tech:
    :param shapes:
    :param debug: Tell DRC cleaner to find unsatisiable core.
    :return:
    """

    # Find minimum area violations.
    # And create a set of whitelisted shapes that are allowed to be changed for DRC cleaning.
    min_area_violations = set()
    for layer, _shapes in shapes.items():
        min_area = tech.min_area.get(layer, 0)
        for shape in _shapes.each():
            area = shape.area()
            if area < min_area:
                min_area_violations.add((layer, shape))

    # TODO: Also whitelist vias connected to the violating shapes.

    if min_area_violations:
        success = drc_cleaner.clean(tech,
                                    shapes=shapes,
                                    white_list=min_area_violations,
                                    enable_min_area=True,
                                    debug=debug
                                    )
        if not success:
            logger.error("Minimum area fixing failed!")
    else:
        logger.info("No minimum area violations.")
