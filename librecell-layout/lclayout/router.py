from .graphrouter.graphrouter import GraphRouter
from .graphrouter.hv_router import HVGraphRouter
from .graphrouter.pathfinder import PathFinderGraphRouter
from .graphrouter.signal_router import DijkstraRouter, ApproxSteinerTreeRouter
from collections import Counter, defaultdict
from .routing_graph import *
from . import tech_util
import networkx as nx
from itertools import chain
import numpy
import klayout.db as db


def _draw_routing_tree(shapes: Dict[str, pya.Shapes],
                       G: nx.Graph,
                       rt: nx.Graph,
                       tech,
                       debug_routing_graph: bool = False):
    """ Draw a routing graph into a layout.
    :param shapes: Mapping from layer name to pya.Shapes object
    :param G: Full graph of routing grid
    :param rt: Graph representing the wires
    :param tech: module containing technology information
    :param debug_routing_graph: Draw narrower wires for easier visual inspection
    :return:
    """

    def is_virtual_node(n):
        return n[0].startswith('virtual')

    def is_virtual_edge(e):
        return is_virtual_node(e[0]) or is_virtual_node(e[1])

    logger.debug("Drawing wires")

    # Loop through all edges of the routing tree and draw them individually.
    for a, b in rt.edges:

        if not is_virtual_edge((a, b)):

            l1, (x1, y1) = a
            l2, (x2, y2) = b

            data = G[a][b]

            if l1 == l2:
                # On the same layer -> wire

                w = data.get('wire_width', tech.wire_width[l1])

                ext = w // 2

                is_horizontal = y1 == y2 and x1 != x2

                if is_horizontal:
                    w = tech.wire_width_horizontal[l1]

                if debug_routing_graph:
                    w = min(tech.routing_grid_pitch_x, tech.routing_grid_pitch_y) // 16

                path = pya.Path([pya.Point(x1, y1), pya.Point(x2, y2)], w, ext, ext)
                shapes[l1].insert(path)
            else:
                # l1 != l1 -> this looks like a via
                assert x1 == x2
                assert y1 == y2
                # Draw via
                via_layer = via_layers[l1][l2]['layer']
                logger.debug('Draw via: {} ({}, {})'.format(via_layer, x1, y1))

                via_width = tech.via_size[via_layer]

                if debug_routing_graph:
                    via_width = min(tech.routing_grid_pitch_x, tech.routing_grid_pitch_y) // 16

                w = via_width // 2
                via = pya.Box(pya.Point(x1 - w, y1 - w),
                              pya.Point(x1 + w, y1 + w))
                shapes[via_layer].insert(via)

                # Ensure minimum via enclosure.
                if not debug_routing_graph:
                    for l in (l1, l2):
                        # TODO: Check on which sides minimum enclosure is not yet satisfied by some wire.

                        neighbors = rt.neighbors((l, (x1, y1)))
                        neighbors = [n for n in neighbors if n[0] == l]

                        w_ext = via_width // 2 + tech.minimum_enclosure[(l, via_layer)]
                        w_noext = via_width // 2

                        # Check on which sides the enclosure must be extended.
                        # Some sides will already be covered by a routing wire.
                        ext_right = w_ext
                        ext_upper = w_ext
                        ext_left = w_ext
                        ext_lower = w_ext
                        # TODO
                        # for _, (n_x, n_y) in neighbors:
                        #     if n_x == x1:
                        #         if n_y < y1:
                        #             ext_lower = w_noext
                        #         if n_y > y1:
                        #             ext_upper = w_noext
                        #     if n_y == y1:
                        #         if n_x < x1:
                        #             ext_left = w_noext
                        #         if n_x > x1:
                        #             ext_right = w_noext

                        enc = pya.Box(
                            pya.Point(x1 - ext_left, y1 - ext_lower),
                            pya.Point(x1 + ext_right, y1 + ext_upper)
                        )
                        shapes[l].insert(enc)


class DefaultRouter():

    def __init__(self,
                 graph_router: GraphRouter,
                 tech,
                 debug_routing_graph: bool = False):
        self.tech = tech  # Technology data.
        self.debug_routing_graph = debug_routing_graph
        self.router = graph_router

        # Load spacing rules in form of a graph.
        self._spacing_graph = tech_util.spacing_graph(self.tech.min_spacing)

    def route(self, shapes: Dict[str, db.Shapes],
              io_pins: List[str],
              transistor_layouts: Dict[Transistor, TransistorLayout],
              routing_terminal_debug_layers: Dict[str, str] = None,
              top_cell: db.Cell = None
              ):
        routing_trees = self._06_route(shapes, io_pins, transistor_layouts,
                       routing_terminal_debug_layers, top_cell)
        self._08_draw_routes(shapes, routing_trees)
        return routing_trees

    def _06_route(self, shapes: Dict[str, db.Shapes],
                  io_pins: List[str],
                  transistor_layouts: Dict[Transistor, TransistorLayout],
                  routing_terminal_debug_layers: Dict[str, str] = None,
                  top_cell: db.Cell = None):
        # TODO: Move as much as possible of the grid construction into a router specific class.
        tech = self.tech

        cell_width = db.Region(shapes[l_abutment_box]).bbox().width()

        # Construct two dimensional grid which defines the routing graph on a single layer.
        grid = Grid2D((tech.grid_offset_x, tech.grid_offset_y),
                      (
                          tech.grid_offset_x + cell_width - tech.grid_offset_x,
                          tech.grid_offset_y + tech.unit_cell_height),
                      (tech.routing_grid_pitch_x, tech.routing_grid_pitch_y))

        # Create base graph
        graph = create_routing_graph_base(grid, tech)

        # Remove illegal routing nodes from graph and get a dict of legal routing nodes per layer.
        remove_illegal_routing_edges(graph, shapes, tech)

        # if not debug_routing_graph:
        #     assert nx.is_connected(graph)

        # Remove pre-routed edges from graph.
        remove_existing_routing_edges(graph, shapes, tech)

        # Create a list of terminal areas: [(net, layer, [terminal, ...]), ...]
        terminals_by_net = extract_terminal_nodes(graph, shapes, tech)

        # Embed transistor terminal nodes in to routing graph.
        terminals_by_net.extend(embed_transistor_terminal_nodes(graph, transistor_layouts, tech))

        # Remove terminals of nets with only one terminal. They need not be routed.
        # This can happen if a net is already connected by abutment of two transistors.
        # Count terminals of a net.
        num_appearance = Counter(chain((net for net, _, _ in terminals_by_net), io_pins))
        terminals_by_net = [t for t in terminals_by_net if num_appearance[t[0]] > 1]

        # Check if each net really has a routing terminal.
        # It can happen that there is none due to spacing issues.
        # First find all net names in the layout.
        all_net_names = {s.property('net') for _, _shapes in shapes.items() for s in _shapes.each()}
        all_net_names -= {None}

        error = False
        # Check if each net has at least a terminal.
        for net_name in all_net_names:
            num_terminals = num_appearance.get(net_name)
            if num_terminals is None or num_terminals == 0:
                logger.error("Net '{}' has no routing terminal.".format(net_name))
                error = True

        if not self.debug_routing_graph:
            assert not error, "Nets without terminals. Check the routing graph (--debug-routing-graph)!"

        # Create virtual graph nodes for each net terminal.
        virtual_terminal_nodes = create_virtual_terminal_nodes(graph, terminals_by_net, io_pins, tech)

        if self.debug_routing_graph:
            # Display terminals on layout.
            routing_terminal_shapes = {
                l: top_cell.shapes(routing_terminal_debug_layers[l]) for l in tech.routing_layers.keys()
            }
            for net, layer, ts in terminals_by_net:
                for x, y in ts:
                    d = tech.routing_grid_pitch_x // 16
                    routing_terminal_shapes[layer].insert(pya.Box(pya.Point(x - d, y - d), pya.Point(x + d, y + d)))

        # Remove nodes that will not be used for routing.
        # Iteratively remove nodes of degree 1.
        while True:
            unused_nodes = set()
            for n in graph:
                if nx.degree(graph, n) <= 1:
                    if not _is_virtual_node_fn(n):
                        unused_nodes.add(n)
            if len(unused_nodes) == 0:
                break
            graph.remove_nodes_from(unused_nodes)

        if not nx.is_connected(graph):
            assert False, 'Routing graph is not connected.'

        self._routing_graph = graph

        # TODO: SPLIT HERE
        # def _07_route(self):

        tech = self.tech
        spacing_graph = self._spacing_graph
        graph = self._routing_graph

        # Route
        if self.debug_routing_graph:
            # Write the full routing graph to GDS.
            logger.info("Skip routing and plot routing graph.")
            self._routing_trees = {'graph': self._routing_graph}
        else:
            logger.info("Start routing")
            # For each routing node find other nodes that are close enough that they cannot be used
            # both for routing. This is used to avoid spacing violations during routing.
            logger.debug("Find conflicting nodes.")
            conflicts = dict()
            # Loop through all nodes in the routing graph graph.
            for n in graph:
                # Skip virtual nodes which have no physical representation.
                if not _is_virtual_node_fn(n):
                    layer, point = n
                    wire_width1 = tech.wire_width.get(layer, 0) // 2
                    node_conflicts = set()
                    if layer in spacing_graph:
                        # If there is a spacing rule defined involving `layer` then
                        # loop through all layers that have a spacing rule defined
                        # relative to the layer of the current node n.
                        for other_layer in spacing_graph[layer]:
                            if other_layer in tech.routing_layers:
                                # Find minimal spacing of nodes such that spacing rule is asserted.
                                wire_width2 = tech.wire_width.get(other_layer, 0) // 2
                                min_spacing = spacing_graph[layer][other_layer]['min_spacing']
                                margin = (wire_width1 + wire_width2 + min_spacing)

                                # Find nodes that are closer than the minimal spacing.
                                # conflict_points = grid.neigborhood(point, margin, norm_ord=1)
                                potential_conflicts = [x for x in graph if x[0] == other_layer]
                                conflict_points = [p for (_, p) in potential_conflicts
                                                   if numpy.linalg.norm(numpy.array(p) - numpy.array(point),
                                                                        ord=1) < margin
                                                   ]
                                # Construct the lookup table for conflicting nodes.
                                for p in conflict_points:
                                    conflict_node = other_layer, p
                                    if conflict_node in graph:
                                        node_conflicts.add(conflict_node)
                    if node_conflicts:
                        conflicts[n] = node_conflicts

            # Find routing nodes that are reserved for a net. They cannot be used to route other nets.
            # (For instance the ends of a gate stripe.)
            reserved_nodes = defaultdict(set)
            for net, layer, terminals in terminals_by_net:
                for p in terminals:
                    n = layer, p
                    reserved = reserved_nodes[net]
                    reserved.add(n)
                    if n in conflicts:
                        for c in conflicts[n]:  # Also reserve nodes that would cause a spacing violation.
                            reserved.add(c)

            assert nx.is_connected(graph)

            # Invoke router and store result.
            routing_trees = self.router.route(graph,
                                                    signals=virtual_terminal_nodes,
                                                    reserved_nodes=reserved_nodes,
                                                    node_conflict=conflicts,
                                                    is_virtual_node_fn=_is_virtual_node_fn
                                                    )

            # TODO: Sanity check on result.
            return routing_trees

    def _08_draw_routes(self, shapes: Dict[str, db.Shapes], routing_trees: Dict[str, nx.Graph]):
        # Draw the layout of the routes.
        if routing_trees:
            for signal_name, rt in routing_trees.items():
                _draw_routing_tree(shapes, self._routing_graph, rt, self.tech, self.debug_routing_graph)

            # Merge the polygons on all layers.
            _merge_all_layers(shapes)


def _merge_all_layers(shapes: Dict[str, db.Shapes]):
    """
    Merge all polygons on all layers.
    """
    for layer_name, s in shapes.items():
        if '_label' not in layer_name:
            r = pya.Region(s)
            r.merge()
            s.clear()
            s.insert(r)


def _is_virtual_node_fn(n) -> bool:
    """
    Check if the node is virtual and has no direct physical representation.
    :param n:
    :return:
    """
    return n[0].startswith('virtual')


def _is_virtual_edge_fn(e) -> bool:
    """
    Check if the edge connects to at least one virtual node.
    :param n:
    :return:
    """
    a, b = e
    return _is_virtual_node_fn(a) or _is_virtual_node_fn(b)
