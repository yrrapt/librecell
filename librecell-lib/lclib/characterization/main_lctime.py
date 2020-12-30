#
# Copyright (c) 2019-2020 Thomas Kramer.
#
# This file is part of librecell 
# (see https://codeberg.org/tok/librecell).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
Main program file for the `lctime` standard-cell characterization tool.
"""

import os
import argparse
import joblib
import tempfile

import liberty.parser as liberty_parser
from liberty.types import *

from PySpice.Unit import *

from ..logic.util import is_unate_in_xi
from ..liberty import util as liberty_util
from ..logic import functional_abstraction

from .util import *
from .timing_combinatorial import characterize_comb_cell
from .input_capacitance import characterize_input_capacitances

from copy import deepcopy

from lccommon import net_util
from lccommon.net_util import load_transistor_netlist, is_ground_net, is_supply_net
import networkx as nx
import sympy
from typing import Iterable


def _boolean_to_lambda(boolean: sympy.boolalg.Boolean):
    """
    Convert a sympy.boolalg.Boolean expression into a Python lambda function.
    :param boolean:
    :return:
    """
    simple = sympy.simplify(boolean)
    f = sympy.lambdify(boolean.atoms(), simple)
    return f


from PySpice.Spice.Parser import SpiceParser
import logging


def main():
    """
    Command-line tool for cell characterization.
    Currently only combinatorial cells are supported excluding tri-state cells.
    :return:
    """

    logger = logging.getLogger(__name__)
    logger.info("lctime main function")

    parser = argparse.ArgumentParser(
        description='Characterize the timing of a combinatorial cell based on a SPICE netlist. '
                    'The resulting liberty file will contain the data of the input liberty file '
                    'plus the updated characteristics of the selected cell.',
        epilog='Example: lctime --liberty specification.lib --cell INVX1 AND2X1 --spice netlists.sp -I '
               'transistor_model.m --output mylib.lib')

    parser.add_argument('-l', '--liberty', required=True, metavar='LIBERTY', type=str,
                        help='Liberty file. This must contain all necessary specifications '
                             'needed to characterize the cell.')

    parser.add_argument('--cell', required=True, metavar='CELL_NAME', type=str,
                        action='append',
                        nargs='+',
                        help='Names of cells to be characterized.')

    parser.add_argument('--spice', required=True, metavar='SPICE', type=str,
                        action='append',
                        nargs='+',
                        help='SPICE netlist containing a subcircuit with the same name as the cell.')

    parser.add_argument('-I', '--include', required=False, action='append', metavar='SPICE_INCLUDE', type=str,
                        help='SPICE files to include such as transistor models.')

    parser.add_argument('--calc-mode', metavar='CALC_MODE', type=str, choices=['worst', 'typical', 'best'],
                        default='typical',
                        help='Calculation mode for computing the default timing arc'
                             ' based on the conditional timing arcs. "worst", "typical" (average) or "best".')

    parser.add_argument('-o', '--output', required=True, metavar='LIBERTY_OUT', type=str, help='Output liberty file.')

    parser.add_argument('--workingdir', required=False, metavar='WORKDIR', type=str,
                        help="Directory for ngspice simulation scripts and raw results.")

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (more verbose logging and plotting waveforms).')

    # Parse arguments
    args = parser.parse_args()

    DEBUG = args.debug
    log_level = logging.DEBUG if DEBUG else logging.INFO

    if DEBUG:
        log_format = '%(module)16s %(levelname)8s: %(message)s'
    else:
        # Also output name of function in DEBUG mode.
        log_format = '%(module)16s %(funcName)16s %(levelname)8s: %(message)s'

    logging.basicConfig(format=log_format, level=log_level)

    workingdir = args.workingdir
    if workingdir is None:
        workingdir = tempfile.mkdtemp(prefix="lctime-")

    # Get list of cell names to be characterized.
    cell_names = [n for names in args.cell for n in names]  # Flatten the nested list.

    # Get list of user-provided netlist files.
    netlist_files = [n for names in args.spice for n in names]  # Flatten the nested list.

    # Generate a lookup-table which tells for each cell name which netlist file to use.
    netlist_file_table: Dict[str, str] = dict()
    for netlist_file in netlist_files:
        logger.info("Load SPICE netlist: {}".format(netlist_file))
        parser = SpiceParser(path=netlist_file)
        for sub in parser.subcircuits:
            if sub.name in netlist_file_table:
                # Abort if a sub circuit is defined in multiple netlists.
                logger.warning(
                    f"Sub-circuit '{sub.name}' is defined in multiple netlists: {netlist_file_table[sub.name]}, {netlist_file}")
                exit(1)
            netlist_file_table[sub.name] = netlist_file

    # Test if all cell names can be found in the netlist files.
    cell_names_not_found = set(cell_names) - netlist_file_table.keys()
    if cell_names_not_found:
        logger.error("Cell name not found in netlists: {}".format(", ".join(cell_names_not_found)))
        exit(1)

    # Load liberty file.
    lib_file = args.liberty
    logger.info("Reading liberty: {}".format(lib_file))
    with open(lib_file) as f:
        data = f.read()
    library = liberty_parser.parse_liberty(data)

    # Check if the delay model is supported.
    delay_model = library['delay_model']
    supported_delay_models = ['table_lookup']
    if delay_model not in supported_delay_models:
        msg = "Delay model not supported: '{}'. Must be one of {}.".format(delay_model,
                                                                           ", ".join(supported_delay_models))
        logger.error(msg)
        assert False, msg

    # Make independent copies of the library.
    new_library = deepcopy(library)
    # Strip all cell groups.
    new_library.groups = [g for g in new_library.groups if g.group_name != 'cell']

    # Load operation voltage and temperature.
    # TODO: load voltage/temperature from operating_conditions group
    supply_voltage = library['nom_voltage']
    temperature = library['nom_temperature']
    logger.info('Supply voltage = {:f} V'.format(supply_voltage))
    logger.info('Temperature = {:f} V'.format(temperature))

    # Units
    # TODO: choose correct unit from liberty file
    capacitance_unit_scale_factor = 1e12
    # TODO: get correct unit from liberty file.
    time_unit_scale_factor = 1e9

    # Get timing corner from liberty file.
    # Find definitions of operating conditions and sort them by name.
    operating_conditions_list = library.get_groups('operating_conditions')
    # Put into a dict by name.
    operating_conditions: Dict[str, Group] = {g.args[0]: g for g in operating_conditions_list}

    logger.info("Operating conditions: {}".format(set(operating_conditions.keys())))

    """
    TODO: Use the information from the operating conditions.
    Example:
    operating_conditions (MPSS) {
        calc_mode : worst ;
        process : 1.5 ;
        process_label : "ss" ;
        temperature : 70 ;
        voltage : 4.75 ;
        tree_type : worse_case_tree ;
    }
    """

    def _transistors2multigraph(transistors) -> nx.MultiGraph:
        """ Create a graph representing the transistor network.
            Each edge corresponds to a transistor, each node to a net.
        """
        G = nx.MultiGraph()
        for t in transistors:
            G.add_edge(t.source_net, t.drain_net, (t.gate_net, t.channel_type))
        assert nx.is_connected(G)
        return G

    # Get timing corner from liberty file.
    # TODO: let user overwrite it.
    calc_modes = {
        'typical': CalcMode.TYPICAL,
        'worst': CalcMode.WORST,
        'best': CalcMode.BEST,
    }

    # TODO: Make use of this.
    default_operating_conditions = library['default_operating_conditions']
    logger.info("Default operating conditions: {}".format(default_operating_conditions))

    assert args.calc_mode in calc_modes, "Unknown calculation mode: {}".format(args.calc_mode)

    calc_mode = calc_modes[args.calc_mode]
    logger.info("calc_mode: {}".format(calc_mode.name))

    # Read trip points from liberty file.
    trip_points = read_trip_points_from_liberty(library)

    logger.debug(trip_points)

    spice_includes = args.include if args.include else []
    if len(spice_includes) == 0:
        logger.warning("No transistor model supplied. Use --include or -I.")

    # TODO: No hardcoded data here!
    output_capacitances = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6]) * 1e-12  # pf
    input_transition_times = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2]) * 1e-9  # ns

    # Characterize all cells in the list.
    def characterize_cell(cell_name: str) -> Group:

        # Create working directory if it does not exist yet.
        cell_workingdir = os.path.join(workingdir, cell_name)
        if not os.path.exists(cell_workingdir):
            os.mkdir(cell_workingdir)

        # Get netlist and liberty group.
        netlist_file = netlist_file_table[cell_name]
        cell_group = select_cell(library, cell_name)
        # Check that the name matches.
        assert cell_group.args[0] == cell_name

        logger.info("Cell: {}".format(cell_name))
        logger.info("Netlist: {}".format(netlist_file))

        # Get information on pins
        input_pins, output_pins, output_functions_user = liberty_util.get_pin_information(cell_group)

        # Load netlist of cell
        # TODO: Load all netlists at the beginning.
        logger.info('Load netlist: %s', netlist_file)
        transistors_abstract, cell_pins = load_transistor_netlist(netlist_file, cell_name)
        io_pins = net_util.get_io_pins(cell_pins)

        # Detect power pins.
        # TODO: don't decide based only on net name.
        power_pins = [p for p in cell_pins if net_util.is_power_net(p)]
        assert len(power_pins) == 2, "Expected to have 2 power pins."
        vdd_pins = [p for p in power_pins if net_util.is_supply_net(p)]
        gnd_pins = [p for p in power_pins if net_util.is_ground_net(p)]
        assert len(vdd_pins) == 1, "Expected to find one VDD pin but found: {}".format(vdd_pins)
        assert len(gnd_pins) == 1, "Expected to find one GND pin but found: {}".format(gnd_pins)
        vdd_pin = vdd_pins[0]
        gnd_pin = gnd_pins[0]

        # Derive boolean functions for the outputs from the netlist.
        logger.info("Derive boolean functions for the outputs based on the netlist.")
        transistor_graph = _transistors2multigraph(transistors_abstract)
        output_functions_deduced = functional_abstraction.analyze_circuit_graph(graph=transistor_graph,
                                                                                pins_of_interest=io_pins,
                                                                                constant_input_pins={vdd_pin: True,
                                                                                                     gnd_pin: False},
                                                                                user_input_nets=input_pins)
        # Convert keys into strings (they are `sympy.Symbol`s now)
        output_functions_deduced = {output.name: function for output, function in output_functions_deduced.items()}
        output_functions_symbolic = output_functions_deduced

        # Log deduced output functions.
        for output_name, function in output_functions_deduced.items():
            logger.info("Deduced output function: {} = {}".format(output_name, function))

        # Merge deduced output functions with the ones read from the liberty file and perform consistency check.
        for output_name, function in output_functions_user.items():
            logger.info("User supplied output function: {} = {}".format(output_name, function))
            assert output_name in output_functions_deduced, "No function has been deduced for output pin '{}'.".format(
                output_name)
            # Consistency check: verify that the deduced output formula is equal to the one defined in the liberty file.
            equal = functional_abstraction.bool_equals(function, output_functions_deduced[output_name])
            if not equal:
                msg = "User supplied function does not match the deduced function for pin '{}'".format(output_name)
                logger.error(msg)

            if equal:
                # Take the function defined by the liberty file.
                # This might be desired because it is in another form (CND, DNF,...).
                output_functions_symbolic[output_name] = function

        # Convert deduced output functions into Python lambda functions.
        output_functions = {
            name: _boolean_to_lambda(f)
            for name, f in output_functions_symbolic.items()
        }

        # Sanity check.
        if len(input_pins) == 0:
            msg = "Cell has no input pins."
            logger.error(msg)
            assert False, msg

        # Sanity check.
        if len(output_pins) == 0:
            msg = "Cell has no output pins."
            logger.error(msg)
            assert False, msg

        # Add groups for the cell to be characterized.
        new_cell_group = deepcopy(select_cell(library, cell_name))

        # Strip away timing groups. They will be replaced by the new characterization.
        for pin_group in new_cell_group.get_groups('pin'):
            pin_group.groups = [g for g in pin_group.groups if g.group_name != 'timing']

        logger.info("Run characterization.")

        # TODO: Make time resolution parametrizable.
        time_resolution_seconds = 50e-12
        logger.info("Time resolution = {}s".format(time_resolution_seconds))

        # Measure input pin capacitances.
        logger.debug("Measuring input pin capacitances.")
        for input_pin in input_pins:
            logger.info("Measuring input capacitance: {}".format(input_pin))
            input_pin_group = new_cell_group.get_group('pin', input_pin)

            result = characterize_input_capacitances(
                cell_name=cell_name,
                input_pins=input_pins,
                active_pin=input_pin,
                output_pins=output_pins,
                supply_voltage=supply_voltage,
                trip_points=trip_points,
                timing_corner=calc_mode,
                spice_netlist_file=netlist_file_table[cell_name],
                spice_include_files=spice_includes,

                time_resolution=time_resolution_seconds,
                temperature=temperature,

                workingdir=cell_workingdir,

                debug=args.debug
            )

            input_pin_group['rise_capacitance'] = result['rise_capacitance'] * capacitance_unit_scale_factor
            input_pin_group['fall_capacitance'] = result['fall_capacitance'] * capacitance_unit_scale_factor
            input_pin_group['capacitance'] = result['capacitance'] * capacitance_unit_scale_factor

        # Measure timing for all input-output arcs.
        logger.debug("Measuring timing.")
        for output_pin in output_pins:
            output_pin_group = new_cell_group.get_group('pin', output_pin)

            # Insert boolean function of output.
            output_pin_group.set_boolean_function('function', output_functions_symbolic[output_pin])

            for related_pin in input_pins:
                logger.info("Timing arc: {} -> {}".format(related_pin, output_pin))

                # Get timing sense of this arc.
                timing_sense = is_unate_in_xi(output_functions[output_pin], related_pin).name.lower()
                logger.info("Timing sense: {}".format(timing_sense))

                result = characterize_comb_cell(
                    cell_name=cell_name,
                    input_pins=input_pins,
                    output_pin=output_pin,
                    related_pin=related_pin,
                    output_functions=output_functions,
                    supply_voltage=supply_voltage,
                    trip_points=trip_points,
                    timing_corner=calc_mode,

                    total_output_net_capacitance=output_capacitances,
                    input_net_transition=input_transition_times,

                    spice_netlist_file=netlist_file_table[cell_name],
                    spice_include_files=spice_includes,

                    time_resolution=time_resolution_seconds,
                    temperature=temperature,

                    workingdir=cell_workingdir,

                    debug=args.debug
                )

                # Get the table indices.
                # TODO: get correct index/variable mapping from liberty file.
                index_1 = result['total_output_net_capacitance'] * capacitance_unit_scale_factor
                index_2 = result['input_net_transition'] * time_unit_scale_factor
                # TODO: remember all necessary templates and create template tables.
                table_template_name = 'delay_template_{}x{}'.format(len(index_1), len(index_2))

                # Create liberty timing tables.
                timing_tables = []
                for table_name in ['cell_rise', 'cell_fall', 'rise_transition', 'fall_transition']:
                    table = Group(
                        table_name,
                        args=[table_template_name],
                    )

                    table.set_array('index_1', index_1)
                    table.set_array('index_2', index_2)
                    table.set_array('values', result[table_name] * time_unit_scale_factor)

                    timing_tables.append(table)

                    # Create the liberty timing group.
                    timing_group = Group(
                        'timing',
                        attributes={
                            'related_pin': [EscapedString(related_pin)],
                            'timing_sense': [timing_sense]
                        },
                        groups=timing_tables
                    )

                    # Attach timing group to output pin group.
                    output_pin_group.groups.append(timing_group)

        assert isinstance(new_cell_group, Group)
        return new_cell_group

    # Characterize cells in parallel.
    new_cell_groups = joblib.Parallel(n_jobs=-1, prefer='threads') \
        (joblib.delayed(characterize_cell)(cell_name) for cell_name in cell_names)

    for new_cell_group in new_cell_groups:
        new_library.groups.append(new_cell_group)

    with open(args.output, 'w') as f:
        logger.info("Write liberty: {}".format(args.output))
        f.write(str(new_library))
