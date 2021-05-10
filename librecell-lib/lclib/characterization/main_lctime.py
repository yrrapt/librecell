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
from ..cell_types import Combinational, SingleEdgeDFF, Latch

from PySpice.Unit import *

from ..logic.util import is_unate_in_xi
from ..liberty import util as liberty_util
from ..logic import functional_abstraction
from ..logic import seq_recognition

from .util import *
from .timing_combinatorial import characterize_comb_cell
from .timing_sequential import *
from .input_capacitance import characterize_input_capacitances

from copy import deepcopy

from lccommon import net_util
from lccommon.net_util import load_transistor_netlist, is_ground_net, is_supply_net
import networkx as nx
import sympy.logic.boolalg

from PySpice.Spice.Parser import SpiceParser
import logging


def _boolean_to_lambda(boolean: boolalg.Boolean):
    """
    Convert a sympy.boolalg.Boolean expression into a Python lambda function.
    :param boolean:
    :return:
    """
    simple = sympy.simplify(boolean)
    f = sympy.lambdify(boolean.atoms(), simple)
    return f


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

    parser.add_argument('-L', '--library', required=False, action='append', metavar='SPICE_LIB', type=str,
                        help='SPICE .LIB statements defining each a path to the library and a library name.'
                             'Example: --library "/path/to/lib libraryName".')

    parser.add_argument('--calc-mode', metavar='CALC_MODE', type=str, choices=['worst', 'typical', 'best'],
                        default='typical',
                        help='Calculation mode for computing the default timing arc'
                             ' based on the conditional timing arcs. "worst", "typical" (average) or "best".')

    parser.add_argument('-o', '--output', required=True, metavar='LIBERTY_OUT', type=str, help='Output liberty file.')

    parser.add_argument('--workingdir', required=False, metavar='WORKDIR', type=str,
                        help="Directory for ngspice simulation scripts and raw results.")

    parser.add_argument('--output-loads', required=True, metavar='CAPACITANCES', type=str,
                        help="List of output load capacitances for the cells. In pico Farads."
                             " List must be quoted, elements must be separated by a comma."
                             " Example: '0.05, 0.1, 0.2'")

    parser.add_argument('--slew-times', required=True, metavar='SLEWTIMES', type=str,
                        help="List of slew times of the input signals nano seconds."
                             " List must be quoted, elements must be separated by a comma."
                             " Example: '0.05, 0.1, 0.2'")

    parser.add_argument('--analyze-cell-function', action='store_true',
                        help='Derive the logical function of the cell from the SPICE netlist (experimental).')

    parser.add_argument('--diff', required=False,
                        nargs="+",
                        metavar='DIFFERENTIAL_PATTERN',
                        type=str,
                        help='Specify differential inputs as "NonInverting,Inverting" tuples.'
                             'The placeholder "%" can be used like "%_P,%_N" or "%,%_Diff", ...')

    parser.add_argument('--time-step', default=10e-12,
                        metavar='TIME_STEP',
                        type=float,
                        help='Specify the simulation time-step in seconds. Default is 10e-12.')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (more verbose logging).')

    parser.add_argument('--debug-plots', action='store_true',
                        help='Create debug plots of simulation waveforms.')

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

    # Sanitize include paths.
    input_argument_error = False
    for path in spice_includes:
        if not os.path.isfile(path):
            logger.error(f"Include file does not exist: {path}")
            input_argument_error = True

    spice_libraries_raw: List[str] = args.library if args.library else []
    # Split library statements into path and library name.
    spice_libraries: List[Tuple[str, str]] = [tuple(s.strip() for s in l.split(" ", maxsplit=2))
                                              for l in spice_libraries_raw
                                              ]
    # Sanitize the library arguments.
    for lib, raw in zip(spice_libraries, spice_libraries_raw):
        if len(lib) != 2 or not lib[0] or not lib[1]:
            logger.error('Library statements must be of the format "/path/to/library libraryName". Found: "{}".'
                         .format(raw))
            exit(1)

        path, name = lib
        if not os.path.isfile(path):
            logger.error(f"Library file does not exist: {path}")
            input_argument_error = True

    # Exit if some input arguments were obviously invalid.
    if input_argument_error:
        logger.info("Exit because of invalid arguments.")
        exit(1)

    # .LIB statements
    library_statements = [f".LIB {path} {name}" for path, name in spice_libraries]

    # .INCLUDE statements
    include_statements = [f".include {i}" for i in spice_includes]

    setup_statements = library_statements + include_statements

    # Setup array of output capacitances and input slews.
    output_capacitances = np.array([float(s.strip()) for s in args.output_loads.split(",")]) * 1e-12  # pF
    input_transition_times = np.array([float(s.strip()) for s in args.slew_times.split(",")]) * 1e-9  # ns

    logger.info(f"Output capacitances [pF]: {output_capacitances * 1e12}")
    logger.info(f"Input slew times [ns]: {input_transition_times * 1e9}")

    # TODO: Make time resolution parametrizable.
    time_resolution_seconds = float(args.time_step)
    logger.info("Time resolution = {}s".format(time_resolution_seconds))
    if time_resolution_seconds <= 0:
        logger.error('Time steop must be larger than zero.')
        exit(1)
    if time_resolution_seconds > 1e-9:
        logger.warning(f"Timestep is larger than 1ns: {time_resolution_seconds}s")

    # Setup configuration struct.
    conf = CharacterizationConfig()
    conf.supply_voltage = supply_voltage
    conf.trip_points = trip_points
    conf.timing_corner = calc_mode
    conf.setup_statements = setup_statements
    conf.time_resolution = time_resolution_seconds
    conf.temperature = temperature
    conf.workingdir = workingdir
    conf.debug = args.debug
    conf.debug_plots = args.debug_plots

    # Characterize all cells in the list.
    def characterize_cell(cell_name: str) -> Group:

        # Create working directory if it does not exist yet.
        cell_workingdir = os.path.join(conf.workingdir, cell_name)
        if not os.path.exists(cell_workingdir):
            os.mkdir(cell_workingdir)

        # Get netlist and liberty group.
        netlist_file = netlist_file_table[cell_name]
        cell_group = select_cell(library, cell_name)
        # Check that the name matches.
        assert cell_group.args[0] == cell_name, "Cell name does not match."

        logger.info("Cell: {}".format(cell_name))
        logger.info("Netlist: {}".format(netlist_file))

        # Get information on pins
        input_pins, output_pins, output_functions_user = liberty_util.get_pin_information(cell_group)

        logger.info(f"Input pins as defined in liberty: {input_pins}")
        logger.info(f"Output pins as defined in liberty: {output_pins}")

        # Load netlist of cell
        # TODO: Load all netlists at the beginning.
        logger.info('Load netlist: %s', netlist_file)
        transistors_abstract, cell_pins = load_transistor_netlist(netlist_file, cell_name, force_lowercase=True)
        io_pins = net_util.get_io_pins(cell_pins)

        # Get pin ordering of spice circuit.
        spice_ports = get_subcircuit_ports(netlist_file, cell_name)
        logger.debug(f"Spice subcircuit ports: {spice_ports}")

        if len(transistors_abstract) == 0:
            msg = "No transistors found in cell. (The netlist must be flattened, sub-circuits are not resolved)"
            logger.error(msg)
            exit(1)

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

        # Sanity check: All pins defined in liberty must appear in the SPICE netlist.
        all_liberty_pins = set()
        for pin in cell_group.get_groups("pin"):
            assert isinstance(pin, liberty_parser.Group)
            pin_name = pin.args[0]
            all_liberty_pins.add(pin_name.lower())
            complementary_pin = pin.get("complementary_pin")
            if complementary_pin is not None:
                all_liberty_pins.add(complementary_pin.lower())
        all_spice_pins = set(cell_pins)
        pins_not_in_spice = sorted(all_liberty_pins - all_spice_pins)
        if pins_not_in_spice:
            logger.error(f"Pins defined in liberty but not in SPICE netlist: {', '.join(pins_not_in_spice)}")
            exit(1)

        # Convert the transistor network into its multi-graph representation.
        # This is used for a formal analysis of the network.
        transistor_graph = _transistors2multigraph(transistors_abstract)

        # Detect input nets.
        if args.analyze_cell_function:
            logger.debug("Detect input nets from the circuit.")
            detected_inputs = functional_abstraction.find_input_gates(transistor_graph)
            # Detect nets that must be inputs (connected to gates only but do not
            # appear in the list of pins in the SPICE circuit definition.
            inputs_missing_in_spice = detected_inputs - all_spice_pins
            if inputs_missing_in_spice:
                logger.warning(f"The circuit has gate nets that must be inputs "
                               f"but are not in the pin definition of the SPICE circuit: "
                               f"{', '.join(sorted(inputs_missing_in_spice))}")
            # Same check for pins declared in liberty template.
            inputs_missing_in_liberty = detected_inputs - all_liberty_pins
            if inputs_missing_in_liberty:
                logger.info(f"The circuit has gate nets that must be inputs "
                            f"but are not declared as a pin in the liberty template: "
                            f"{', '.join(sorted(inputs_missing_in_liberty))}")

            # Add detected input pins.
            diff = detected_inputs - set(input_pins)
            logger.info(f"Also include detected pins: {', '.join(sorted(diff))}")
            input_pins.extend(detected_inputs)

            # Find pins that are defined in the SPICE circuit but are not inputs nor power.
            maybe_outputs = all_spice_pins - set(input_pins) - set(power_pins)
            logger.info(f"Potential output pins: {', '.join(sorted(maybe_outputs))}")
            output_pins.append(maybe_outputs)

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

        # Extract differential pairs from liberty.
        logger.debug("Load complementary pins from liberty.")
        differential_inputs_liberty = dict()
        for pin in cell_group.get_groups("pin"):
            assert isinstance(pin, liberty_parser.Group)
            pin_name = pin.args[0]
            complementary_pin = pin.get("complementary_pin")
            if complementary_pin is not None:
                differential_inputs_liberty[pin_name] = complementary_pin

        # Match differential inputs.
        if args.diff is not None:
            logger.debug("Match complementary pins from user-defined pattern.")
            differential_inputs_from_pattern = find_differential_inputs_by_pattern(args.diff, input_pins)
        else:
            differential_inputs_from_pattern = dict()

        differential_inputs_liberty.update(differential_inputs_from_pattern)
        differential_inputs = differential_inputs_liberty

        # Sanity checks on complementary pins.
        # Complementary pin should not be defined as pin group in liberty file.
        for pin in cell_group.get_groups("pin"):
            assert isinstance(pin, liberty_parser.Group)
            pin_name = pin.args[0]
            if pin_name in differential_inputs.values():
                logger.warning(
                    f"Complementary pin is modelled in the liberty file but will not be characterized: {pin_name}")

        for noninv, inv in differential_inputs.items():
            logger.info(f"Differential input (+,-): {noninv}, {inv}")

        # Find all input pins that are not inverted inputs of a differential pair.
        inverted_pins = differential_inputs.values()
        input_pins_non_inverted = [p for p in input_pins if p not in inverted_pins]

        # Find from liberty if the cell is sequential.
        cell_type = None
        ff_group = cell_group.get_groups("ff")
        latch_group = cell_group.get_groups("latch")
        if ff_group and latch_group:
            logger.error("Cell contains a 'ff' and 'latch' description.")
            assert False, "Cannot characterize cells with both 'ff' and 'latch'."
        elif ff_group:
            if len(ff_group) != 1:
                assert False, "Cannot characterize cells with more than one 'ff' group."
            logger.info("'ff' group found. Cell is expected to be a flip-flop.")
            ff_group = ff_group[0]
            assert isinstance(ff_group, Group)

            # Get state names.
            iq, iqn = ff_group.args
            clocked_on = ff_group.get_boolean_function('clocked_on')
            next_state = ff_group.get_boolean_function('next_state')

            clear = ff_group.get_boolean_function('clear')
            preset = ff_group.get_boolean_function('preset')

            clear_preset_var1 = ff_group.get_boolean_function('clear_preset_var1')
            clear_preset_var2 = ff_group.get_boolean_function('clear_preset_var2')

            clocked_on_also = ff_group.get_boolean_function('clocked_on_also')
            power_down_function = ff_group.get_boolean_function('power_down_function')

            cell_type = SingleEdgeDFF()
            cell_type.clocked_on = clocked_on
            cell_type.next_state = next_state
            cell_type.async_preset = preset
            cell_type.async_clear = clear

        elif latch_group:
            if len(latch_group) != 1:
                assert False, "Cannot characterize cells with more than one 'latch' group."
            logger.info("'latch' group found. Cell is expected to be a latch.")
            cell_type = Latch()
        else:
            # No sequential element.
            cell_type = Combinational()

        if args.analyze_cell_function:
            # Derive boolean functions for the outputs from the netlist.
            logger.info("Derive boolean functions for the outputs based on the netlist.")

            abstracted_circuit = functional_abstraction.analyze_circuit_graph(
                graph=transistor_graph,
                pins_of_interest=io_pins,
                constant_input_pins={
                    vdd_pin: True,
                    gnd_pin: False},
                differential_inputs=differential_inputs,
                user_input_nets=None
            )

            if abstracted_circuit.latches:
                # There's some feedback loops in the circuit.

                # Try to recognize sequential cells.
                detected_cell_type = seq_recognition.extract_sequential_circuit(abstracted_circuit)

                if detected_cell_type:
                    print("Detected sequential circuit:")
                    print(detected_cell_type)

            else:
                logger.info("Detected purely combinational circuit.")
                detected_cell_type = Combinational()
                if cell_type is None:
                    cell_type = Combinational()

            if cell_type is None:
                cell_type = detected_cell_type
            else:
                # Sanity check: Detected cell type (combinational, latch, ff) must match with the liberty file.
                if type(detected_cell_type) is not type(cell_type):
                    msg = f"Mismatch: Detected cell type is {type(detected_cell_type)} " \
                          f"but liberty says {type(cell_type)}."
                    logger.error(msg)
                    assert False, msg

            output_functions_deduced = abstracted_circuit.outputs

            # Convert keys into strings (they are `sympy.Symbol`s now)
            output_functions_deduced = {output.name: comb.function for output, comb in output_functions_deduced.items()}
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
                logger.info("Check equality of boolean function in liberty file and derived function.")
                equal = functional_abstraction.bool_equals(function, output_functions_deduced[output_name])
                if not equal:
                    msg = "User supplied function does not match the deduced function for pin '{}'".format(output_name)
                    logger.error(msg)

                if equal:
                    # Take the function defined by the liberty file.
                    # This might be desired because it is in another form (CND, DNF,...).
                    output_functions_symbolic[output_name] = function
        else:
            # Skip functional abstraction and take the functions provided in the liberty file.
            output_functions_symbolic = output_functions_user

        # Convert deduced output functions into Python lambda functions.
        output_functions = {
            name: _boolean_to_lambda(f)
            for name, f in output_functions_symbolic.items()
        }

        # Add groups for the cell to be characterized.
        new_cell_group = deepcopy(select_cell(library, cell_name))

        # Strip away timing groups. They will be replaced by the new characterization.
        for pin_group in new_cell_group.get_groups('pin'):
            pin_group.groups = [g for g in pin_group.groups if g.group_name != 'timing']

        # Create missing pin groups.
        for pin in input_pins_non_inverted + output_pins:
            pin_group = new_cell_group.get_group('pin', pin)
            if pin_group is None:
                pin_group = Group('pin', pin)
                new_cell_group.groups.append(pin_group)

        # Set 'direction' attribute of input pins.
        for pin in input_pins_non_inverted:
            pin_group = new_cell_group.get_group('pin', pin)
            if 'direction' not in pin_group:
                pin_group['direction'] = 'input'

        # Set 'direction' attribute of output pins.
        for pin in output_pins:
            pin_group = new_cell_group.get_group('pin', pin)
            if 'direction' not in pin_group:
                pin_group['direction'] = 'output'

        # Create 'complementary_pin' attribute for the inverted pin of differential pairs.
        for input_pin in input_pins_non_inverted:
            input_pin_group = new_cell_group.get_group('pin', input_pin)
            # Create link to inverted pin for differential inputs.
            input_pin_inverted = differential_inputs.get(input_pin)
            if input_pin_inverted:
                input_pin_group['complementary_pin'] = [EscapedString(input_pin_inverted)]

        logger.info("Run characterization.")

        # Setup cell specific configuration.
        cell_conf = CellConfig()
        cell_conf.cell_name = cell_name
        cell_conf.global_conf = conf
        cell_conf.complementary_pins = differential_inputs
        cell_conf.ground_net = gnd_pin
        cell_conf.supply_net = vdd_pin
        cell_conf.workingdir = cell_workingdir
        cell_conf.spice_netlist_file = netlist_file_table[cell_name]
        cell_conf.spice_ports = spice_ports

        # Measure input pin capacitances.
        logger.debug(f"Measuring input pin capacitances of cell {cell_name}.")
        for input_pin in input_pins_non_inverted:
            # Input capacitances are not measured for the inverting inputs of differential pairs.
            logger.info("Measuring input capacitance: {} {}".format(cell_name, input_pin))
            input_pin_group = new_cell_group.get_group('pin', input_pin)

            result = characterize_input_capacitances(
                input_pins=input_pins,
                active_pin=input_pin,
                output_pins=output_pins,
                cell_conf=cell_conf
            )

            input_pin_group['rise_capacitance'] = result['rise_capacitance'] * capacitance_unit_scale_factor
            input_pin_group['fall_capacitance'] = result['fall_capacitance'] * capacitance_unit_scale_factor
            input_pin_group['capacitance'] = result['capacitance'] * capacitance_unit_scale_factor

        if isinstance(cell_type, Combinational):
            # Measure timing for all input-output arcs.
            logger.debug("Measuring combinational delay arcs.")
            for output_pin in output_pins:
                output_pin_group = new_cell_group.get_group('pin', output_pin)

                # Insert boolean function of output.
                output_pin_group.set_boolean_function('function', output_functions_symbolic[output_pin])

                for related_pin in input_pins_non_inverted:

                    related_pin_inverted = differential_inputs.get(related_pin)
                    if related_pin_inverted:
                        logger.info("Timing arc (differential input): ({}, {}) -> {}"
                                    .format(related_pin, related_pin_inverted, output_pin))
                    else:
                        logger.info("Timing arc: {} -> {}".format(related_pin, output_pin))

                    # Get timing sense of this arc.
                    timing_sense = is_unate_in_xi(output_functions[output_pin], related_pin).name.lower()
                    logger.info("Timing sense: {}".format(timing_sense))

                    result = characterize_comb_cell(
                        input_pins=input_pins,
                        output_pin=output_pin,
                        related_pin=related_pin,
                        output_functions=output_functions,

                        total_output_net_capacitance=output_capacitances,
                        input_net_transition=input_transition_times,

                        cell_conf=cell_conf
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
                    timing_attributes = {
                        'related_pin': [EscapedString(related_pin)],
                        'timing_sense': [timing_sense]
                    }

                    timing_group = Group(
                        'timing',
                        attributes=timing_attributes,
                        groups=timing_tables
                    )

                    # Attach timing group to output pin group.
                    output_pin_group.groups.append(timing_group)
        elif isinstance(cell_type, SingleEdgeDFF):
            logger.info("Characterize single-edge triggered flip-flop.")

            data_in_pin = 'D'
            data_out_pin = 'Q'
            clock_pin = 'CLK'

            clock_edge_polarity = True  # TODO: Move to cell_type.

            # min_clock_pulse = find_minimum_pulse_width(
            #     cell_config=cell_conf,
            #     clock_input=clock_pin,
            #     data_in=data_in_pin,
            #     data_out=data_out_pin,
            #     setup_time=100e-12,
            #     clock_pulse_polarity=False,
            #     rising_data_edge=False,
            #     clock_rise_time=10e-12,  # TODO: Something fails when 0.
            #     clock_fall_time=10e-12,
            #     output_load_capacitances={'Q': 0},
            #     clock_pulse_width_guess=100e-12,
            #     max_simulation_time=1e-8,
            #     static_input_voltages=None,
            # )
            # logger.info(f'min_clock_pulse = {min_clock_pulse}')
            # exit()

            result = characterize_flip_flop_setup_hold(
                cell_conf=cell_conf,
                data_in_pin=data_in_pin,
                data_out_pin=data_out_pin,
                clock_pin=clock_pin,
                clock_edge_polarity=clock_edge_polarity,
                clock_transition_time=1e-12,  # TODO

                total_output_net_capacitance=output_capacitances,
                input_net_transition=input_transition_times,

                static_input_voltages=None  # TODO: Disable preset/clear/scan_en.
            )

            # Get the table indices.
            # TODO: get correct index/variable mapping from liberty file.
            index_1 = result['total_output_net_capacitance'] * capacitance_unit_scale_factor
            index_2 = result['input_net_transition'] * time_unit_scale_factor
            # TODO: remember all necessary templates and create template tables.

            input_pin_group = new_cell_group.get_group('pin', data_in_pin)

            clock_edge = 'rising' if clock_edge_polarity else 'falling'

            for contraint_type in ['hold', 'setup']:
                # TODO: Create missing template tables.
                table_template_name = f'{constraint_type}_template_{len(index_1)}x{len(index_2)}'

                rise_constraint = Group('rise_constraint', args=[table_template_name])
                rise_constraint.set_array('index_1', index_1)
                rise_constraint.set_array('index_2', index_2)
                rise_constraint.set_array(
                    'values',
                    result[f'{contraint_type}_rise_constraint'] * time_unit_scale_factor
                )

                fall_constraint = Group('fall_constraint', args=[table_template_name])
                fall_constraint.set_array('index_1', index_1)
                fall_constraint.set_array('index_2', index_2)
                fall_constraint.set_array(
                    'values',
                    result[f'{contraint_type}_fall_constraint'] * time_unit_scale_factor
                )

                timing_group = Group(
                    'timing',
                    attributes={
                        'timing_type': [f'{contraint_type}_{clock_edge}'],
                        'related_pin': [EscapedString(clock_pin)]
                    },
                    groups=[rise_constraint, fall_constraint]
                )
                input_pin_group.groups.append(timing_group)

        elif isinstance(cell_type, Latch):
            logger.info("Characterize latch.")
            assert False, "Characterization of latches is not yet supported."
        else:
            assert False, f"Unsupported cell type: {type(cell_type)}"

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
