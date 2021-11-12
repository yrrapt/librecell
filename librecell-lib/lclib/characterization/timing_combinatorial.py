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
Characterization functions for combinatorial cells.
"""

from typing import List, Dict, Callable, Optional

from itertools import product
import os
import tempfile
from .util import *
from .piece_wise_linear import *
from .simulation_subprocess import simulate_cell
from lccommon.net_util import get_subcircuit_ports

import logging

logger = logging.getLogger(__name__)


def characterize_comb_cell(
        input_pins: List[str],
        output_pin: str,
        related_pin: str,
        output_functions: Dict[str, Callable],

        total_output_net_capacitance: np.ndarray,
        input_net_transition: np.ndarray,

        cell_conf: CellConfig,

        simulator: str,
        constant_inputs: Dict[str, bool] = dict(),

) -> Dict[str, np.ndarray]:
    """
    Calculate the NDLM timing table of a cell for a given timing arc.
    :param constant_inputs: A truth-assignment for input pins that are static. Used for tri-state enable pins.
    :param cell_conf: Parameters and configuration for the characterization.
    :param input_net_transition: Transition times of input signals in seconds.
    :param total_output_net_capacitance: Load capacitance in Farads.
    :param input_pins: List of input pins.
    :param output_pin: The output pin of the timing arc.
    :param related_pin: The input pin of the timing arc.
    :param output_functions: A dict mapping output pin names to corresponding boolean functions.
    :param simulator: Specify the simulator to use.

    :return: Returns the NDLM timing tables wrapped in a dict:
    {'cell_rise': 2d-np.ndarray, 'cell_fall': 2d-np.ndarray, ... }
    """

    assert isinstance(cell_conf, CellConfig)
    assert isinstance(cell_conf.global_conf, CharacterizationConfig)
    cfg = cell_conf.global_conf

    inputs_inverted = cell_conf.complementary_pins.values()
    assert related_pin not in inputs_inverted, f"Active pin '{related_pin}' must not be an inverted pin of a differential pair."
    input_pins_non_inverted = [p for p in input_pins if p not in inputs_inverted]
    related_pin_inverted = cell_conf.complementary_pins.get(related_pin)

    # Find ports of the SPICE netlist.
    ports = get_subcircuit_ports(cell_conf.spice_netlist_file, cell_conf.cell_name)
    logger.debug("Subcircuit ports: {}".format(", ".join(ports)))

    vdd = cfg.supply_voltage

    # TODO
    # Maximum simulation time.
    time_max = cfg.time_step * 1e4

    # Find function to summarize different timing arcs.
    # TODO: Make this directly parametrizable by caller.
    reduction_function = {
        CalcMode.WORST: max,
        CalcMode.BEST: min,
        CalcMode.TYPICAL: np.mean
    }[cfg.timing_corner]

    # Create a list of include files.
    setup_statements = cfg.setup_statements
    setup_statements['include'] += [cell_conf.spice_netlist_file]

    # Get all input nets that are not toggled during a simulation run.
    logger.debug("Get all input nets that are not toggled during a simulation run.")
    static_input_nets = [i for i in input_pins_non_inverted if i != related_pin]
    # Get a list of all input combinations that will be used for measuring conditional timing arcs.
    num_inputs = len(static_input_nets)
    static_inputs = list(product(*([[0, 1]] * num_inputs)))

    # Get boolean function of output pin.
    assert output_pin in output_functions, \
        "Boolean function not defined for output pin '{}'".format(output_pin)
    output_function = output_functions[output_pin]

    def f(input_transition_time: float, output_cap: float):
        """
        Evaluate cell timing at a single input-transition-time/output-capacitance point.
        :param input_transition_time: Transition time (slew) of the active input signal.
        :param output_cap: Load capacitance at the output.
        :return:
        """

        # Empty results.
        rise_transition_durations = []
        fall_transition_durations = []
        rise_delays = []
        fall_delays = []

        rise_powers = []
        fall_powers = []

        for static_input in static_inputs:

            # Check if the output is controllable with this static input.
            bool_inputs = {net: value > 0 for net, value in zip(static_input_nets, static_input)}

            bool_inputs[related_pin] = False
            output_when_false = output_function(**bool_inputs)
            bool_inputs[related_pin] = True
            output_when_true = output_function(**bool_inputs)

            if output_when_false == output_when_true:
                # The output will not change when this input is changed.
                # Simulation of this conditional arc can be skipped.
                logger.debug("Simulation skipped for conditional arc (output does not toggle): {}".format(bool_inputs))
                continue

            # Get voltages at static inputs.
            input_voltages = {net: cfg.supply_voltage * value for net, value in zip(static_input_nets, static_input)}
            # Add supply voltage.
            input_voltages[cell_conf.supply_net] = cfg.supply_voltage
            # Add input voltages for inverted inputs of differential pairs.
            for p in static_input_nets:
                inv = cell_conf.complementary_pins.get(p)
                if inv is not None:
                    assert inv not in input_voltages
                    # Add the inverted input voltage.
                    input_voltages[inv] = cfg.supply_voltage - input_voltages[p]

            # Add constant voltages (tri-state enable pin).
            for pin, value in constant_inputs.items():
                input_voltages[pin] = cfg.supply_voltage if value else 0.0

            logger.debug("Static input voltages: {}".format(input_voltages))

            for input_rising in [True, False]:

                # Simulation script file path.
                file_name = f"lctime_combinational_" \
                            f"slew={input_transition_time}_" \
                            f"load={output_cap}" \
                            f"{''.join((f'{net}={v}' for net, v in input_voltages.items() if isinstance(v, float)))}_" \
                            f"{'rising' if input_rising else 'falling'}"
                sim_file = os.path.join(cfg.workingdir, f"{file_name}.sp")

                # Output file for simulation results.
                sim_output_file = os.path.join(cfg.workingdir, f"{file_name}_output.txt")
                # File for debug plot of the waveforms.
                sim_plot_file = os.path.join(cfg.workingdir, f"{file_name}_plot.svg")

                bool_inputs[related_pin] = input_rising
                expected_output = output_function(**bool_inputs)
                initial_output_voltage = 0 if expected_output else vdd

                # # Get voltages at static inputs.
                # input_voltages = {net: vdd * value for net, value in zip(static_input_nets, static_input)}
                # # Add supply voltage.
                # input_voltages[supply_net] = supply_voltage
                #
                # # Add input voltages for inverted inputs of differential pairs.
                # for p in static_input_nets:
                #     inv = complementary_pins.get(p)
                #     if inv is not None:
                #         assert inv not in input_voltages
                #         # Add the inverted input voltage.
                #         input_voltages[inv] = supply_voltage - input_voltages[p]
                #
                # logger.debug("Voltages at static inputs: {}".format(input_voltages))

                # Get stimulus signal for related pin.
                input_wave = StepWave(start_time=0, polarity=input_rising,
                                      rise_threshold=0,
                                      fall_threshold=1,
                                      transition_time=input_transition_time)
                input_wave.y = input_wave.y * vdd
                input_voltages[related_pin] = input_wave
                # Get stimulus signal for the inverted pin (if any).
                if related_pin_inverted:
                    input_wave_inverted = vdd - input_wave
                    input_voltages[related_pin_inverted] = input_wave_inverted

                    # Get the breakpoint condition.
                if expected_output:
                    breakpoint_statement = f"stop when v({output_pin}) > {vdd * 0.99}"
                else:
                    breakpoint_statement = f"stop when v({output_pin}) < {vdd * 0.01}"
                breakpoint_statements = [breakpoint_statement]

                # Initial node voltages.
                initial_conditions = {
                    output_pin: initial_output_voltage
                }
                simulation_title = f"Timing simulation for pin '{related_pin}', input_rising={input_rising}."

                time, voltages, currents = simulate_cell(
                    cell_name=cell_conf.cell_name,
                    cell_ports=ports,
                    input_voltages=input_voltages,
                    initial_voltages=initial_conditions,
                    breakpoint_statements=breakpoint_statements,
                    output_voltages=[related_pin, output_pin],
                    output_currents=[cell_conf.supply_net],
                    simulation_file=sim_file,
                    simulation_output_file=sim_output_file,
                    max_simulation_time=time_max,
                    simulation_title=simulation_title,
                    temperature=cfg.temperature,
                    output_load_capacitances={output_pin: output_cap},
                    time_step=cfg.time_step,
                    setup_statements=setup_statements,
                    ground_net=cell_conf.ground_net,
                    simulator=simulator,
                    debug=cfg.debug,
                )

                # Retrieve data.
                try:
                    supply_current = currents['v'+cell_conf.supply_net.lower()]
                except:
                    pass
                try:
                    supply_current = currents['v'+cell_conf.supply_net.lower()+'#branch']
                except:
                    pass
                
                input_voltage = voltages[related_pin.lower()]
                output_voltage = voltages[output_pin.lower()]

                if cfg.debug_plots:
                    logger.debug("Create plot of waveforms: {}".format(sim_plot_file))
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    plt.close()
                    plt.title(f"")
                    plt.plot(time, input_voltage, 'x-', label='input voltage')
                    plt.plot(time, output_voltage, label='output voltage')
                    plt.plot(time, supply_current, label='supply current')
                    plt.legend()
                    plt.savefig(sim_plot_file)
                    plt.close()

                # TODO: What unit does rise_power/fall_power have in liberty files???
                # Is it really power or energy?
                switching_power = np.mean(supply_current * vdd)
                switching_energy = switching_power * (time[-1] - time[0])

                v_thresh = 0.5 * vdd

                # Get input signal before switching and after.
                input_a = input_voltage[0] > v_thresh
                input_b = input_voltage[-1] > v_thresh
                assert input_a != input_b

                # Get output signal before switching and after.
                output_a = output_voltage[0] > v_thresh
                output_b = output_voltage[-1] > v_thresh

                # There should be an edge in the output signal.
                # Because the input signals have been chosen that way.
                assert output_a != output_b, "Supplied boolean function and simulation are inconsistent."

                # Check if output signal toggles.
                assert output_a != output_b, "Output did not toggle."

                output_rising = output_b

                # Normalize input/output such that both have a rising edge.
                input_voltage = input_voltage if input_rising else vdd - input_voltage
                output_voltage = output_voltage if output_rising else vdd - output_voltage
                # Normalize to range [0, ..., 1]
                input_voltage /= vdd
                output_voltage /= vdd

                # Check if signals are already stabilized after one `period`.
                # assert abs(input_voltage[0]) < 0.01, "Input signal not yet stable at start."
                # assert abs(1 - input_voltage[-1]) < 0.01, "Input signal not yet stable at end."
                if input_rising:
                    output_threshold = cfg.trip_points.output_threshold_rise
                else:
                    output_threshold = cfg.trip_points.output_threshold_fall
                assert abs(output_voltage[0]) <= output_threshold, "Output signal not yet stable at start."
                assert abs(1 - output_voltage[-1]) <= output_threshold, "Output signal not yet stable at end."

                # Calculate the output slew time: the time the output signal takes to change from
                # `slew_lower_threshold` to `slew_upper_threshold`.
                output_transition_duration = get_slew_time(time, output_voltage, trip_points=cfg.trip_points)

                # Calculate delay from the moment the input signal crosses `input_threshold` to the moment the output
                # signal crosses `output_threshold`.
                cell_delay = get_input_to_output_delay(time, input_voltage, output_voltage,
                                                       trip_points=cfg.trip_points)

                if output_rising:
                    rise_delays.append(cell_delay)
                    rise_transition_durations.append(output_transition_duration)
                else:
                    fall_delays.append(cell_delay)
                    fall_transition_durations.append(output_transition_duration)

                if input_rising:
                    rise_powers.append(switching_energy)
                else:
                    fall_powers.append(switching_energy)

        return (reduction_function(np.array(rise_delays)),
                reduction_function(np.array(fall_delays)),
                reduction_function(np.array(rise_transition_durations)),
                reduction_function(np.array(fall_transition_durations)),
                reduction_function(np.array(rise_powers)),
                reduction_function(np.array(fall_powers)),
                )

    f_vec = np.vectorize(f, cache=True)

    xx, yy = np.meshgrid(input_net_transition, total_output_net_capacitance)

    # Evaluate timing on the grid.
    cell_rise, cell_fall, rise_transition, fall_transition, rise_power, fall_power = f_vec(xx, yy)

    # Return the tables by liberty naming scheme.
    return {
        'total_output_net_capacitance': total_output_net_capacitance,
        'input_net_transition': input_net_transition,
        'cell_rise': cell_rise,
        'cell_fall': cell_fall,
        'rise_transition': rise_transition,
        'fall_transition': fall_transition,
        'rise_power': rise_power,
        'fall_power': fall_power
    }
