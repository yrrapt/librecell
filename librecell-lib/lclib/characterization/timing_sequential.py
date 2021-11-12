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
Characterization functions for sequential cells.
"""

import os
import tempfile
from itertools import count

from .simulation_subprocess import simulate_cell

from .util import *
from lccommon.net_util import get_subcircuit_ports
from .piece_wise_linear import *

from scipy import optimize
import math

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def ff_find_stabilization_time(
        cell_config: CellConfig,
        cell_ports: List[str],
        clock_input: str,
        data_in: str,
        data_out: str,
        supply_voltage: float,
        setup_time: float,
        clock_edge_polarity: bool,
        rising_data_edge: bool,
        clock_rise_time: float,
        clock_fall_time: float,
        simulator: str,
        output_load_capacitances: Dict[str, float] = None,
        time_step: float = 100.0e-12,
        max_simulation_time: float = 1e-7,
) -> float:
    """Find the time it takes for the data output signal of a flip-flop to stabilize after an active clock edge.
    This is used to estimate the order of magnitude of the switching speed which will be used in subsequent simulations.
    
    :param cell_ports: All circuit pins/ports in the same ordering as used in the SPICE circuit model.
    :param clock_input: Name of the clock pin ('related pin').
    :param data_in: Name of the data-in pin ('constrained pin').
    :param data_out: Name of the data-out pin.
    :param supply_voltage: Supply voltage in volts.
    :param clock_rise_time: Rise time of the clock signal.
    :param clock_fall_time: Fall time of the clock signal.
    :param output_load_capacitances: A dict with (net, capacitance) pairs which defines the load capacitances attached to certain nets.
    :param time_step: Simulation time step.
    :param simulator: Simulator to use.
    """

    cfg = cell_config.global_conf

    t_clock_edge = time_step * 16  # Rough estimate of when to start the clock edge.

    trip_points = cfg.trip_points

    # Generate the clock edge relative to which the delay will be measured.
    clock_edge = StepWave(
        start_time=t_clock_edge,
        polarity=clock_edge_polarity,
        transition_time=clock_rise_time if clock_edge_polarity else clock_fall_time,
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )
    clock_edge *= supply_voltage

    threshold = 0.5
    breakpoint = f"stop when v({data_out}) > {supply_voltage * threshold} when time > {t_clock_edge}"
    breakpoints = [breakpoint]

    simulation_title = "Estimate flip-flop propagation speed (CLK->D_Out)."

    time, voltages, currents = simulate_cell(
        cell_name=cell_config.cell_name,
        cell_ports=cell_ports,
        input_voltages=input_voltages,
        initial_voltages=initial_conditions,
        breakpoint_statements=breakpoints,
        output_voltages=[data_in, clock_input, data_out],
        output_currents=[cell_config.supply_net],
        simulation_file=sim_file,
        simulation_output_file=sim_output_file,
        max_simulation_time=max_simulation_time,
        simulation_title=simulation_title,
        temperature=cfg.temperature,
        output_load_capacitances=output_load_capacitances,
        time_step=time_step,
        setup_statements=cfg.setup_statements,
        ground_net=cfg.ground_net,
        simulator=simulator,
        debug=cfg.debug,
    )

    raise NotImplementedError()


def find_minimum_pulse_width(
        cell_config: CellConfig,
        ff_clock_edge_polarity: bool,
        clock_input: str,
        data_in: str,
        data_out: str,
        setup_time: float,
        clock_pulse_polarity: bool,
        rising_data_edge: bool,
        clock_rise_time: float,
        clock_fall_time: float,
        clock_pulse_width_guess: float,
        simulator: str,
        output_load_capacitances: Dict[str, float] = None,
        max_delay_estimation: float = 1e-7,
        static_input_voltages: Dict[str, float] = None,
) -> Tuple[float, float]:
    """Find the minimum clock pulse width such that the data is sampled.

    :param ff_clock_edge_polarity: Type of active clock edge. Rising edge: True, Falling edge: False.
    :param rising_data_edge:
    :param clock_pulse_polarity:
    :param setup_time:
    :param cell_config:
    :param clock_pulse_width_guess: Initial value for the minimal clock-pulse length.
        The search for the minimum will be started from there. The closer the guess is, the
        faster the search will terminate.
    :param max_delay_estimation: Upper bound for the clock-to-output delay. For example 10e-9 s.
    :param clock_input: Name of the clock pin ('related pin').
    :param data_in: Name of the data-in pin ('constrained pin').
    :param data_out: Name of the data-out pin.
    :param clock_rise_time: Rise time of the input signal (clock and data).
    :param clock_fall_time: Fall time of the input signal (clock and data).
    :param output_load_capacitances: A dict with (net, capacitance) pairs which defines the load capacitances attached to certain nets.
    :param static_input_voltages: Static input voltages.
        This can be used to set the voltage of static input signals such as scan-enable.
    :param simulator: The simulator to use.
    :return: Returns a tuple of the minimal clock pulse width such that the data signal is sampled and the corresponding delay.
        (min clock pulse width, clock-to-output delay)
    """
    assert isinstance(cell_config, CellConfig)
    cfg = cell_config.global_conf
    assert isinstance(cfg, CharacterizationConfig)
    workingdir = cell_config.workingdir

    logger.debug("Find minimum clock pulse width.")

    supply_voltage = cfg.supply_voltage

    # # SPICE include files.
    # includes = [f"cell_config.spice_netlist_file"]
    # includes += cfg.setup_statements
    # setup_statements = {'library' : [],
    #                     'include' : includes}

    # Create a list of include files.
    setup_statements = cfg.setup_statements
    setup_statements['include'] += [cell_config.spice_netlist_file]


    # Load capacitance statements.
    if output_load_capacitances is None:
        output_load_capacitances = dict()
    else:
        assert isinstance(output_load_capacitances, dict)

    if static_input_voltages is None:
        static_input_voltages = dict()
    else:
        static_input_voltages = static_input_voltages.copy()

    delay_function_cache = dict()

    def delay_function(clock_pulse_width: float, max_allowed_delay: float = max_delay_estimation) -> float:
        """
        Compute the delay from the clock edge to the data output edge.
        If there's no edge at the data output within the maximal simulation time 'Inf' is returned.
        :param clock_pulse_width: Width of the clock pulse.
        :param max_allowed_delay: Maximum allowed clock-to-output delay. The simulation will be stopped
         when the output does not change after `max_allowed_delay` seconds. A delay of `Inf` will be returned.
        :return: Returns the delay from the clock edge to the data output edge or `Inf` if the data edge does not come
        during the maximal allowed delay time.
        """

        def delay_function_uncached(clock_pulse_width: float, max_allowed_delay: float = max_delay_estimation) -> float:

            # Generate the pulse waveform of the clock.
            clock_pulse_shape = PulseWave(
                start_time=0,
                duration=clock_pulse_width,
                polarity=clock_pulse_polarity,
                rise_time=clock_rise_time,
                fall_time=clock_fall_time,
                rise_threshold=cfg.trip_points.input_threshold_rise,
                fall_threshold=cfg.trip_points.input_threshold_fall
            )

            t_pulse1 = setup_time

            clock_pulse1 = clock_pulse_shape.delayed(t_pulse1)

            # Create delayed second pulse.
            time_between_pulses = 4 * pulse_width + 1e-9
            t_pulse2 = t_pulse1 + time_between_pulses
            clock_pulse2 = clock_pulse_shape.delayed(t_pulse2)

            clock_pulse = clock_pulse1 + clock_pulse2
            if not clock_pulse_polarity:
                clock_pulse -= 1

            # Set voltage.
            clock_pulse *= cfg.supply_voltage

            simulation_duration = t_pulse2 + clock_pulse_width + max_allowed_delay

            # Create input signal that first resets the flip-flop to a certain state at the first clock pulse
            # and then switches to the other rail such that the flip-flop changes state with the second clock pulse.

            input_toggle_time = setup_time + time_between_pulses / 2
            input_signal = StepWave(
                start_time=input_toggle_time,
                polarity=rising_data_edge,
                transition_time=1e-15,  # Almost instant.
            )
            input_signal *= cfg.supply_voltage

            # All input voltage signals.
            input_voltages = static_input_voltages.copy()
            input_voltages.update({
                cell_config.supply_net: supply_voltage,
                clock_input: clock_pulse,
                data_in: input_signal
            })

            # Initial voltages of output nodes.
            initial_conditions = {
                data_out: cfg.supply_voltage - input_signal(0)  # The inverse of data_in.
            }

            # Simulate until output signal clearly reaches the other rail.
            # Waiting until the signal reaches ~50% is not sufficient because
            # it might be in a meta stable state.
            threshold = 0.98 if rising_data_edge else 0.02
            cmp = ">" if rising_data_edge else "<"
            breakpoint_statement = f"stop when v({data_out}) {cmp} {supply_voltage * threshold} " \
                                   f"when time > {setup_time + time_between_pulses}"
            breakpoints = [breakpoint_statement]

            # Simulation script file path.
            file_name = f"lctime_min_clk_pulse_width_" \
                        f"{'pos_pulse' if clock_pulse_polarity else 'neg_pulse'}_" \
                        f"{'data_rising' if rising_data_edge else 'data_falling'}"
            sim_file = os.path.join(workingdir, f"{file_name}.sp")

            # Output file for simulation results.
            sim_output_file = os.path.join(workingdir, f"{file_name}_output.txt")
            # File for debug plot of the waveforms.
            sim_plot_file = os.path.join(workingdir, f"{file_name}_plot.svg")

            simulation_title = f"Find minimum clock pulse width: '{data_in}'-'{clock_input}'->'{data_out}', " \
                               f"pulse polarity={clock_pulse_polarity}."

            time, voltages, currents = simulate_cell(
                cell_name=cell_config.cell_name,
                cell_ports=cell_config.spice_ports,
                input_voltages=input_voltages,
                initial_voltages=initial_conditions,
                breakpoint_statements=breakpoints,
                output_voltages=[data_in, clock_input, data_out],
                output_currents=[cell_config.supply_net],
                simulation_file=sim_file,
                simulation_output_file=sim_output_file,
                max_simulation_time=simulation_duration,
                simulation_title=simulation_title,
                temperature=cfg.temperature,
                output_load_capacitances=output_load_capacitances,
                time_step=cfg.time_step,
                setup_statements=setup_statements,
                ground_net=cell_config.ground_net,
                simulator=simulator,
                debug=cfg.debug,
            )

            supply_current = currents['v'+cell_config.supply_net.lower()]
            input_voltage = voltages[data_in.lower()]
            clock_voltage = voltages[clock_input.lower()]
            output_voltage = voltages[data_out.lower()]

            # Cut away the signals of the setup procedure.
            start_time = input_toggle_time
                        
            assert start_time <= time[-1]
            start_index = np.arange(len(time))[time > start_time][0]
            time = time[start_index:]
            supply_current = supply_current[start_index:]
            input_voltage = input_voltage[start_index:]
            clock_voltage = clock_voltage[start_index:]
            output_voltage = output_voltage[start_index:]

            if cfg.debug_plots:
                # Plot data in debug mode.
                logger.debug("Create plot of waveforms: {}".format(sim_plot_file))
                import matplotlib
                # matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.close()
                plt.title("Clock to output delay")
                plt.plot(time, clock_voltage, label='clock')
                plt.plot(time, input_voltage, label='data_in')
                plt.plot(time, output_voltage, label='data_out')
                plt.plot(time, supply_current, label='supply_current')
                plt.legend()
                plt.savefig(sim_plot_file)
                # plt.show()
                plt.close()

            # Normalize
            clock_voltage /= supply_voltage
            input_voltage /= supply_voltage
            output_voltage /= supply_voltage

            # Turn a falling edge into a rising edge by flipping the signal.
            # This makes measurement of the delay easier.
            if not rising_data_edge:
                output_voltage = 1 - output_voltage

            # Get decision thresholds.
            if rising_data_edge:
                output_threshold = cfg.trip_points.output_threshold_rise
            else:
                output_threshold = cfg.trip_points.output_threshold_fall

            # Get logical values at start and end.
            logic_out_start = output_voltage[0] > output_threshold
            logic_out_end = output_voltage[-1] > output_threshold

            # The delay can only be measured if there is a rising edge in the output.
            if not logic_out_start and logic_out_end:
                # Output has rising edge
                # Get first clock edge.
                thresh_clk = cfg.trip_points.input_threshold_rise if clock_pulse_polarity else cfg.trip_points.input_threshold_fall
                t_active_clock_edge = transition_time(voltage=clock_voltage, time=time, n=0,
                                                      assert_one_crossing=True,
                                                      threshold=thresh_clk,
                                                      find_falling_edges=not ff_clock_edge_polarity,
                                                      find_rising_edges=ff_clock_edge_polarity)

                assert t_active_clock_edge is not None

                # Get first output data edge.
                thresh_data = cfg.trip_points.input_threshold_rise if rising_data_edge else cfg.trip_points.input_threshold_fall
                t_output_data_edge = transition_time(voltage=output_voltage, time=time, n=0,
                                                     threshold=thresh_data)
                assert t_output_data_edge is not None

                # Compute the delay from the clock edge to the output data edge.
                delay = t_output_data_edge - t_active_clock_edge
            else:
                # There's no edge in the output.
                delay = float('Inf')

            return delay

        # Check cache.
        cache_tag = (clock_pulse_width, max_allowed_delay)
        result = delay_function_cache.get(cache_tag)
        if result is None:
            result = delay_function_uncached(clock_pulse_width, max_allowed_delay)
            delay_function_cache[cache_tag] = result

        return result

    pulse_width = clock_pulse_width_guess
    delay_estimation = max_delay_estimation  # Estimation of the order of magnitude of the clock-to-output delay.
    # Find a pulse width that is long enough.
    while True:
        delay = delay_function(pulse_width)
        logger.debug(f"Pulse width = {pulse_width}, Delay = {delay}")
        if math.isinf(delay):
            pulse_width = pulse_width * 2
            assert pulse_width < 1e-6, "Does not converge."
        else:
            assert not math.isinf(delay)
            assert delay > 0
            delay_estimation = delay * 10  # Store the delay as an estimation of the delay.
            break

    # Find a pulse width such that the minimal clock-to-output delay is met.
    delay_prev = delay
    pulse_width = pulse_width * 2
    while True:
        delay = delay_function(pulse_width, delay_estimation)
        err = abs(delay - delay_prev)
        if err > 1e-12:
            pulse_width = pulse_width * 2
            assert pulse_width < 1e-6, "Does not converge."
        else:
            break
        delay_prev = delay

    # Remember the upper bound of the pulse width.
    upper_bound = pulse_width
    reference_delay = delay_function(upper_bound, delay_estimation)
    assert not math.isinf(reference_delay)

    # Find a pulse width that is too short.
    pulse_width = clock_pulse_width_guess
    while True:
        delay = delay_function(pulse_width, delay_estimation)
        logger.debug(f"Pulse width = {pulse_width}, Delay = {delay}")
        if not math.isinf(delay):
            pulse_width = pulse_width / 2
        else:
            break
    lower_bound = pulse_width

    logger.debug(f"Search for minimum pulse width in the interval [{lower_bound}, {upper_bound}].")

    max_delay = reference_delay + cfg.max_pushout_time

    def f(pulse_width) -> float:
        result = delay_function(pulse_width, delay_estimation) - max_delay
        logger.debug(f"f({pulse_width}) == {result}")
        return result

    assert f(lower_bound) > 0, f"f({lower_bound}) = {f(lower_bound)}"
    assert f(upper_bound) < 0, f"f({upper_bound}) = {f(upper_bound)}"

    min_pulse_width = optimize.brentq(f, lower_bound, upper_bound)
    assert isinstance(min_pulse_width, float)

    logger.debug(f"Minimal clock pulse: {min_pulse_width}s")
    corresponding_delay = delay_function(min_pulse_width, delay_estimation)
    assert isinstance(corresponding_delay, float)
    logger.debug(f"Corresponding delay: {corresponding_delay}")
    assert not math.isinf(corresponding_delay)

    return min_pulse_width, corresponding_delay


# def test_find_min_pulse_width():
#     trip_points = TripPoints(
#         input_threshold_rise=0.5,
#         input_threshold_fall=0.5,
#         output_threshold_rise=0.5,
#         output_threshold_fall=0.5,
#
#         slew_lower_threshold_rise=0.2,
#         slew_upper_threshold_rise=0.8,
#         slew_lower_threshold_fall=0.2,
#         slew_upper_threshold_fall=0.8
#     )
#
#     subckt_name = 'DFFPOSX1'
#
#     include_file = f'../../test_data/freepdk45/netlists_pex/{subckt_name}.pex.netlist'
#     model_file = f'../../test_data/freepdk45/gpdk45nm.m'
#
#     ports = get_subcircuit_ports(include_file, subckt_name)
#     print("Ports: ", ports)
#     data_in = 'D'
#     clock = 'CLK'
#     data_out = 'Q'
#     ground = 'GND'
#     supply = 'VDD'
#
#     input_rise_time = 0.000e-9
#     input_fall_time = 0.000e-9
#
#     temperature = 27
#     logger.info(f"Temperature: {temperature} C")
#
#     output_load_capacitances = {data_out: 0.06e-12}
#     logger.info(f"Output load capacitance: {output_load_capacitances} [F]")
#
#     time_step = 10e-12
#     logger.info(f"Time step: {time_step} s")
#
#     # TODO: find appropriate simulation_duration_hint
#     simulation_duration_hint = 250e-12
#
#     # SPICE include files.
#     includes = [include_file, model_file]
#
#     vdd = 1.1
#     logger.info(f"Supply voltage: {vdd} V")
#
#     setup_time = 1e-9  # Choose big enough such that initial disturbances settle down.
#     clock_pulse_polarity = True
#     rising_data_edge = True
#
#     # Voltage sources for input signals.
#     # input_sources = [circuit.V('in_{}'.format(inp), inp, circuit.gnd, 'dc 0 external') for inp in inputs]
#
#     pos_edge_flipflop = True
#
#     def _min_pulse_width(pulse_polarity: bool) -> float:
#         return find_minimum_pulse_width(
#             cell_name=subckt_name,
#             cell_ports=ports,
#             clock_input=clock,
#             data_in=data_in,
#             data_out=data_out,
#             setup_time=setup_time,
#             clock_pulse_polarity=pulse_polarity,
#             rising_data_edge=rising_data_edge,
#             supply_voltage=vdd,
#             input_rise_time=input_rise_time,
#             input_fall_time=input_fall_time,
#             trip_points=trip_points,
#             temperature=temperature,
#             output_load_capacitances=output_load_capacitances,
#             time_step=time_step,
#             setup_statements=includes,
#             ground_net=ground,
#             supply_net=supply,
#             # debug=True
#         )
#
#     clock_pulse_polarity = False
#     min_pulse_width_low = _min_pulse_width(pulse_polarity=clock_pulse_polarity)
#     clock_pulse_polarity = True
#     min_pulse_width_high = _min_pulse_width(pulse_polarity=clock_pulse_polarity)
#
#     print(f"min_pulse_width_high = {min_pulse_width_high}")
#     print(f"min_pulse_width_low = {min_pulse_width_low}")
#     assert isinstance(min_pulse_width_high, float)


def get_clock_to_output_delay(
        cell_conf: CellConfig,
        clock_input: str,
        data_in: str,
        data_out: str,
        setup_time: float,
        hold_time: float,
        rising_clock_edge: bool,
        rising_data_edge: bool,
        input_rise_time: float,
        input_fall_time: float,
        clock_rise_time: float,
        clock_fall_time: float,
        clock_cycle_hint: float,
        output_load_capacitances: Dict[str, float] = None,
        setup_statements: List[str] = None,
        input_voltages: Dict[str, float] = None,
        include_slew: bool = False,
) -> float:
    """Get the delay from the clock edge to the output edge.

    :param input_voltages: Static input voltages.
        This can be used to set the voltage of static input signals such as scan-enable.
    :param clock_input: Name of the clock pin ('related pin').
    :param data_in: Name of the data-in pin ('constrained pin').
    :param data_out: Name of the data-out pin.
    :param setup_time: Delay from data input `D` edge to rising clock edge.
    :param hold_time: Delay from clock edge to data input edge.
    :param rising_clock_edge: `True` = use rising clock edge, `False` = use falling clock edge.
    :param rising_data_edge: `True` = use rising data edge, `False` = use falling data edge.
    :param input_rise_time: Rise time of the input signal (clock and data).
    :param input_fall_time: Fall time of the input signal (clock and data).
    :param output_load_capacitances: A dict with (net, capacitance) pairs which defines the load capacitances attached to certain nets.
    :param clock_cycle_hint: Run the simulation for at least this amount of time.
    :param setup_statements: SPICE statements that are included at the beginning of the simulation.
        This should be used for .INCLUDE and .LIB statements.
    :param include_slew: If set to True return a tuple of (delay time, slew time).
    :return: Returns the delay from the clock edge to the data edge.
     Returns `Inf` if the output does not toggle within the maximum simulation time.
    """

    assert isinstance(cell_conf, CellConfig)
    assert cell_conf.workingdir is not None
    workingdir = cell_conf.workingdir

    cfg = cell_conf.global_conf
    assert isinstance(cfg, CharacterizationConfig)

    trip_points = cfg.trip_points
    assert isinstance(trip_points, TripPoints)

    supply_voltage = cfg.supply_voltage
    supply_net = cell_conf.supply_net
    ground_net = cell_conf.ground_net

    logger.debug("get_clock_to_output_delay() ...")

    # Load include files.
    if setup_statements is None:
        setup_statements = {'library'   :   [],
                            'include'   :   []}

    period = max(clock_cycle_hint, input_rise_time + input_fall_time, clock_rise_time + clock_fall_time)

    # Generate the wave form of the clock.
    # First a clock pulse makes sure that the right state is sampled into the cell.
    clock_pulse1 = PulseWave(
        start_time=period,
        duration=period,
        polarity=rising_clock_edge,
        rise_time=clock_rise_time,
        fall_time=clock_fall_time,
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )

    t_clock_edge = 4 * period + max(setup_time, 0)

    assert t_clock_edge > 0

    # Generate the clock edge relative to which the delay will be measured.
    clock_transition_time = clock_rise_time if rising_clock_edge else clock_fall_time
    clock_edge = StepWave(
        start_time=t_clock_edge,
        polarity=rising_clock_edge,
        transition_time=clock_transition_time,
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )

    # Sanity check:
    assert math.isclose(clock_edge(t_clock_edge),
                        trip_points.input_threshold_rise if rising_clock_edge
                        else trip_points.input_threshold_fall)

    clk_wave = clock_pulse1 + clock_edge

    if not rising_clock_edge:
        # Compensate for offset.
        clk_wave -= 1

    simulation_end = t_clock_edge + 4 * period

    # Create data pulse.
    logger.debug("Create data pulse.")
    input_wave = PulseWave(
        start_time=t_clock_edge - setup_time,
        duration=setup_time + hold_time,
        polarity=rising_data_edge,
        rise_time=input_rise_time,
        fall_time=input_fall_time,
        rise_threshold=trip_points.input_threshold_rise,
        fall_threshold=trip_points.input_threshold_fall
    )

    input_wave *= supply_voltage
    clk_wave *= supply_voltage

    if input_voltages is None:
        input_voltages = dict()
    else:
        input_voltages = input_voltages.copy()

    # Set the data and clock signals.
    input_voltages.update({
        supply_net: supply_voltage,
        clock_input: clk_wave,
        data_in: input_wave
    })

    # Load capacitance statements.
    if output_load_capacitances is None:
        output_load_capacitances = dict()
    else:
        assert isinstance(output_load_capacitances, dict)

    # Initial voltages of output nodes..
    initial_conditions = {
        data_out: 0 if rising_data_edge else supply_voltage
    }

    # Simulate only until output reaches threshold.
    # Compute stopping voltages of the output signal.
    if rising_data_edge:
        # Rising edge.
        # Add a margin on the threshold to simulate a bit longer.
        threshold = 1 - 0.1 * (1 - trip_points.output_threshold_rise)
    else:
        # Falling edge.
        # Subtract a margin on the threshold to simulate a bit longer.
        threshold = 0.1 * trip_points.output_threshold_fall

    # Set a breakpoint that is active only short before the actual clock edge.
    # This way it ignores signal changes during the initialization.
    cmp = '>' if rising_data_edge else '<'
    assert t_clock_edge - period / 2 > 0
    breakpoint_statement = f"stop when v({data_out}) {cmp} {supply_voltage * threshold} " \
                           f"when time > {t_clock_edge + clock_transition_time * 0.05}"

    breakpoints = [breakpoint_statement]

    # Simulation script file path.
    # file_name = f"lctime_clock_to_output_delay_" \
    #             f"{''.join((f'{net}={v}' for net, v in input_voltages.items()))}_" \
    #             f"{'clk_rising' if rising_clock_edge else 'clk_falling'}_" \
    #             f"{'data_rising' if rising_data_edge else 'data_falling'}"

    file_name = f"lctime_clock_to_output_delay_" \
                f"{'clk_rising' if rising_clock_edge else 'clk_falling'}_" \
                f"{'data_rising' if rising_data_edge else 'data_falling'}"
    sim_file = os.path.join(workingdir, f"{file_name}.sp")

    # Output file for simulation results.
    sim_output_file = os.path.join(workingdir, f"{file_name}_output.txt")
    # File for debug plot of the waveforms.
    sim_plot_file = os.path.join(workingdir, f"{file_name}_plot.svg")

    simulation_title = f"Measure constraint '{data_in}'-'{clock_input}'->'{data_out}', rising_clock_edge={rising_clock_edge}."

    # Choose time step adaptively relative to the smalles input transition time.
    # Upper bound: set by global config.
    time_step = max(cfg.time_step, min(clock_rise_time, input_rise_time) / 128)

    time, voltages, currents = simulate_cell(
        cell_name=cell_conf.cell_name,
        cell_ports=cell_conf.spice_ports,
        input_voltages=input_voltages,
        initial_voltages=initial_conditions,
        breakpoint_statements=breakpoints,
        output_voltages=[data_in, clock_input, data_out],
        output_currents=[supply_net],
        simulation_file=sim_file,
        simulation_output_file=sim_output_file,
        max_simulation_time=simulation_end,
        simulation_title=simulation_title,
        temperature=cfg.temperature,
        output_load_capacitances=output_load_capacitances,
        time_step=time_step,
        setup_statements=setup_statements,
        ground_net=ground_net,
        debug=cfg.debug,
    )

    supply_current = currents['v'+supply_net.lower()]
    input_voltage = voltages[data_in.lower()]
    clock_voltage = voltages[clock_input.lower()]
    output_voltage = voltages[data_out.lower()]

    if cfg.debug_plots:
        # Plot data in debug mode.
        logger.debug("Create plot of waveforms: {}".format(sim_plot_file))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.close()
        plt.title("Clock to output delay")
        plt.plot(time, clock_voltage, label='clock')
        plt.plot(time, input_voltage, label='data_in')
        plt.plot(time, output_voltage, label='data_out')
        plt.plot(time, supply_current, label='supply_current')
        plt.legend()
        plt.savefig(sim_plot_file)
        plt.close()

    # Start of interesting interval
    start_time = t_clock_edge - period / 2
    assert start_time < time[-1]
    start_index = np.arange(len(time))[time > start_time][0]
    # Cut away initialization signals.
    time = time[start_index:]
    clock_voltage = clock_voltage[start_index:]
    input_voltage = input_voltage[start_index:]
    output_voltage = output_voltage[start_index:]

    # import matplotlib.pyplot as plt
    # plt.plot(time, clock_voltage, label='clock')
    # plt.plot(time, input_voltage)
    # plt.plot(time, output_voltage)
    # plt.legend()
    # plt.show()

    # Normalize voltages (divide by VDD).
    clock_voltage /= supply_voltage
    input_voltage /= supply_voltage
    output_voltage /= supply_voltage

    # Turn a falling edge into a rising edge by flipping the signal.
    # This makes measurement of the delay easier.
    if not rising_data_edge:
        output_voltage = 1 - output_voltage

    # Get decision thresholds.
    if rising_data_edge:
        output_threshold = trip_points.output_threshold_rise
    else:
        # Invert threshold because the signal gets also normalized.
        output_threshold = 1 - trip_points.output_threshold_fall

    # Get logical values at start and end.
    logic_out_start = output_voltage[0] > output_threshold
    logic_out_end = output_voltage[-1] > output_threshold

    # The delay can only be measured if there is a rising edge in the output.
    if not logic_out_start and logic_out_end:
        # Output has rising edge
        delay = get_input_to_output_delay(time=time, input_signal=clock_voltage,
                                          output_signal=output_voltage, trip_points=trip_points)
        slew = get_slew_time(time=time, voltage=output_voltage, trip_points=trip_points)
    else:
        # There's no edge in the output. Delay is infinite.
        delay = float('Inf')
        slew = float('Inf')

    if include_slew:
        return delay, slew
    else:
        # Default:
        return delay


def characterize_flip_flop_setup_hold(
        cell_conf: CellConfig,

        data_in_pin: str,
        data_out_pin: str,
        clock_pin: str,
        clock_edge_polarity: bool,

        related_pin_transition: np.ndarray,
        constrained_pin_transition: np.ndarray,

        static_input_voltages: Dict[str, float] = None,
        output_load_capacitance: float = 0,

) -> Dict[str, np.ndarray]:
    """
    Find the setup and hold constraints between the `data_in_pin` and the `clock_pin` such
    that the output on `data_output_pin` is correct.
    This is done for all combinations of output capacitances and input transition times.

    :param cell_conf:
    :param data_in_pin: The constrained pin.
    :param data_out_pin: Data output pin of the flip-flop.
    :param clock_pin: The related pin.
    :param related_pin_transition: List of clock transition times.
    :param constrained_pin_transition: List of input transition times.
    :param static_input_voltages: Static input signals other than VDD/GND.
    This should include clear/preset and scan_enable signals, if there are any.
    :return:
    """

    def f(input_transition_time: float, clock_transition_time: float):
        result = measure_flip_flop_setup_hold(
            cell_conf=cell_conf,
            data_in_pin=data_in_pin,
            data_out_pin=data_out_pin,
            clock_pin=clock_pin,
            clock_edge_polarity=clock_edge_polarity,
            output_load_capacitances={data_out_pin: output_load_capacitance},
            data_rise_time=input_transition_time,
            data_fall_time=input_transition_time,
            clock_transition_time=clock_transition_time,
            static_input_voltages=static_input_voltages
        )

        return (
            # Setup
            result.dependent_setup_time_rise,
            result.dependent_setup_time_fall,
            result.dependent_setup_delay_rise,
            result.dependent_setup_delay_fall,
            # Hold
            result.dependent_hold_time_rise,
            result.dependent_hold_time_fall,
            result.dependent_hold_delay_rise,
            result.dependent_hold_delay_fall,
        )

    f_vec = np.vectorize(f, cache=True)

    xx, yy = np.meshgrid(constrained_pin_transition, related_pin_transition)

    # Evaluate timing on the input_slew*load_capacitance grid.
    (setup_time_rise, setup_time_fall, setup_delay_rise, setup_delay_fall,
     hold_time_rise, hold_time_fall, hold_delay_rise, hold_delay_fall) = f_vec(xx, yy)

    # TODO: Which delay should be chosen? Min, max, average?
    cell_rise = setup_delay_rise
    cell_fall = setup_delay_fall

    return {
        'related_pin_transition': related_pin_transition,
        'constrained_pin_transition': constrained_pin_transition,
        'setup_rise_constraint': setup_time_rise,
        'setup_fall_constraint': setup_time_fall,
        'hold_rise_constraint': hold_time_rise,
        'hold_fall_constraint': hold_time_fall,
        'cell_rise': cell_rise,
        'cell_fall': cell_fall,
    }


class FFSetupHoldResult:
    """
    Bundle setup/hold times and corresponding delays for single-edge triggered flip-flops.
    This values are valid for a certain input slew / output load combination.
    """

    def __init__(self):
        self.dependent_setup_time_rise = 0
        "Setup time for a rising data signal."
        self.dependent_setup_time_fall = 0
        "Setup time for a falling data signal."
        self.dependent_setup_delay_rise = 0
        "Data output delay for a rising data signal when the setup time is `dependent_setup_time_rise`."
        self.dependent_setup_delay_fall = 0
        "Data output delay for a falling data signal when the setup time is `dependent_setup_time_fall`."

        self.dependent_hold_time_rise = 0
        "Hold time for a rising data signal."
        self.dependent_hold_time_fall = 0
        "Hold time for a falling data signal."
        self.dependent_hold_delay_rise = 0
        "Data output delay for a rising data signal when the hold time is `dependent_hold_time_rise`."
        self.dependent_hold_delay_fall = 0
        "Data output delay for a falling data signal when the hold time is `dependent_hold_time_fall`."

        self.minimum_delay_rise = 0
        """
        Absolute minimal clock-to-output delay that can be achieved with large setup and hold windows.
        """
        self.minimum_delay_fall = 0
        """
        Absolute minimal clock-to-output delay that can be achieved with large setup and hold windows.
        """


def measure_flip_flop_setup_hold(
        cell_conf: CellConfig,
        data_in_pin: str,
        data_out_pin: str,
        clock_pin: str,
        clock_edge_polarity: bool,
        output_load_capacitances: Dict[str, float],
        data_rise_time: float,
        data_fall_time: float,
        clock_transition_time: float,
        hold_margin: float = 10e-12,
        setup_margin: float = 10e-12,
        static_input_voltages: Dict[str, float] = None
) -> FFSetupHoldResult:
    """
    Measure constraints (setup, hold) of a flip-flop.
    :param cell_conf:
    :param hold_margin: Safety margin to put on the hold times. In seconds. Default is 1e-12s.
        Should be larger than 0. Possibly in the order of a pico-second.
    :param setup_margin: Safety margin to put on the hold times. In seconds. Default is 1e-12s.
        Should be larger than 0. Possibly in the order of a pico-second.
    :return: Returns a `FFSetupHoldResult` object that bundles the results.
    """
    assert isinstance(cell_conf, CellConfig)
    cfg = cell_conf.global_conf

    assert hold_margin > 0
    assert setup_margin > 0

    # TODO: find appropriate simulation_duration_hint
    clock_cycle_hint = 1e-9
    logger.debug(f"simulation_duration_hint = {clock_cycle_hint}")

    clock_rise_time = clock_transition_time
    clock_fall_time = clock_transition_time

    logger.debug(f"Output load capacitance: {output_load_capacitances} [F]")

    # Create a list of include files.
    setup_statements = cfg.setup_statements
    setup_statements['include'] += [cell_conf.spice_netlist_file]

    vdd = cfg.supply_voltage

    # Cache for faster re-evaluation of `delay_f`
    cache = dict()

    def delay_f(
            setup_time: float,
            hold_time: float,
            rising_clock_edge: bool,
            rising_data_edge: bool
    ):
        """
        Wrapper around `get_clock_to_output_delay()`. Results are cached such that a further call with same arguments returns the
        cached value of the first call.
        :param setup_time:
        :param hold_time:
        :param rising_clock_edge:
        :param rising_data_edge:
        :return:
        """
        logger.debug(f"evaluate delay_f({setup_time}, {hold_time}, {rising_clock_edge}, {rising_data_edge})")

        cache_tag = (setup_time, hold_time, rising_clock_edge, rising_data_edge)
        result = cache.get(cache_tag)
        if result is None:
            result = get_clock_to_output_delay(
                cell_conf=cell_conf,
                clock_input=clock_pin,
                data_in=data_in_pin,
                data_out=data_out_pin,
                setup_time=setup_time,
                hold_time=hold_time,
                rising_clock_edge=rising_clock_edge,
                rising_data_edge=rising_data_edge,
                input_rise_time=data_rise_time,
                input_fall_time=data_fall_time,
                clock_rise_time=clock_rise_time,
                clock_fall_time=clock_fall_time,
                output_load_capacitances=output_load_capacitances,
                clock_cycle_hint=clock_cycle_hint,
                setup_statements=setup_statements,
                input_voltages=static_input_voltages,
            )
            cache[cache_tag] = result
        else:
            logger.debug('Cache hit.')
        return result

    def find_min_data_delay(rising_data_edge: bool, abstol: float = 1e-12) -> Tuple[float, Tuple[float, float]]:
        """ Find minimum clock->data delay (with large setup/hold window).

        Procedure is as follows: Setup and hold time are increased until the data delay reaches a stable value.

        The tolerance should be chosen in the order of the accepted absolute error.
        """

        # TODO: Find a estimate start value for setup and hold times.
        setup_time_guess = data_rise_time
        hold_time_guess = data_fall_time

        setup_time = setup_time_guess
        hold_time = hold_time_guess

        assert setup_time != 0, "Setup time guess cannot be 0."  # Does not terminate otherwise.
        assert hold_time != 0, "Hold time guess cannot be 0."  # Does not terminate otherwise.

        prev_delay = None
        delay = None
        ctr = count()
        for _ in ctr:
            delay = delay_f(setup_time, hold_time,
                            rising_clock_edge=clock_edge_polarity,
                            rising_data_edge=rising_data_edge)

            if prev_delay is not None and delay != float('Inf'):
                diff = abs(delay - prev_delay)
                if diff < abstol:
                    # Close enough.
                    break
            setup_time = setup_time * 2
            hold_time = hold_time * 2

            assert setup_time < 1e-6, "Does not converge."
            assert hold_time < 1e-6, "Does not converge."

            prev_delay = delay

        logger.debug(f"Minimum clock to data delay: {delay}. "
                    f"(Iterations = {next(ctr)}, "
                    f"setup_time = {setup_time}, hold_time = {hold_time}, "
                    f"absolute error = {diff})")

        # Return the minimum delay and setup/hold times that lead to it.
        # setup/hold times are divided by 2 because the previous values actually lead to a delay that is close enough.
        return delay, (setup_time / 2, hold_time / 2)

    # Find the minimum data delays.
    # They are used to determine the target data delays when finding
    # setup and hold times.
    logger.debug("Find minimal propagation delays.")
    tolerance = 1e-12
    min_rise_delay, (setup_guess_rise, hold_guess_rise) = find_min_data_delay(rising_data_edge=True, abstol=tolerance)
    min_fall_delay, (setup_guess_fall, hold_guess_fall) = find_min_data_delay(rising_data_edge=False, abstol=tolerance)

    logger.debug(f"min_rise_delay = {min_rise_delay}")
    logger.debug(f"min_fall_delay = {min_fall_delay}")

    logger.debug(f"setup_guess_rise = {setup_guess_rise}")
    logger.debug(f"hold_guess_rise = {hold_guess_rise}")

    logger.debug(f"max. allowed clock-to-output push-out time = {cfg.max_pushout_time}")

    # Define flip flop failure: FF fails if delay is larger than max_accepted_{rise,fall}_delay
    max_rise_delay = min_rise_delay + cfg.max_pushout_time
    max_fall_delay = min_fall_delay + cfg.max_pushout_time

    # Compute the closest delay from rising to falling data edge based on the slew.
    dr = cfg.trip_points.slew_upper_threshold_rise - cfg.trip_points.slew_lower_threshold_rise
    edge_duration_rise = data_rise_time / dr  # Time of edge going fully from 0 to 1.
    df = cfg.trip_points.slew_upper_threshold_fall - cfg.trip_points.slew_lower_threshold_fall
    edge_duration_fall = data_fall_time / df  # Time of edge going fully from 1 to 0.

    min_data_edge_separation = edge_duration_rise * (1 - cfg.trip_points.input_threshold_rise) + \
                               edge_duration_fall * (1 - cfg.trip_points.input_threshold_fall)
    # min_data_edge_separation *= 1.01
    # print(f"dr = {dr}")
    # print(f"df = {df}")
    # print(f"edge_duration_rise = {edge_duration_rise}")
    # print(f"min_separation = {min_data_edge_separation}")

    # Tolerances for bisection root finding algorithm.
    xtol = 1e-15
    rtol = 1e-6

    def find_min_setup(rising_data_edge: bool,
                       hold_time: float) -> Tuple[float, float]:
        """
        Find minimal setup time given a fixed hold time.
        Set `hold_time` to a very large value to find the independent minimal setup time.
        :param rising_data_edge: True = rising data edge, False = falling data edge.
        :param hold_time: Fixed hold time.
        :return:
        """
        max_delay = max_rise_delay if rising_data_edge else max_fall_delay
        setup_guess = setup_guess_rise if rising_data_edge else setup_guess_fall

        logger.debug(f"Find min. setup time. Hold time = {hold_time}. rising_data_edge = {rising_data_edge}")

        def f(setup_time: float) -> float:
            """
            Optimization function objective.
            For finding `setup_time` such that the delay equals the maximum allowed delay.
            :param setup_time:
            :return:
            """
            # print('eval f', setup_time)
            # assert setup_time + hold_time >= input_rise_time + input_fall_time
            delay = delay_f(setup_time=setup_time, hold_time=hold_time,
                            rising_clock_edge=clock_edge_polarity,
                            rising_data_edge=rising_data_edge)
            return delay - max_delay

        # Determine min and max setup time for binary search.
        # shortest = -hold_time + data_rise_time + data_fall_time
        shortest = -hold_time
        longest = max(setup_guess_rise, shortest)
        assert shortest <= longest
        a = f(shortest)
        b = f(longest)
        # Make sure that sign(f(a)) != sign(f(b)) such that the zero can be found with a binary search.
        while not a > 0:
            shortest = shortest / 2
            a = f(shortest)

        assert a > 0

        while not b < 0:
            longest = longest * 2
            assert longest < 1e-6, "Does not converge."
            b = f(longest)

        assert b < 0

        # if cfg.debug_plots:
        # Plot data in debug mode.
        # logger.debug("Create plot of waveforms: {}".format(sim_plot_file))
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        #
        # t_su = np.linspace(shortest, longest, num=100)
        # err = np.vectorize(f)(t_su)
        # plt.plot(t_su, err)
        # plt.show()

        min_setup_time = optimize.brentq(f, shortest, longest, xtol=xtol, rtol=rtol)
        assert isinstance(min_setup_time, float)
        if math.isclose(min_setup_time, shortest) or math.isclose(min_setup_time, longest):
            logger.warning("Result of binary search is on bounds. Optimal setup-time not found.")

        delay_err = f(min_setup_time)
        # Check if we really found the root of `f`.
        logger.debug(f"min_setup_time = {min_setup_time}, delay_err = {delay_err}, max_delay = {max_delay}")
        assert np.allclose(0, delay_err, atol=10e-12), "Failed to find solution for minimal setup time." \
                                                       " Try to decrease the simulation time step."

        return min_setup_time, delay_err + max_delay

    def find_min_hold(rising_data_edge: bool,
                      setup_time: float) -> Tuple[float, float]:
        """
        Find minimal hold time given a fixed setup time.
        Set `setup_time` to a very large value to find the independent minimal hold time.
        :param rising_data_edge: True = rising data edge, False = falling data edge.
        :param setup_time: Fixed setup time.
        :return: Minimal hold time.
        """
        max_delay = max_rise_delay if rising_data_edge else max_fall_delay
        hold_guess = hold_guess_rise if rising_data_edge else hold_guess_fall

        logger.debug(f"Find min. hold time. Setup time = {setup_time}. rising_data_edge = {rising_data_edge}")

        def f(hold_time: float) -> float:
            """
            Function to find zero.
            :param hold_time:
            :return:
            """
            # print('eval f', hold_time)
            delay = delay_f(setup_time=setup_time,
                            hold_time=hold_time,
                            rising_clock_edge=clock_edge_polarity,
                            rising_data_edge=rising_data_edge)
            return delay - max_delay

        # Determine min and max hold time for binary search.
        # shortest = -setup_time + data_rise_time + data_fall_time
        shortest = -setup_time
        longest = hold_guess
        a = f(shortest)
        b = f(longest)
        assert a > 0
        # Make sure that sign(f(a)) != sign(f(b)) such that the zero can be found with a binary search.
        while not b < 0:
            longest = longest * 2
            assert longest < 1e-6, "Does not converge."
            b = f(longest)

        assert b < 0

        min_hold_time = optimize.brentq(f, shortest, longest, xtol=xtol, rtol=rtol)
        assert isinstance(min_hold_time, float)
        delay_err = f(min_hold_time)
        # Check if we really found the root of `f`.
        logger.debug(f"min_hold_time = {min_hold_time}, delay_err = {delay_err}, max_delay = {max_delay}")
        assert np.allclose(0, delay_err, atol=10e-12), "Failed to find solution for minimal hold time." \
                                                       " Try to decrease the simulation time step."

        return min_hold_time, f(min_hold_time) + max_delay

    def find_min_setup_plus_hold(rising_data_edge: bool, setup_guess: float, hold_guess: float) -> Tuple[float, float]:
        """
        Find the smallest `setup_time + hold_time` such that the flip-flop is operational and the output delay
        corresponds to the maximum allowed delay.

        Proceeds as follows:
        1) Start with a guess of the time window. Set window center to 0.
        2) Shift the center of the window such that the output delay is minimized. Update the window center.
        3) Find the window width such that the resulting delay equals the maximum allowed delay. Update the window width.
        4) If tolerance is not met: Continue at 2. Otherwise return the result.

        :param rising_data_edge: True = rising data edge, False = falling data edge.
        :param setup_guess: Initial value for the setup time.
        :param hold_guess: Initial value for the hold time.
        :return:
        """
        max_delay = max_rise_delay if rising_data_edge else max_fall_delay

        logger.debug(f"Find min. setup plus hold time."
                    f"Initial setup = {setup_guess}, initial hold = {hold_guess}. "
                    f"rising_data_edge = {rising_data_edge}")

        def f(window_width: float, center: float) -> float:
            """
            Optimization function objective.
            Setup and hold are represented as a time window around the clock edge where the data signal needs
            to be stable.
            The clock edge is assumed to come at time 0. The shift relative to the clock edge
            is denoted by `center`.
            :param window_width: setup_time + hold_time
            :param center: (hold_time - setup_time) / 2
            :return:
            """
            setup_time = window_width / 2 - center
            hold_time = center + window_width / 2
            delay = delay_f(setup_time=setup_time, hold_time=hold_time,
                            rising_clock_edge=clock_edge_polarity,
                            rising_data_edge=rising_data_edge)
            return delay - max_delay

        center = 0
        window_width = setup_guess + hold_guess

        max_iter = 100  # Limit the number of iterations.
        for i in range(100):
            logger.debug(f"Find minimal setup+hold: iteration {i}")
            # Remember window width to find out when to terminate the loop.
            previous_width = window_width

            # Minimize the output delay by adjusting the center of the window.
            def f_center(center: float) -> float:
                return f(window_width, center)

            center_opt_result = optimize.minimize_scalar(
                f_center,
                bounds=[center - window_width / 2, center + window_width / 2],
                method='bounded',
                tol=xtol
            )
            assert center_opt_result.success, f"Optimization failed: {center_opt_result.message}"
            center = center_opt_result.x

            def f_width(window_width: float):
                return f(window_width, center)

            # Find window width such that the output delay is equal to the maximum allowed output delay.

            width_lower_bound = window_width / 2
            width_upper_bound = window_width

            a = f_width(width_lower_bound)
            b = f_width(width_upper_bound)

            # Find the two bounds for the bisection search.
            # Make sure that sign(f(a)) != sign(f(b)) such that the zero can be found with a binary search.
            while a <= 0:
                width_lower_bound *= 0.5
                a = f_width(width_lower_bound)

            while b >= 0:
                width_upper_bound *= 2
                b = f_width(width_upper_bound)

            assert b < 0 and a > 0

            new_window_width = optimize.brentq(f_width, width_lower_bound, width_upper_bound, xtol=xtol, rtol=rtol)
            assert isinstance(new_window_width, float)
            window_width = new_window_width

            step = window_width - previous_width

            if abs(step) < 1e-14:  # TODO: What tolerance to choose here?
                setup_time = window_width / 2 - center
                hold_time = center + window_width / 2
                logger.debug(f"Found minimal setup + hold: "
                             f"setup = {setup_time}, "
                             f"hold = {hold_time}, "
                             f"setup+hold={window_width}")
                return setup_time, hold_time

        assert False, f"Iteration limit reached: {max_iter}"

    if False:
        logger.info("Find minimal setup + hold time.")
        min_setup_plus_hold = find_min_setup_plus_hold(rising_data_edge=True,
                                                       setup_guess=setup_guess_rise,
                                                       hold_guess=hold_guess_rise
                                                       )
        min_setup, min_hold = min_setup_plus_hold
        logger.info(f"Minimal setup+hold: setup = {min_setup}, hold = {min_hold}")

        # TODO: Store minimal setup+hold

    logger.info("Measure unconditional minimal setup time.")

    """
    The unconditional minimal setup time is the minimal setup time that can be achieved
    when the hold time is infinitely long, i.e. when the input data signal remains
    stable forever after the clock edge.
    """
    hold_time_guess = max(hold_guess_rise, hold_guess_fall) * 8
    min_setup_time_uncond_rise, min_setup_delay_rise = find_min_setup(rising_data_edge=True,
                                                                      hold_time=hold_time_guess)
    min_setup_time_uncond_fall, min_setup_delay_fall = find_min_setup(rising_data_edge=False,
                                                                      hold_time=hold_time_guess)

    logger.info(f"unconditional min. setup time rise: {min_setup_time_uncond_rise}")
    logger.info(f"unconditional min. setup time fall: {min_setup_time_uncond_fall}")
    logger.info(f"max delays (rise): {min_setup_delay_rise}")
    logger.info(f"max delays (fall): {min_setup_delay_fall}")

    logger.info("Measure unconditional minimal hold time.")
    """
    The unconditional minimal hold time is the minimal hold time that can be achieved
    when the setup time is infinitely long, i.e. when the input data signal is already stable
    since an infinite time before the clock edge.
    """
    setup_time_guess = max(setup_guess_rise, setup_guess_fall) * 8  # TODO: How to choose the initial value?
    min_hold_time_uncond_rise, min_hold_delay_rise = find_min_hold(rising_data_edge=True,
                                                                   setup_time=setup_time_guess)
    min_hold_time_uncond_fall, min_hold_delay_fall = find_min_hold(rising_data_edge=False,
                                                                   setup_time=setup_time_guess)

    logger.info(f"unconditional min. hold time rise: {min_hold_time_uncond_rise}")
    logger.info(f"unconditional min. hold time fall: {min_hold_time_uncond_fall}")
    logger.info(f"max delays (rise): {min_hold_delay_rise}")
    logger.info(f"max delays (fall): {min_hold_delay_fall}")

    # Find dependent setup time.
    logger.info("Measure dependent minimal setup time.")
    dependent_setup_time_rise, dependent_setup_delay_rise = \
        find_min_setup(rising_data_edge=True,
                       hold_time=min_hold_time_uncond_rise + hold_margin)

    dependent_setup_time_fall, dependent_setup_delay_fall = \
        find_min_setup(rising_data_edge=False,
                       hold_time=min_hold_time_uncond_fall + hold_margin)

    # Find dependent hold time.
    logger.info("Measure dependent minimal hold time.")
    dependent_hold_time_rise, dependent_hold_delay_rise = \
        find_min_hold(rising_data_edge=True,
                      setup_time=min_setup_time_uncond_rise + setup_margin)

    dependent_hold_time_fall, dependent_hold_delay_fall = \
        find_min_hold(rising_data_edge=False,
                      setup_time=min_setup_time_uncond_fall + setup_margin)

    logger.info(f"dep. setup (rise, fall): {dependent_setup_time_rise}, {dependent_setup_time_fall}")
    logger.info(f"dep. setup delay (rise, fall): {dependent_setup_delay_rise}, {dependent_setup_delay_fall}")

    logger.info(f"dep. hold (rise, fall): {dependent_hold_time_rise}, {dependent_hold_time_fall}")
    logger.info(f"dep. hold delay (rise, fall): {dependent_hold_delay_rise}, {dependent_hold_delay_fall}")

    result = FFSetupHoldResult()

    result.dependent_setup_time_rise = dependent_setup_time_rise
    result.dependent_setup_time_fall = dependent_setup_time_fall
    result.dependent_setup_delay_rise = dependent_setup_delay_rise
    result.dependent_setup_delay_fall = dependent_setup_delay_fall

    result.dependent_hold_time_rise = dependent_hold_time_rise
    result.dependent_hold_time_fall = dependent_hold_time_fall
    result.dependent_hold_delay_rise = dependent_hold_delay_rise
    result.dependent_hold_delay_fall = dependent_hold_delay_fall

    result.minimum_delay_rise = min_rise_delay
    result.minimum_delay_fall = min_fall_delay

    return result
