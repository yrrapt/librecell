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
Measurement of the input capacitance by driving the input pin with a constant current.
"""

import os
from typing import List, Optional

from itertools import product

from .util import *
from .piece_wise_linear import *
from .simulation_subprocess import *
from lccommon.net_util import get_subcircuit_ports
import tempfile
import logging

from analog_sim.spice.generic import create_sim_object

from scipy import interpolate

logger = logging.getLogger(__name__)


def characterize_input_capacitances(
        input_pins: List[str],
        active_pin: str,
        output_pins: List[str],
        cell_conf: CellConfig,
        simulator: str
):
    """
    Estimate the input capacitance of the `active_pin`.
    The estimation is done by simulating a constant current flowing into an input and measuring the
    time it takes for the input to go from high to low or low to high. This time multiplied by the current
    yields the transported charge which together with the voltage difference tells the capacitance.
    The measurement is done for all combinations of static inputs (all other inputs that are not measured).

    :param input_pins: List of all input pin names.
    :param active_pin: Name of the pin to be measured.
    :param output_pins: List of cell output pins.
    :param config: Parameters for the characterization.
    :param simulator: Specify the simulator to use

    """

    assert isinstance(cell_conf, CellConfig)
    assert isinstance(cell_conf.global_conf, CharacterizationConfig)
    cfg = cell_conf.global_conf

    inputs_inverted = cell_conf.complementary_pins.values()
    assert active_pin not in inputs_inverted, f"Active pin '{active_pin}' must not be an inverted pin of a differential pair."
    input_pins_non_inverted = [p for p in input_pins if p not in inputs_inverted]
    active_pin_inverted = cell_conf.complementary_pins.get(active_pin)

    logger.debug("characterize_input_capacitances()")
    # Find ports of the SPICE netlist.
    ports = get_subcircuit_ports(cell_conf.spice_netlist_file, cell_conf.cell_name)
    logger.debug("Subcircuit ports: {}".format(", ".join(ports)))

    logger.debug("Ground net: {}".format(cell_conf.ground_net))
    logger.debug("Supply net: {}".format(cell_conf.supply_net))

    vdd = cfg.supply_voltage
    logger.debug("Vdd: {} V".format(vdd))

    # create the simulation object
    analog_sim_obj = create_sim_object(simulator)


    # Add output load capacitance. Right now this is 0F.

    # Choose a maximum time to run the simulation.
    time_max = cfg.time_step * 1e3

    # Find function to summarize different timing arcs.
    reduction_function = {
        CalcMode.WORST: max,
        CalcMode.BEST: min,
        CalcMode.TYPICAL: np.mean
    }[cfg.timing_corner]
    logger.debug("Reduction function for summarizing multiple timing arcs: {}".format(reduction_function.__name__))

    logger.debug("Measuring input capacitance.")

    # Generate all possible input combinations for the static input pins.
    static_input_nets = [i for i in input_pins_non_inverted if i != active_pin]
    num_inputs = len(static_input_nets)

    static_inputs = list(product(*([[0, 1]] * num_inputs)))
    logger.debug(f"Number of static input combinations: {len(static_inputs)}")

    input_current = cfg.input_current_for_capacitance_measurement
    logger.debug("Input current: {}".format(input_current))

    # Loop through all combinations of inputs.
    capacitances_rising = []
    capacitances_falling = []
    for static_input in static_inputs:
        for input_rising in [True, False]:

            # Get voltages at static inputs.
            input_voltages = {net: cfg.supply_voltage * value for net, value in zip(static_input_nets, static_input)}

            # Add input voltages for inverted inputs of differential pairs.
            for p in static_input_nets:
                inv = cell_conf.complementary_pins.get(p)
                if inv is not None:
                    assert inv not in input_voltages
                    # Add the inverted input voltage.
                    input_voltages[inv] = cfg.supply_voltage - input_voltages[p]

            logger.debug("Static input voltages: {}".format(input_voltages))

            # Simulation script file path.
            file_name = f"lctime_input_capacitance_" \
                        f"{''.join((f'{net}={v}' for net, v in input_voltages.items()))}_" \
                        f"{'rising' if input_rising else 'falling'}"
            sim_file = os.path.join(cfg.workingdir, f"{file_name}.sp")

            # Output file for simulation results.
            sim_output_file = os.path.join(cfg.workingdir, f"{file_name}_output.txt")
            # File for debug plot of the waveforms.
            sim_plot_file = os.path.join(cfg.workingdir, f"{file_name}_plot.svg")

            # Switch polarity of current for falling edges.
            _input_current = input_current if input_rising else -input_current

            # Get initial voltage of active pin.
            initial_voltage = 0 if input_rising else vdd
            initial_voltage_inv = vdd - initial_voltage

            # Get the breakpoint condition.
            if input_rising:
                breakpoint_statement = f"stop when v({active_pin}) > {vdd * 0.9}"
            else:
                breakpoint_statement = f"stop when v({active_pin}) < {vdd * 0.1}"


            # Initial node voltages.
            initial_conditions = {
                active_pin: initial_voltage,
                cell_conf.supply_net: cfg.supply_voltage
            }
            # Add static input voltages
            initial_conditions.update(input_voltages)
            # Add initial voltage of inverted input pin (if any).
            if active_pin_inverted:
                initial_conditions[active_pin_inverted] = initial_voltage_inv



            # Create ngspice simulation script.
            sim_netlist  = analog_sim_obj.netlist_comment(f"""librecell {__name__}""") + "\n"
            sim_netlist += analog_sim_obj.netlist_title(f"""Measure input capacitance of pin {active_pin}""") + "\n\n"
            sim_netlist += analog_sim_obj.netlist_temperature(cfg.temperature) + "\n\n"


            # Add include and library definitions.
            if len(cfg.setup_statements['library']) > 0:
                for statement in cfg.setup_statements['library']:
                    sim_netlist += analog_sim_obj.netlist_library(statement[0], statement[1]) + "\n"

            if len(cfg.setup_statements['include']) > 0:
                for statement in cfg.setup_statements['include']:
                    sim_netlist += analog_sim_obj.netlist_include(statement) + "\n"
            sim_netlist += "\n"
            sim_netlist += f".include {cell_conf.spice_netlist_file}\n"

            # instantiate the DUT
            sim_netlist += f"""Xcircuit_under_test {" ".join(ports)} {cell_conf.cell_name}\n\n"""

            # add pin capacotance
            for pin in output_pins:
                sim_netlist += analog_sim_obj.netlist_capacitor(name         =  f"Cload_{pin}", 
                                                                positive_net = pin, 
                                                                negative_net = cell_conf.ground_net,
                                                                capacitance  = 0) + "\n"
            
            sim_netlist += analog_sim_obj.netlist_voltage_dc(cell_conf.supply_net, cfg.supply_voltage, negative=cell_conf.ground_net) + "\n"
            sim_netlist += analog_sim_obj.netlist_voltage_dc(cell_conf.ground_net, 0, negative='0') + "\n"


            # Create SPICE statements for the input current sources that drive the active pin.
            sim_netlist += analog_sim_obj.netlist_current_pulse(name        = cell_conf.ground_net,
                                                                value0      = 0, 
                                                                value1      = _input_current, 
                                                                delay       = 1e-9, 
                                                                rise_time   = 10e-12, 
                                                                fall_time   = 0, 
                                                                pulse_width = 100, 
                                                                negative    = active_pin) + "\n"

            if active_pin_inverted is not None:
                sim_netlist += analog_sim_obj.netlist_current_pulse(name        = active_pin_inverted,
                                                                    value0      = 0,
                                                                    value1      = -_input_current,
                                                                    delay       = 1e-9,
                                                                    rise_time   = 10e-12,
                                                                    fall_time   = 0,
                                                                    pulse_width = 100,
                                                                    negative    = cell_conf.ground_net) + "\n"
            
            # add static voltage source
            for net, voltage in input_voltages.items():
                sim_netlist += analog_sim_obj.netlist_voltage_dc(net, voltage, negative=cell_conf.ground_net) + "\n"
            sim_netlist += "\n"

            # set inital conditions
            sim_netlist += analog_sim_obj.netlist_initial_conditions(initial_conditions) + "\n\n"

            sim_netlist += analog_sim_obj.netlist_sim_tran(time_max, cfg.time_step, True) + "\n"
            sim_netlist += analog_sim_obj.netlist_end() + "\n"

            # Dump netlist.
            logger.debug(sim_netlist)

            # Write the netlist to file
            analog_sim_obj.write_netlist(sim_netlist)

            # Start simulation.
            logger.debug("Run simulation.")
            analog_sim_obj.run_simulation()

            # extract the single dataset
            # data = analog_sim_obj.simulation_data[next(iter(analog_sim_obj.simulation_data))]
            data = analog_sim_obj.simulation_data

            # Extract the time.
            # TODO should find a more efficient way to deal with complex numbers from analog_sim library
            time = np.array([np.real(_) for _ in data['time']['data']])
            # time = np.array([np.real(_) for _ in data['data']])

            # Collect the voltages.
            # TODO should find a more efficient way to deal with complex numbers from analog_sim library
            voltages = {}
            for key in data.keys():
                if key.startswith('v('):
                    voltages[key[2:-1]] = np.array([np.real(_) for _ in data[key]])
                if data[key]['units'] == 'voltage':
                    voltages[key] = np.array([np.real(_) for _ in data[key]['data']])

            # Collect the currents.
            # TODO should find a more efficient way to deal with complex numbers from analog_sim library
            currents = {}
            for key in data.keys():
                if key.startswith('i('):
                    currents[key[2:-1]] = np.array([np.real(_) for _ in data[key]])
                if data[key]['units'] == 'current':
                    currents[key] = np.array([np.real(_) for _ in data[key]['data']])

            # Select simulation results.
            input_voltage = voltages[active_pin.lower()]

            if cfg.debug_plots:
                logger.debug("Create plot of waveforms: {}".format(sim_plot_file))
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.close()
                plt.title(f"Measure input capacitance of pin {active_pin}.")
                plt.plot(time, input_voltage, label=active_pin)
                plt.legend()
                plt.savefig(sim_plot_file)
                plt.close()

            # Calculate average derivative of voltage by finding the slope of the line
            # through the crossing point of the voltage with the two thresholds.
            #
            # TODO: How to chose the thresholds?
            if input_rising:
                thresh1 = vdd * cfg.trip_points.slew_lower_threshold_rise
                thresh2 = vdd * cfg.trip_points.slew_upper_threshold_rise
                assert thresh1 < thresh2
            else:
                thresh1 = vdd * cfg.trip_points.slew_upper_threshold_fall
                thresh2 = vdd * cfg.trip_points.slew_lower_threshold_fall
                assert thresh1 > thresh2

            # Find transition times for both thresholds.
            transition_time1 = transition_time(input_voltage, time, threshold=thresh1, n=-1)
            transition_time2 = transition_time(input_voltage, time, threshold=thresh2, n=-1)
            assert transition_time2 > transition_time1

            # Compute deltas of time and voltage between the crossing of the two thresholds.
            f_input_voltage = interpolate.interp1d(x=time, y=input_voltage)
            dt = transition_time2 - transition_time1
            dv = f_input_voltage(transition_time2) - f_input_voltage(transition_time1)
            # dv = input_voltage[-1] - input_voltage[0]
            # dt = time[-1] - time[0]

            # Compute capacitance.
            capacitance = float(_input_current) / (float(dv) / float(dt))

            logger.debug("dV: {}".format(dv))
            logger.debug("dt: {}".format(dt))
            logger.debug("I: {}".format(input_current))
            logger.debug("Input capacitance {}: {} F".format(active_pin, capacitance))

            if input_rising:
                capacitances_rising.append(capacitance)
            else:
                capacitances_falling.append(capacitance)

    logger.debug("Characterizing input capacitances: Done")

    # Find max, min or average depending on 'reduction_function'.
    logger.debug(
        "Convert capacitances of all timing arcs into the default capacitance ({})".format(reduction_function.__name__))
    final_capacitance_falling = reduction_function(capacitances_falling)
    final_capacitance_rising = reduction_function(capacitances_rising)
    final_capacitance = reduction_function([final_capacitance_falling, final_capacitance_rising])

    return {
        'rise_capacitance': final_capacitance_falling,
        'fall_capacitance': final_capacitance_rising,
        'capacitance': final_capacitance
    }
