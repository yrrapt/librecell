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
Simple sub-process based simulation binding.
"""

import os
import subprocess
import queue
import threading
import logging
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from .piece_wise_linear import PieceWiseLinear

from analog_sim.spice.generic import create_sim_object


logger = logging.getLogger(__name__)


# def run_simulation(sim_file: str, ngspice_executable: str = 'ngspice'):
#     """
#     Invoke 'ngspice' to run the `sim_file`.
#     :param sim_file: Path to ngspice simulation file.
#     :return: Returns (stdout, stderr) outputs of ngspice.
#     """
#     logger.debug(f"Run simulation: {sim_file}")
#     try:
#         ret = subprocess.run([ngspice_executable, sim_file], capture_output=True)
#         # proc = subprocess.Popen([ngspice_executable, sim_file])
#         # logger.debug(f"Subprocess return value: {ret}")
#         if ret.returncode != 0:
#             ngspice_err_message = ret.stderr.decode("utf-8")
#             logger.error(f"ngspice simulation failed: {ngspice_err_message}")
#             raise Exception(f"ngspice simulation failed: {ngspice_err_message}")

#         return ret.stdout.decode("utf-8"), ret.stderr.decode("utf-8")
#     except FileNotFoundError as e:
#         msg = f"SPICE simulator executable not found. Make sure it is in the current path: {ngspice_executable}"
#         logger.error(msg)
#         raise FileNotFoundError(msg)


def simulate_cell(
        cell_name: str,
        cell_ports: List[str],
        input_voltages: Dict[str, Union[PieceWiseLinear, float]],
        initial_voltages: Dict[str, float],
        breakpoint_statements: List[str],
        output_voltages: List[str],
        output_currents: List[str],
        simulation_file: str,
        simulation_output_file: str,
        max_simulation_time: float,
        simulator: str,
        simulation_title: str = "<UNTITLED SIMULATION>",
        temperature: float = 25,
        output_load_capacitances: Dict[str, float] = None,
        time_step: float = 100.0e-12,
        setup_statements: List[str] = None,
        ground_net: str = 'GND',
        debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Simulate a circuit with given input signals and measure voltages and currents.

    :param cell_name: Name of the cell to be characterized. Must match with the name used in netlist and liberty.
    :param cell_ports: All circuit pins/ports in the same ordering as used in the SPICE circuit model.
    :param input_voltages: Dictionary holding input voltages or waveforms per net.
    :param initial_voltages: Additional initial voltages per net name at the beginning of the simulation.
        By default initial voltages are derived from the input voltages. This is useful for specifying initial voltages of outputs.
    :param breakpoint_statements: List of ngspice breakpoint statements.
    :param output_voltages: List of voltages that should be in the output of this function.
    :param output_currents: List of sources whose current should be in the output of this function.
    :param simulation_file: Path to the SPICE file which will be generated.
    :param simulation_output_file: Path of the simulation output data.
    :param max_simulation_time: Maximal simulation time in seconds.
    :param simulation_title: Title of the simulation that will be written into the simulation file.
    :param temperature: Temperature of the simulation.
    :param output_load_capacitances: A dict with (net, capacitance) pairs which defines the load capacitances attached to certain nets.
    :param time_step: Simulation time step.
    :param setup_statements: SPICE statements that are included at the beginning of the simulation.
        This should be used for .INCLUDE and .LIB statements.
    :param ground_net: The name of the ground net.
    :param simulator: Specify the simulator to use, defaults to ngspice
    :param debug: Enable more verbose debugging output such as plots of the simulations.
    :return: Returns tuple (time, Dict[net, voltage], Dict[source, current]).
    The output voltages and currents correspond to the values specified in `output_voltages` and `output_currents`.
    """

    logger.debug(f"Simulate cell: {cell_name}")
    logger.debug(f"Cell ports: {cell_ports}")
    logger.debug(f"Initial voltages: {initial_voltages} [V]")
    logger.debug(f"Ground net: {ground_net}")
    logger.debug(f"Temperature: {temperature} C")
    logger.debug(f"Max. simulation time: {max_simulation_time} s")
    logger.debug(f"Simulator: {simulator}")

    # create the simulation object
    analog_sim_obj = create_sim_object(simulator)

    # Add include and library definitions.
    setup_statements_string = ''
    if len(setup_statements['library']) > 0:
        for statement in setup_statements['library']:
            setup_statements_string += "\n" + analog_sim_obj.netlist_library(statement[0], statement[1])

    if len(setup_statements['include']) > 0:
        for statement in setup_statements['include']:
            setup_statements_string += "\n" + analog_sim_obj.netlist_include(statement)

    input_voltages_static = dict()
    input_voltages_active = dict()

    # Split input signals into static and active signals
    for net, voltage in input_voltages.items():
        if isinstance(voltage, PieceWiseLinear):
            input_voltages_active[net] = voltage
        else:
            assert isinstance(voltage, float), "Voltage needs to be a float or a piecewise linear function."
            input_voltages_static[net] = voltage

    logger.debug("Static input voltages: {}".format(sorted(list(input_voltages_static.keys()))))
    logger.debug("Dynamic input voltages: {}".format(sorted(list(input_voltages_active.keys()))))

    def create_voltage_source_statement(net: str, voltage: Union[float, PieceWiseLinear]) -> str:
        """
        Create a SPICE statement for a voltage source driving the 'net' with a voltage.
        The voltage can be either a static value or a `PieceWiseLinear` function.
        """
        if isinstance(voltage, PieceWiseLinear):
            return analog_sim_obj.netlist_voltage_pwl(net, voltage.to_spice_pwl_string(), negative=ground_net, dc=0)
        elif isinstance(voltage, float):
            return analog_sim_obj.netlist_voltage_dc(net, voltage, negative=ground_net)
        else:
            assert False, "`voltage` must be either a float or {}".format(PieceWiseLinear)


    # Initial node voltages.
    initial_conditions = initial_voltages

    # Calculate initial conditions for all sources.
    for net, voltage in input_voltages.items():
        if isinstance(voltage, PieceWiseLinear):
            initial_conditions[net] = voltage(0)
        else:
            assert isinstance(voltage, float)
            initial_conditions[net] = voltage

    # Format breakpoint statements.
    breakpoint_statements = '\n'.join(breakpoint_statements)

    # Generate output definitions for voltages.
    output_voltage_defs = " ".join((f"v({v})" for v in output_voltages))

    # Can measure currents only through sources.
    # Check that specified output currents match with specified voltage sources.
    for current in output_currents:
        if current not in input_voltages:
            msg = "Can measure currents only through voltage sources. " \
                  f"'{current}' does not correspond to a source."
            logger.error(msg)
            assert False, msg

    # Generate output definitions for currents.
    output_current_defs = " ".join((f"i(v{source})" for source in output_currents))

    # Create ngspice simulation script.
    sim_netlist  = analog_sim_obj.netlist_comment(f"""librecell {__name__}""") + "\n"
    sim_netlist += analog_sim_obj.netlist_title(simulation_title) + "\n\n"
    sim_netlist += analog_sim_obj.netlist_temperature(temperature) + "\n\n"

    # Add include and library definitions.
    if len(setup_statements['library']) > 0:
        for statement in setup_statements['library']:
            sim_netlist += analog_sim_obj.netlist_library(statement[0], statement[1]) + "\n"

    if len(setup_statements['include']) > 0:
        for statement in setup_statements['include']:
            sim_netlist += analog_sim_obj.netlist_include(statement) + "\n"
    sim_netlist += "\n"

    # instantiate the DUT
    sim_netlist += f"""Xcircuit_under_test {" ".join(cell_ports)} {cell_name}\n\n"""

    # Load capacitance statements.
    sim_netlist += analog_sim_obj.netlist_comment("Output load capacitances.") + "\n"
    if output_load_capacitances is None:
        output_load_capacitances = dict()
    else:
        assert isinstance(output_load_capacitances, dict)

        for net, load in output_load_capacitances.items():
            sim_netlist += analog_sim_obj.netlist_capacitor(name         =  f"Cload_{net}", 
                                                            positive_net = net, 
                                                            negative_net = ground_net,
                                                            capacitance  = load) + "\n"
    sim_netlist += "\n"

    # Add the static voltages.
    sim_netlist += analog_sim_obj.netlist_comment("Static input and supply voltages.") + "\n"
    for net, voltage in input_voltages_static.items():
        sim_netlist += create_voltage_source_statement(net, voltage) + "\n"
    sim_netlist += analog_sim_obj.netlist_voltage_dc(ground_net, 0, negative='0') + "\n"
    sim_netlist += "\n"

    # Add the actice voltages.
    sim_netlist += analog_sim_obj.netlist_comment("Active input signals (clock & data_in).") + "\n"
    for net, wave in input_voltages_active.items():
        sim_netlist += create_voltage_source_statement(net, wave) + "\n"
    sim_netlist += "\n"

    # Add initial conditions
    sim_netlist += analog_sim_obj.netlist_comment("Initial conditions.") + "\n"
    sim_netlist += analog_sim_obj.netlist_comment("Also all voltages of DC sources must be here if they are needed to compute the initial conditions.") + "\n"
    sim_netlist += analog_sim_obj.netlist_initial_conditions(initial_conditions) + "\n\n"
    sim_netlist += analog_sim_obj.netlist_sim_tran(max_simulation_time, time_step, True) + "\n"
    sim_netlist += analog_sim_obj.netlist_end() + "\n"

    # Write the netlist to file
    analog_sim_obj.write_netlist(sim_netlist)

    # Start simulation.
    logger.debug("Run simulation.")
    # stdout, stderr = analog_sim_obj.run_simulation()
    analog_sim_obj.run_simulation()

    # extract the single dataset
    data = analog_sim_obj.simulation_data

    # Extract the time.
    # TODO should find a more efficient way to deal with complex numbers from analog_sim library
    time = np.array([np.real(_) for _ in data['time']['data']])

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

    return time, voltages, currents
