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
Utility functions for handling signals such as finding signal edges, measuring delays.
"""

import numpy as np
from scipy import interpolate, optimize
from enum import Enum
from collections import namedtuple
from liberty.types import Group
import logging
from typing import Dict, Iterable, Optional, List
import re

logger = logging.getLogger(__name__)


class CalcMode(Enum):
    WORST = 1
    TYPICAL = 2
    BEST = 3


TripPoints = namedtuple("TripPoints", [
    "input_threshold_rise",
    "input_threshold_fall",
    "output_threshold_rise",
    "output_threshold_fall",

    "slew_lower_threshold_rise",
    "slew_upper_threshold_rise",
    "slew_lower_threshold_fall",
    "slew_upper_threshold_fall"
])


class CellConfig:
    """
    Characterization settings specific for a single cell.
    """

    def __init__(self):
        self.global_conf: CharacterizationConfig = None
        """
        General characterization configuration.
        """

        self.cell_name: str = None
        """
        Name of the cell, consistent with the name in the liberty file.
        """

        self.spice_netlist_file: str = None
        """
        Path to SPICE netlist containing the subcircuit of the cell.
        """

        self.spice_ports: List[str] = None
        """
        Ordering of pins of the spice subcircuit that describes this cell.
        """

        self.complementary_pins: Dict[str, str] = dict()
        """Mapping of non-inverting pin name to its complementary pin name of differential pairs.
        Dict[non inverting pin, inverting pin]."""

        self.workingdir: Optional[str] = None
        "Directory where simulation files are put. Best on a ram-disk."

        self.ground_net: str = 'GND'
        "Name of the ground net."

        self.supply_net: str = 'VDD'
        "Name of the power supply net."


class CharacterizationConfig:
    """
    General settings for the standard-cell characterization runs and simulations.
    """

    def __init__(self):
        self.supply_voltage: float = 0.0
        "Supply voltage in volts."

        self.trip_points: TripPoints = None
        """
        Trip-point object which specifies the voltage thresholds of the logical values.
        """

        self.timing_corner: CalcMode = None
        """
        Specify whether to take the maximum, minimum or average capacitance value. (Over all static input combinations).
        
        One of TimingCorner.WORST, TimingCorner.BEST or TimingCorner.TYPICAL
        This defines how the default timing arc is calculated from all the conditional timing arcs.
        WORST: max
        BEST: min
        TYPICAL: np.mean
        """

        self.setup_statements: List[str] = list()
        """SPICE statements that are included at the beginning of the simulation. 
        This should be used for .INCLUDE and .LIB statements."""

        self.time_step: float = 1e-12
        "Time resolution of the simulation."

        self.temperature = 27
        "Temperature of the simulated circuit."

        self.workingdir: Optional[str] = None
        "Directory where simulation files are put. Best on a ram-disk."

        self.debug: bool = False
        "Enable more verbose debugging output."

        self.debug_plots: bool = False
        "Enable more verbose debugging output such as plots of the simulations."

        # self.roll_off_factor: float = 0.10
        # """
        # OBSOLETE
        # For characterization of setup/hold times. Must be a positive and non-zero value.
        #
        # Larger values lead to smaller setup/hold window but to increased delay.
        #
        # Define how much data delay increase is tolerated.
        # The minimal delay at a sequential cell is achieved when the input signal remains
        # stable in a very wide window around the clock edge. This enlarges
        # the required setup and hold time. To make a trade-off, a degradation of the
        # delay time is accepted by the `roll_off_factor`. If the minimal delay is `d`,
        # then a delay of `d*(1+roll_off_factor)` is targeted for the characterization
        # of setup and hold times.
        # """

        self.max_pushout_time: float = 10e-12
        """
        For characterization of setup/hold times. Must be a positive and non-zero value.
        Larger values lead to smaller setup/hold window but to increased delay.
        
        Define how much increase of the clock-to-output delay is tolerated.
        
        The minimal delay at a sequential cell is achieved when the input signal remains
        stable in a very wide window around the clock edge. This enlarges
        the required setup and hold time. To make a trade-off, a degradation of the 
        delay time is accepted by the `max_pushout_time`. If the minimal delay is `d`,
        then a delay of `d + max_pushout_time` is targeted for the characterization
        of setup and hold times.
        """

        self.input_current_for_capacitance_measurement = 10000e-9  # [A]
        """
        Define the current that is pushed into an input to measure the input capacitance.
        """


#
# # TODO: Add type hints for Python 3.6.
# class TripPoints(NamedTuple):
#     input_threshold_rise = 0.5
#     input_threshold_fall = 0.5
#     output_threshold_rise = 0.5
#     output_threshold_fall = 0.5
#
#     slew_lower_threshold_rise = 0.2
#     slew_upper_threshold_rise = 0.8
#     slew_lower_threshold_fall = 0.2
#     slew_upper_threshold_fall = 0.8


def is_rising_edge(voltage: np.ndarray, threshold: float = 0.5) -> bool:
    """ Check if the signal rises by comparing first and last value.
    :param voltage: Signal.
    :param threshold: Decision threshold for HIGH/LOW value.
    :return: True iff the signal rises.
    """
    return voltage[0] < threshold <= voltage[-1]


def is_falling_edge(voltage: np.ndarray, threshold: float = 0.5) -> bool:
    """ Check if the signal falls by comparing first and last value.
    :param voltage: Signal.
    :param threshold: Decision threshold for HIGH/LOW value.
    :return: True iff the signal falls.
    """
    return voltage[0] > threshold >= voltage[-1]


def transition_time(voltage: np.ndarray,
                    time: np.ndarray,
                    threshold: float,
                    n: int = -1,
                    assert_one_crossing: bool = False,
                    find_falling_edges: bool = True,
                    find_rising_edges: bool = True) -> Optional[float]:
    """ Find time of the n-th event when the signal crosses the threshold.
    :param find_rising_edges: Detect rising edges. Default is `True`.
    :param find_falling_edges: Detect falling edges. Default is `True`.
    :param voltage: np.ndarray holding voltage values.
    :param time: np.ndarray holding time values.
    :param threshold:
    :param n: Selects the event if there are multiple. 0: first event, -1: last event.
    :param assert_one_crossing: If set, then assert that the signal crosses the threshold exactly once.
    :return: Time when the signal crosses the threshold for the n-th time or `None` if there's no crossing.
    """

    y_shifted = voltage - threshold

    # Find zero-crossings.
    # 0: no zero crossing here
    # 1: crossing from negative to positive
    # -1: crossing from positive to negative
    transitions = np.sign(np.diff(np.sign(y_shifted)))
    if not find_falling_edges:
        transitions[transitions == -1] = 0 # Disable falling edges.
    if not find_rising_edges:
        transitions[transitions == 1] = 0 # Disable rising edges.
    index = np.arange(len(transitions))
    # Get indices of crossings.
    transition_indices = index[transitions != 0]

    if n >= len(transition_indices):
        # There's no such crossing.
        return None

    transition_idx = transition_indices[n]

    if assert_one_crossing:
        # Count number of zero-crossings. There should be exactly one.
        assert np.sum(transitions != 0) > 0, "Signal does not cross threshold."
        assert np.sum(transitions != 0) == 1, "Signal crosses threshold multiple times."

    is_rising = transitions[transition_idx] == 1

    if not is_rising:
        # Normalize to rising edge.
        y_shifted = -y_shifted

    # Interpolate: Find zero between the both samples.
    # y1 and y2 don't have the same sign. Find the zero-crossing inbetween.
    y1 = y_shifted[transition_idx]
    y2 = y_shifted[transition_idx + 1]
    assert y1 <= 0 <= y2
    t1 = time[transition_idx]
    t2 = time[transition_idx + 1]

    dydt = (y2 - y1) / (t2 - t1)
    # 0 = y1 + delta_t * dydt
    # delta_t = -y1/dydt
    delta_t = -y1 / dydt
    t_threshold_crossing = t1 + delta_t

    return t_threshold_crossing


def get_slew_time(time: np.ndarray, voltage: np.ndarray,
                  trip_points: TripPoints) -> float:
    """
    Calculate the slew time of `voltage` signal.
    :param time: Time axis.
    :param voltage:
    :param trip_points:
    :return: Slew time in same units as `time`.
    """

    if is_falling_edge(voltage, trip_points.slew_lower_threshold_fall):
        threshold1 = trip_points.slew_upper_threshold_fall
        threshold2 = trip_points.slew_lower_threshold_fall
    elif is_rising_edge(voltage, trip_points.slew_upper_threshold_rise):
        threshold1 = trip_points.slew_lower_threshold_rise
        threshold2 = trip_points.slew_upper_threshold_rise
    else:
        assert False, "Signal has neither a rising edge nor a falling edge."

    slew = transition_time(voltage, time, threshold2) - transition_time(voltage, time, threshold1)

    assert slew >= 0, "Slew can't be negative."

    return slew


def get_input_to_output_delay(time: np.ndarray, input_signal: np.ndarray, output_signal: np.ndarray,
                              trip_points: TripPoints) -> float:
    """Calculate delay from the moment the input signal crosses `trip_points.input_threshold_{rise,fall}`
    to the moment the output signal crosses `trip_points.output_threshold_{rise,fall}`.

    Rise/fall thresholds are selected automatically depending on start and end values of the signals.

    :param time: Time axis.
    :param input_signal: Voltage of input signal (probably normalized to [0,1]).
    :param output_signal: Voltage of output signal (probably normalized to [0,1]).
    :param trip_points: TripPoints object holding threshold values.
    :return: Delay in same units as `time`.
    """

    if is_falling_edge(input_signal, trip_points.input_threshold_fall):
        i_threshold = trip_points.input_threshold_fall
    elif is_rising_edge(input_signal, trip_points.input_threshold_rise):
        i_threshold = trip_points.input_threshold_rise
    else:
        assert False, "Input signal has neither a rising edge nor a falling edge."

    if is_falling_edge(output_signal, trip_points.output_threshold_fall):
        o_threshold = trip_points.output_threshold_fall
    elif is_rising_edge(output_signal, trip_points.output_threshold_rise):
        o_threshold = trip_points.output_threshold_rise
    else:
        assert False, "Output signal has neither a rising edge nor a falling edge."

    delay = transition_time(output_signal, time, o_threshold) - transition_time(input_signal, time, i_threshold)
    assert isinstance(delay, float)
    return delay


def read_trip_points_from_liberty(library_group: Group) -> TripPoints:
    """
    Read trip points from a liberty library Group object.
    :param library_group:
    :return:
    """
    assert library_group.group_name == 'library', "Expected a `library` group but got `{}`".format(
        library_group.group_name)

    trip_points = TripPoints(
        input_threshold_rise=library_group['input_threshold_pct_rise'] * 0.01,
        input_threshold_fall=library_group['input_threshold_pct_fall'] * 0.01,
        output_threshold_rise=library_group['output_threshold_pct_rise'] * 0.01,
        output_threshold_fall=library_group['output_threshold_pct_fall'] * 0.01,

        slew_lower_threshold_rise=library_group['slew_lower_threshold_pct_rise'] * 0.01,
        slew_upper_threshold_rise=library_group['slew_upper_threshold_pct_rise'] * 0.01,
        slew_lower_threshold_fall=library_group['slew_lower_threshold_pct_fall'] * 0.01,
        slew_upper_threshold_fall=library_group['slew_upper_threshold_pct_fall'] * 0.01,
    )

    return trip_points


def find_differential_inputs_by_pattern(patterns: Iterable[str], input_pins: Iterable[str]) -> Dict[str, str]:
    """
    Find pairs of differential inputs based on user-specified patterns.
    The patterns should have the form `['PinA,PinANeg', '%_P,%_N']`.
    A comma is used to separate the non-inverting pin name from the inverting pin name.
    A '%' can be used as a placeholder.
    :param patterns: Patterns that associate non-inverting and inverting pins.
    :param input_pins: All input pin names.
    :return: Returns a dict with the mapping Dict[non-inverting pin, inverting pin]
    """
    # Store mapping of non-inverting input to inverting input.
    differential_inputs = dict()  # Dict[non-inverting, inverting]
    # Match differential inputs.

    logger.info(f"Specified differential inputs: {patterns}")
    # Split at ','
    patterns = [tuple(d.split(',', 1)) for d in patterns]
    # Convert to regex.
    patterns = [(noninv.replace('%', "(.*)"), inv) for noninv, inv in patterns]
    patterns = [(re.compile(noninv), inv) for noninv, inv in patterns]

    for pin in input_pins:
        # Try to match a pattern.
        for p, inverted_name_template in patterns:
            result = p.search(pin)
            if result:
                if len(result.groups()) == 0:
                    basename = result.group(0)
                else:
                    basename = result.group(1)
                inv_name = inverted_name_template.replace("%", basename)
                if pin in differential_inputs:
                    # Sanity check.
                    msg = f"Multiple matches for non-inverting input '{pin}'."
                    logger.error(msg)
                    raise Exception(msg)
                # Store the mapping.
                differential_inputs[pin] = inv_name

    logger.info(f"Mapping of differential inputs: {differential_inputs}")

    # Sanity check.
    if len(set(differential_inputs.keys())) != len(set(differential_inputs.values())):
        logger.error(f"Mismatch in the mapping of differential inputs.")

    return differential_inputs
