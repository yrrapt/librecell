##
## Copyright (c) 2021 Thomas Kramer.
##
## This file is part of librecell-lib
## (see https://codeberg.org/tok/librecell/src/branch/master/librecell-lib).
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##

from typing import Dict, List
import sympy
from sympy.logic import boolalg


class CellType:
    """
    Description of a standard-cell type.
    A `CellType` object should contain information that is necessary for the characterization of this type of cells.
    """

    def human_readable_description(self) -> str:
        raise NotImplementedError()


class Combinational(CellType):
    """
    Purely combinational cell without any feed-back loops.
    """

    def __init__(self):
        self.inputs: List[sympy.Symbol] = None
        self.outputs: List[sympy.Symbol] = None
        self.output_functions: Dict[sympy.Symbol, boolalg.Boolean] = None


class Latch(CellType):

    def __init__(self):
        self.data_in = None
        self.enable = None  # Write condition / clock.
        self.clear = None  # Clear condition.
        self.preset = None  # Preset condition.

    def __str__(self):
        return self.human_readable_description()

    def human_readable_description(self) -> str:
        return f"""Latch {{
    write data: {self.data_in}
    write enable: {self.enable}
    clear: {self.clear}
    preset: {self.preset}
}}"""


class SingleEdgeDFF(CellType):
    """
    Single-edge triggered delay flip-flop.
    """

    def __init__(self):
        self.clock_signal = None
        "Name of the clock signal."
        self.clock_edge_polarity = None
        "True = rising, False = falling"
        self.clocked_on: boolalg.Boolean = None
        "Clocked when the value of the boolean expression rises to true."
        self.next_state: boolalg.Boolean = None
        "Next state that follows a clock edge."

        self.data_in = None
        "Expression for the input data."
        self.data_out = None
        "Name of the non-inverted data output net."
        self.data_out_inv = None
        "Name of the inverted data output net (if any)."

        self.scan_enable = None
        "Name of the scan-enable input."
        self.scan_in = None

        self.async_preset = None
        "Name of the asynchronous preset signal."
        self.async_set_polarity = None
        "Polarity of the signal (False: active low, True: active high)."

        self.async_clear = None
        "Name of the asynchronous clear signal."
        self.async_reset_polarity = None
        "Polarity of the signal (False: active low, True: active high)."

    def __str__(self):
        return self.human_readable_description()

    def human_readable_description(self) -> str:

        preset_polarity = ""
        if self.async_set_polarity is not None:
            preset_polarity = "HIGH" if self.async_set_polarity else "LOW"

        clear_polarity = ""
        if self.async_reset_polarity is not None:
            clear_polarity = "HIGH" if self.async_reset_polarity else "LOW"

        return f"""SingleEdgeDFF {{
    clock: {self.clock_signal}
    active clock edge: {"rising" if self.clock_edge_polarity else "falling"}
    output: {self.data_out}
    inverted output: {self.data_out_inv}
    next data: {self.data_in}

    asynchronous preset: {self.async_preset} {preset_polarity}
    asynchronous clear: {self.async_clear} {clear_polarity}

    scan enable: {self.scan_enable}
    scan input: {self.scan_in}
}}"""
