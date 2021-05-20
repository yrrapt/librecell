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

        self.internal_state: sympy.Symbol = None
        """
        Variable for the current internal state.
        """

        self.clocked_on: boolalg.Boolean = sympy.false
        "Clocked when the value of the boolean expression rises to true."
        self.next_state: boolalg.Boolean = sympy.false
        "Next state that follows a clock edge."

        self.outputs: Dict[sympy.Symbol, boolalg.Boolean] = dict()
        """
        Boolean functions for all outputs. The output functions
        are functions of the primary inputs and of the `internal_state` variable.
        """

        self.scan_enable = None
        "Name of the scan-enable input."
        self.scan_in = None

        self.async_preset: boolalg.Boolean = sympy.false
        "Preset condition."

        self.async_clear: boolalg.Boolean = sympy.false
        "Clear condition."
    def __str__(self):
        return self.human_readable_description()

    def clock_signal(self) -> sympy.Symbol:
        """
        Return the clock signal if there is exactly one clock signal.
        :return:
        """
        atoms = list(self.clocked_on.atoms(sympy.Symbol))
        if len(atoms) == 1:
            return atoms[0]
        else:
            return None

    def clock_edge_polarity(self) -> bool:
        """
        Get the polarity of the clock edge if there is exactly one clock signal.
        If there are multiple or no clock signal, return `None`.
        """
        clock = self.clock_signal()
        if clock is None:
            return None
        return self.clocked_on.subs({clock: True}) == True

    def human_readable_description(self) -> str:

        return f"""SingleEdgeDFF {{
    internal_state: {self.internal_state}
    next_state: {self.next_state}
    clocked_on: {self.clocked_on}
    active clock edge: {"rising" if self.clock_edge_polarity() else "falling"}
    outputs: {self.outputs}

    asynchronous preset: {self.async_preset} 
    asynchronous clear: {self.async_clear} 

    scan enable: {self.scan_enable}
    scan input: {self.scan_in}
}}"""
