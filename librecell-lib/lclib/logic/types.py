#
# Copyright (c) 2019-2021 Thomas Kramer.
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

from sympy.logic import satisfiable
from sympy.logic import boolalg


class CombinationalOutput:
    """
    Description of an output signal of a combinatorial circuit.
    """

    def __init__(self, function: boolalg.Boolean, high_impedance: boolalg.Boolean):
        self.function: boolalg.Boolean = function
        "Boolean expression for the logic output."
        self.high_impedance: boolalg.Boolean = high_impedance
        "Boolean expression which tells when the output is in high-impedance state."

    def is_tristate(self):
        """
        Check if the output have be high-impedance.
        Check if the high-impedance condition is satisfiable.
        :return: bool
        """
        return satisfiable(self.high_impedance)

    def __str__(self):
        return "CombinationalOutput(f = {}, Z = {})".format(self.function, self.high_impedance)

    def __repr__(self):
        return str(self)
