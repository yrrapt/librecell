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
from liberty.types import Group, Attribute
import logging
from typing import Optional, Tuple, Dict, List
import numpy as np
from ..logic.types import CombinationalOutput

logger = logging.getLogger(__name__)


def get_pin_information(cell_group: Group) -> Tuple[List[str], Dict[str, CombinationalOutput]]:
    """
    Get a list of input pins, output pins and the logic functions of output pins.
    :param cell_group:
    :return: (list of input pins, list of output pins, Dict[output pin, logic function], Dict[output pin, tri-state function])
    """
    input_pins = []
    outputs = dict()
    for pin_group in cell_group.get_groups('pin'):
        # Get pin name
        pin_name = pin_group.args[0]

        # Get direction of pin (input/output)
        direction = pin_group.get('direction', None)

        # Get boolean function of pin (for outputs).
        expr = pin_group.get_boolean_function('function')
        if expr is None:
            # Assert that for all output pins the logic function is defined.
            if direction == 'output':
                msg = f'Output pin has no function defined: {pin_name}'
                logger.warning(msg)

        # Get boolean expression for tri-state condition.
        tri_state = pin_group.get_boolean_function('three_state')
        if tri_state is not None:
            if direction != 'output':
                logger.error(f"Found non-output that with tri-state: {pin_name}")

        logger.info(f"Pin '{pin_name}' {direction} {expr}")

        # Check that pin direction is defined.
        if direction is None:
            logger.warning("Pin has undefined direction: {}/{}".format(cell_group.args[0], pin_name))

        # Remember input and output pins.
        if direction == 'input':
            input_pins.append(pin_name)
        elif direction == 'output':
            comb = CombinationalOutput(function=expr, high_impedance=tri_state)
            outputs[pin_name] = comb
        else:
            logger.warning("Pin direction type not handled: {}".format(direction))

    return input_pins, outputs


def create_table_template_if_not_exists(library: Group,
                                        table_type: str,
                                        name: str,
                                        n: int, m: int,
                                        variable1: str,
                                        variable2: Optional[str] = None,
                                        ) -> Group:
    """
    Create a lookup table template if it does not exist.

    The table name will be `'{name}_template_{n}x{m}'`.

    :param library: Library group where the template is added.
    :param table_type: Group name of the template table. One of ['lu_table_template', 'power_lut_template'].
    :param name:
    :param n: Length of index_1.
    :param m: Length of index_2
    :param variable1:
    :param variable2:
    :return:
    """

    table_types = ['lu_table_template', 'power_lut_template']
    assert table_type in table_types, f"table_type must be one of {table_type}."
    names = ['delay', 'energy', 'setup', 'hold', 'passive_energy', 'recovery', 'removal']
    assert name in names, f"name must be one of {names}."

    full_name = f"{name}_template_{n}x{m}"

    logger.debug(f"Create table template: {table_type}({full_name})")

    # Test if table already exists.
    table = library.get_groups(table_type, argument=full_name)
    if table:
        return table[0]

    else:
        # Create table.
        attributes = [
            Attribute('variable_1', variable1)
        ]
        if m > 1:
            assert variable2 is not None
            attributes.append(Attribute('variable_2', variable2))

        table = Group(table_type,
                      args=[full_name],
                      attributes=attributes
                      )

        index_1 = np.arange(n, dtype=float) + 1000

        table.set_array('index_1', index_1)
        if m > 1:
            index_2 = np.arange(m, dtype=float) + 1000
            table.set_array('index_2', index_2)

        library.groups.append(table)

        return table


def create_delay_template_table(library: Group, n: int, m: int) -> Group:
    """
    Create a `lu_table_template(delay_template_nxm)` group and add it to the library.
    :param library:
    :param n:
    :param m:
    :return:
    """
    return create_table_template_if_not_exists(
        library=library,
        table_type='lu_table_template',
        name='delay',
        n=n,
        m=m,
        variable1='total_output_net_capacitance',
        variable2='input_net_transition'
    )


def create_power_template_table(library: Group, n: int, m: int) -> Group:
    """
    Create a `power_lut_template(energy_template_nxm)` group and add it to the library.
    :param library:
    :param n:
    :param m:
    :return:
    """
    return create_table_template_if_not_exists(
        library=library,
        table_type='power_lut_template',
        name='energy',
        n=n,
        m=m,
        variable1='total_output_net_capacitance',
        variable2='input_net_transition'
    )


def create_constraint_template_table(library: Group, name: str, n: int, m: int) -> Group:
    """
    Create a `lu_table_template(delay_template_nxm)` group and add it to the library.
    :param library:
    :param name: 'setup', 'hold', 'removal' or 'recovery'
    :param n:
    :param m:
    :return:
    """
    return create_table_template_if_not_exists(
        library=library,
        table_type='lu_table_template',
        name=name,
        n=n,
        m=m,
        variable1='related_pin_transition',
        variable2='constrained_pin_transition'
    )
