#
# Copyright 2019-2021 Thomas Kramer.
#
# This source describes Open Hardware and is licensed under the CERN-OHL-S v2.
#
# You may redistribute and modify this documentation and make products using it
# under the terms of the CERN-OHL-S v2 (https:/cern.ch/cern-ohl).
# This documentation is distributed WITHOUT ANY EXPRESS OR IMPLIED WARRANTY,
# INCLUDING OF MERCHANTABILITY, SATISFACTORY QUALITY AND FITNESS FOR A PARTICULAR PURPOSE.
# Please see the CERN-OHL-S v2 for applicable conditions.
#
# Source location: https://codeberg.org/tok/librecell
#


from .layers import *
from typing import Any, Dict, List, Optional, Set, Tuple
import klayout.db as db


class WellTapLayout:
    """ Implementations of this class are responsible for drawing a welltap to the layout.
    The function `draw()` must be implemented.
    """

    def __init__(self, well_type: str, location: Tuple[int, int], tech):
        """
        Create the layout representation of a well-tap based on the location within the cell
        and design rules.
        :param well_type: Either 'nwell' or 'pwell'.
        :param location: Location in the cell matrix.
        :param tech: Technology specific design rules.
        """
        raise NotImplemented()

    def terminal_nodes(self) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:
        """
        Get point-like terminal nodes in the form `{net name: {(layer name, (x, y)), ...}}`.

        This function allows to define point-like terminals at precise locations additionally to the terminals
        defined by polygons in the layout.

        This could be used for instance if a net region does not touch any grid points. Hence it is possible to insert
        off-grid routing terminals.
        """
        return dict()

    def draw(self, shapes: Dict[Any, db.Shapes]) -> None:
        """ Draw the layout of the well tap.

        Routing terminals must be labelled with the `'net'` property.

        Example
        =======
        To insert the gate of a transistor:

        `shapes[l_nplus].insert(tap_shape).set_property('net', vdd_net)`

        :param shapes: Dict[layer name, db.Shapes]
          A dict mapping layer names to db.Shapes.
        """
        raise NotImplemented()
