from liberty.types import Group
from liberty.parser import parse_boolean_function
import sympy
from sympy.utilities.lambdify import lambdify
import logging

logger = logging.getLogger(__name__)


def get_pin_information(cell_group: Group):
    """
    Get a list of input pins, output pins and the logic functions of output pins.
    :param cell_group:
    :return: (list of input pins, list of output pins, Dict[output pin, logic function])
    """
    input_pins = []
    output_pins = []
    output_functions = dict()
    for pin_group in cell_group.get_groups('pin'):
        # Get pin name
        pin_name = pin_group.args[0]

        # Get direction of pin (input/output)
        direction = pin_group.get('direction', None)

        # Get boolean function of pin (for outputs).
        function_str = pin_group.get('function', None)
        if function_str is not None:
            expr = parse_boolean_function(function_str.value)
            # Convert expression into a Python lambda function.
            # ! `lambdify` calls eval in the background. `expr` should never come from
            # some where else than `parse_boolean_function`.
            simple = sympy.simplify(expr)
            f = lambdify(expr.atoms(), simple)
            output_functions[pin_name] = f
        else:
            # Assert that for all output pins the logic function is defined.
            if direction == 'output':
                msg = 'Output pin has no function defined: {}'.format(pin_name)
                logger.error(msg)
                assert False, msg
            expr = ''

        logger.info("Pin '{}' {} {}".
                    format(pin_name, direction, expr)
                    )

        # Check that pin direction is defined.
        if direction is None:
            logger.warning("Pin has undefined direction: {}/{}".format(cell_group.args[0], pin_name))

        # Remember input and output pins.
        if direction == 'input':
            input_pins.append(pin_name)
        elif direction == 'output':
            output_pins.append(pin_name)
        else:
            logger.warning("Pin direction type not handled: {}".format(direction))

    return input_pins, output_pins, output_functions
