import warnings

import numpy as np

"""
Define functions used thought MCNP related code.
"""

class MCNPNumberMapping(dict):
    """Used for automatically assigning numbers to MCNP cells and surfaces. """

    def __init__(self, class_name, starting_number: int, step=1):
        self.step = step
        self.class_name: type = class_name
        self.starting_number: int = starting_number
        super(MCNPNumberMapping, self).__init__()  # initialize empty dict
        self.auto_picked_numbers = []  # numbers that have been chosen automatically.

    def number_getter(self, item):
        if self.class_name == 'Cell':
            return getattr(item, 'cell_number')
        elif self.class_name == 'Surface':
            return getattr(item, 'surface_number')
        elif self.class_name == 'F4Tally':
            return getattr(item, 'tally_number')
        elif self.class_name == 'Material':
            return getattr(item, 'mat_number')
        else:
            assert False

    def number_setter(self, item, num):
        if self.class_name == 'Cell':
            return setattr(item, 'cell_number', num)
        elif self.class_name == 'Surface':
            return setattr(item, 'surface_number', num)
        elif self.class_name == 'F4Tally':
            return setattr(item, 'tally_number', num)
        elif self.class_name == 'Material':
            return setattr(item, 'mat_number', num)
        else:
            assert False

    def get_number_auto(self):
        if self.starting_number not in self:
            num = self.starting_number
        else:
            for i in sorted(self.keys()):
                i += 1
                if i not in self.keys():
                    num = i
                    break
            else:
                assert False, 'WTF?!'

        self.auto_picked_numbers.append(num)
        return num

    def names_used(self):
        return [o.__name__ for o in self.values() if o.__name__ is not None]

    def __setitem__(self, number, item):
        assert isinstance(number, (int, type(None))), '{} number must be an integer'.format(self.class_name)

        if isinstance(item.__name__, str):
            assert len(item.__name__) > 0, 'Blank name used in {} {}'.format(self.class_name, item)
        if item.__name__ is not None:
            if item.__name__ in self.names_used():
                old_name = item.__name__
                i = 1
                while (new_name := f'{item.__name__}_{i}') in self.names_used():
                    i += 1
                item.__name__ = new_name
                warnings.warn(f'{self.class_name} name `{old_name}` has already been used.\n'
                              f'Changing to "{item.__name__}"')
        # else:
        #     item.__name__ = None
        if number is not None:  # do not pick number automatically
            if number in self.keys():  # there is a numbering conflict, not allowed in MCNP. Try to resolve conflict
                conflicting_item = self[number]  # Item with numbering conflict.

                if number in self.auto_picked_numbers:  # Can we fix the numbering conflict by changing a number?
                    # Yes, we can, because the conflicting instance's number was not user chosen, but this one was.
                    self.auto_picked_numbers.remove(number)
                    new_number = self.get_number_auto()
                    del self[number]  # unlink conflicting item from its original number to avoid naming conflict later.
                    self[new_number] = conflicting_item  # re-assign old cell to new number, resolving the conflict
                    self.number_setter(conflicting_item, new_number)  # change conflicting instances number

                else:  # cannot fix the numbering conflict, raise an Exception
                    raise Exception('{} number {} has already been used (by {})'.format(self.class_name, number,
                                                                                        conflicting_item))
        else:
            number = self.get_number_auto()

        super(MCNPNumberMapping, self).__setitem__(number, item)  # re-assign current cell to number
        self.number_setter(item, number)  # set the cell or surf instance's number attribute to the correct value


def get_comment(comment, name):
    if name is not None:
        return_comment = " name: {}".format(name)
    else:
        return_comment = ''
    if comment is not None:
        return_comment = '{} {}'.format(comment, return_comment)

    if len(return_comment):
        return_comment = " $ " + return_comment
    return return_comment


def get_rotation_matrix(theta, vx=0, vy=0, vz=1, _round=3):
    """Return the rotation matrix for a rotation around an arbitrary axis. theta is in degrees.
    """
    v = np.array([vx, vy, vz], dtype=float)
    v /= np.linalg.norm(v)
    vx, vy, vz = v

    out = np.array(list([list([vx ** 2 + (vy ** 2 + vz ** 2) * np.cos((np.pi * theta) / 180.),
        vx * vy - vx * vy * np.cos((np.pi * theta) / 180.) - vz * np.sqrt(
        vx ** 2 + vy ** 2 + vz ** 2) * np.sin((np.pi * theta) / 180.),
        vx * vz - vx * vz * np.cos((np.pi * theta) / 180.) + vy * np.sqrt(
        vx ** 2 + vy ** 2 + vz ** 2) * np.sin((np.pi * theta) / 180.)]), list([
        vx * vy - vx * vy * np.cos((np.pi * theta) / 180.) + vz * np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * np.sin(
            (np.pi * theta) / 180.), vy ** 2 + (vx ** 2 + vz ** 2) * np.cos((np.pi * theta) / 180.),
        vy * vz - vy * vz * np.cos((np.pi * theta) / 180.) - vx * np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * np.sin(
            (np.pi * theta) / 180.)]), list([
        vx * vz - vx * vz * np.cos((np.pi * theta) / 180.) - vy * np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * np.sin(
            (np.pi * theta) / 180.),
        vy * vz - vy * vz * np.cos((np.pi * theta) / 180.) + vx * np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) * np.sin(
            (np.pi * theta) / 180.), vz ** 2 + (vx ** 2 + vy ** 2) * np.cos((np.pi * theta) / 180.)])]))
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            out[i][j] = round(out[i][j], _round)
            if out[i][j] == 0.0:
                out[i][j] = 0

    return out


class BinaryOperator:
    n = 0

    def __init__(self, left, right, operator, compliment=1):
        self.left = left
        self.right = right
        self.operator = operator
        self.compliment = compliment

        self.use_parentheses = False
        for thing in [self.left, self.right]:
            if self.operator == 'and' and isinstance(thing, BinaryOperator):
                if thing.operator == 'or':
                    thing.use_parentheses = True

    def __to_str__(self):
        if not isinstance(self.left, str):
            left = self.left.__to_str__()
        else:
            left = self.left
        if not isinstance(self.right, str):
            right = self.right.__to_str__()
        else:
            right = self.right

        operator = ' ' if self.operator == 'and' else ' : '

        if self.compliment == -1:
            out = "#({}{}{})".format(left, operator, right)
        else:

            if self.use_parentheses:
                out = "({}{}{})".format(left, operator, right)
            else:
                out = "{}{}{}".format(left, operator, right)

        return out

    def __str__(self):
        return self.__to_str__()

    def __and__(self, other):
        return BinaryOperator(self, other, 'and')

    def __or__(self, other):
        return BinaryOperator(self, other, 'or')

    def __invert__(self):
        return BinaryOperator(self.left, self.right, self.operator, self.compliment * -1)

    def __neg__(self):
        return ~self


class GeomSpecMixin:
    def __init__(self, surf_number=None, cell_number=None, __compliment__=1):
        self.__compliment = __compliment__
        if not hasattr(self, 'surface_number'):
            self.__surface_number = surf_number
        else:
            self.__surface_number = self.surface_number

        if not hasattr(self, 'cell_number'):
            self.__cell_number = cell_number
        else:
            self.__cell_number = self.cell_number

    def __invert__(self):
        assert self.__cell_number is not None, 'Compliment operator, "~", can only be used on cell instances.'
        return GeomSpecMixin(None, self.__cell_number, self.__compliment * -1)

    def __neg__(self):
        assert self.__surface_number is not None, 'negative operator, "-", can only be used on surface instances.'
        return GeomSpecMixin(self.__surface_number, None, self.__compliment * -1)

    def __pos__(self):
        assert self.__surface_number is not None, 'The (optional) positive operator, "+", can only be used on ' \
                                                  'surface instances.'
        return GeomSpecMixin(self.__surface_number, None, self.__compliment)

    def __to_str__(self):
        if self.__surface_number is not None:

            return "{}{}".format("" if self.__compliment == 1 else '-', self.__surface_number)

        else:
            assert self.__cell_number is not None
            assert self.__compliment == -1, '\nCell instance can only be used in geometry specification\nif a compliment' \
                                          ' operator precedes it, like with "cell_10"\nin the following valid ' \
                                          'example,\n\t>>>~cell_10 & -surf_1\n\t>>>#10 -1\nand not like in the' \
                                          ' following, invalid example:\n\t>>>cell_10 & -surf_1'
            return '#{}'.format(self.__cell_number)
            # return '{}{}'.format("#" if self.__compliment == -1 else '', self.__cell_number)

    def __and__(self, other):
        return BinaryOperator(self, other, 'and')

    def __or__(self, other):
        return BinaryOperator(self, other, 'or')

    def __str__(self):
        return self.__to_str__()
