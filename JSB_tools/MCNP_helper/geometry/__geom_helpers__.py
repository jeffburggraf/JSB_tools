import numpy as np


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
        self.compliment = __compliment__
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
        return GeomSpecMixin(None, self.__cell_number, self.compliment * -1)

    def __neg__(self):
        assert self.__surface_number is not None, 'negative operator, "-", can only be used on surface instances.'
        return GeomSpecMixin(self.__surface_number, None, self.compliment * -1)

    def __pos__(self):
        assert self.__surface_number is not None, 'The (optional) positive operator, "+", can only be used on ' \
                                                  'surface instances.'
        return GeomSpecMixin(self.__surface_number, None, self.compliment)

    def __to_str__(self):
        if self.__surface_number is not None:

            return "{}{}".format("" if self.compliment == 1 else '-', self.__surface_number)

        else:
            assert self.__cell_number is not None
            assert self.compliment == -1, '\nCell instance can only be used in geometry specification\nif a compliment' \
                                          ' operator precedes it, like with "cell_10"\nin the following valid ' \
                                          'example,\n\t>>>~cell_10 & -surf_1\n\t>>>#10 -1\nand not like in the' \
                                          ' following, invalid example:\n\t>>>cell_10 & -surf_1'
            return '#{}'.format(self.__cell_number)

    def __and__(self, other):
        return BinaryOperator(self, other, 'and')

    def __or__(self, other):
        return BinaryOperator(self, other, 'or')

    def __str__(self):
        return self.__to_str__()
