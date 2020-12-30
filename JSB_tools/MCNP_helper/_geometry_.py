from __future__ import annotations
from typing import Tuple, List, Dict, Union, Iterable, Sized

#
# class OperatorMixin:
#     def __and__(self, other):
#         return BinaryOperator(self, other, 'and')
#
#     def __or__(self, other):
#         return BinaryOperator(self, other, "or")
#
#     def __invert__(self):
#         assert isinstance(self, BinaryOperator)
#         return BinaryOperator(self.left, self.right, self.operator, self.compliment * -1)
#
#
# class BinaryOperator(OperatorMixin):
#     def __init__(self, left, right, operator, compliment=1):
#         self.left = left
#         self.right = right
#         self.operator = operator
#         self.compliment = compliment
#
#     def __str__(self):
#         if self.compliment == -1:
#             compliment = '#'
#         else:
#             compliment = ''
#
#         left = str(self.left)
#         right = str(self.right)
#         # eval_order = {}#  {'parentheses':None, 'no parentheses': None}
#         # is_binary_array = [isinstance(self.left, BinaryOperator), isinstance(self.right, BinaryOperator)]
#         # if is_binary_array == [True, True]:
#         #     eval_order['both parentheses'] = [self.left, self.right, self]
#         #     eval_order['no left parentheses'] = [sel]
#         # elif is_binary_array == [True, False]:
#         #     eval_order['both parentheses'] = [self.left, self]
#         # elif is_binary_array == [False, True]:
#         #     eval_order['both parentheses'] = [self.right, self]
#         # else:
#         #     eval_order['both parentheses'] = [self.left, self.right]
#
#         if self.operator == 'and':
#             out = "{}({} {})".format(compliment, left, right)
#         elif self.operator == 'or':
#             out = "{}({} : {})".format(compliment, left, right)
#         else:
#             assert False, "Invalid operator: {}".format(self.operator)
#
#         # print('Binary Operator,  compliment:{}  operator={}\nleft: {}; right: {}\nresult:{}\n'
#         #       .format(self.compliment, self.operator, self.left, self.right, out))
#         return out
#
#
# class SurfaceSpec(OperatorMixin):
#     def __init__(self, number, sense=1):
#         self.number = number
#         self.sense = sense
#
#     def __invert__(self):
#         assert False, 'Compliment (e.g. __invert__) operator, "~", not allowed on surfaces. Use "-". '
#
#     def __neg__(self):
#         return SurfaceSpec(self.number, -1 * self.sense)
#
#     def __str__(self):
#         if self.sense == 1:
#             return str(self.number)
#         elif self.sense == -1:
#             return "-{}".format(self.number)
#         else:
#             assert False
#
#
# class CellSpec(OperatorMixin):
#     def __init__(self, number, compliment=1):
#         self.number = number
#         self.compliment_int = compliment
#
#     def __invert__(self):
#         return CellSpec(self.number, -1 * self.compliment_int)
#
#     def __neg__(self):
#         assert False, 'Negative (e.g. __neg__) operator, "-", not allowed on cells. Use compliment, "~". '
#
#     def __str__(self):
#         assert self.compliment_int == -1, "CellSpec instance is only valid when used as a compliment"
#         return "#{}".format(self.number)
#

#
# class OperatorMixin:
#     def __and__(self, other):
#         return BinaryOperator(self, other, 'and')
#
#     def __or__(self, other):
#         return BinaryOperator(self, other, "or")
#
#     def __invert__(self):
#         assert isinstance(self, BinaryOperator)
#         return BinaryOperator(self.left, self.right, self.operator, self.compliment * -1)
#


class BinaryOperator:
    n = 0

    def __init__(self, left, right, operator, compliment=1):
        self.left = left
        self.right = right
        self.operator = operator
        self.compliment = compliment
        if operator == 'and':
            self.__precedence__ = 0
        else:
            self.__precedence__ = 1

        self.__n__ = BinaryOperator.n
        BinaryOperator.n += 1

        self.use_parentheses = False
        for thing in [self.left, self.right]:
            if self.operator == 'and' and isinstance(thing, BinaryOperator):
                if thing.operator == 'or' or thing.compliment == True:
                    thing.use_parentheses = True

        print('Call to init, operator: {}'.format(self.operator))
        print('left: ', self.left, 'left type: ', type(self.left))
        print('left: ', self.right, 'right type: ', type(self.right), '\n')
        # print('left number: {}'.format(self.left.number if not isinstance(self.left, BinaryOperator) else None))
        # print('Right number: {}'.format(self.right.number if not isinstance(self.right, (BinaryOperator, GeomSpecMixin)) else None))

    def __str__(self, verbose=False):
        left = str(self.left)
        right = str(self.right)
        use_parentheses = False

        operator = ' ' if self.operator == 'and' else ' : '


        if self.compliment == -1:
            out = "#({}{}{})".format(left, operator, right)
        else:

            if self.use_parentheses:
                out = "({}{}{})".format(left, operator, right)
            else:
                out = "{}{}{}".format(left, operator, right)

        return out

    def __and__(self, other):
        return BinaryOperator(self, other, 'and')

    def __or__(self, other):
        return BinaryOperator(self, other, 'or')

    def __invert__(self):
        return BinaryOperator(self.left, self.right, self.operator, self.compliment * -1)

    def __neg__(self):
        return ~self


class GeomSpecMixin:
    def __init__(self, number, init_type, compliment=1):
        self.compliment = compliment
        self.init_type = init_type
        if not hasattr(self, 'number'):
            self.number = number
        self.__precedence__ = 0

    def __invert__(self):
        assert self.init_type == Cell,  '"~" operator can only be used on Cell instances, not Surface'
        assert isinstance(self, GeomSpecMixin)
        return GeomSpecMixin(self.number, self.init_type, self.compliment * -1)

    def __neg__(self):
        assert self.init_type == Surface, '"-" operator can only be used on Surface instances, not Cell'
        assert isinstance(self, GeomSpecMixin)
        return GeomSpecMixin(self.number, self.init_type, self.compliment*-1)

    def __str__(self, verbose=False):
        if self.init_type == Cell:
            assert isinstance(self, GeomSpecMixin)
            assert self.compliment == -1, '\nCell instance can only be used in geometry specification\nif a compliment' \
                                          ' operator precedes it, like with "cell_10"\nin the following valid ' \
                                          'example,\n\t>>>~cell_10 & -surf_1\n\t>>>#10 -1\nnot like in the following ' \
                                          'invalid example,\n\t>>>cell_10 & surf_1'
            return '#{}'.format(self.number)
        elif self.init_type == Surface:
            assert isinstance(self, GeomSpecMixin)
            return "{}{}".format("" if self.compliment == 1 else '-', self.number)
        else:
            assert False, 'Invalid type: {}'.format(self.init_type)

    def __and__(self, other):
        return BinaryOperator(self, other, 'and')

    def __or__(self, other):
        return BinaryOperator(self, other, 'or')


class Cell(GeomSpecMixin):
    def __init__(self, number):
        super().__init__(number, Cell)
        self.number = number


class Surface(GeomSpecMixin):
    def __init__(self, number):
        super().__init__(number, Surface)
        self.number = number


cell1 = Cell(10)
cell2 = Cell(20)

surf1 = Surface(1)
surf2 = Surface(2)
surf3 = Surface(3)
surf4 = Surface(4)
surf5 = Surface(5)

# s = ~cell1 | (surf1 & surf2)
s2 = (-surf3 | surf4) & (~(~cell1 | surf4 & surf1) | surf5)
# print(s)
print(s2)
# print('type(s): ', type(s))
# print(type(str(s)))