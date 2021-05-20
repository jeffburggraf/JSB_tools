

import copy

# Convenience to work with both normal and mutable numbers
def _get_value(obj):
    try:
        return obj.value[0]
    except:
        return obj

class MutableFLoat(object):
    def __init__(self, value):
        # Mutable storage because `list` defines a location
        self.value = [value]

    # Define the comparison interface
    def __eq__(self, other):
        return _get_value(self) == _get_value(other)

    def __ne__(self, other):
        return _get_value(self) != _get_value(other)

    # Define the numerical operator interface, returning new instances
    # of mutable_number
    def __add__(self, other):
        return MutableFLoat(self.value[0] + _get_value(other))

    def __mul__(self, other):
        return MutableFLoat(self.value[0] * _get_value(other))

    # In-place operations alter the shared location
    def __iadd__(self, other):
        self.value[0] += _get_value(other)
        return self

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other - self

    def __imul__(self, other):
        self.value[0] *= _get_value(other)
        return self

    # Define the copy interface
    def __copy__(self):
        new = MutableFLoat(0)
        new.value = self.value
        return new

    def __repr__(self):
        return repr(self.value[0])


a = MutableFLoat(10)

b = 1 + a
print(b)

a += 20
print(b)

list_1 = [MutableFLoat(i) for i in [1, 2, 3, 4]]
list_2 = copy.copy(list_1)
list_1[0] *= 100
print(list_1)
print(list_2)