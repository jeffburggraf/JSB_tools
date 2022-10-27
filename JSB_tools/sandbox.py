import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange
from analysis import Shot
import numba as nb
from numba.typed import List

l = Shot(134).list
# nb.types.List([1,2,3])
energy_binned_times = List(l.energy_binned_times)


max_time = Shot(134).list.times[-1]


def get_cut(method=0):
    # a = List(a)

    if method == 0:
        def f(a, v):
            out = np.zeros(len(a))

            for i in range(len(a)):
                i1, i2 = np.searchsorted(a[i], v)
                out[i] = (i2 - i1)
            return out

        return f

    elif method == 1:
        @jit(nopython=True)
        def f(a, v):
            out = np.zeros(len(a))

            for i in range(len(a)):
                i1, i2 = np.searchsorted(a[i], v)
                out[i] = i2 - i1

            return out

        return f
    else:
        assert False


n_events = len(l.times)
dq = 0.05
ranges = []

for q in np.arange(dq, 0.5, dq):
    ranges.append([l.times[int(q * n_events)], l.times[int((1 - q) * n_events)]])

print(f'N ranges per loop: {len(ranges)}')


v = np.array([60, 120])
funcs = {0: get_cut(0), 1: get_cut(1)}
results = {}


def run(method=0):
    # f = get_cut(method)
    r = funcs[method](energy_binned_times, v)
    if method not in results:
        results[method] = r


def plot_results():
    for k, v in results.items():
        plt.plot(v, label=k)
        plt.legend()
#
print()