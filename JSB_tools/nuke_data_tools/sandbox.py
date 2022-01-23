import matplotlib.pyplot as plt
import numpy as np
from openmc.data import FissionProductYields, Evaluation, Reaction, Tabulated1D, Product, IncidentNeutron
from  openmc.data.ace import get_libraries_from_xsdata
from openmc.data.endf import get_evaluations
from pathlib import Path
from JSB_tools.nuke_data_tools import talys_calculation
from JSB_tools.nuke_data_tools import Nuclide
import scipy


driving_term = 1
init_term = 0

lambda_matrix = np.array([[-1, 0], [1, -2]])
decay_rate = False
column_labels = ['1', '2']

eig_vals, eig_vecs = np.linalg.eig(lambda_matrix)

eig_vecs = eig_vecs.T
# if not decay_rate:
    # initial condition: fraction of parent nuclide is 1. 0 for the rest
b = [init_term] + [0]*(len(eig_vals) - 1)

if driving_term != 0:
    # coefficients of the particular solution (which is added to homo. sol.)
    particular_coeffs = np.linalg.solve(-lambda_matrix, [driving_term] + [0]*(len(eig_vals) - 1))
else:
    particular_coeffs = np.zeros_like(eig_vals)  # has no effect. No driving term.


coeffs = np.linalg.solve(eig_vecs.T, b-particular_coeffs)  # solve for initial conditions


def func(ts, plot=False):
    if hasattr(ts, '__iter__'):
        iter_flag = True
    else:
        iter_flag = False
        ts = [ts]

    yields = [np.sum([c * vec * np.e ** (val * t) for c, vec, val in
                      zip(coeffs, eig_vecs, eig_vals)], axis=0) for t in ts] + particular_coeffs
    yields = np.array(yields).T
    if not decay_rate:
        out = {name: rel_yield for name, rel_yield in zip(column_labels, yields)}
    else:
        out = {name: rel_yield*rate for name, rel_yield, rate in
                zip(column_labels, yields, np.abs(np.diagonal(lambda_matrix)))}

    if not iter_flag:
        for k, v in out.items():
            out[k] = v[0]

    if plot:
        # if not (plot is True)
        assert iter_flag, 'Cannot plot for only one time'
        plt.figure()
        for k, v in out.items():
            plt.plot(ts, v, label=k)
        if decay_rate:
            plt.ylabel("Decays/s")
        else:
            plt.ylabel("Rel. abundance")
        plt.xlabel('Time [s]')
        plt.legend()

    return out


ts = np.arange(0, 10, 0.05)

for k,v in func(ts).items():
    plt.plot(ts, v, label=k)
plt.legend()
plt.show()