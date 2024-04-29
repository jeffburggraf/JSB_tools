from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from typing import Callable


# ====================================
nuclide_names = "Ba140", "La140",  "Ce140"

lambda_matrix = [[-6.2907E-7,  0          , 0],  # from half lives
                 [ 6.2907E-7, -4.77937e-06, 0],
                 [ 0.0       , 4.77937e-06, 0]]

sigma_phi = 2.5  # production rate

irrad_time = 36 * 60 ** 2

def driving_term(t):  # returns rate of nuclei creation [s^-1]
    out = np.zeros(len(lambda_matrix[0]))

    if t < irrad_time:  # "running" for 12 hrs
        out[0] = sigma_phi  # sigma_phi

    return out


init_conditions = [1E4, 0, 0]  # starts with 10,000 Ba140

plot_times = np.linspace(0, 20 * 24 * 60**2, 10000)  #
# ====================================


def J(y, t):
    return lambda_matrix


def func(y, t):
    out = np.matmul(lambda_matrix, y)

    if isinstance(driving_term, Callable):
        out += driving_term(t)
    else:
        out += driving_term

    return out


yields, infodict = odeint(func, init_conditions, t=plot_times, Dfun=J, full_output=True)

yields = yields.transpose()

solution = {name: yield_ for name, yield_ in zip(nuclide_names, yields)}

fig, ax = plt.subplots()
fig.suptitle(f"Irradiating something that makes Ba-140 at {sigma_phi} atoms per second")

for k, v in solution.items():
    x = plot_times/24/60**2  # seconds to days
    ax.plot(x, v, label=k)

ax.axvspan(0, irrad_time/24/60**2, label="Irradiation time", color ="black", alpha=0.1)

ax.set_xlabel("Time [days]")
ax.set_ylabel("# of atoms")

ax.legend()

plt.show()
