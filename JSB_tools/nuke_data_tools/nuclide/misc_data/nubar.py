import pickle
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
from uncertainties import ufloat
matplotlib.use('Qt5agg')
cwd = Path(__file__).parent


def load_fit(ax=None):
    data = {}
    As = []
    Ns = []
    nubars = []

    with open("nubar") as f:
        lines = f.readlines()[1:]
        for line in lines:
            Z, s, A, _, nubar = line.split()

            nubar = float(nubar)
            A = int(A)
            Z = int(Z)
            N = A - Z
            As.append(A)
            Ns.append(N)

            nubars.append(nubar)

            data[Z, A] = nubar

    with open("nubars.pickle", 'wb') as f:
        pickle.dump(data, f)

    fit = np.polyfit(As, nubars, deg=1)[::-1]

    if ax is not None:
        _as = np.linspace(min(As), max(As))

        ax.plot(As, nubars, ls='None', marker='o',  label='Actual data')

        ax.plot(_as, fit[0] + _as * fit[1], label='fit')

        ax.set_xlabel("A")
        ax.set_ylabel("Nu bar")
        ax.legend()

    print(f"FIT = {fit}")

    return data, fit


FIT = [-24.22479668, 0.11086718]  # linear fit form A to nubar for SF

with open(cwd / 'nubars.pickle', 'rb') as f:
    data = pickle.load(f)


def get_nubar(Z, A):
    if Z == 252 and A == 98:
        return ufloat(3.7573, 0.0056)

    try:
        return data[Z, A]
    except KeyError:
        return FIT[0] + FIT[1] * A


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2)

    data, fit = load_fit(axs[0])

    nubars = data.values()
    As, Zs = [], []

    for k in data.keys():
        Zs.append(k[0])
        As.append(k[1])

    axs[1].plot(As, nubars, ls='None', marker='o', label="Actual data")

    test_x = []
    test_y = []

    aovera = 232/92
    for z in range(90, 99):
        a0 = int(aovera * z)
        for A in range(a0 - 8, a0 + 8):
            nubar = get_nubar(z, A)
            test_x.append(A)
            test_y.append(nubar)

    axs[1].set_xlabel("A")
    axs[1].set_ylabel("Nu bar")

    axs[1].plot(test_x, test_y, ls='None', marker='o', markersize=1, label='Data test')
    axs[1].legend()



    plt.show()


