import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5agg')
cwd = Path(__file__).parent


import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox
from JSB_tools import mpl_hist_from_data, mpl_hist
from scipy.integrate import quad
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline


path = Path(__file__).parent/'Cf252_PFNS.pickle'

try:
    with open(path, 'rb') as f:
        nx, ny = pickle.load(f)
except FileNotFoundError:
    nx, ny = [], []

    with open(path.with_suffix(''), 'r') as f:
        for line in f.readlines():
            x, y = map(float, line.split())
            nx.append(x)
            ny.append(y)

    with open(path, 'wb') as f:
        pickle.dump((nx, ny), f)


def Cf252_watt_spectrum(ergs, binsQ=False):
    if binsQ:
        f = InterpolatedUnivariateSpline(nx, ny, k=1)

        out = [f.integral(e0, e1) for e0, e1 in zip(ergs[:-1], ergs[1:])]

        return np.array(out)
    else:
        out = np.interp(ergs, nx, ny)
        return out

