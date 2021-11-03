import matplotlib.pyplot as plt
import numpy as np
from openmc.data import FissionProductYields, Evaluation, Reaction, Tabulated1D, Product
from  openmc.data.ace import get_libraries_from_xsdata
from openmc.data.endf import get_evaluations
from pathlib import Path
from JSB_tools.nuke_data_tools import Nuclide
import scipy



func = decay(Nuclide.from_symbol('Kr89'))
x = np.linspace(0, 60*60*24, 10000)
out = func(x)
for label, values in out.items():
    plt.plot(x, values, label=label)
plt.legend()
plt.show()
