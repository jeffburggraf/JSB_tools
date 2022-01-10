import matplotlib.pyplot as plt
import numpy as np
from openmc.data import FissionProductYields, Evaluation, Reaction, Tabulated1D, Product
from openmc.data.ace import get_libraries_from_xsdata
from openmc.data.endf import get_evaluations
from pathlib import Path
from JSB_tools.nuke_data_tools import Nuclide
import scipy


n = Nuclide.from_symbol('Cu63')


def find_activation_candidates(nuclide):
    for k,v in nuclide.get_incident_proton_daughters().items():
        if v.xs.threshold_erg() < 40 and 20*60 < v.half_life < 24*60*60:
            print(k, v, f'thresh erg: {v.xs.threshold_erg():.1f}')
            v.xs.plot()


find_activation_candidates(n)
find_activation_candidates(Nuclide.from_symbol('Cu65'))
plt.show()