# from __future__ import annotations
#
# import pickle
#
# import matplotlib.pyplot as plt
# import numpy as np
# from openmc.data import Evaluation, Reaction, Tabulated1D, Product, IncidentNeutron, Decay
# from pathlib import Path
# from uncertainties import UFloat
# from uncertainties import unumpy as unp
# import re
import matplotlib.pyplot as plt

from JSB_tools import Nuclide, TabPlot


import numpy as np
from GlobalValues import get_proton_erg_prob_1

ergs = np.linspace(2, 60)

weights = get_proton_erg_prob_1(ergs)
weights *= 1.0/sum(weights)


