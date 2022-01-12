import matplotlib.pyplot as plt
import numpy as np
from openmc.data import FissionProductYields, Evaluation, Reaction, Tabulated1D, Product, IncidentNeutron
from  openmc.data.ace import get_libraries_from_xsdata
from openmc.data.endf import get_evaluations
from pathlib import Path
from JSB_tools.nuke_data_tools import talys_calculation
from JSB_tools.nuke_data_tools import Nuclide
import scipy


# a = talys_calculation('C13', 'n')
e = Evaluation('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/TENDL-g/g-C013.tendl')
r = Reaction.from_endf(e, 5)
print()
# i = IncidentNeutron.from_endf(e)
# print()
