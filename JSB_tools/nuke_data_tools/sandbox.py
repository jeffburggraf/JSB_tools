import matplotlib.pyplot as plt
import numpy as np
from openmc.data import FissionProductYields, Evaluation, Reaction, Tabulated1D, Product
from  openmc.data.ace import get_libraries_from_xsdata
from openmc.data.endf import get_evaluations
from pathlib import Path
from JSB_tools.nuke_data_tools import Nuclide


n =Nuclide.from_symbol('Ar40')

e = Evaluation('/Users/burggraf1/Downloads/iaea-pd2019/g_92-U-238_9237.endf')
print(Reaction.from_endf(e, 18).xs)
