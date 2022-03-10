import matplotlib.pyplot as plt
import numpy as np
from openmc.data import FissionProductYields, Evaluation, Reaction, Tabulated1D, Product, IncidentNeutron, Decay
from  openmc.data.ace import get_libraries_from_xsdata
from openmc.data.endf import get_evaluations
from pathlib import Path
from JSB_tools.nuke_data_tools import talys_calculation
from JSB_tools.nuke_data_tools import Nuclide
from JSB_tools import mpl_hist
import scipy







