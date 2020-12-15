import numpy as np
from warnings import warn
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc
from pathlib import Path
import sys
import time
from openmc.data import ATOMIC_SYMBOL, atomic_weight
import re
from typing import List

# try:
#     import soerp
# except ModuleNotFoundError as e:
#     warn("No soerp module available. Analytical 2nd order error propagation not available.")
# try:
#     import mcerp
#     from mcerp import UncertainFunction
# except ModuleNotFoundError as e:
#     warn("No mcerp module available. Analytical monte carlo error propagation not available.")
#
# from matplotlib import pyplot as plt
# import time
# import scipy.stats
# from numbers import Number
