import matplotlib.pyplot as plt

from JSB_tools import calc_background, mpl_hist
import numpy as np
from lmfit.models import GaussianModel
from lmfit.model import CompositeModel
from scipy.signal import find_peaks


from JSB_tools