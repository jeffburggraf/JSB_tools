# from JSB_tools.nuke_data_tools.gamma_spec import exp_decay_maximum_likely_hood
from __future__ import annotations
import numpy as np
from JSB_tools.TH1 import TH1F, rolling_median
from matplotlib import pyplot as plt
from JSB_tools.nuke_data_tools import Nuclide
from JSB_tools.nuke_data_tools.__init__ import DECAY_PICKLE_DIR
import marshal
import pickle
from JSB_tools import closest, interp1d_errors
import ROOT
from sortedcontainers import SortedDict
# from PHELIXDataTTree import get_global_energy_bins, get_time_bins
from numbers import Number
# from GlobalValues import shot_groups
from typing import Collection, Union, Tuple
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional
import warnings
import uncertainties.unumpy as unp
from uncertainties import ufloat, UFloat
# from JSB_tools import PolyFit, LogPolyFit, ROOTFitBase
from JSB_tools.regression import PeakFit, LogPolyFit, PolyFit, PolyFitODR
import re
from scipy.signal import find_peaks, peak_widths
from JSB_tools import human_friendly_time
data_dir = DECAY_PICKLE_DIR/'__fast__gamma_dict__.marshal'

cwd = Path(__file__).parent
DATA = None

data_save_path = cwd/'spectroscopy_saves'
if not data_save_path.exists():
    Path.mkdir(data_save_path)


class _CommonDecayNuclides:
    """
    Container of nuclides with a similar decay line.

    Gamma data is extracted from __fast__gamma_dict__.marshal via the static method from_key_and_values.
    Data structure of __fast__gamma_dict__.marshal:
        {
         g_erg_1: ([name1, name2, ...], [intensity1, intensity2, ...], [half_life1, half_life2, ...]),
         g_erg_2: (...)
         }
    """
    def __init__(self, name, intensity, hl, erg):
        self.name = name
        self.hl = hl
        self.intensity = intensity
        self.erg = erg
        self.__rank__ = 0

    @staticmethod
    def from_key_and_values(key, *values) -> List[_CommonDecayNuclides]:
        """
        Takes a `key` (which is a gamma energy) and the dict values from __fast__gamma_dict__.marshal and creates a
            list of ValueStruct for each nuclide that has a decay energy according to `key`.
        Args:
            key:
            *values:

        Returns:

        """
        outs = []
        for name, i, hl in zip(*values):
            outs.append(_CommonDecayNuclides(name, i, hl, key))
        return outs

    def __repr__(self):
        return f"{self.name: <8} rel γ rate: {self.__rank__:.3e}; " \
               f"gamma erg/intensity: {self.erg:.1f} KeV/{int(100*self.intensity)}%; " \
               f"hl: {human_friendly_time(self.hl)};"


def gamma_search(erg_center: float,
                 e_sigma: float = 1,
                 start_time: Union[Number, None] = None,
                 end_time: Union[Number, None] = None,
                 half_life_min:  Union[Number, None] = None,
                 half_life_max:  Union[Number, None] = None,
                 nuclide_weighting_function: Callable = lambda x: 1,
                 rank_threshold=0.001) -> List[_CommonDecayNuclides]:
    """
    Search for nuclides that produce gamma decays in the neighborhood of `erg_center` (+/- sigma_erg). The nuclides are
     sorted from most to least number of decay events that would occur over the course of data acquisition as specified
     by start_time and end_time.
    Args:
        erg_center: Center of energy window
        e_sigma: half width of energy window

        start_time: Time elapsed between nuclide creation and the *start* of data acquisition. If None, then  assumed
         to be zero
        end_time: Time elapsed between nuclide creation and the *end* of data acquisition. If None, then assumed to
         be infinity.

        half_life_min: Min cut off for half life
        half_life_max:  Max cut off for half life

        nuclide_weighting_function: Must be a weighting function that accepts a nuclide name (str) and returns a number
            used to weight the sorter. e.g., weight by fission yield when searching for fission fragments.

        rank_threshold: Threshold for including in results. e.g. 0.001 means anything smaller than 1/1000th of the most
            prominent is not included.

    Returns:

    """
    assert isinstance(start_time, (type(None), Number))
    assert isinstance(end_time, (type(None), Number))
    assert isinstance(erg_center, Number)
    assert isinstance(e_sigma, Number)
    assert isinstance(nuclide_weighting_function, Callable)

    erg_range = erg_center - e_sigma, erg_center + e_sigma

    if end_time is None:
        end_time = np.inf
    if start_time is None:
        start_time = 0

    assert end_time > start_time, "`end_time` must be greater than `start_time`"

    global DATA
    if DATA is None:
        with open(data_dir, 'rb') as f:
            DATA = SortedDict(marshal.load(f))

    def hl_filter(cls: _CommonDecayNuclides):
        if half_life_max is not None:
            if cls.hl > half_life_max:
                return False
        if half_life_min is not None:
            if cls.hl < half_life_min:
                return False

        return True

    def sorter(cls: _CommonDecayNuclides):
        fraction_decayed = 0.5**(start_time/cls.hl) - 0.5**(end_time/cls.hl)
        _out = -cls.intensity * fraction_decayed
        try:
            weight = nuclide_weighting_function(cls.name)
        except Exception:
            warnings.warn('Error in nuclide_weighting_function(). Make sure the function accepts a nuclide name (str), '
                          'and returns a float. ')
            raise
        _out *= weight
        cls.__rank__ = abs(_out)
        return _out

    out: List[_CommonDecayNuclides] = []
    for key in DATA.irange(*erg_range):
        out.extend(_CommonDecayNuclides.from_key_and_values(key, *DATA[key]))

    if half_life_min is not None or half_life_max is not None:
        out = list(filter(hl_filter, out))

    out = sorted(out, key=sorter)
    ranks = [cls.__rank__ for cls in out]
    max_rank = max(ranks)
    for cls in out:
        cls.__rank__ /= max_rank

    def threshold_filter(cc: _CommonDecayNuclides):
        if cc.__rank__.n < rank_threshold:
            return False
        return True

    out = list(filter(threshold_filter, out))

    return out


if __name__ == '__main__':
    pass
