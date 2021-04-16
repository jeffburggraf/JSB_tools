# from JSB_tools.nuke_data_tools.gamma_spec import exp_decay_maximum_likely_hood
from __future__ import annotations
import numpy as np
from JSB_tools.TH1 import TH1F
from matplotlib import pyplot as plt
from JSB_tools.nuke_data_tools import Nuclide
from JSB_tools.nuke_data_tools.__init__ import DECAY_PICKLE_DIR
import marshal
from JSB_tools import closest
import ROOT
from sortedcontainers import SortedDict
# from PHELIXDataTTree import get_global_energy_bins, get_time_bins
from numbers import Number
# from GlobalValues import shot_groups
from typing import Collection, Union, Tuple
from pathlib import Path
from typing import List, Dict, Callable
import warnings
import uncertainties.unumpy as unp

data_dir = DECAY_PICKLE_DIR/'__fast__gamma_dict__.marshal'


DATA = None


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
        Takes a `key` (which is a gamma energy) and its `values` from __fast__gamma_dict__.marshal and creates a list of
        ValueStruct for each nuclide that has a decay energy according to `key`.
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
        return f"{self.name}: erg: {self.erg}, intensity: {self.intensity}, hl: {self.hl} rel. # of events: {self.__rank__}"


def gamma_search(erg_center: float,
                 e_sigma: float = 1,
                 start_time: Union[Number, None] = None,
                 end_time: Union[Number, None] = None,
                 half_life_min:  Union[Number, None] = None,
                 half_life_max:  Union[Number, None] = None,
                 nuclide_weighting_function: Callable[[str], float] = lambda x: 1) -> List[_CommonDecayNuclides]:
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
        nuclide_weighting_function: A weighting function that accepts a nuclide name (str) and returns a number used to weight
         the sorter. e.g.: weight by fission yield when searching for fission fragments.

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
        if half_life_min is not None:
            if not half_life_min < cls.hl:
                return False
        if half_life_min is not None:
            if not cls.hl < half_life_max:
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

    return out


required_spec_attribs = ['erg', 't', 'eff']


class ROOTSpectrum:
    """
    For analysis of spectra that are stored in a ROOT file in a standardized manner.
    Instances can be built from either a path_to_root_file (using __init__), or from a TTree/TChain
    (using cls.from_tree).

    File format standard:
        The tree must have the following branches:
            'erg':  Gamma energy
            't': time of gamma detection
            'eff' efficiency of detector at energy erg.
        The associated ROOT file may contain the following two histograms:
            "ch_2_erg_hist":  Bin left edges are channels, bin values are energy and sigma energy
            "erg_2_eff_hist": Bin left edges are energies, bin values are efficiency and sigma efficiency
        It's fine if these arent there. In this case, ROOTSpec

    Todo:
        -Make ch_to_erg a TF1.
        -Implement a way to change the coeffs of the TF1 in the file (all files in the case of a TChain)
        -A function that accepts a cut (str) and returns a UFloat equal to the efficiency of the selection plus
            uncertainties
        -Plot time dependence of peak/multiple peaks simultaneously. Maybe options for several methods of background
            subtraction.
            --Make it easy to overlay two spectra in a ROOT TCanvas
            --Auto / manual selection of non-uniform time bins
        -Area under peak calculations
        -Energy spectrum plotting – overlaying of multiple ROOTSpecrtrum instances
        -Energy calibration tool
        -Efficiency calibration tool. Option for no erg/eff data saved to ROOT files.
    """

    def __init__(self, root_file):
        assert isinstance(root_file, ROOT.TFile)
        self.root_file = root_file
        # root_file = ROOT.TFile(str(path_to_root_file))
        self.tree = root_file.Get('tree')
        self.ch_2_erg_hist, self.erg_2_eff_hist = self.__get_saved_histograms__()
        self.ch_2_erg_hist.plot()
        self.erg_2_eff_hist.plot()

    def __get_saved_histograms__(self) -> Tuple[TH1F, TH1F]:
        """Used to check for ch_2_erg_hist and erg_2_eff_hist histograms in the root file."""

        keys = [k.GetName() for k in self.root_file.GetListOfKeys()]
        assert 'ch_2_erg_hist' in keys, 'ROOT file does not contain "ch_2_erg_hist" histogram, a histogram channel vs' \
                                        ' gamma energy + uncertainty.'
        assert 'erg_2_eff_hist' in keys, 'ROOT file does not contain "erg_2_eff_hist" histogram,' \
                                         ' a histogram of gamma energy vs efficiency + uncertainty.'
        ch_2_erg_hist = TH1F.from_native_ROOT_hist(self.root_file.Get('ch_2_erg_hist'))
        erg_2_eff_hist = TH1F.from_native_ROOT_hist(self.root_file.Get('erg_2_eff_hist'))
        return ch_2_erg_hist, erg_2_eff_hist

    @classmethod
    def from_path(cls, path):
        return cls(ROOT.TFile(str(path)))

    @classmethod
    def from_tree(cls, tree):
        if isinstance(tree, ROOT.TChain):
            file = tree.GetFile()
        elif isinstance(tree, ROOT.TTree):
            file = tree.GetCurrentFile()
        else:
            assert False, '`tree` is not a ROOT.TTree or ROOT.TChain instance'

        return cls(root_file=file)

    @property
    def ergs(self):
        return unp.nominal_values(self.ch_2_erg_hist.bin_values)

    @property
    def __erg_sigmas(self):
        return unp.std_devs(self.ch_2_erg_hist.bin_values)

    @property
    def __erg_fwhms(self):
        return unp.std_devs(self.ch_2_erg_hist.bin_values)*2.355

    def erg_fwhm(self, erg):
        i = np.searchsorted(self.ergs, erg)
        s = slice(i-5, i+5)
        return np.interp(erg, self.ergs[s], self.__erg_fwhms[s])







if __name__ == '__main__':
    from GlobalValues import shot_groups
    tree = shot_groups['He+Ar'].tree
    s = ROOTSpectrum.from_tree(tree)
    # s2 = ROOTSpectrum('/Users/burggraf1/PycharmProjects/PHELIX/Shot_data_analysis/PHELIX2019ROOT_TREES/_shot42.root')
    print(s.erg_fwhm(50))


    # n = Nuclide.from_symbol('U238')
    # fiss_yields = n.independent_gamma_fission_yield()
    #
    # def weighting(name):
    #     try:
    #         return fiss_yields.yields[name]
    #     except KeyError:
    #         return 0
    #
    # print(gamma_search(218, 2, 20, 60*3, nuclide_weighting_function=weighting))
    # Gdelta_t = get_time_bins()[-1]
    # data_dir = DECAY_PICKLE_DIR / '__fast__gamma_dict__.marshal'
    #
    # tree = shot_groups['He+Ar'].tree
    # hist = TH1F(bin_left_edges=get_global_energy_bins())
    # hist.Project(tree, 'erg', weight='1.0/eff')
    # hist.plot()
    # print(Nuclide.from_symbol('U238').independent_gamma_fission_yield_gef()[1])
    # # for n, yield_ in Nuclide.from_symbol('U238').independent_gamma_fission_yield_gef().items():
    # #     print(n, yield_)
    # for i in gamma_search(2612, e_sigma=4, delta_t=Gdelta_t):
    #     print(i)
    plt.show()