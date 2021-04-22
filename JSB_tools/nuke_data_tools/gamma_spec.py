# from JSB_tools.nuke_data_tools.gamma_spec import exp_decay_maximum_likely_hood
from __future__ import annotations
import numpy as np
from JSB_tools.TH1 import TH1F
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
from typing import List, Dict, Callable
import warnings
import uncertainties.unumpy as unp
from uncertainties import ufloat, UFloat
import uncertainties.umath as umath
from JSB_tools import PolyFit, PeakFit
import re

data_dir = DECAY_PICKLE_DIR/'__fast__gamma_dict__.marshal'

cwd = Path(__file__).parent
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


# class PeakResult:
#     """
#     Return value from peak_fit
#
#     """
#     def __init__(self, amp, center, sigma, bg, y_fit_result, hist: TH1F):
#         """
#
#         Args:
#             amp: Gauss amplitude, i.e. number of events in peak
#             center: peak center
#             sigma: peak sigma
#             bg: bg parameter
#             y_fit_result: Fiy function evalueated at xs
#             hist: Histogram that contained fitted peak.
#         """
#         self.center = center
#         self.sigma = sigma
#         self.gb = bg
#         self.y = y_fit_result
#         self.amplitude = amp
#         self.hist = hist
#
#     def plot(self, ax=None):
#         if ax is None:
#             ax = plt.gca()
#         self.hist.plot(ax)
#         ax.plot(self.hist.bin_centers, self.y, label='Fit')
#         return ax

#
# def peak_fit(hist: TH1F, center_guess, min_x=None, max_x=None, method='ROOT',
#              sigma_guess=1) -> PeakResult:
#     """
#     Fits a peak within a histogram.
#     Args:
#        hist:
#        center_guess:
#        min_x:
#        max_x:
#        method:
#        sigma_guess:
#
#     Returns:
#
#     """
#
#     assert method in ['ROOT']
#
#     if isinstance(center_guess, UFloat):
#         center_guess = center_guess.n
#     assert isinstance(center_guess, (float, int))
#     if not hist.is_density:
#         warnings.warn('Histogram passed to fit_peak may not have density as bin values!'
#                       ' To fix this, do hist /= hist.binvalues before passing to peak_fit')
#     # Set background guess before range cut is applied. This could be useful when one wants the background estimate to
#     # be over a larger range that may include other interfering peaks
#     bg_guess = hist.median_y.n
#
#     if min_x is not None or max_x is not None:  # if limits are used, make new histogram cut to range.
#         bins = hist.__bin_left_edges__
#         if max_x is None:
#             max_x = bins[-1]
#         if min_x is None:
#             min_x = bins[0]
#
#         max_bin = np.searchsorted(hist.__bin_left_edges__, max_x) + 1
#         min_bin = np.searchsorted(hist.__bin_left_edges__, min_x)
#         new_bins = bins[min_bin: max_bin]
#         old_hist = hist
#         hist = TH1F(bin_left_edges=new_bins)
#         hist += old_hist.bin_values[min_bin: max_bin-1]
#
#     if method == 'ROOT':
#         func = ROOT.TF1('peak_fit', '[0]*TMath::Gaus(x,[1],[2], kTRUE) + [3]')
#         amp_guess = np.sum(hist.bin_widths*(hist.bin_values-hist.median_y)).n
#         func.SetParameter(0, amp_guess)
#
#         func.SetParameter(1, center_guess)
#         func.SetParameter(2, sigma_guess)
#         func.SetParameter(3, bg_guess)
#         hist.__ROOT_hist__.Fit('peak_fit')
#         amp = ufloat(func.GetParameter(0), func.GetParError(0))
#         center = ufloat(func.GetParameter(1), func.GetParError(1))
#         sigma = ufloat(func.GetParameter(2), func.GetParError(2))
#         bg = ufloat(func.GetParameter(3), func.GetParError(3))
#         xs = hist.bin_centers
#         ys = np.array([func.Eval(x) for x in xs])
#         out = PeakResult(amp, center, sigma, bg, ys, hist)
#         return out


class MakeROOTSpectrum:

    data_dir = cwd/"Spectra_data"

    def __init__(self, path_name):
        """
        Used to create spectra in the standardized format.
        Args:
            path_name: Name of the specrum to be saved to file.
            n_channels: Number of channels.
            ch_2_erg_coefficients: ch_2_erg coeffs. If None, they can be determined using methods.
            fwhm_coefficients:
        """
        path_name = Path(path_name)
        assert path_name.parent.exists()
        if not MakeROOTSpectrum.data_dir.exists():
            MakeROOTSpectrum.data_dir.mkdir()
        path = MakeROOTSpectrum.data_dir/path_name

        self.n_channels = None
        self.root_file = ROOT.TFile(str(path), 'recreate')
        self.tree = ROOT.TTree('spectrum', 'spectrum')
        self.erg_br = np.array([0], dtype=np.float)
        self.eff_br = np.array([0], dtype=np.float)
        self.ch_br = np.array([0], dtype=np.float)
        self.t_br = np.array([0], dtype=np.float)
        self.tree.Branch('erg', self.erg_br, 'erg/F')
        self.tree.Branch('eff', self.eff_br, 'eff/F')
        self.tree.Branch('ch', self.ch_br, 'ch/F')
        self.tree.Branch('t', self.t_br, 't/F')

        self.ch_2_erg_coeffs = None
        self.shape_coefficients = None
        self.erg_bins = None
        self.efficiencies = None
        self.efficiency_errors = None

        self.erg_bin_centers = None

    def do_erg_calibration(self, counts_array, channels_to_ergs: List[Tuple], window_size=15,
                           order=1, plot=True, background_counts=None):
        """
        Args:
            counts_array: An array of counts for each channel
            channels_to_ergs:
            window_size: int or array of ints. Size of window (in channels) for gaus fits.
            order: Order of PolyFit for energy calibration
            plot: whether or not to plot
            background_counts: n array to be subtracted from counts_array

        Returns:

        """
        counts_array = np.array(counts_array)
        self.n_channels = len(counts_array)
        try:
            assert hasattr(channels_to_ergs, '__iter__')
            for _ in channels_to_ergs:
                assert hasattr(_, '__len__')
                assert len(_) == 2
        except AssertionError:
            raise AssertionError('`channels_to_ergs` not of the correct format. Example:'
                                 '\n[(approx_ch_1, actual_erg_1),...,approx_ch_n, actual_erg_n)]')

        # the "- 0.5" so that bin centers are on whole numbers, starting with 0
        channel_hist = TH1F(bin_left_edges=(np.arange(self.n_channels+1) - 0.5))
        channel_hist += unp.uarray(counts_array, np.sqrt(counts_array))
        channel_hist /= channel_hist.bin_widths
        if background_counts is not None:
            assert len(background_counts) == len(channel_hist)
            channel_hist -= background_counts

        if not hasattr(window_size, '__iter__'):
            window_size = [window_size]*len(channels_to_ergs)

        fit_channels = []
        fit_ergs = []
        fit_peak_sigmas = []

        for index, (ch_center_guess, erg) in enumerate(channels_to_ergs):
            w = window_size[index]
            fit = PeakFit(channel_hist, ch_center_guess, min_x=ch_center_guess-w//2, max_x=ch_center_guess+w//2)
            fit_channels.append(fit.center)
            if not isinstance(erg, UFloat):
                erg = ufloat(erg, 0)
            fit_ergs.append(erg)
            fit_peak_sigmas.append(fit.sigma)

            if plot:
                ax = fit.plot_fit()
                ax.set_title(f'Given erg: {erg}KeV;  Resulting channel: {fit.center}')

        channels_errors = np.array([x.std_dev for x in fit_channels])
        fit_channels = np.array([x.n for x in fit_channels])
        ergs_errors = np.array([y.std_dev for y in fit_ergs])
        fit_ergs = np.array([y.n for y in fit_ergs])

        fit_shape_ys = np.array([y.n for y in fit_peak_sigmas])
        # fit_shape_errors_ys = np.array([y.std_dev for y in fit_peak_sigmas])

        erg_fit = PolyFit(fit_channels, fit_ergs, channels_errors, ergs_errors, order=order)
        shape_fit = PolyFit(fit_ergs, fit_shape_ys, order=order)  # shape coeffs are a function of energy
        erg_fit.plot_fit(title="ch to erg")
        shape_fit.plot_fit(title="erg to Shape")

        self.ch_2_erg_coeffs = [i.n for i in erg_fit.coeffs]
        self.shape_coefficients = [i.n for i in shape_fit.coeffs]
        self.erg_bins = erg_fit.eval_fit(channel_hist.__bin_left_edges__)
        self.erg_bin_centers = unp.nominal_values(erg_fit.eval_fit(channel_hist.bin_centers))
        print(self.erg_bin_centers)
        print(channel_hist.bin_centers)

    def do_efficiency_calibration(self, ergs, n_counts_meas, n_count_true):
        assert all([hasattr(s, '__iter__') for s in [ergs, n_counts_meas, n_count_true]])
        assert len(ergs) == len(n_count_true) == len(n_counts_meas)
        assert self.n_channels is not None, "Do energy calibration before doing eff. calibration"
        assert not isinstance(n_counts_meas[0], UFloat),\
            "Don't include errors in measured counts. This is done automatically"
        ergs = [0] + list(ergs) + [ergs[-1]*3]
        # n_count_true = np.array(n_count_true)
        # n_counts_meas = [0] + list(n_counts_meas)
        effs_n = np.array(n_counts_meas)/np.array(n_count_true)
        effs_std = np.sqrt(n_counts_meas) / np.array(n_count_true)
        effs = unp.uarray([0] + list(effs_n) + [0], [0] + list(effs_std)+ [0])
        interp = interp1d_errors(ergs, effs, self.erg_bin_centers, order=2)
        self.efficiencies = unp.nominal_values(interp)
        self.efficiency_errors = unp.std_devs(interp)
        plt.figure()
        plt.errorbar(self.erg_bin_centers, self.efficiencies, yerr=self.efficiency_errors,
                     label="efficiency calibration")
        plt.errorbar(ergs[:-1], unp.nominal_values(effs[:-1]), yerr= unp.std_devs(effs[:-1]),
                     label="Data points")
        plt.legend()


class ROOTSpectrum:
    """
    For analysis of spectra that are stored in a ROOT file in a standardized manner.
    Instances can be built from either a path_to_root_file (using __init__), or from a TTree/TChain
    (using cls.from_tree).

    ROOT file format standard:
        The tree, named 'spectrum', must have (at least) the following branches:
            'ch': Channel
            't': time of gamma detection
            'eff' efficiency of detector at energy erg.
            'erg': Energy
        As well as an associated pickle file with the following:
            "ch_2_erg":  Array of length n_channels and values equal to energies
            "erg_bins": An array to be used to specify the bins of a histogram. Length = n_channels + 1
            "efficiencies": an array of efficiencies with length equal to n_channels
            "efficiency_errors": an array of efficiency errors with length equal to n_channels
            "ch_2_erg_coeffs": coeffs for converting channel to energy, i.e. a*np.arange(n_channels) + b
            "shape_coefficients": coefffs of fit for peak sigmas as a function of energy (NOT CHANNEL!).  Length = n_channels
    """
    @staticmethod
    def __load_pickle_data__(name):
        a = ["ch_2_erg", "erg_bins", "efficiencies", "efficiency_errors", "ch_2_erg_coeffs", "shape_coefficients"]
        pickle_path = cwd/'Spectra_data'/f'{name}.pickle'
        assert pickle_path.exists(), f"'{pickle_path}' does not exist"
        with open(pickle_path, 'rb') as f:
            _d = pickle.load(f)
            assert all([s in _d for s in a])
        return {_d[k]: v for k, v in _d.items() if k in a}

    def __add_root_file__(self, name):
        root_path = cwd/'Spectra_data'/f'{name}.root'
        assert root_path.exists(), f"'{root_path}' does not exist"
        self.tree.Add(root_path)

    def __init__(self, *names):
        self.tree = ROOT.TChain('spectrum')
        test_dict = None
        for name in names:
            data = self.__load_pickle_data__(name)
            if test_dict is None:
                test_dict = self.__load_pickle_data__(name)
            assert data == test_dict
            self.__add_root_file__(name)



if __name__ == '__main__':

    p_name = '7_Loop_26s_150s_000.Spe'
    counts = np.zeros(8192)
    channels = np.arange(len(counts), dtype=np.float) + 0.5

    for path in Path('/Users/burggraf1/PycharmProjects/PHELIX/PHELIX_data/data').iterdir():
        if __name__ == '__main__':
            if m := re.match('([0-9]+)_Shot', path.name):
                shot_num = int(m.groups()[0])
                if 42 <= shot_num <= 52:
                    paths = [p.name for p in path.iterdir()]
                    assert p_name in paths
                    with open(path/p_name) as f:
                        lines = f.readlines()
                        index_start = lines.index('$DATA:\n') + 2
                        index_stop = lines.index('$ROI:\n')
                        data_lines = lines[index_start: index_stop]
                        _new_counts = np.array([int(s.split()[0]) for s in data_lines])
                        counts += np.array([int(s.split()[0]) for s in data_lines])
                        # a = np.array([int(s.split()[0]) for s in data_lines])
                        # print(a.shape)
    ch_2_erg = [(393.0, 218), (917.6, 511)]  # ch -> actual energy
    m = MakeROOTSpectrum('test')
    m.do_erg_calibration(counts, ch_2_erg)
    ergs = np.array([59.9, 88.4, 122, 166, 392, 514, 661, 898, 1173, 1332, 1835], dtype=np.float)
    effs = np.array([0.06, 0.1, 0.144, 0.157, 0.1, 0.07, 0.05, 0.04, 0.03, 0.027, 0.018])
    counts_true = 10000*np.ones_like(effs)
    counts_measured = effs*counts_true
    m.do_efficiency_calibration(ergs,counts_measured, counts_true)
    plt.show()



# #
# #
#
# class ROOTSpectrum:
#     """
#     For analysis of spectra that are stored in a ROOT file in a standardized manner.
#     Instances can be built from either a path_to_root_file (using __init__), or from a TTree/TChain
#     (using cls.from_tree).
#
#     ROOT file format standard:
#         The tree, named 'spectrum', must have the following branches:
#             'ch': Channel
#             't': time of gamma detection
#             'eff' efficiency of detector at energy erg.
#         The associated ROOT file may contain the following two histograms:
#             "ch_2_erg":  TH1, quadratic polynomial mapping channel to gamma energy. Parameters are coeffs and par errors
#                 are energy uncertainties.
#             "erg_2_eff_hist": Bin left edges are energies, bin values are efficiency and sigma efficiency
#             "channels": sorted TArrayI of all channels.
#                 Example creation:
#                     np_arr = np.array(channels, dtype=np.int)
#                     arr = ROOT.TArrayI(len(a))
#                     for i in range(np_arr):
#                         arr[i] = np_arr[i]
#                     arr.WriteObject(arr, 'channels')
#         It's fine if these arent there. In this case, ROOTSpec
# #
# #     Todo:
# #         -Make ch_to_erg a TF1.
# #         -Implement a way to change the coeffs of the TF1 in the file (all files in the case of a TChain)
# #         -A function that accepts a cut (str) and returns a UFloat equal to the efficiency of the selection plus
# #             uncertainties
# #         -Plot time dependence of peak/multiple peaks simultaneously. Maybe options for several methods of background
# #             subtraction.
# #             --Make it easy to overlay two spectra in a ROOT TCanvas
# #             --Auto / manual selection of non-uniform time bins
# #         -Area under peak calculations
# #         -Energy spectrum plotting – overlaying of multiple ROOTSpecrtrum instances
# #         -Energy calibration tool
# #         -Efficiency calibration tool. Option for no erg/eff data saved to ROOT files.
# #     """
# #     def __init__(self, root_file_path, overwrite=False):
# #         self.erg_coeffs = [0, 1, 0]
# #         self.shape_cal_coeffs = [1,0, 0]
# #
# #
# #     #
# #     # def __init__(self, root_file):
# #     #     assert isinstance(root_file, ROOT.TFile)
# #     #     self.root_file = root_file
# #     #     # root_file = ROOT.TFile(str(path_to_root_file))
# #     #     self.tree = root_file.Get('tree')
# #     #     self.erg_2_eff_hist = self.__load_erg_2_eff_hist__()
# #     #     self.energy_coeffs: List[UFloat] = self.__load_erg_calibration__()
# #     #
# #     # def __load_erg_calibration__(self) -> List[UFloat]:
# #     #     keys = self.root_file.GetListOfKeys()
# #     #     assert 'ch_2_erg' in [k.GetName() for k in keys], 'No energy calibration available. '
# #     #     tf1 = self.root_file.Get('ch_2_erg')
# #     #     # print([(tf1.GetParameter(i), (tf1.GetParError(i))) for i in range(3)])
# #     #     coeffs = [ufloat(tf1.GetParameter(i), tf1.GetParError(i)) for i in range(3)]
# #     #     return coeffs
# #     #
# #     # def __load_erg_2_eff_hist__(self) ->TH1F:
# #     #     """Used to check for ch_2_erg_hist and erg_2_eff_hist histograms in the root file."""
# #     #
# #     #     keys = [k.GetName() for k in self.root_file.GetListOfKeys()]
# #     #     assert 'erg_2_eff_hist' in keys, f'ROOT file does not contain "ch_2_erg_hist" histogram, a histogram channel vs' \
# #     #                                     f' gamma energy + uncertainty. Available keys:\n{keys}'
# #     #     # assert 'erg_2_eff_hist' in keys, 'ROOT file does not contain "erg_2_eff_hist" histogram,' \
# #     #     #                                  ' a histogram of gamma energy vs efficiency + uncertainty.'
# #     #     erg_2_eff_hist = TH1F.from_native_ROOT_hist(self.root_file.Get('erg_2_eff_hist'))
# #     #     return erg_2_eff_hist
# #     #
# #     # @classmethod
# #     # def from_path(cls, path):
# #     #     return cls(ROOT.TFile(str(path)))
# #     #
# #     # @classmethod
# #     # def from_tree(cls, tree):
# #     #     if isinstance(tree, ROOT.TChain):
# #     #         file = tree.GetFile()
# #     #     elif isinstance(tree, ROOT.TTree):
# #     #         file = tree.GetCurrentFile()
# #     #     else:
# #     #         assert False, '`tree` is not a ROOT.TTree or ROOT.TChain instance'
# #     #
# #     #     return cls(root_file=file)
# #     #
# #     # @property
# #     # def ergs(self):
# #     #     return unp.nominal_values(self.erg_2_eff_hist.bin_values)
# #     #
# #     # @property
# #     # def __erg_sigmas(self):
# #     #     return unp.std_devs(self.ch_2_erg_hist.bin_values)
# #     #
# #     # @property
# #     # def __erg_fwhms(self):
# #     #     return unp.std_devs(self.ch_2_erg_hist.bin_values)*2.355
# #     #
# #     # def erg_fwhm(self, erg):
# #     #     i = np.searchsorted(self.ergs, erg)
# #     #     s = slice(i-5, i+5)
# #     #     return np.interp(erg, self.ergs[s], self.__erg_fwhms[s])
# #     #
# #
# #
# #
# #
# #
# #
# # if __name__ == '__main__':
# #     # from GlobalValues import shot_groups
# #     # tree = shot_groups['He+Ar'].tree
# #     # # s = ROOTSpectrum.from_tree(tree)
# #     # s2 = ROOTSpectrum.from_path('/Users/burggraf1/PycharmProjects/PHELIX/Shot_data_analysis/PHELIX2019ROOT_TREES/_shot42.root')
# #     # # print(s.erg_fwhm(50))
#