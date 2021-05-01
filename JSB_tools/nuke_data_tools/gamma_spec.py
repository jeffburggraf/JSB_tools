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
from JSB_tools import PolyFit, PeakFit, LogPolyFit
import re
from scipy.signal import find_peaks, peak_widths
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
        return f"{self.name}: erg: {self.erg}, intensity: {self.intensity}, hl: {self.hl}, rel. # of gammas: {self.__rank__}"


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
         the sorter. e.g.: weight by fissionXS yield when searching for fissionXS fragments.

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


class PrepareGammaSpec:

    data_dir = cwd/"Spectra_data"

    def __init__(self, path_name, n_channels):
        """
        Used to create spectra in the standardized format.
        Args:
            path_name: Name of the specrum to be saved to file.
        """
        path_name = Path(path_name)
        assert path_name.parent.exists()
        if not PrepareGammaSpec.data_dir.exists():
            PrepareGammaSpec.data_dir.mkdir()
        path = PrepareGammaSpec.data_dir / path_name

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

        self.calibration_spectra = []  # various count spectra for energy calibration.
        self.calibration_points: List[Dict[str, Union[List, Tuple]]] = []  # Calibration points.

        self.ch_2_erg_coeffs = None
        self.shape_coefficients = None
        self.erg_bins = None
        self.efficiencies = None
        self.efficiency_errors = None

        self.__channel_bins__ = np.arange(self.n_channels+1) - 0.5  # for use in specifying histogram with bin centers
        #                                                              equal to channels, starting at 0

        self.erg_bin_centers = None

    def plot_channel_spectrum(self):pass
        # hist = TH1F(bin_left_edges=self.__channel_bins__)
        # hist += self.n_counts_array
        # return hist.plot()

    @property
    def n_channels(self):
        # assert hasattr(self.__n_counts_array__, '__iter__')
        # return len(self.__n_counts_array__)
        return  len(self.__channel_bins__) - 1

    # @property
    # def n_counts_array(self):
    #     assert self.__n_counts_array__ is not None
    #     return unp.uarray(self.__n_counts_array__, np.sqrt(self.__n_counts_array__))

    def add_peaks_4_calibration(self, counts_array: Collection,
                                ch_erg_TrueCounts: List[Tuple[int, float, Optional[float]]]):
        """
        Add a observed peak(s) as seen in the spectrum specified by`counts_array`.
        You must provide the true energy and approx location of the peak in the channel spectrum (aka `counts_array`).
        Each peak will be used for energy calibration, and optionally, the efficiency calibration.

        Args:
            counts_array: An array of length n_channels, and is the number of counts in each channel.
            ch_erg_TrueCounts: Each entry in the list has the form:
                    (channel_peak_estimate, true_energy, true_counts)]
                 where,
                    "channel_peak_estimate" is your estimate of which channel the peak center lies in;
                    "true_energy" is the known energy of the peak;
                    "true_counts" is optional, and if provided, will be used for efficiency calibration as well.
        """
        assert len(counts_array) == self.n_channels
        self.calibration_points.append({'counts': counts_array, 'points': []})
        data_dict = self.calibration_points[-1]
        format_msg = 'Incorrect format for `ch_erg_TrueCounts`. `ch_erg_TrueCounts` is a list of 3-tuples' \
                     ', with the form\n\t (channel_peak_estimate, true_energy, <true_counts>)],\n where ' \
                     '\n\t channel_peak_estimate is your estimate of which channel the peak center lies is.' \
                     '\n\t true_energy is the known energy of the peak' \
                     '\n\t true_counts is optional, and if provided will be used for efficiency calibration. '

        try:
            for _ in ch_erg_TrueCounts:
                assert isinstance(_, tuple)
                assert len(_) >= 2
                assert isinstance(_[0], int)
                point = list(_[:2])
                try:
                    point.append(_[2])
                except IndexError:
                    point.append(None)
                data_dict['points'].append(point)
        except AssertionError:
            assert False, format_msg

    def do_erg_calibration(self, channels_to_ergs: List[Tuple], window_size=20,
                           order=1, plot=True, background_counts=None):
        """
        Args:
            channels_to_ergs:
            window_size: int or array of ints. Size of window (in channels) for gaus fits.
            order: Order of PolyFit for energy calibration
            plot: whether or not to plot
            background_counts: n array to be subtracted from counts_array

        Returns:

        """

        try:
            assert hasattr(channels_to_ergs, '__iter__')
            for _ in channels_to_ergs:
                assert hasattr(_, '__len__')
                assert len(_) == 2
        except AssertionError:
            raise AssertionError('`channels_to_ergs` not of the correct format. Example:'
                                 '\n[(approx_ch_1, actual_erg_1),...,approx_ch_n, actual_erg_n)]')
        __ergs__ = [_[1] for _ in channels_to_ergs]
        if 1460.83 not in __ergs__:
            warnings.warn("Don't forget to look for the common K-40 line at 1460.83 KeV. ")
        if 511 not in __ergs__:
            warnings.warn("Don't forget to look for the 511 KeV annihilation line")

        # the "- 0.5" so that bin centers are on whole numbers, starting with 0
        channel_hist = TH1F(bin_left_edges=self.__channel_bins__ )
        channel_hist += self.n_counts_array
        channel_hist /= channel_hist.bin_widths
        if background_counts is not None:
            assert len(background_counts) == len(channel_hist)
            channel_hist -= background_counts

        if not hasattr(window_size, '__iter__'):
            window_size = [window_size]*len(channels_to_ergs)

        fit_channels = []  # peak channel center
        fit_ergs = []  # peak center with errors
        fit_fwhms = []  # peak shape (gaussian sigma), with errors

        peak_fits = []

        for index, (ch_center_guess, erg_true) in enumerate(channels_to_ergs):
            ch_index = channel_hist.find_bin_index(ch_center_guess)

            w = window_size[index]
            # make sure to cover the whole peak plus some for bg guess
            large_window_hist = channel_hist.remove_bins_outside_range(ch_index-2*w, ch_index+2*w)
            # ch_center_guess = large_window_hist.find_bin_index(ch_center_guess)
            fit = PeakFit(large_window_hist, ch_center_guess, min_x=ch_center_guess-w//2, max_x=ch_center_guess+w//2)
            peak_fits.append(fit)
            fit_channels.append(fit.center)
            if not isinstance(erg_true, UFloat):
                erg_true = ufloat(erg_true, 0)
            fit_ergs.append(erg_true)
            if erg_true != 511:
                fit_fwhms.append(2.355*fit.sigma)

            if plot:
                ax = fit.plot_fit(x_label='channel', y_label="counts")
                ax.set_title(f'Given erg: {erg_true}KeV;  fit channel: {fit.center};\ngaus sigma: {fit.sigma}')

        channels_errors = np.array([x.std_dev for x in fit_channels])
        fit_channels = np.array([x.n for x in fit_channels])

        ergs_errors = np.array([y.std_dev for y in fit_ergs])
        fit_ergs = np.array([y.n for y in fit_ergs])

        fit_fwhms_errors = [y.std_dev for y in fit_fwhms]
        fit_fwhms = [y.n for y in fit_fwhms]

        erg_fit = PolyFit(fit_channels, fit_ergs, channels_errors, ergs_errors, order=order)

        shape_fit = PolyFit(list(filter(lambda x: x != 511, fit_ergs)), fit_fwhms, y_err=fit_fwhms_errors, order=order)
        """
            "erg_bins":List[float] An array to be used to specify the bins of a histogram. Length = n_channels + 1
            "ch_2_erg_coeffs": List[float], coeffs for converting channel to energy, i.e. a*np.arange(n_channels) + b
            "erg_2_eff_coeffs": List[UFloat], PolyFit coeffs (UFloat instances),
                                e.g. e^(c0 + c1*log(x) + c2*log(x)^2 + c3*log(x)^3 ... cn*log(x)^n)
            "shape_coefficients": List[UFloat], Polynomial coeffs mapping energy to peak FWHM, length = n_channels
            """
        self.erg_bins = unp.nominal_values(erg_fit.eval_fit(channel_hist.__bin_left_edges__))
        self.ch_2_erg_coeffs = erg_fit.coeffs
        self.shape_coefficients = shape_fit.coeffs
        self.erg_bin_centers = unp.nominal_values(erg_fit.eval_fit(channel_hist.bin_centers))

        if plot:
            erg_fit.plot_fit(x=np.arange(self.n_channels), title="ch to erg", x_label="channel", y_label='energy [KeV]')
            shape_fit.plot_fit(x=self.erg_bin_centers, title="erg to shape",  x_label="energy [KeV]",
                               y_label='counts')

    def do_efficiency_calibration(self, ergs, n_counts_meas, n_count_true, order=3):
        """

        Args:
            ergs: Energies of measures gamma line
            n_counts_meas: Measured counts of gamma lines
            n_count_true: Theoretical counts of gamma lines
            order: Order of LogPolyFit. 3 is a good number.

        Returns:

        """
        assert all([hasattr(s, '__iter__') for s in [ergs, n_counts_meas, n_count_true]])
        assert len(ergs) == len(n_count_true) == len(n_counts_meas)
        assert self.erg_bins is not None, "Do energy calibration before doing eff. calibration"
        assert not isinstance(n_counts_meas[0], UFloat),\
            "Don't include errors in measured counts. This is done automatically"
        # appending an anchor point (x=y=0) below
        ergs = [1E-10] + list(ergs)
        effs_n = [1E-10] + list(np.array(n_counts_meas)/np.array(n_count_true))
        effs_std = [0] + list(np.sqrt(n_counts_meas) / np.array(n_count_true))
        fit = LogPolyFit(ergs, effs_n, y_err=effs_std, order=order, fix_coeffs_to_guess=[order])
        fit.plot_fit(title="Efficiency calibration")
        efficiencies = fit.eval_fit(self.erg_bin_centers)
        self.efficiencies = unp.nominal_values(efficiencies)
        self.efficiency_errors = unp.std_devs(efficiencies)

    def plot_erg_spectrum(self, min_erg=None, max_erg=None, eff_correction=False):

        assert self.__n_counts_array__ is not None, "Do energy calibration before calling obj.plot_erg_spectrum() !"
        hist = TH1F(bin_left_edges=self.erg_bins)
        hist += self.n_counts_array
        if eff_correction:
            if self.efficiencies is None:
                warnings.warn("No efficiency calibration. Setting `eff_correction` to False ")
            else:
                hist /= self.efficiencies
        ax = hist.plot(xmax=max_erg, xmin=min_erg)

        def ch_2_erg(chs):
            return np.interp(chs, np.arange(self.n_channels), self.erg_bin_centers)

        def erg_2_ch(ergs):
            return np.interp(ergs, self.erg_bin_centers, np.arange(self.n_channels))
        ax.secondary_xaxis('top', functions=(erg_2_ch, ch_2_erg)).set_xlabel("channel")
        ax.set_xlabel("energy [KeV]")

        return ax


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

        As well as an associated pickle file with the following keys:
            "erg_bins": List[float] An array to be used to specify the bins of a histogram. Length = n_channels + 1
            "ch_2_erg_coeffs": List[float], coeffs for converting channel to energy, i.e. a*np.arange(n_channels) + b
            "erg_2_eff_coeffs": List[UFloat], PolyFit coeffs (UFloat instances),
                                e.g. e^(c0 + c1*log(x) + c2*log(x)^2 + c3*log(x)^3 ... cn*log(x)^n)
            "shape_coefficients": List[UFloat], Polynomial coeffs mapping energy to peak FWHM, length = n_channels
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

    p_name = '10_Loop_596s_2400s_000.Spe'
    counts = np.zeros(8192)
    channels = np.arange(len(counts), dtype=np.float) + 0.5

    for path in Path('/Users/burggraf1/PycharmProjects/PHELIX/PHELIX_data/data').iterdir():
        if __name__ == '__main__':
            if m := re.match('([0-9]+)_Shot', path.name):
                shot_num = int(m.groups()[0])
                if 42 <= shot_num <= 52:
                    paths = [p.name for p in path.iterdir()]
                    if p_name not in paths:
                        continue
                    with open(path/p_name) as f:
                        lines = f.readlines()
                        index_start = lines.index('$DATA:\n') + 2
                        index_stop = lines.index('$ROI:\n')
                        data_lines = lines[index_start: index_stop]
                        _new_counts = np.array([int(s.split()[0]) for s in data_lines])
                        counts += np.array([int(s.split()[0]) for s in data_lines])
                        # a = np.array([int(s.split()[0]) for s in data_lines])
                        # print(a.shape)

    ch_2_erg = [(393.0, 218.6), (315, 175), (532, 296.5), (917.6, 511), (2623, 1460.83)]  # ch -> actual energy
    ergs = np.array([59.9, 88.4, 122, 166, 392, 514, 661, 898, 1173, 1332, 1835], dtype=np.float)
    effs = np.array([0.06, 0.1, 0.144, 0.157, 0.1, 0.07, 0.05, 0.04, 0.03, 0.027, 0.018])
    counts_true = 10000 * np.ones_like(effs)
    counts_measured = effs * counts_true
    m = PrepareGammaSpec('test', counts)
    m.do_erg_calibration(ch_2_erg)
    m.plot_erg_spectrum(100)
    #
    m.do_efficiency_calibration(ergs,counts_measured, counts_true, order=3)
    m.plot_erg_spectrum(100, eff_correction=True)
    # from JSB_tools.nuke_data_tools import Nuclide
    print(Nuclide.from_symbol('Xe139').decay_gamma_lines)
    #

    #
    plt.show()
