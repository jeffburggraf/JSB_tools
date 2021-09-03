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
        nuclide_weighting_function: Must be a weighting function that accepts a nuclide name (str) and returns a number
        used to weight the sorter. e.g., weight by fission yield when searching for fission fragments.

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
    data_path = cwd.parent / 'user_saved_data' / 'spectra_calibrations'

    def __init__(self, n_channels):
        """
        Used to calibrate a HPGe detector and save in a standardized format.
        Args:
            n_channels: number of channels.
        """
        self.calibration_points: List[PrepareGammaSpec.CalibrationPeak] = []  # Calibration points.

        self.erg_fit: PolyFit = None
        self.eff_fit: LogPolyFit = None
        self.fwhm_coeffs = None
        self.eff_fit_method: str = None
        self.__channel_bins__ = np.arange(n_channels+1) - 0.5  # for use in specifying histogram with bin centers
                                                                     # equal to channels, starting at 0
        self.erg_bin_centers = None
        self.erg_bins = None

    @property
    def n_channels(self):
        # assert hasattr(self.__n_counts_array__, '__iter__')
        # return len(self.__n_counts_array__)
        return len(self.__channel_bins__) - 1

    class CalibrationPeak:
        """
        A container for fitted peaks.
        Attributes:
            self.fit: FitPeak instance.
            self.true_counts: actual number of counts if efficiency calibration is to be performed

        """
        def __init__(self, gamma_spec: PrepareGammaSpec, counts_array, channel_guess, energy, true_counts, window_width):
            assert len(counts_array) == gamma_spec.n_channels, "All calibration data must be a full spectrum, i.e." \
                                                               "there must be a value for every channel"
            if not isinstance(counts_array[0], UFloat):
                self.count_array = unp.uarray(counts_array, np.sqrt(counts_array))
            else:
                self.count_array = counts_array
            self.channel_hist = TH1F(bin_left_edges=gamma_spec.__channel_bins__)
            self.channel_hist += self.count_array
            # self.channel_hist = self.channel_hist.remove_bins_outside_range(channel_guess - 2*window_width,
            #                                                                 channel_guess+2*window_width)
            self.channel_hist /= self.channel_hist.bin_widths
            self.fit = PeakFit(peak_center_guess=channel_guess, x=self.channel_hist.bin_centers, y=counts_array,
                               window_width=window_width)
            self.true_counts = true_counts
            self.energy = energy
            if isinstance(self.energy, UFloat):
                self.energy = self.energy.n

        def plot(self):
            ax = self.fit.plot_fit(xlabel="Channel", ylabel="counts (raw)")
            t = f'Gamma energy: {self.energy}, n_counts: {self.fit.amp: .2e}\n' \
                f'peak FWHM (chs): {self.fit.fwhm}'

            if self.true_counts is not None:
                plt.subplots_adjust(top=0.81)
                t += f'\nefficiency: \n {self.fit.amp/self.true_counts:.2f}'
            ax.set_title(t)

    def add_peaks_4_calibration(self, counts_array: Collection,
                                channel_guesses: Collection[int, float],
                                gamma_energies=Collection[float],
                                true_counts: Collection[float, None] = None,
                                fit_width=20, plot=True):
        """
        Add a observed peak(s) as seen in the spectrum specified by`counts_array`.
        You must provide the true energy and approx location of the peak in the channel spectrum (aka `counts_array`).
        Each peak will be used for energy calibration, and optionally, the efficiency calibration.

        Args:
            counts_array: An array of length n_channels, and is the number of counts in each channel.
            channel_guesses: A list of your estimates of which channel each peak center lies in
            gamma_energies: Actual energies of peak
            true_counts: Optional, or list. If provided, peaks for which `true_counts` is *not* None will be used for
                efficiency calibration as well as energy calibration.
            fit_width: Should be the smallest window that can contain an entire peak.
            plot: If true, plot fits.

        """
        if hasattr(self, 'is_loaded'):
            raise NotImplementedError('Calibration already complete (this calibration was loaded from disk)')
        assert len(counts_array) == self.n_channels

        if true_counts is None:
            true_counts = [None]*len(channel_guesses)
        for ch_guess, energy, n_counts in zip(channel_guesses, gamma_energies, true_counts):
            z = PrepareGammaSpec.CalibrationPeak(self, counts_array=counts_array, channel_guess=ch_guess, energy=energy,
                                                 true_counts=n_counts, window_width=fit_width)
            self.calibration_points.append(z)
            if plot:
                z.plot()

    def compute_calibration(self, efficiency_order=3, fit_last_eff_coeff=False, erg_order=1):
        """
        When you're done adding calibration peaks, call this function.
        Args:
            efficiency_order: Order of the efficiency fit.
            fit_last_eff_coeff: Fix the largest efficiency fit coefficients to the initial guess. See LogPolyFit docs
            erg_order: Order of energy calibration.

        Returns:

        """
        if hasattr(self, 'is_loaded'):
            raise NotImplementedError('Calibration already computed (this calibration was loaded from disk)')
        channel_peak_centers = []  # center of peaks (the channel!)
        energies = []
        _eff_energies = []  # energies only of those peaks for which efficiency calibration will be perfomred
        efficiencies = []
        for p in self.calibration_points:
            _peak_center = ufloat(p.fit.center.n, p.fit.sigma.n/np.sqrt(p.fit.amp.n))
            channel_peak_centers.append(_peak_center)
            energies.append(p.energy)
            if p.true_counts is not None:
                eff = p.fit.amp/p.true_counts
                _eff_energies.append(p.energy)
                efficiencies.append(eff)
        energies = np.array(energies)
        channel_peak_centers = np.array(channel_peak_centers)
        srt = np.argsort(channel_peak_centers)
        energies = energies[srt]
        channel_peak_centers = channel_peak_centers[srt]

        if not any(np.isclose(energies, 1460.83, atol=1)):
            warnings.warn("Don't forget to use the (very common) K-40 line at 1460.83 KeV. ")
        if 511 not in energies:
            warnings.warn("Don't forget to use the (very common) 511 KeV annihilation line")
        self.erg_fit = PolyFitODR(x=channel_peak_centers, y=energies, order=erg_order)
        ax = self.erg_fit.plot_fit(xlabel='Channels', ylabel='Energies [Kev]')
        ax.set_title("Energy Calibration Result")
        self.erg_bin_centers = unp.nominal_values(self.erg_fit.eval_fit(np.arange(self.n_channels)))
        self.erg_bins = unp.nominal_values(self.erg_fit.eval_fit(self.__channel_bins__))
        # plt.figure()
        # plt.title('FWHM')
        # _x = np.arange(self.n_channels)
        # plt.plot(_x, 2.34*unp.std_devs(self.erg_calibration.eval_fit(_x)), label='Err')
        # plt.legend()
        # plt.plot(_x, 2.34*(self.erg_calibration.coeffs[0].std_dev + _x*self.erg_calibration.coeffs[1].std_dev))

        if len(efficiencies):
            efficiencies = [ufloat(1E-10, 0)] + list(efficiencies)
            _eff_energies = [1E-10] + list(_eff_energies)
            assert efficiency_order <= len(efficiencies), "Lower efficiency order. Not enough points."

            self.eff_fit = LogPolyFit(_eff_energies, efficiencies, order=efficiency_order,
                                      fix_highest_coeff=fit_last_eff_coeff)

            self.eff_fit.plot_fit(fit_x=self.erg_bin_centers, xlabel='Energy [KeV]', ylabel='Efficiency')
        else:
            warnings.warn("No points for efficiency calibration.")

    def plot_erg_spectrum(self, min_erg=None, max_erg=None, eff_correction=False):
        if hasattr(self, 'is_loaded'):
            raise NotImplementedError('Cannot plot calibration spectrum for loaded calibration data.')

        assert self.erg_bin_centers is not None, "Run compute_calibration before calling obj.plot_erg_spectrum() !"
        hist = TH1F(bin_left_edges=self.erg_bins)  # start from second entry to avoid dividing by zero
        for p in self.calibration_points:
            hist += p.count_array
        hist /= len(self.calibration_points)
        if eff_correction:
            _effs = self.eff_fit.eval_fit(hist.bin_centers)
            _effs = np.where(unp.nominal_values(_effs) != 0, _effs, unp.uarray(np.ones(len(_effs)), np.zeros_like(len(_effs))))
            hist /= _effs
        hist.SetTitle("Erg vs (raw) counts")
        ax = hist.plot(xmax=max_erg, xmin=min_erg)

        def ch_2_erg(chs):
            return np.interp(chs, np.arange(self.n_channels), self.erg_bin_centers)

        def erg_2_ch(ergs):
            return np.interp(ergs, self.erg_bin_centers, np.arange(self.n_channels))
        ax.secondary_xaxis('top', functions=(erg_2_ch, ch_2_erg)).set_xlabel("channel")
        ax.set_xlabel("Energy [KeV]")
        ax.set_ylabel("Counts")

        return ax

    def save_calibration(self, name: str):
        assert self.erg_bin_centers is not None, "Run self.compute_calibration before saving"
        assert isinstance(name, str)
        data_path = PrepareGammaSpec.data_path
        Path(data_path).mkdir(parents=True, exist_ok=True)
        data = {'erg_bins': self.erg_bins, 'erg_bin_centers': self.erg_bin_centers,
                '__channel_bins__': self.__channel_bins__}  # maybe add more to this? e.g. raw calibration data
        with open(data_path/(name + '.pickle'), 'wb') as f:
            pickle.dump(data, f)

        self.eff_fit.save(name)
        self.erg_fit.save(name)

    @classmethod
    def load_calibration(cls, name: str):
        assert isinstance(name, str)
        data_path = PrepareGammaSpec.data_path/f"{name}.pickle"
        assert data_path.exists(), f'No saved calibration named, "{name}"'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        result = cls.__new__(cls)
        result.erg_fit = PolyFitODR.load(name)
        result.eff_fit = LogPolyFit.load(name)
        result.erg_bins = data['erg_bins']
        result.erg_bin_centers = data['erg_bin_centers']
        result.__channel_bins__ = data['__channel_bins__']
        result.is_loaded = True
        return result


class BuildSpectrum:
    """
    This is the next step after saving a calibration.
    Using the desired calibration, sen all the data of a given spectrum to this class, where it can be saved to a ROOT
    tree for analysis.
    """
    def __init__(self, name, name_of_calibration):
        self.__name_of_calibration = name_of_calibration
        self.name = name
        path = cwd.parent/'user_saved_data'/'spectra_trees'
        path.mkdir(parents=True)
        path = path/name
        if path.exists():
            warnings.warn(f'Overwriting spectrum named "{name}"')

        self.root_file = ROOT.TFile(str(path), 'recreate')
        self.tree = ROOT.TTree('spectrum', 'spectrum')
        self.erg_br = np.array([0], dtype=float)
        self.eff_br = np.array([0], dtype=float)
        self.ch_br = np.array([0], dtype=float)
        self.t_br = np.array([0], dtype=float)
        self.tree.Branch('erg', self.erg_br, 'erg/F')
        self.tree.Branch('eff', self.eff_br, 'eff/F')
        self.tree.Branch('ch', self.ch_br, 'ch/F')
        self.tree.Branch('t', self.t_br, 't/F')

    def fill(self, channel, time):pass


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
    # c = PrepareGammaSpec.load_calibration('PHELIX_test')
    # print(c.erg_calibration)
    # print(c.eff_fit)
    # print(c.erg_bins)
    # c.eff_fit.plot_fit()
    # c.erg_calibration.plot_fit()
    # plt.show()
    p_name = '10_Loop_596s_2400s_000.Spe'
    counts = np.zeros(8192)
    channels = np.arange(len(counts), dtype=float) + 0.5

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

    def erg_2_channel(erg):
        return (erg-0.119)/0.556999

    def channel_2_erg(ch):
        return ch*0.556999 + 0.119


    ch_2_erg = [(393.0, 218.6), (315, 175), (532, 296.5), (917.6, 511), (2623, 1460.83)]  # ch -> actual energy
    ergs = np.array([59.9, 88.4, 122,   166,   392, 514, 661, 898, 1173, 1332, 1835])
    effs = np.array([0.06, 0.1,  0.144, 0.157, 0.1, 0.07, 0.05, 0.04, 0.03, 0.027, 0.018])
    # counts_true = 10000 * np.ones_like(effs)
    # counts_measured = effs * counts_true

    m = PrepareGammaSpec(len(counts))
    _channels = [393.0, 315, 532, 917.6, 2623]
    _energies = [218.6, 175, 296.5, 511, 1460.83]
    #  ============================  Add fake peaks =====================
    # fake_erg = 1600
    # c = PrepareGammaSpec.load_calibration('PHELIX_test')
    # ch = np.searchsorted(c.erg_bin_centers, fake_erg)
    # fake_eff = 1.2*np.interp(fake_erg, ergs, effs)
    # hist = TH1F(bin_left_edges=c.erg_bins)
    # for n in np.random.normal(fake_erg, 3, int(fake_eff*10000)):
    #     hist.Fill(n)
    # counts += hist.nominal_bin_values
    # _energies.append(fake_erg)
    # _channels.append(ch)
    # # plt.plot(c.erg_bin_centers, counts)
    # # plt.show()

    #  ======================================================

    _true_counts = list(10000*np.interp(_energies, ergs, effs))
    _true_counts[_energies.index(511)] = None
    _true_counts[_energies.index(1460.83)] = None
    m.add_peaks_4_calibration(counts, _channels, _energies, _true_counts)
    m.compute_calibration()
    m.plot_erg_spectrum(min_erg=50, max_erg=2000)
    m.save_calibration('PHELIX_test')

    c = PrepareGammaSpec.load_calibration('PHELIX_test')
    print(c.erg_fit)
    plt.show()
#
#
# #  Todo figure out FWHM