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
from uncertainties import ufloat, UFloat
import uncertainties.umath as umath

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


required_spec_attribs = ['erg', 't', 'eff', 'ch']
from sklearn.linear_model import LinearRegression
class LogPolyFit:
    def __init__(self, x, y, order=2):
        assert isinstance(order, int)
        if not isinstance(x[0], UFloat):
            x = unp.uarray(x, np.zeros_like(x))
        self.x = x
        self.y = y

        # x = unp.nominal_values(x)
        x_log = unp.log(x)
        x_err_log = np.array(unp.std_devs(x_log), dtype=np.float)
        x_log = np.array(unp.nominal_values(x_log), dtype=np.float)
        # y = np.array(unp.nominal_values(y), dtype=np.float)
        y_log = unp.log(y)
        y_err_log = np.array(unp.std_devs(y_log), dtype=np.float)
        y_log = np.array(unp.nominal_values(y_log), dtype=np.float)

        log_tgraph = ROOT.TGraphErrors(len(x), x_log, y_log, x_err_log, y_err_log)

        f_name = f'pol{order}'
        log_tgraph.Fit(f_name)
        __log_tf1__ = log_tgraph.GetListOfFunctions().FindObject(f_name)
        log_parameters = [ufloat(__log_tf1__.GetParameter(i), __log_tf1__.GetParError(i))
                           for i in range(order+1)]
        sum_terms = ['[0]'] + [f"TMath::Log(x)*[{i}]" for i in range(1, order+1)]
        func_str = f"{round(np.e, 4)}**({'+'.join(sum_terms)})"
        final_tf1 = ROOT.TF1('log_fit', func_str)
        for i in range(order+1):
            final_tf1.SetParameter(i, log_parameters[i].n)

        tgraph = ROOT.TGraphErrors(len(x), unp.nominal_values(x), unp.nominal_values(y), unp.std_devs(x), unp.std_devs(y))
        tgraph.Fit("log_fit")
        tgraph.Draw()
        ROOT_loop()
        print(func_str)

    def predict(self, x, nominal_values=False):
        return_array = True
        if not hasattr(x, '__iter__'):
            x = [x]
            return_array = False

        pred = np.zeros_like(x)
        pred = unp.uarray(pred, pred)
        for i, c in enumerate(self.parameters):
            pred += unp.log(x)**i*c
        out = np.e**pred
        if nominal_values:
            out = unp.nominal_values(out)
        if not return_array:
            assert len(out) == 1
            return out[0]
        else:
            return out
        #
        #
        # def predict(_x):
        #     pred = np.zeros_like(_x)
        #     for i in range(order+1):
        #         print(self.__log_tf1__.GetParameter(i))
        #         pred += self.__log_tf1__.GetParameter(i)*np.log(_x)**i
        #     return np.e**pred
        # sum_terms = ['[0]'] + [f"TMath::Log(x)^{i}*[{i}]" for i in range(1, order + 1)]
        # tf1_string = f'{round(np.e,4)}**({"+".join(sum_terms)})'
        # final_tf1 = ROOT.TF1('log_fit', tf1_string)
        # for i in range(order+1):
        #     print(i, self.__log_tf1__.GetParameter(i),  self.__log_tf1__.GetParError(i))
        #     final_tf1.SetParameter(i, self.__log_tf1__.GetParameter(i))
        #     final_tf1.SetParError(i, self.__log_tf1__.GetParError(i))
        #
        # plt.plot(ergs_all, predict(ergs_all), label='MOdel')
        #
        # plt.errorbar(ergs_all, [final_tf1.Eval(_x) for _x in ergs_all], label='test_model')
    # plt.plot(ergs_all, predict(ergs_all), label='MOdel')
    # ROOT_loop()


def peak_fit(hist: TH1F, center_guess, min_x=None, max_x=None, method='ROOT',
             sigma_guess=1, ):
    """
    Fits a peak.
    """
    assert method in ['ROOT']

    if isinstance(center_guess, UFloat):
        center_guess = center_guess.n
    assert isinstance(center_guess, (float, int))
    if not hist.is_density:
        warnings.warn('Histogram passed to fit_peak may not have density as bin values!'
                      ' To fix this, do hist /= hist.binvalues before passing to peak_fit')
    # Set background guess before range cut is applied. This could be useful when one wants the background estimate to
    # be over a larger range that may include other interfering peaks
    bg_guess = hist.median_y.n

    if min_x is not None or max_x is not None:
        bins = hist.__bin_left_edges__
        if max_x is None:
            max_x = bins[-1]
        if min_x is None:
            min_x = bins[0]

        select = np.where((bins >= min_x) & (bins <= max_x ))
        new_bins = bins[select]
        old_hist = hist
        hist = TH1F(bin_left_edges=new_bins)
        hist += old_hist.bin_values[select]

    if method == 'ROOT':
        func = ROOT.TF1('peak_fit', '[0]*TMath::Gaus(x,[1],[2], kTRUE) + [3]')
        amp_guess = np.sum(hist.bin_widths*(hist.bin_values-hist.median_y)).n
        func.SetParameter(0, amp_guess)
        print('amp_guess', amp_guess)
        print('center_guess', center_guess)
        print('sigma_guess', sigma_guess)
        print('bg_guess',bg_guess)
        func.SetParameter(1, center_guess)
        func.SetParameter(2, sigma_guess)
        func.SetParameter(3, bg_guess)
        hist.__ROOT_hist__.Fit('peak_fit')



from GlobalValues import shot_groups
from Shot_data_analysis.PHELIXDataTTree import get_global_energy_bins
from JSB_tools import ROOT_loop

def erg_efficiency(erg):
    a = ufloat(45.6, 1.5)                           # From Pascal's thesis
    b = ufloat(3.63E-3, 0.13E-3)                    # From Pascal's thesis
    c = ufloat(2.43, 0.12)                        # From Pascal's thesis
    return (a*np.e**(-b*erg)+c)/100.           # From Pascal's thesis

ergs_all = np.array(get_global_energy_bins(200))
selector =  list(map(int, (len(ergs_all)-1)*np.linspace(0.2, 1, 10)))
ergs = ergs_all[selector]
effs = erg_efficiency(ergs)
# print(effs)
g = LogPolyFit(ergs, effs)
plt.plot(ergs_all, unp.nominal_values(erg_efficiency(ergs_all)), label='Actual')
# y = g.predict(ergs_all, False)
# plt.errorbar(ergs_all, unp.nominal_values(y), unp.std_devs(y))
plt.legend()
plt.show()
# g.Draw()
# ROOT_loop()
# plt.show()
# tree = shot_groups['He+Ar'].tree
# center = 297
# dx = 10
# _min, _max = center-dx, center+dx
# hist = TH1F(bin_left_edges=get_global_energy_bins(_min, _max))
# hist.Project(tree, 'erg', '(t<300)*(1.0/eff)')
# hist /= hist.bin_widths
#
# peak_fit(hist, center, )
# hist.Draw()
# ROOT_loop()

#
# class MakeROOTSpectrum:
#     data_dir = cwd/"Spectra_data"
#
#     def __init__(self, path, channels, ch_2_erg_coefficients, fwhm_coefficients):
#         path = Path(path)
#         assert path.parent.exists()
#         if not MakeROOTSpectrum.data_dir.exists():
#             MakeROOTSpectrum.data_dir.mkdir()
#         self.root_file = ROOT.TFile(path, 'recreate')
#         self.tree = ROOT.TTree('spectrum', 'spectrum')
#         self.ch_2_erg_coefficients = ch_2_erg_coefficients
#         self.fwhm_coefficients = fwhm_coefficients
#         # self.
#
#
#
#
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
#
#     Todo:
#         -Make ch_to_erg a TF1.
#         -Implement a way to change the coeffs of the TF1 in the file (all files in the case of a TChain)
#         -A function that accepts a cut (str) and returns a UFloat equal to the efficiency of the selection plus
#             uncertainties
#         -Plot time dependence of peak/multiple peaks simultaneously. Maybe options for several methods of background
#             subtraction.
#             --Make it easy to overlay two spectra in a ROOT TCanvas
#             --Auto / manual selection of non-uniform time bins
#         -Area under peak calculations
#         -Energy spectrum plotting – overlaying of multiple ROOTSpecrtrum instances
#         -Energy calibration tool
#         -Efficiency calibration tool. Option for no erg/eff data saved to ROOT files.
#     """
#     def __init__(self, root_file_path, overwrite=False):
#         self.erg_coeffs = [0, 1, 0]
#         self.shape_cal_coeffs = [1,0, 0]
#
#
#     #
#     # def __init__(self, root_file):
#     #     assert isinstance(root_file, ROOT.TFile)
#     #     self.root_file = root_file
#     #     # root_file = ROOT.TFile(str(path_to_root_file))
#     #     self.tree = root_file.Get('tree')
#     #     self.erg_2_eff_hist = self.__load_erg_2_eff_hist__()
#     #     self.energy_coeffs: List[UFloat] = self.__load_erg_calibration__()
#     #
#     # def __load_erg_calibration__(self) -> List[UFloat]:
#     #     keys = self.root_file.GetListOfKeys()
#     #     assert 'ch_2_erg' in [k.GetName() for k in keys], 'No energy calibration available. '
#     #     tf1 = self.root_file.Get('ch_2_erg')
#     #     # print([(tf1.GetParameter(i), (tf1.GetParError(i))) for i in range(3)])
#     #     coeffs = [ufloat(tf1.GetParameter(i), tf1.GetParError(i)) for i in range(3)]
#     #     return coeffs
#     #
#     # def __load_erg_2_eff_hist__(self) ->TH1F:
#     #     """Used to check for ch_2_erg_hist and erg_2_eff_hist histograms in the root file."""
#     #
#     #     keys = [k.GetName() for k in self.root_file.GetListOfKeys()]
#     #     assert 'erg_2_eff_hist' in keys, f'ROOT file does not contain "ch_2_erg_hist" histogram, a histogram channel vs' \
#     #                                     f' gamma energy + uncertainty. Available keys:\n{keys}'
#     #     # assert 'erg_2_eff_hist' in keys, 'ROOT file does not contain "erg_2_eff_hist" histogram,' \
#     #     #                                  ' a histogram of gamma energy vs efficiency + uncertainty.'
#     #     erg_2_eff_hist = TH1F.from_native_ROOT_hist(self.root_file.Get('erg_2_eff_hist'))
#     #     return erg_2_eff_hist
#     #
#     # @classmethod
#     # def from_path(cls, path):
#     #     return cls(ROOT.TFile(str(path)))
#     #
#     # @classmethod
#     # def from_tree(cls, tree):
#     #     if isinstance(tree, ROOT.TChain):
#     #         file = tree.GetFile()
#     #     elif isinstance(tree, ROOT.TTree):
#     #         file = tree.GetCurrentFile()
#     #     else:
#     #         assert False, '`tree` is not a ROOT.TTree or ROOT.TChain instance'
#     #
#     #     return cls(root_file=file)
#     #
#     # @property
#     # def ergs(self):
#     #     return unp.nominal_values(self.erg_2_eff_hist.bin_values)
#     #
#     # @property
#     # def __erg_sigmas(self):
#     #     return unp.std_devs(self.ch_2_erg_hist.bin_values)
#     #
#     # @property
#     # def __erg_fwhms(self):
#     #     return unp.std_devs(self.ch_2_erg_hist.bin_values)*2.355
#     #
#     # def erg_fwhm(self, erg):
#     #     i = np.searchsorted(self.ergs, erg)
#     #     s = slice(i-5, i+5)
#     #     return np.interp(erg, self.ergs[s], self.__erg_fwhms[s])
#     #
#
#
#
#
#
#
# if __name__ == '__main__':
#     # from GlobalValues import shot_groups
#     # tree = shot_groups['He+Ar'].tree
#     # # s = ROOTSpectrum.from_tree(tree)
#     # s2 = ROOTSpectrum.from_path('/Users/burggraf1/PycharmProjects/PHELIX/Shot_data_analysis/PHELIX2019ROOT_TREES/_shot42.root')
#     # # print(s.erg_fwhm(50))
#     # print (s2.energy_coeffs)
#     # s2.erg_2_eff_hist.plot()
#     #
#     # def erg_efficiency(erg):
#     #     a = ufloat(45.6, 1.5)  # From Pascal's thesis
#     #     b = ufloat(3.63E-3, 0.13E-3)  # From Pascal's thesis
#     #     c = ufloat(2.43, 0.12)  # From Pascal's thesis
#     #     return (a * np.e ** (-b * erg) + c) / 100.  # From Pascal's thesis
#     #
#     # x = np.linspace(0, 4000, 300)
#     # plt.figure()
#     # plt.plot(x, unp.nominal_values(erg_efficiency(x)))
#
#     # @ROOT.Numba.Declare(['float'], 'float')
#     # def func(x):
#     #     return x**2
#     # tree.Draw("Numba::func(erg)")
#
#     tfile = ROOT.TFile(str(cwd/'del'), 'recreate')
#     a = np.array([1,2,3,4,5], dtype=np.int)
#     aR = ROOT.TArrayI(len(a))
#     for i in range(len(a)):
#         aR[i] = a[i]
#     for i in aR:
#         print(i)
#     tfile.WriteObject(aR, 'omhomhomh')
#     for k in tfile.GetListOfKeys():
#         print('key', k)
#     # print(aR)
#     #
#     # tree.Draw('ch_2erg(ch)')
#     #
#     # for k in file.GetListOfKeys():
#     #     print(k.GetName(), (k.GetCycle()))
#     #     if k.GetCycle() == 2:
#     #         print('Deleting')
#     #         del k
#     # for k in file.GetListOfKeys():
#     #     print(k.GetName(), k.GetCycle())
#     # n = Nuclide.from_symbol('U238')
#     # fiss_yields = n.independent_gamma_fission_yield()
#     #
#     # def weighting(name):
#     #     try:
#     #         return fiss_yields.yields[name]
#     #     except KeyError:
#     #         return 0
#     #
#     # print(gamma_search(218, 2, 20, 60*3, nuclide_weighting_function=weighting))
#     # Gdelta_t = get_time_bins()[-1]
#     # data_dir = DECAY_PICKLE_DIR / '__fast__gamma_dict__.marshal'
#     #
#     # tree = shot_groups['He+Ar'].tree
#     # hist = TH1F(bin_left_edges=get_global_energy_bins())
#     # hist.Project(tree, 'erg', weight='1.0/eff')
#     # hist.plot()
#     # print(Nuclide.from_symbol('U238').independent_gamma_fission_yield_gef()[1])
#     # # for n, yield_ in Nuclide.from_symbol('U238').independent_gamma_fission_yield_gef().items():
#     # #     print(n, yield_)
#     # for i in gamma_search(2612, e_sigma=4, delta_t=Gdelta_t):
#     #     print(i)
#     # plt.show()