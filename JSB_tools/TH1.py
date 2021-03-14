from __future__ import annotations
try:
    import ROOT
except ModuleNotFoundError:
    assert False, 'PyROOT must be installed and working in order to use TH1 module'

import numpy as np
from uncertainties import unumpy as unp
from uncertainties import ufloat, UFloat
from uncertainties.core import Variable, AffineScalarFunc, wrap
from numbers import Number
from warnings import warn
from lmfit.models import GaussianModel, ConstantModel, LinearModel
from lmfit import Model, fit_report
from scipy.signal import find_peaks
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List, Union, Sequence
import re
import os
import matplotlib.offsetbox as offsetbox
from scipy.stats import norm


class HistoBinMerger:
    def __init__(self, start_stop_tuples, new_bin_left_edges, old_hist):
        self.__start_stop_tuples__ = start_stop_tuples
        assert isinstance(old_hist, TH1F)
        self.__old_bin_left_edges__ = old_hist.__bin_left_edges__
        self.__new_bin_left_edges__ = new_bin_left_edges


def binned_median(bin_left_edges, weights):
    x = bin_left_edges
    assert len(bin_left_edges) == len(weights) + 1
    if not isinstance(weights, list):
        weights = list(weights)
    weights.insert(0, 0)
    cum_sum = np.cumsum(weights)
    median_w = cum_sum[-1]/2
    index = np.searchsorted(cum_sum, median_w) - 1
    dy = cum_sum[index + 1] - cum_sum[index]
    b_width = x[index+1] - x[index]
    error = median_w - cum_sum[index]
    dx = error/dy*b_width
    return x[index] + dx


def rolling_median(window_width, values):
    """
    Rolling median over a uniform window. Window is clipped at the edges.
    Args:
        window_width: Size of independent arrays for median calculations.
        values: array of values

    Returns:

    """
    n = min([window_width, len(values)])
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    window_indicies = (range(max([0, i - n // 2]), min([len(values) - 1, i + n // 2])) for i in range(len(values)))

    medians = np.array([np.median(values[idx]) for idx in window_indicies], dtype=np.ndarray)

    return medians


def MAD(values):
    """
    MAD (Median Absolute Deviation).
    Args:
        values: array of values.

    Returns: MAD

    """
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    median = np.median(values)
    return np.median(np.abs(values - median))


def rolling_MAD(window_width, values):
    """
    Calculate the median absolute deviation over a uniform rolling window. Window is clipped at the edges.
    Args:
        window_width: Width of window
        values: Array-like

    Returns: Result.

    """
    n = min([window_width, len(values)])
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    window_indicies = (range(max([0, i - n // 2]), min([len(values) - 1, i + n // 2])) for i in range(len(values)))
    print(list(window_indicies))
    out = np.array([np.median(np.abs(values[idx] - np.median(values[idx]))) for idx in window_indicies],
                   dtype=np.ndarray)

    return out


def convolve_uniform(window_width: int, values):
    """
    Convolve array with a square pulse. Sum of array values is preserved.
    Args:
        window_width: size of square wave for convolution
        values: Array to convolve

    Returns:

    """
    assert isinstance(window_width, int), '`window_width` must be expressed in a number of bins,' \
                                          ' and so must be an integer'
    v = np.ones(window_width)
    v /= len(v)
    return np.convolve(values, v, mode='same')


def convolve_gaus(sigma: float, values):
    """
    Convolve array of values with a gaussian.  Sum of array values is preserved.
    Args:
        sigma: The width of window (number of bins).
        values: Array to convolve

    Returns: result

    """
    x = np.arange(int(-6*sigma), int(6*sigma)+1)
    v = norm.pdf(x=x, scale=sigma)
    v /= sum(v)
    return np.convolve(values, v, mode='same')


class TH1F:
    title_number = 0
    titles = []
    ROOT_histos = []
    __histo_pad_dict__ = {}

    def __init__(self, min_bin=None, max_bin=None, nbins=None, bin_left_edges=None, bin_width=None, title=None, ROOT_hist=None):
        if ROOT_hist is not None:
            self.__ROOT_hist__ = ROOT_hist
            self.__bin_left_edges__ = np.array(
                [ROOT_hist.GetBinLowEdge(i) for i in range(1, ROOT_hist.GetNbinsX() + 2)], dtype=np.float)
            self.title = ROOT_hist.GetName()
            self.n_bins = len(self.__bin_left_edges__) - 1
        else:
            if bin_left_edges is None:
                arg_error_msg = "\nIf not passing `bin_left_edges` to TH1F, you must pass `min_bin`, `max_bin`" \
                                ", *and* `nbins` or `binwidth` (but not both `nbins` and " \
                                "`binwidth`)"
                assert None not in [min_bin, max_bin], arg_error_msg
                if bin_width is None:
                    assert nbins is not None, arg_error_msg
                    assert isinstance(nbins, int), '`nbins` arg must be an integer'
                    self.__bin_left_edges__ = np.linspace(min_bin, max_bin, nbins + 1, dtype=np.float)
                else:
                    assert nbins is None, arg_error_msg
                    assert isinstance(bin_width, Number), '`bin_width` arg must be a number'
                    self.__bin_left_edges__ = np.arange(min_bin, max_bin + bin_width, bin_width, dtype=np.float)
            else:
                assert len(bin_left_edges) >= 2, "`bin_left_edges` argument must be iterable of length greater than 1"
                assert all([isinstance(x, Number) for x in bin_left_edges]), 'All values of `bin_left_edges` must be a'\
                                                                             ' number'
                assert all([x is None for x in [bin_width, min_bin, max_bin, nbins]]), '`bin_left_edges` was passed to'\
                                                                                       ' TH1F. Bins are fullly' \
                                                                                       ' specified.'\
                                                                                       ' No other bin specification '\
                                                                                       'arguments allowed in this case.'

                self.__bin_left_edges__ = np.array(bin_left_edges, dtype=np.float)

            if title is None:
                title = "hist{0}".format(TH1F.title_number)
                TH1F.title_number += 1
            else:
                title = str(title)
                assert title not in TH1F.titles

            self.n_bins = len(self.__bin_left_edges__) - 1
            self.__ROOT_hist__ = ROOT.TH1F(title, title, self.n_bins, self.__bin_left_edges__)
            self.title = title

        self.bin_centers = np.array(
            [0.5 * (b2 + b1) for b1, b2 in zip(self.__bin_left_edges__[:-1], self.__bin_left_edges__[1:])])

        self.bin_widths = np.array(
            [c2 - c1 for c1, c2 in zip(self.__bin_left_edges__[:-1], self.__bin_left_edges__[1:])])

        self.draw_expression = None
        self.cut = None
        self.draw_weight = None

    @classmethod
    def from_data_points(cls, data: Sequence[Number], bins=None, weights=None):
        if bins is None:
            bins = 'auto'
        bin_values, bin_lef_edges = np.histogram(data, bins=bins, weights=weights)
        hist = cls(bin_left_edges=bin_lef_edges)
        bin_errors = np.sqrt(bin_values)
        bin_values = unp.uarray(bin_values, bin_errors)
        hist.__set_bin_values__(bin_values)
        return hist

    @classmethod
    def from_x_and_y(cls, x: Sequence[float], y: Sequence[Union[UFloat, float]]) -> TH1F:
        """
        Generate a histogram from x and y points. This generally shouldn't be used.

        Args:
            x: bin centers
            y: bin values (can be floats or UFloats)

        Returns: histogram.

        """
        assert isinstance(x, Sequence) and len(x) >= 2, '`x` must be a collection with minimum length of 2'
        assert not isinstance(x[0], UFloat), 'x bust be a Number, not UFloat'
        y = [i.n if isinstance(i, UFloat) else i for i in y]
        y_err = [i.std_dev if isinstance(i, UFloat) else 0 for i in y]
        bin_left_edges = [x[0] - (x[1]-x[0])/2]
        for xi in x:
            bin_left_edges.append(-bin_left_edges[-1] + 2*xi)
        assert all([x1-x0 > 0 for x0, x1 in zip(bin_left_edges[:-1], bin_left_edges[1:])]),\
            'Provided bins are not increasing monotonically!'

        bin_values = unp.uarray(y, y_err)
        hist = cls(bin_left_edges=bin_left_edges)
        hist.__set_bin_values__(bin_values)
        return hist

    def __uniform_binsQ__(self):
        """"
        test whether bin widths are all equal.
        """
        uniform_bins = True
        for b in self.bin_widths:
            if not np.isclose(b, self.bin_widths[0]):
                uniform_bins = False
                break
        return uniform_bins

    def convolve_uniform(self, window_width):
        """
        Convolve the bin values with a uniform square pulse with width `window_width`.
        Args:
            window_width: The width of window (number of bins).

        Returns: None, modifies the histogram.

        """
        self.__set_bin_values__(convolve_uniform(window_width, self.bin_values))

    def convolve_gaus(self, sigma: float):
        """
        Convolve the bin values with a gaussian pulse.
        Args:
            sigma: The width of window (number of bins).

        Returns: None, modifies the histogram.

        """
        self.__set_bin_values__(convolve_gaus(sigma, self.bin_values))

    @property
    def title(self) -> str:
        return self.__ROOT_hist__.GetTitle()

    @title.setter
    def title(self, title: str):
        self.__ROOT_hist__.SetTitle(str(title))
        return title

    @property
    def bin_width(self):
        assert len(set(
            self.bin_widths)) == 1, \
                "No bin_width parameter with histogram of varying bin_widths. Use self.bin_widths instead" \
                "\n\tbin_widths={0}".format(self.bin_widths)
        return self.bin_widths[0]

    def merge_bins(self, merge_instance):
        new_bin_values = []
        assert isinstance(merge_instance, HistoBinMerger)
        assert np.allclose(self.__bin_left_edges__, merge_instance.__old_bin_left_edges__), \
            "Attempted to merge incompatible histogram bin specs!\nbins1 = {0}\nbins2 = {1}" \
                .format(self.__bin_left_edges__,
                        merge_instance.__old_bin_left_edges__)
        start_stops_tuples = merge_instance.__start_stop_tuples__

        for index_start, index_stop in start_stops_tuples:
            assert 0 <= index_stop < len(self), "Merging bins out of range (index_stop = {})".format(index_stop)
            assert 0 <= index_start < len(self), "Merging bins out of range (index_start = {})".format(index_start)
            assert index_start <= index_stop, \
                "Attempted to merge bins with an index_stop less than (or = to) index_start"
            new_bin_values.append(np.sum(self.bin_values[index_start: index_stop + 1]))

        _new_hist = TH1F(bin_left_edges=merge_instance.__new_bin_left_edges__)
        _new_hist.__set_bin_values__(new_bin_values)
        return _new_hist

    def set_min_bin_value(self, y=0):
        bin_values_n = np.where(self.bin_values >= y, unp.nominal_values(self.bin_values), y)
        new_bin_values = unp.uarray(bin_values_n, self.bin_std_devs)
        self.__set_bin_values__(new_bin_values)

    def peak_fit(self, peak_center=None, model="gaussian", background="constant", sigma_fix=None, amplitude_fix=None, c_fix=None,
                 divide_by_bin_width=False, full_range=None):
        if isinstance(background, str):
            background = background.lower()
        if divide_by_bin_width:
            self /= self.bin_widths

        def lin_func(x, slope, intercept):
            return intercept + slope * (x - peak_center)

        if background is None:
            model = GaussianModel()
        elif isinstance(background, str):
            if background == "constant":
                model = GaussianModel() + Model(lin_func)
            if background == "linear":
                model = GaussianModel() + Model(lin_func)
        else:
            assert False, "Invalid background. Must be None, 'linear', or 'constant'. "

        if full_range is False:
            peaks_ix, peak_infos = find_peaks(unp.nominal_values(self.bin_values),
                                              prominence=unp.std_devs(self.bin_values),
                                              width=0, height=0)
            prominences = peak_infos["prominences"]

            widths = peak_infos["widths"]

            if peak_center is None:
                assert len(prominences) != 0, "No peaks found to fit"
                peak_info_ix = np.argmax(prominences)
            else:
                peak_info_ix = np.abs(self.bin_centers[peaks_ix] - peak_center).argmin()
            best_peak_ix = peaks_ix[peak_info_ix]
            peak_center = self.bin_centers[best_peak_ix]
            peak_width_ix = int(round(widths[peak_info_ix]))
            peak_width = widths[peak_info_ix]*self.bin_widths[best_peak_ix]
            peak_height = prominences[peak_info_ix]

            valleys_ix, _ = find_peaks(-unp.nominal_values(self.bin_values) + max(unp.nominal_values(self.bin_values)),
                                    prominence=np.abs(peak_height*0.2-2*unp.std_devs(self.bin_values)),
                                              width=0, height=0)
            if len(np.where(valleys_ix < best_peak_ix)[0]):
                min_x_ix = valleys_ix[np.where(valleys_ix < best_peak_ix)[0][-1]]
                if min_x_ix > best_peak_ix - peak_width_ix:
                    min_x_ix = best_peak_ix - peak_width_ix

            else:
                min_x_ix = best_peak_ix - peak_width_ix

            if len(np.where(valleys_ix > best_peak_ix)[0]):
                max_x_ix = valleys_ix[np.where(valleys_ix > best_peak_ix)[0][0]]
                if max_x_ix < best_peak_ix + peak_width_ix:
                    max_x_ix = best_peak_ix + peak_width_ix
            else:
                max_x_ix = best_peak_ix + peak_width_ix
            diff_ix = max_x_ix - min_x_ix

            if diff_ix < 7:
                max_x_ix += diff_ix//2
                min_x_ix -= diff_ix//2
                max_x_ix += diff_ix%2

            max_x_ix = min([len(self) - 1, max_x_ix])
            min_x_ix = max([0, min_x_ix])
            min_x = self.bin_centers[min_x_ix]
            max_x = self.bin_centers[max_x_ix]

            fit_selector = np.where((self.bin_centers >= min_x) &
                                    (self.bin_centers <= max_x))

            _x = self.bin_centers[fit_selector]
            _y = unp.nominal_values(self.bin_values[fit_selector])
            _y_err = unp.std_devs(self.bin_values[fit_selector])

            model_params = model.make_params()
            sigma_guess = peak_width / 2.
            amplitude_guess = peak_height / 0.3989423 / sigma_guess
            c0_guess = unp.nominal_values(self.bin_values)[peaks_ix[peak_info_ix]] - peak_height
            model_params["amplitude"].set(value=amplitude_guess)
            model_params["sigma"].set(value=sigma_guess, min=1)
            if background == "constant":
                model_params["intercept"].set(value=c0_guess)
                model_params["slope"].set(value=0, vary=False)
            if background == "linear":
                dx = self.bin_centers[fit_selector][-1] - self.bin_centers[fit_selector][0]
                dy = unp.nominal_values(self.bin_values[fit_selector[0][-1]]) - \
                     unp.nominal_values(self.bin_values[fit_selector[0][0]])
                model_params["intercept"].set(value=c0_guess)
                model_params["slope"].set(value=dy / dx)
            model_params["center"].set(value=peak_center)
        else:
            model_params = model.make_params()
            sigma_guess = 2
            amplitude_guess = np.max(self.nominal_bin_values)
            c0_guess = 0
            model_params["amplitude"].set(value=amplitude_guess)
            model_params["sigma"].set(value=sigma_guess, min=1)
            if background == "constant":
                model_params["intercept"].set(value=c0_guess)
                model_params["slope"].set(value=0, vary=False)
            if background == "linear":

                model_params["intercept"].set(value=c0_guess)
                model_params["slope"].set(value=0)
            model_params["center"].set(value=peak_center, min=peak_center - 3.5, max=peak_center + 3.5)
            _x = self.bin_centers
            _y = unp.nominal_values(self.bin_values)
            _y_err = unp.std_devs(self.bin_values)

        weights = 1.0/np.where(abs(_y_err)>0, abs(_y_err),  1)
        fit_result = model.fit(_y, x=_x, weights=weights,
                               params=model_params)
        eval_fit = lambda xs: model.eval(fit_result.params, x=xs)
        if divide_by_bin_width:
            self *= self.bin_widths

        return fit_result, eval_fit

    def get_stats_text(self):
        counts = self.__ROOT_hist__.GetEntries()
        quantiles = self.quantiles(4)
        s = ["counts {:.2e}".format(counts)]
        s += ["mean  {:.2e}".format(self.mean.n)]
        s += ["std   {:.2e}".format(self.std.n)]
        s += ["25%   {:.2e}".format(quantiles[0])]
        s += ["50%   {:.2e}".format(quantiles[1])]
        s += ["75%   {:.2e}".format(quantiles[2])]
        return s

    def plot(self, ax=None, logy=False, logx=False, xmax=None, xmin=None, leg_label=None, xlabel=None,
             ylabel=None, show_stats=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        else:
            if ax is plt:
                ax = ax.gca()
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if logy:
            ax.set_yscale(value='log')
        if logx:
            ax.set_xscale(value='log')

        if self.title is not None:
            ax.set_title(self.title)

        if xmin is None:
            xmin = self.__bin_left_edges__[0]
        if xmax is None:
            xmax = self.__bin_left_edges__[-1]
        s = np.where((self.bin_centers <= xmax) & (xmin <= self.bin_centers))

        ax.errorbar(self.bin_centers[s], self.nominal_bin_values[s],
                    yerr=self.bin_std_devs[s], ds="steps-mid", label=leg_label, **kwargs)
        if show_stats:
            text = self.get_stats_text()
            text = '\n'.join(text)
            # ax.text(0.8, 0.95-0.03*h, text,  transform=ax.transAxes)
            ob = offsetbox.AnchoredText(text, loc=1)
            ax.add_artist(ob)

        return ax

    def convolve_median(self, window_width):
        """
        Replace the value in each bin[i] with the median of bins within a rolling window of width
        `window_width` centered around bin[i].
        Args:
            window_width: width of rolling window

        Returns: None, modifies hist.
        """
        new_bin_values = rolling_median(window_width, self.bin_values)
        self.__set_bin_values__(new_bin_values)

    def dist_from_rolling_median(self, window_width: float):
        """
        Takes the median of a rolling window and then subtract the result from the values in each bin.
        This operation has the tendency of removing backgrounds/baselines.
        Setting `window_width` to None will pick a `window_width` that is about 6 times the typical FWHM
        of your spectra's features.
        Args:
            window_width: Width of window (in number of bins).

        Returns: None, Modifies the histogram.

        """
        # if window_width is None:
        #     peak_idx, peak_infos = find_peaks(self.nominal_bin_values,
        #                                       prominence=2.5 * self.std_errs,
        #                                       plateau_size=[0, 20],
        #                                       width=2)
        nom_vals = self.nominal_bin_values  # for optimization
        medians = rolling_median(window_width, nom_vals)
        self.__isub__(medians)

    def get_merge_obj_max_rel_error(self, max_rel_error, merge_range_x=None):
        bin_start_stops = []
        if merge_range_x is not None:
            assert len(merge_range_x) == 2

        def get_rel_error(_value):
            if _value.n == 0:
                return 0
            return abs(_value.std_dev / _value.n)

        start = None
        cum_value = None
        new_bin_left_edges = []
        for index, value in enumerate(self.bin_values):
            if cum_value is None:
                cum_value = value
                start = index
                new_bin_left_edges.append(self.__bin_left_edges__[index])
            else:
                cum_value += value

            if merge_range_x is not None:
                is_mergeable = merge_range_x[0] <= self.bin_centers[index] <= merge_range_x[1]
            else:
                is_mergeable = True
            if (not is_mergeable) or (get_rel_error(cum_value) <= max_rel_error) or (index + 1 == len(self)):
                cum_value = None
                bin_start_stops.append((start, index))
                if index + 1 == len(self):
                    new_bin_left_edges.append(self.__bin_left_edges__[index + 1])
        new_bin_left_edges = np.array(new_bin_left_edges)
        return HistoBinMerger(np.array(bin_start_stops), new_bin_left_edges, self)

    @staticmethod
    def get_n_events_with_cut(tree, cut, weight: str = None):
        assert isinstance(cut, str)
        _hist = TH1F(0, 1, 1)
        if weight is not None:
            assert isinstance(weight, str)
            cut = '({0})*({1})'.format(weight, cut)
        assert isinstance(tree, ROOT.TTree), '"tree" must be a TTree instance.'
        tree.Project(_hist.__ROOT_hist__.GetName(), '0.5', cut)
        return _hist.bin_values[0]

    def Project(self, tree, drw_exp, cut=None, max_events=None, options="", weight=None, start=None):
        assert isinstance(drw_exp, str), '`drw_exp` must be a string'
        assert isinstance(tree, ROOT.TTree), '"tree" must be a TTree instance.'

        ROOT_hist = self.__ROOT_hist__

        if isinstance(weight, str) and len(weight) != 0:
            if cut is None:
                cut = "({weight})".format(weight=weight)
            else:
                cut = "({weight})*({cut})".format(weight=weight, cut=cut)
        else:
            if cut is None:
                cut = ''

        for b in tree.GetListOfBranches():
            if 'nps' in b.GetName():
                if 'wgt' not in cut:
                    warn(
                        '\n\nWhen analysing MCNP PTRAC, weights need to be included!\n'
                        'Cuts should be of the form "wgt*(<cut>)".\n')
                break

        if max_events is not None:
            result = tree.Project(ROOT_hist.GetName(), drw_exp, cut, options, int(max_events),
                                  0 if start is None else start)
        else:
            result = tree.Project(ROOT_hist.GetName(), drw_exp, cut, options)
        if isinstance(weight, (Number, AffineScalarFunc, Variable)):
            self *= weight

        # ROOT_hist.update_bin_containers_from_hist()
        # self.__ROOT_hist__.SetEntries(self.nEntries)

        return result

    def GetEntries(self):
        return self.__ROOT_hist__.GetEntries()

    def SetLineColor(self, color, cycler=None):
        """kWhite  = 0,   kBlack  = 1,   kGray    = 920,  kRed    = 632,  kGreen  = 416,
        kBlue   = 600, kYellow = 400, kMagenta = 616,  kCyan   = 432,  kOrange = 800,
        kSpring = 820, kTeal   = 840, kAzure   =  860, kViolet = 880,  kPink   = 900"""
        colors = {"black": ROOT.kBlack, "red": ROOT.kRed, "blue": ROOT.kBlue, "yellow": ROOT.kYellow,
                  "green": ROOT.kGreen}
        color = colors[color]

        self.__ROOT_hist__.SetLineColor(color)

    def __Cycle_line_styles__(self, i):
        cs = [4, ROOT.kRed, ROOT.kBlack, ROOT.kGray, ROOT.kGreen, ROOT.kYellow, ROOT.kOrange, ROOT.kPink]
        stylers = [1, 2, 9, 10]
        line_style = stylers[i // len(cs)]
        color_i = i % len(cs)
        color = cs[color_i]

        self.__ROOT_hist__.SetLineColor(color)
        self.__ROOT_hist__.SetLineStyle(line_style)

    def FillRandom(self, func="gaus", n=10000):
        self.__ROOT_hist__.FillRandom(func, n)

    def Fill(self, value, weight=1):
        self.__ROOT_hist__.Fill(value, weight)

    def Rebin(self, n):
        new_hist = self.__copy__()
        new_hist.__ROOT_hist__.Rebin(n)
        return TH1F(ROOT_hist=new_hist.__ROOT_hist__)

    @staticmethod
    def __update_pads__():
        for key, value in TH1F.__histo_pad_dict__.items():
            _max = None
            _min = None
            for hist in value:
                assert isinstance(hist, TH1F)
                if _max is None:
                    _max = max(hist.nominal_bin_values)
                elif max(hist.nominal_bin_values) > _max:
                    _max = max(hist.nominal_bin_values)
                if _min is None:
                    _min = min(hist.nominal_bin_values)
                elif min(hist.nominal_bin_values) < _min:
                    _min = min(hist.nominal_bin_values)
            value[0].__ROOT_hist__.SetMaximum(_max + (0.15 * abs(_max)))
            if _min < 0:
                value[0].__ROOT_hist__.SetMinimum(_min - (0.15 * abs(_min)))

    def Draw(self, options=""):
        self.__ROOT_hist__.Draw(options)
        if ROOT.gPad.GetName() not in TH1F.__histo_pad_dict__:
            TH1F.__histo_pad_dict__[ROOT.gPad.GetName()] = [self]
        else:
            TH1F.__histo_pad_dict__[ROOT.gPad.GetName()].append(self)
        self.__Cycle_line_styles__(len(TH1F.__histo_pad_dict__[ROOT.gPad.GetName()]) - 1)
        self.__update_pads__()

    def find_bin_index(self, x: float) -> int:
        """Return bin index for the bin containing the value x"""
        assert self.__bin_left_edges__[0] <= x <= self.__bin_left_edges__[-1], \
            "x of {0} out of range, between {1} and {2}!".format(x, self.__bin_left_edges__[0],
                                                                 self.__bin_left_edges__[1])
        return self.__ROOT_hist__.FindBin(x) - 1

    def SetTitle(self, title):

        self.__ROOT_hist__.SetTitle(str(title))

    def GetTitle(self):
        return self.__ROOT_hist__.GetTitle()

    def __get_bin_values_from_ROOT_hist__(self, hist):
        values = np.zeros(hist.GetNbinsX())
        errors = np.zeros(hist.GetNbinsX())
        assert len(errors) == len(self), "{0}, {1}".format(len(errors), len(self))
        for raw_index, b in enumerate(self.bin_centers):
            bin_index = hist.FindBin(b)
            assert np.isclose(b, hist.GetBinCenter(bin_index)), \
                "incompatible bin centers. '{0}' should equal '{1}'. This should not happen"\
                     .format(b,
                             hist.GetBinCenter(
                                bin_index))
            values[raw_index] = hist.GetBinContent(bin_index)
            errors[raw_index] = hist.GetBinError(bin_index)
        return unp.uarray(values, errors)

    @property
    def mean(self) -> UFloat:
        if np.sum(unp.nominal_values(self.bin_values)) == 0:
            if np.sum(unp.std_devs(self.bin_values)) == 0:
                return ufloat(np.nan, np.nan)
            else:
                out = np.average(self.bin_centers,
                                  weights=unp.uarray(self.bin_std_devs / 2., self.bin_std_devs / 2.))
                out = ufloat(out, 0)
                return out

        return np.average(self.bin_centers, weights=np.abs(self.bin_values))

    @property
    def median(self):
        return binned_median(self.__bin_left_edges__, self.bin_values)

    def quantiles(self, n_quantiles, bin_indicies=False) -> List[float]:
        """
        Returns the bin values which split the hist into sections of equal probabilities. e.g. n_quantiles = 2 gives
         the median
        Args:
            n_quantiles:
            bin_indicies: If True, returned list represents bin indicies. Otherwise, it represents interpolated points
                on the x axis.

        Returns: n_quantiles-1 length list
        """
        cumsum = np.concatenate(([0], np.cumsum(self.nominal_bin_values)))
        sum_points = np.linspace(0, cumsum[-1], n_quantiles+2-1)[1:-1]

        left_best_indicies = np.searchsorted(cumsum, sum_points) - 1
        residue = sum_points - cumsum[left_best_indicies]
        fractions = residue/self.nominal_bin_values[left_best_indicies]
        dxs = fractions*self.bin_widths[left_best_indicies]
        result = self.__bin_left_edges__[left_best_indicies] + dxs

        if bin_indicies:
            result = np.array(map(int, result))

        return result

    def quantiles_plot(self, n_quantiles, ax, label=None):
        x_points = self.quantiles(n_quantiles)
        if ax is plt:
            _min, _max = ax.ylim()
        else:
            _min, _max = ax.get_ylim()
        ax.vlines(x_points, _min, _max, label=label, ls='--')

    @property
    def rel_errors(self):
        return np.array([abs(e.std_dev / e.n) if e.n != 0 else 0 for e in self.bin_values ])

    @property
    def skewness(self):
        assert sum(unp.nominal_values(self.bin_values)) != 0
        return np.average(((self.bin_centers - self.mean) / self.std) ** 3, weights=np.abs(self.bin_values))

    @property
    def std(self):
        _sqrt = wrap(np.sqrt)

        if np.sum(unp.nominal_values(self.bin_values)) == 0:
            if np.sum(unp.std_devs(self.bin_values)) == 0:
                return ufloat(np.nan, 0)
            else:
                variance = np.average((self.bin_centers - self.mean) ** 2,
                                      weights=unp.uarray(self.bin_std_devs / 2., self.bin_std_devs / 2.))
        else:
            variance = np.average((self.bin_centers - self.mean) ** 2, weights=np.abs(self.bin_values))
        return _sqrt(variance)

    @property
    def bin_values(self):
        return self.__get_bin_values_from_ROOT_hist__(self.__ROOT_hist__)

    @property
    def nominal_bin_values(self):
        return unp.nominal_values(self.bin_values)

    @property
    def n(self): # short for nominal_bin_values
        return self.nominal_bin_values

    @property
    def bin_std_devs(self):
        return unp.std_devs(self.bin_values)

    @property  # short for bin_std_devs
    def std_errs(self):
        return self.bin_std_devs

    def __set_bin_values__(self, other):
        assert isinstance(other[0], AffineScalarFunc)
        entries = self.__ROOT_hist__.GetEntries()
        nominal_values = unp.nominal_values(other)
        errors = unp.std_devs(other)
        root_hist = self.__ROOT_hist__
        for raw_index, b in enumerate(self.bin_centers):
            bin_index = root_hist.FindBin(b)
            root_hist.SetBinContent(bin_index, nominal_values[raw_index])
            root_hist.SetBinError(bin_index, errors[raw_index])
        self.__ROOT_hist__.SetEntries(entries)

    def __copy__(self):
        new_ROOT_hist = self.__ROOT_hist__.Clone("hist{0}".format(TH1F.title_number))
        TH1F.title_number += 1
        return TH1F(ROOT_hist=new_ROOT_hist)

    def copy(self):
        return self.__copy__()

    def __convert_other_for_operator__(self, other):
        if hasattr(other, "__iter__"):
            assert len(other) == len(self), "Multiplying hist of len {0} by iterator of len {1}".format(len(self),
                                                                                                        len(other))
            assert len(other), "Cannot operate on array opf length zero"
            if isinstance(other[0], Variable):
                other = unp.uarray([v.n for v in other], [v.std_dev for v in other])
            else:
                other = np.array(other)
        if isinstance(other, ROOT.TH1):
            other = self.__get_bin_values_from_ROOT_hist__(other)
        elif isinstance(other, TH1F):
            other = self.__get_bin_values_from_ROOT_hist__(other.__ROOT_hist__)
        elif isinstance(other, (Number, np.ndarray, Variable, AffineScalarFunc)):
            pass
        else:
            assert False, "Type {0} has not been implemented for multiply".format(type(other))
        return other

    def __rmul__(self, other):
        other = self.__convert_other_for_operator__(other)
        result = other * self.bin_values
        self.__set_bin_values__(result)
        return self

    def __mul__(self, other):
        new = self.__copy__()
        new *= other
        return new

    def __imul__(self, other):
        other = self.__convert_other_for_operator__(other)
        result = self.bin_values * other
        self.__set_bin_values__(result)
        return self

    def __neg__(self):
        new = self.__copy__()
        new.__set_bin_values__(-new.bin_values)
        return new

    def __radd__(self, other):
        return other + self

    def __add__(self, other):
        new = self.__copy__()
        new += other
        return new

    def __iadd__(self, other):
        other = self.__convert_other_for_operator__(other)
        result = self.bin_values + other
        self.__set_bin_values__(result)
        return self

    def __rsub__(self, other):
        other = self.__convert_other_for_operator__(other)
        result = other - self.bin_values
        self.__set_bin_values__(result)
        return self

    def __sub__(self, other):
        new = self.__copy__()
        new -= other
        return new

    def __isub__(self, other):
        other = self.__convert_other_for_operator__(other)
        result = self.bin_values - other
        self.__set_bin_values__(result)
        return self

    def __truediv__(self, other):
        new = self.__copy__()
        new /= other
        return new

    def __itruediv__(self, other):
        other = self.__convert_other_for_operator__(other)
        result = self.bin_values / other
        self.__set_bin_values__(result)
        return self

    def __pow__(self, other):
        new = self.__copy__()
        new **= other
        return new

    def __rpow__(self, other):
        other = self.__convert_other_for_operator__(other)
        result = other ** self.bin_values
        self.__set_bin_values__(result)
        return self

    def __ipow__(self, other):
        other = self.__convert_other_for_operator__(other)
        result = self.bin_values ** other
        self.__set_bin_values__(result)
        return self

    def __len__(self):
        return self.n_bins

    def __abs__(self):
        self.__set_bin_values__(abs(self.bin_values))
        return self

    def __repr__(self):
        return '<TH1F object, title: "{}">'.format(self.title)

    def shift_left_or_right(self, value):
        new_bin_values = unp.uarray(np.zeros(len(self)), np.zeros(len(self)))
        for b_center, b_value in zip(self.bin_centers, self.bin_values):
            new_b_center = b_center + value
            new_bin_index = np.argmax(self.__bin_left_edges__ > new_b_center)-1
            if not 0 <= new_bin_index < len(self):
                continue
            new_bin_values[new_bin_index] = b_value
        self.__set_bin_values__(new_bin_values)

        return self

    def set_draw_expression(self, draw_expression, cut='', weight=1):
        assert isinstance(draw_expression, str)
        assert isinstance(cut, str)
        assert isinstance(weight, (Number, str))
        self.draw_expression = draw_expression
        self.cut = cut
        if weight != 1:
            self.draw_weight = str(weight)

    n_multi_fill_calls = 0

    @staticmethod
    def multi_fill(tree, histos, max_entries=None, delete_c_files=True):
        """
        Generates and runs a C file that efficiently fills multiple histograms in a single loop through a TTree.
        The cuts and the expressions/weights that will be passed to TH1F->Fill are set by calling
        TH1F.set_draw_expression(exp, cut, weight) on each histogram in `histos` argument prior to using this method.

        Parameters:
                tree (ROOT.TTree): A ROOT tree (or TChain)
                histos (List[TH1F]): An iterable of TH1F instances (not the native ROOT TH1F, but the ROOT inspired
                TH1F class defined here).
                max_entries (int): Maximum number of entries to loop through.
        Returns:
                None
        """
        assert hasattr(histos, '__iter__')
        assert all([isinstance(h, TH1F) for h in histos]), '`histos` arg must ba an iterator of TH1F types'
        expressions = [h.draw_expression for h in histos]
        cuts = [h.cut for h in histos]
        assert isinstance(tree, ROOT.TTree), '`tree` arg must be ROOT.TTree instance'
        if max_entries is None:
            max_entries = tree.GetEntries()
        d = os.getcwd()
        os.chdir(Path(__file__).parent)
        __temp__name__ = '__temp__{}'.format(TH1F.n_multi_fill_calls)
        tree.MakeClass(__temp__name__)
        os.chdir(d)

        header_file_path = Path(__file__).parent/'{}.h'.format(__temp__name__)

        with open(header_file_path) as header_file:
            header_lines = header_file.readlines()
        new_header_lines = []
        histo_args = ['TH1F *h{0}'.format(i) for i in range(len(histos))]
        histo_args += ['int max_entries']
        histo_args = ', '.join(histo_args)

        for line in header_lines:
            if re.match(' +virtual void + Loop\(\);', line):
                new_line = '   virtual void     Loop({0});'.format(histo_args)+'\n'
                new_header_lines.append(new_line)
            else:
                new_header_lines.append(line)

        with open(header_file_path, 'w') as header_file:
            for line in new_header_lines:
                header_file.write(line)

        c_file_path = Path(__file__).parent/'{}.C'.format(__temp__name__)

        with open(c_file_path) as c_file:
            c_lines = c_file.readlines()
        new_c_lines = []

        for line in c_lines:
            if re.match(r'.+\/\/ if \(Cut\(ientry\) < 0\) continue', line):
                analysis_lines = ['\n']
                for index, (expression, cut) in enumerate(zip(expressions, cuts)):
                    if histos[index].draw_weight is not None:
                        expression = '{1}, {0}'.format(histos[index].draw_weight, expression)
                    fill_line = 'h{0}->Fill({1});'.format(index, expression)

                    if cut.rstrip() != '':
                        if_block = '\t  if ({0}){{{fill_line}}}\n'.format(cut, fill_line=fill_line)
                        analysis_lines.append(if_block)
                    else:
                        analysis_lines.append(fill_line)
                new_c_lines.append('\n'.join(analysis_lines))
                new_c_lines.append('\t  if (jentry > max_entries){break;}\n')

            elif re.match('void {}::Loop\(\)'.format(__temp__name__), line):
                new_c_lines.append('void {0}::Loop({1})\n'.format(__temp__name__, histo_args))
            else:
                new_c_lines.append(line)

        with open(c_file_path, 'w') as c_file:
            for line in new_c_lines:
                c_file.write(line)
        root_macro_path = str(Path(__file__).parent/'{}.C'.format(__temp__name__))
        ROOT.gROOT.LoadMacro(root_macro_path)
        process_line = 'cls{0} = {1}()'.format(TH1F.n_multi_fill_calls, __temp__name__)
        ROOT.gROOT.ProcessLine(process_line)
        _histos_for_ROOT = [h.__ROOT_hist__ for h in histos]
        getattr(ROOT, 'cls{}'.format(TH1F.n_multi_fill_calls)).Loop(*_histos_for_ROOT, max_entries)
        # ROOT.gROOT.Reset()
        TH1F.n_multi_fill_calls += 1
        if delete_c_files:
            header_file_path.unlink()
            c_file_path.unlink()




def ttree_cut_range(min_max_tuplee, expression, greater_then_or_equal=True, weight=None):
    if not min_max_tuplee[0] < min_max_tuplee[1]:
        warn("cut_rangeAND(): Min is greater than max. Switching values.")
    min_max_tuplee = sorted(min_max_tuplee)
    return "{weight}({min} {op} ({exp}) && ({exp}) {op} {max})".format(min=min_max_tuplee[0], exp=expression,
                                                                       max=min_max_tuplee[1],
                                                                       op="<=" if greater_then_or_equal else "<",
                                                                       weight="" if weight is None else "{0}*".format(
                                                                           weight))


def ttree_and(expressions):
    return " && ".join(expressions)




if __name__ == "__main__":
    import time
    #
    while True:
        ROOT.gSystem.ProcessEvents()
        time.sleep(0.05)
