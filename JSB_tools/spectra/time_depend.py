from __future__ import annotations
import numbers
import os
import re
import sys
import time
import warnings
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.text import Text
from JSB_tools import mpl_hist, convolve_gauss
from pathlib import Path
from matplotlib.patches import Circle
from functools import cached_property, cache
from typing import List, Dict, Union, Optional, Literal
import matplotlib.patches as mpatches
from numbers import Number
from matplotlib.text import Annotation
from matplotlib.transforms import Transform, Bbox
from lmfit import Model, Parameters, Parameter
from lmfit.models import GaussianModel
from lmfit.model import ModelResult
from uncertainties import ufloat, UFloat
import uncertainties.unumpy as unp
from matplotlib.widgets import CheckButtons, Slider, Button, RadioButtons, TextBox
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import logging
import matplotlib.transforms as transforms
from numba.typed import List as numbaList
from numba import jit, prange
from math import erf
from logging import warning
instructions = """
Instructions

Selecting Spectra:
    The radio buttons on the center right can be used to select the active spectrum. All other spectra (if any) besides 
    the selected spectrum will be grey, while the active spectrum blue.

Setting spectra time cuts:
    The time cuts possible are of the form
        c-w/2 <= time <= c + w/2
    where c and w are set by the sliders titled 'Center time' and 'Time window', respectively.
    To disable time cuts, select the check box in the lower left titled "All events" (this is the default).
    
Performing fitsAndNotes:
    While holding the space key, click the canvas one or more times to select peaks to be fit with Gaussians. 
    A multi-gaussian fit is performed for all selected peaks upon release of the space bar. When clicking peaks, 
    a single  mouse click will use the nearest local maxima for initial peak energy value while a double click will 
    constrain the value to within the bin that was clicked.

"Track fitsAndNotes" checkbox: 
    When selected, fitsAndNotes will be re-calculated as time cuts are changed, while preserving the original peak centers and 
    sigmas . 
    
"Clear Fits" button:
    Click the "Clear fitsAndNotes" button in the upper right to clear all fitsAndNotes of current selected spectrum.
    Click a second time to clear all fitsAndNotes.


"Rescale y" button:
    Rescale y to the min/max of selected spectrum. 
    
"All events" button:
    Selecting this button will force all events to be displayed regardless of the state of the time cute sliders.
    
"Make density" button:
    divides the bin values by the bin widths. Automatically selected when a fit is performed. 

"Remove baseline" button:
    This button, along with the corresponding "width (keV)" input field, control the subtraction of background routine.
    The width should be a number that is 4-10x larger than the typical width of the peaks. 
    
Other features:
    - Setting InteractiveSpectra.VERBOSE = True will print a fit report of each fi to cout. 
"""


def text_with_autofit(
        ax: plt.Axes,
        txt: str,
        xy: tuple[float, float],
        width: float, height: float,
        *,
        transform: Optional[Transform] = None,
        ha: Literal['left', 'center', 'right'] = 'center',
        va: Literal['bottom', 'center', 'top'] = 'center',
        show_rect: bool = False,
        **kwargs):
    if transform is None:
        transform = ax.transData

    #  Different alignments give different bottom left and top right anchors.
    x, y = xy
    xa0, xa1 = {
        'center': (x - width / 2, x + width / 2),
        'left': (x, x + width),
        'right': (x - width, x),
    }[ha]
    ya0, ya1 = {
        'center': (y - height / 2, y + height / 2),
        'bottom': (y, y + height),
        'top': (y - height, y),
    }[va]
    a0 = xa0, ya0
    a1 = xa1, ya1

    x0, y0 = transform.transform(a0)
    x1, y1 = transform.transform(a1)
    # rectangle region size to constrain the text in pixel
    rect_width = x1 - x0
    rect_height = y1 - y0

    fig: plt.Figure = ax.get_figure()
    dpi = fig.dpi
    rect_height_inch = rect_height / dpi
    # Initial fontsize according to the height of boxes
    fontsize = rect_height_inch * 72

    text: Annotation = ax.annotate(txt, xy, ha=ha, va=va, xycoords=transform, **kwargs)

    # Adjust the fontsize according to the box size.
    text.set_fontsize(fontsize)
    bbox: Bbox = text.get_window_extent(fig.canvas.get_renderer())
    adjusted_size = fontsize * rect_width / bbox.width
    text.set_fontsize(adjusted_size)

    if show_rect:
        rect = mpatches.Rectangle(a0, width, height, fill=False, ls='--')
        ax.add_patch(rect)

    return text


@jit(nopython=True)
def gaus_kernel(xs, mu, s, sigma_thresh=5):
    i0 = np.searchsorted(xs, mu - sigma_thresh * s)
    i1 = i0 + np.searchsorted(xs[i0:], mu + sigma_thresh * s)

    xmin, xmax = xs[i0], xs[i1]
    # norm = np.sqrt(2/np.pi)/(s*(-erf((mu - xmax)/(np.sqrt(2)*s)) + erf((mu - xmin)/(np.sqrt(2)*s))))
    norm = 1
    out = np.zeros_like(xs)
    out[i0: i1] += norm * np.e**(-(-mu + t)**2/(2*s**2))
    return out


@jit(nopython=True)
def remove_background(values: np.ndarray, median_window=None):
    """

    Args:
        values: Signal to operate on. Can be uarray
        median_window:

    Returns:

    """

    window_width = int(median_window)

    n = min([window_width, len(values)])

    window_indicies = [np.arange(max([0, i - n // 2]), min([len(values) - 1, i + n // 2])) for i in range(len(values))]

    bg = np.array([np.median(values[idx]) for idx in window_indicies])

    return values - bg


@jit(nopython=True)
def get_erg_spec(erg_binned_times: numbaList, time_range: np.ndarray, arr: np.ndarray=None, bg_subtract=False, bg_window=40):
    """
    Return compiled function for generating energy spectrum with time cuts.
    Args:
        erg_binned_times:

        time_range: array like [min_time, max_time]. A value of np.array([]) means no time cuts.

        arr: Array to be filled.

        bg_subtract:

        bg_window:

    Returns: array of shape len(erg_binned_times) representing counts for each energy bin

    """
    if arr is None:
        arr = np.zeros(len(erg_binned_times), dtype=float)

    assert len(arr) == len(erg_binned_times)
    # if n is None:
    #     out = np.zeros(len(erg_binned_times))
    # else:
    #     out = np.zeros(n)

    if len(time_range) == 0:
        full_spec = True
    else:
        full_spec = False

    for i in range(len(erg_binned_times)):
        tot_counts = len(erg_binned_times[i])

        if tot_counts == 0:
            continue

        if full_spec:
            val: float = tot_counts
        else:
            i1, i2 = np.searchsorted(erg_binned_times[i], time_range)
            val: float = i2 - i1

        arr[i] = val

    if bg_subtract:
        arr = remove_background(arr, median_window=bg_window)

    return arr


def erg_cut(bins, erg_binned_times, effs, emin=None, emax=None):
    i0 = (-1 + np.searchsorted(bins, emin, side='right')) if emin is not None else 0
    i1 = (-1 + np.searchsorted(bins, emax, side='right')) if emax is not None else len(bins) - 2
    return erg_binned_times[i0: i1 + 1], bins[i0: i1 + 2], effs[i0: i1 + 1] if effs is not None else None


def find_local_max(a, i0):
    """
    Starting from a[i0], find index of nearest local maxima.
    If there are two local maxima, take the larger of the two.
    Or, if they're the same height, return the left-most index.

    Args:
        a:  Array
        i0: Initial index

    Returns: int

    """
    is_ = []
    a = np.array(a)

    for s in [-1, 1]:
        i = i0
        if i + s <= 0:
            pass
        elif i + s >= len(a):
            pass
        else:  # not at edge of array, increment.
            while a[i + s] > a[i]:
                i += s

        is_.append(i)
    out = is_[np.argmax(a[is_])]
    return out


def get_fit_slice(bins, center_guesses, fit_buffer_window):
    """
    Return slices to reduce energy bins and ydata to a range for fit.
    Args:
        bins:
        center_guesses: initial locations of peaks.
        fit_buffer_window:

    Returns: bins_slice, y_slice

    """
    delta = bins[-1] - bins[-2]
    xmin, xmax = bins[0], bins[-1] - delta * 1E-3
    I0, I1 = np.searchsorted(bins, [max(xmin, min(center_guesses) - fit_buffer_window),
                                    min(xmax, max(center_guesses) + fit_buffer_window)],
                             side='right') - 1
    return (I0, I1 + 2), (I0, I1 + 1)


def linear(x, slope, bg):
    """a line used for fitting background. """
    return slope * (x - x[len(x) // 2]) + bg


class GausFitResult(ModelResult):
    """
    Wrapper for lmfit.ModelResult.

    """
    def __new__(cls, model_result: ModelResult, *args, **kwargs) -> GausFitResult:
        out = object.__new__(cls)
        for k, v in model_result.__dict__.items():
            if k[:2] == '__':
                continue
            try:
                setattr(out, k, v)
            except Exception as e:
                continue
        return out

    def __init__(self, *args, **kwargs):
        self.fit_x = self.userkws['x']
        self.bin_widths = self.userkws['bin_widths']

    @property
    def fit_y(self):
        y = self.data
        return y

    @property
    def fit_yerr(self):
        errs = 1.0/self.weights
        return errs

    @staticmethod
    def _get_param_value(param):
        if param.stderr is None:
            err = np.nan
        else:
            err = param.stderr
        return ufloat(param.value, err)

    def centers(self, i=None):
        if i is None:
            return [self.centers(i) for i in range(len(self))]

        param = self.params[f'_{i}_center']
        return self._get_param_value(param)

    def amplitudes(self, i=None) -> Union[List[UFloat], UFloat]:
        if i is None:
            return [self.amplitudes(i) for i in range(len(self))]

        param = self.params[f'_{i}_amplitude']
        return self._get_param_value(param)

    def sigmas(self, i=None):
        if i is None:
            return [self.sigmas(i) for i in range(len(self))]

        param = self.params[f'_{i}_sigma']
        return self._get_param_value(param)

    def bg(self):
        return self._get_param_value(self.params['bg'])

    def slope(self):
        return self._get_param_value(self.params['slope'])

    def plot_fit_curve(self, *args, **kwargs):  # bw compat.
        return self.plot(*args, **kwargs)

    def density2count_scales(self):
        return self.bin_widths

    def plot(self, ax=None, fill_between=True, npoints=300, plot_data=True, label=None, label_params=False, **plt_kwargs):
        if ax is None:
            _, ax = plt.subplots()

        x = np.linspace(self.fit_x[0], self.fit_x[-1], npoints)
        y = self.eval(x=x)

        if fill_between:
            yerr = self.eval_uncertainty(x=x, **plt_kwargs)
        else:
            yerr = None

        color = plt_kwargs.get('color', plt_kwargs.get('c', None))

        ax.plot(x, y, **plt_kwargs)

        if yerr is not None:
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.6, color=color)
        if plot_data:
            ax.errorbar(self.fit_x, self.fit_y, self.fit_yerr, label=label, ls='None', marker='x')

        if label_params:
            for i, s in enumerate(['E', 'A', r'$\sigma$']):
                x = 0.8 * (i/(3-1) - 0.5) + 0.5
                # print(f'x: {x}')
                ax.text(x, 0.97, s, transform=ax.transAxes, fontsize=13)

            texts = ['\t']
            for i in range(1, len(self) + 1):
                param_i = i-1
                try:
                    sigma = self.sigmas(param_i)
                except KeyError:
                    sigma = self.sigmas(0)

                try:
                    center = self.centers(param_i)
                except KeyError:
                    center = self.centers(0)

                try:
                    texts.append(fr'{center:.2e} {self.amplitudes(param_i):.2e} {sigma:.2e}')
                except KeyError:
                    break
            t = '\n'.join(texts)
            text_with_autofit(ax, t, (0, 0.99), 1, 0.5, transform=ax.transAxes, ha='left', va='top')
        if label is not None:
            ax.legend(loc='upper right' if label_params is None else 'lower left')

        return ax

    def __len__(self):
        out = 0
        for k in self.params.keys():
            if m := re.match("_([0-9]+)_.+", k):
                i = int(m.groups()[0])
                if i > out:
                    out = i

        return out + 1


def multi_guass_fit(bins, y, center_guesses, fixed_in_binQ: List[bool] = None, make_density=False, yerr=None,
                    share_sigma=True, sigma_guesses: Union[list, float] = None, fix_sigmas: bool = False,
                    fix_centers: Union[bool, List[bool]] = False, fix_bg: float = None, poissonian_errs = False,
                    fit_buffer_window: Union[int, None] = 5, nobins=False, bg: Literal['const', 'lin'] = 'lin',
                    **kwargs) -> GausFitResult:
    """
    Perform multi-gaussian fit to data.

    params names are, e.g.:
     amplitude: '_0_amplitude', '_1_amplitude, ...
        center: '_0_center', '_1_center', ...
         sigma: '_0_sigma', '_1_sigma', ...
            bg: 'bg'
         slope: 'slope'

    Notes:
        If y is a UFloat, erry = unp.std_devs(y), else erry = np.sqrt(y).

    Args:
        bins: Bin left edges. Must be len(y) + 1
        y: Bin values. Can be unp.uarray
        center_guesses: Guesses for centers of each peak to be fitted.

        fixed_in_binQ: List of bool. True means energy is constrained to the energy bin according to the
            corresponding element in `center_guesses`. Must be same len as center_guesses.

        make_density:
            In order for the gaussian amplitude to have the correct physical meaning, the units of y must be in per
            bin width. E.g. counts/keV instead of counts. This is remedied by dividing each bin value by the
            corresponding bin width before fitting, or, by dividing the final amplitude parameter by the mean bin
            width (less accurate).

        yerr: Error in y.

        share_sigma: If True, all peaks share the same sigma (as determined by fit).

        sigma_guesses: sigma guess. If number, use same number for all fits.

        fix_sigmas: Force sigmas to sigma_guesses.

        fix_centers: If True, peak centers wil be fixed at center_guesses

        fix_bg: Fix background to this value

        poissonian_errs: Automatically set errors to be np.sqrt(y)

        fit_buffer_window:
            The data will be truncated around peak centers with a buffer of this value (in keV) to the left and right
            Use None to not truncate.

        bg:

        nobins: If True, bins arg refers to bin centers instead.

    Returns: lmfit.ModelResult

    TODO:
        -Expand debugging capabilities:
            Allow for fitsAndNotes to be performed via command that mimic mouse click fitsAndNotes
            Consider using pythons logging module for degub msgs
        -Add energy rebin input box.
        - Add option to fit all peaks at a given energy in one click
        - Add button to toggle visibility of peak fitsAndNotes (gets busy with lots of overlapping peaks)
        - Inpout box to manually specify time range.
        - Option to Draw error bars.
        - Optimize:
            Save calculated energy_binned_times
            Maybe only calculate for current view?
            Figure out why it's slow. Maybe profiler?
    """
    assert bg in ['lin', 'const']
    if nobins:
        assert len(y) == len(bins)
        assert fixed_in_binQ is None
    else:
        assert len(y) == len(bins) - 1

    yfull = y[:]  # y without being cut to only include peaks

    if fit_buffer_window is not None and not nobins:
        bins_slice, y_slice = get_fit_slice(bins=bins, center_guesses=center_guesses,
                                            fit_buffer_window=fit_buffer_window)

        if (y_slice[1] - y_slice[0]) <= 1 + 2 + 3 * len(center_guesses):  # more params than data points. widen window
            if y_slice[0] > 0 and y_slice[1] < len(y):
                print(f"Fit window being increased from {fit_buffer_window} to {fit_buffer_window * 2}")  # todo: make log?
                fit_buffer_window *= 2
                return multi_guass_fit(**locals())

        bins = bins[slice(*bins_slice)]
        y = y[slice(*y_slice)]

    if fix_sigmas:
        assert sigma_guesses is not None, "Use of fix_sigmas requires sigma_guesses argument. "
        if isinstance(sigma_guesses, numbers.Number):
            sigma_guesses = [sigma_guesses] * len(center_guesses)

    if isinstance(fix_centers, bool):
        fix_centers = [fix_centers] * len(center_guesses)
    assert len(fix_centers) == len(center_guesses)

    if nobins:
        x = bins
    else:
        x = 0.5 * (bins[1:] + bins[:-1])

    assert len(center_guesses) != 0, 'Cannot perform fit!'

    if fixed_in_binQ is None:
        fixed_in_binQ = [False] * len(center_guesses)

    center_guesses = [max(bins[0], min(bins[-1] - 1E-10, x)) for x in center_guesses]

    if nobins:
        b_width = np.median(x[1:] - x[:-1])
    else:
        b_width = 0.5 * ((bins[1] - bins[0]) + (bins[-1] - bins[-2]))  # ~mean bin width

    if sigma_guesses is None:
        if share_sigma:
            sigma_guesses = b_width * 3
        else:
            sigma_guesses = [b_width * 3] * len(center_guesses)

    if isinstance(y[0], UFloat):
        yerr = unp.std_devs(y)
        y = unp.nominal_values(y)
    elif poissonian_errs:
        if not make_density:
            if any(int(yi) != yi for yi in y):
                warnings.warn("Assuming auto-generated Poissonian errors. Make sure the histogram values supplied to\n"
                              "multi_gaus_fit are in units of counts (e.g. not NOT a density histogram). \n"
                              "If you want density histogram, supply raw counts and set the `make_density` argument to True ")
        yerr = np.sqrt(y)

    yscale = 1  # used for bg geuss

    if make_density:  # divide bin values by bin widths.
        assert not nobins, 'Cannot make `y` into a density unless bins are used.'
        _bws = (bins[1:] - bins[:-1])  # bin widths
        y = y / _bws
        yscale *= 1/np.mean(_bws)
        if yerr is not None:
            yerr = yerr / _bws

    if yerr is None:
        weights = np.ones_like(y)
    else:
        weights = 1.0/np.where(yerr > 0, yerr, 1)

    model: Model = None
    params: Parameters = None

    max_slope = np.abs(max(y) - min(y)) / (x[-1] - x[0])
    max_slope = max(0.01, max_slope)  # zero max_slope will cause error bc range is ser to (-max_slope, max_slope)

    slope_guess = np.median(np.gradient(y))
    bg_guess = fix_bg if fix_bg is not None else min([np.mean([yfull[i-1], yfull[i], yfull[i + 1]]) for i in range(1, len(yfull) - 1)])

    bg_guess *= yscale
    if isinstance(bg_guess, UFloat):
        bg_guess = bg_guess.n

    min_sigma, max_sigma = b_width/2, b_width * len(y)/2

    def set_sigma(param: Parameter):
        kwargs = {'min': min_sigma, 'max': max_sigma}

        if fix_sigmas:
            kwargs['vary'] = False
        else:
            kwargs['vary'] = True

        if share_sigma:
            if param.name == '_0_sigma':  # this param name labels is the only *real* sigma, so sert value
                if isinstance(sigma_guesses, Number):
                    kwargs['value'] = sigma_guesses
            else:
                kwargs['expr'] = f'_0_sigma'  # param
                kwargs['value'] = None
                kwargs['min'] = None
                kwargs['max'] = None
        else:
            kwargs['value'] = sigma_guesses[i]

        param.set(**kwargs)

    def set_center(param: Parameter):
        kwargs = {'vary': True, 'value': center_guess}

        if fix_centers[i]:
            kwargs['vary'] = False

        elif fixedQ:
            kwargs['min'] = bins[i0]
            kwargs['max'] = bins[i0 + 1]
            assert bins[i0] < center_guess <= bins[i0 + 1], (bins[i0], center_guess, bins[i0 + 1])
        else:
            kwargs['value'] = x[find_local_max(y, i0)]
            kwargs['min'] = x[0]
            kwargs['max'] = x[-1]

        param.set(**kwargs)

    def amp_guess(max_y, sigma):
        out = (max_y - bg_guess) * np.sqrt(2 * np.pi) * sigma
        if isinstance(out, UFloat):
            out = out
        return out

    for i, (fixedQ, center_guess) in enumerate(zip(fixed_in_binQ, center_guesses)):
        i0 = np.searchsorted(bins, center_guess, side='right') - 1

        prefix = f'_{i}_'
        sub_model = GaussianModel(prefix=prefix)
        sub_params = sub_model.make_params()

        if model is None:
            model = Model(linear) + sub_model

            params = model.make_params()
            if bg == 'lin':
                params['slope'].set(value=slope_guess, max=max_slope, min=-max_slope)
            else:
                params['slope'].set(value=0, vary=False)
            params['bg'].set(value=bg_guess, vary=fix_bg is None)
        else:
            model += sub_model
            params.update(sub_params)

        set_sigma(params[f'{prefix}sigma'])

        set_center(params[f'{prefix}center'])

        _amp_guess = amp_guess(y[i0], params[f'{prefix}sigma'].value)

        params[f'{prefix}amplitude'].set(value=amp_guess(y[i0], params[f'{prefix}sigma'].value), min=0)

    fit_result = model.fit(data=y, params=params, weights=weights, x=x)

    fit_result.userkws['b_width'] = None if nobins else b_width
    if not nobins:
        fit_result.userkws['bin_widths'] = bins[1:] - bins[:-1]
    else:
        fit_result.userkws['bin_widths'] = None

    return GausFitResult(fit_result)


def get_param_ufloat(name, params):
    """
    Return param value with error for printing.
    Args:
        name:
        params:

    Returns:

    """
    err = params[name].stderr
    if err is None:
        err = np.nan
    return ufloat(params[name].value, err)


class EnergyBinnedTimes:
    def __init__(self, arr):
        # super().__init__(arr)
        self.arr = numbaList(arr)
        self.i0 = 0
        self.i1 = len(arr)

    def view(self):
        return self[self.i0: self.i1]


class InteractiveSpectra:
    print_click_coords = False  # If True, print mouse click points in fractional Axes coords (used for dev)
    VERBOSE = False  # print fit_report (and other things?)
    Print_FIT_REPORT = True
    FIT_VERBOSE = False  # print fit stuff for debugging

    default_fit_window = 6  # in keV
    default_n_convolve_bins = 3  # amount to convolve signal when guessing guass fit centers.

    cmap = plt.get_cmap("tab10")

    def add_list(self, spec, scale=1, time_max=None, disable_eff_corr=False, title=None, erg_min=None, erg_max=None):
        """
        Same as add_spectra except can be used
        Args:
            spec:
            scale:
            time_max:
            disable_eff_corr:
            title:
            erg_min:
            erg_max:

        Returns:

        """
        assert hasattr(spec, 'energy_binned_times') and hasattr(spec, 'erg_bins'),\
            f'Invalid type, {type(spec)}. The `spec` argument must have a erg_bins and energy_binned_times method.'

        if not hasattr(spec, 'effs') or disable_eff_corr:
            effs = None
        else:
            effs = spec.effs
        if time_max is not None:
            spec.time_cut(tmax=time_max)

        self.add_spectra(energy_binned_times=spec.energy_binned_times, energy_bins=spec.erg_bins, effs=effs,
                         scale=scale, title=title, erg_min=erg_min, erg_max=erg_max)

    def add_static_spectra(self, erg_bins, counts, effs=None, scale=1, title=None, erg_min=80, erg_max=None):
        """
        Sdd spectra that has no time depedance.
        Returns:

        """
        assert len(erg_bins) - 1 == len(counts), (len(erg_bins), len(counts))
        energy_binned_times = [np.array(n) for n in counts]
        return self.add_spectra(energy_binned_times, energy_bins=erg_bins, effs=effs, scale=scale, title=title,
                                erg_min=erg_min, erg_max=erg_max)

    def is_static_spectra(self, index):
        return self.energy_binned_timess[index][0].ndim == 0

    def n_spectra(self):
        return len(self.energy_binned_timess)

    def add_spectra(self, energy_binned_times, energy_bins, effs=None, scale=1, title=None, erg_min=80, erg_max=None):
        """

        Args:
            energy_binned_times:
                A list of lists, such as [[t1_1, t1_2, t1_3], [[t2_1, t2_2, t2_3, t2_4], ... [tn_1, tn_2]]
                Where each ti_j refers to the time of the jth event falling in the ith energy bin.
                I.e, the times are grouped byu energy bin in the order in which they occured.

            energy_bins: Energy bin edges
            effs: Array of efficiencies. len(effs) == len(energy_bins) - 1
            scale: Scale the spectrum. Can be scalar or array.
            title: Title appears next to button that selects this spectra
            erg_min:
            erg_max:

        Returns:

        """
        if isinstance(scale, np.ndarray):
            assert len(scale) == len(energy_binned_times)
        elif isinstance(scale, (float, int)):
            scale = scale * np.ones(len(energy_binned_times), dtype=float)
        else:
            raise ValueError(f"Invalid type, '{type(scale)}', for argument `scale`. Must be a number or an np.ndarray")

        i0 = max(0, (-1 + np.searchsorted(energy_bins, erg_min, side='right'))) if erg_min is not None else 0
        i1 = (np.searchsorted(energy_bins, erg_max, side='right')) if erg_max is not None else \
            (len(energy_bins) - 1)

        energy_binned_times = energy_binned_times[i0: i1]
        effs = effs[i0: i1] if effs is not None else None
        scale = scale[i0: i1] if isinstance(scale, np.ndarray) else scale
        energy_bins = np.array(energy_bins[i0: i1 + 1])

        if effs is not None:
            scale = scale/effs

        if hasattr(scale, '__iter__') and isinstance(scale[0], UFloat):
            scale = unp.nominal_values(scale)
        elif isinstance(scale, UFloat):
            scale = scale.n

        self.scales.append(scale)
        self.energy_binned_timess.append(energy_binned_times)
        self.views.append([0, len(energy_binned_times)])  # initial view is max
        self.erg_binss.append(energy_bins)

        self.erg_bin_widths.append(energy_bins[1:] - energy_bins[:-1])

        if energy_binned_times[0].ndim == 1:
            if self._events_times_range[0] is None:
                self._events_times_range[0] = min([ts[0] for ts in energy_binned_times if len(ts)])
            else:
                self._events_times_range[0] = min(self._events_times_range[0], min([ts[0] for ts in energy_binned_times if len(ts)]))

            if self._events_times_range[-1] is None:
                self._events_times_range[-1] = max([ts[-1] for ts in energy_binned_times if len(ts)])
            else:
                self._events_times_range[-1] = max(self._events_times_range[-1], max([ts[-1] for ts in energy_binned_times if len(ts)]))
        else:
            if self._events_times_range[0] is None:
                self._events_times_range = [0, 0]

        t0, t1 = self._events_times_range
        _y, _ = self._calc_y(len(self), t0, t1, True)
        handle = self.ax.plot(energy_bins, _y, ds='steps-post', c=self.cmap(len(self)))[0]

        self.handles.append(handle)  # this causes self.__len__ to increment

        if title is None:
            title = str(len(self) - 1)

        self.titles.append(title)

    def _mouse_click(self, event):
        """"""
        if self.print_click_coords:
            msg = ''

            if event.xdata is not None:
                msg += f"x, y = {event.xdata:.2f},{event.ydata:.2f} in Axes coords\n"

            x0, x1 = self.fig.get_window_extent().x0, self.fig.get_window_extent().x1
            y0, y1 = self.fig.get_window_extent().y0, self.fig.get_window_extent().y1
            fx, fy = (event.x - x0) / (x1 - x0), (event.y - y0) / (y1 - y0)

            msg += f"x,y = {fx:.2f},{fy:.2f} in rel. Figure coords.\n"

            print(msg)
        x_now = event.xdata

        if x_now is None:
            return

        if self.selecting_peaks:  # fit peak selected
            x_points_array = self.fit_clicks['x_points']  # Previous fit energies selected during current fit.
            fixed_bins_array = self.fit_clicks['fixed_binsQs']

            others = np.array(x_points_array)

            if len(x_points_array) and x_points_array[-1] == x_now and event.dblclick:
                # double click means user wants to perform fixed energy bin fit. Change entry from previous click.
                fixed_bins_array[-1] = True
                self.fit_clicks['axvlines'][-1].set_ls('-')
                self._draw()
                return  # don't add another fit energy

            if np.all(np.abs(x_now - others) > self._mean_bin_width(self._active_spec_index)):
                # Don't fit if energy is too close to previous energies
                x_points_array.append(event.xdata)
                fixed_bins_array.append(False)
                self.fit_clicks['axvlines'].append(self.ax.axvline(x_now, ls='--', c='black', lw=1.5))
                self._draw()
            else:
                i = np.argmin(np.abs(x_now - others))
                logging.warning(
                    f"Did not add fit at {x_now:.2f} keV because is too close to the already selected peak at {others[i]:.2f} keV")

    def _draw(self):
        self.fig.canvas.draw_idle()

    def update_line(self, y, yerr, index):
        """
        Args:
            y: Hist values with y[-1] appended onto end (i.e. must be same length as self.erg_bins[index]).
            yerr:
            index: Index representing the active spectrum

        Returns:

        """
        assert len(y) == len(self.erg_binss[index]), "Bad trouble!"
        line: Line2D = self.handles[index]
        line._yorig = y
        line._yerr = yerr  # custom attribute for now.

        line.stale = True
        line._invalidy = True

    @property
    def time_cut_min(self):
        """
        Min time according to the two time cut sliders
        Returns:

        """
        return self.slider_time_center.val - self.slider_window.val / 2

    @property
    def time_cut_max(self):
        """
        Max time according to the two time cut sliders

        Returns:

        """
        return self.slider_time_center.val + self.slider_window.val / 2

    @property
    def time_range(self):
        """
        Return max and min times in spectrum

        Returns: [min_time, max_time]

        """
        return self._events_times_range

    def _set_ylims(self, all_spectraQ=True, *args, **kwargs):
        """
        Rescale y-axis to fit y-data in current view.
        Use either the current spectra or all spectra for y-data.
        Args:
            all_spectraQ: If True, only scale y to data of current spectra.
            *args:
            **kwargs:

        Returns:

        """
        xlim = self.ax.get_xlim()
        ylim = np.inf, -np.inf

        if all_spectraQ:
            xs, ys = [], []
            for index in range(len(self)):
                x, y = self.handles[index].get_data()
                xs.append(x)
                ys.append(y)
        else:
            x, y = self.handles[self._active_spec_index].get_data()
            xs, ys = [x], [y]

        for x, y in zip(xs, ys):
            # faster, but assumes that x is sorted
            start, stop = np.searchsorted(x, xlim)
            yc = y[max(start - 1, 0):(stop + 1)]
            ylim = min(ylim[0], np.nanmin(yc)), max(ylim[1], np.nanmax(yc))

        self.ax.set_xlim(xlim, emit=False)

        # y axis: set dataLim, make sure that autoscale in 'y' is on
        corners = (xlim[0], ylim[0]), (xlim[1], ylim[1])
        self.ax.dataLim.update_from_data_xy(corners, ignore=True, updatex=False)
        self.ax.autoscale(enable=True, axis='y')
        # cache xlim to mark 'a' as treated
        self.ax.xlim = xlim
        self._draw()

    @property
    def current_y(self):
        """
        Return y, yerr arrays of currently active spectrum.
        Returns:

        """
        index = self._active_spec_index
        return self.get_y(index)

    def get_y(self, index):
        """
        Return y, yerr arrays of currently active spectrum.
        Returns:

        """
        return self.handles[index].get_ydata()[:-1], self.handles[index]._yerr[:-1]

    def _calc_y(self, index, tmin, tmax, append_last_value=False, truncate_xrange=True):
        """

        Args:
            index: Index to select spectrum
            tmin:
            tmax:
            append_last_value: Last bin value can be duplicated in order to draw histogram.
            truncate_xrange: Only calculate spectra current within plot view.

        Returns: y, yerr

        """
        bg_subtractQ = self.checkbox_bg_subtract.get_status()[0]
        bg_window = int(self.bg_textbox.text)
        scales = self.scales[index]
        energy_binned_times = self.energy_binned_timess[index]  # here

        n = len(energy_binned_times)
        if append_last_value:
            n += 1  # array will have one extra value at end.

        out = np.zeros(n)

        if self.is_static_spectra(index):
            out[:-1] += energy_binned_times
            out_err = np.zeros_like(out)
        else:
            if self._time_integratedQ:
                time_range = np.array([])  # accept all events.
            else:
                time_range = np.array([tmin, tmax])  # perform time cut.

            if truncate_xrange and False:  # disabled for now.
                I0, I1 = np.searchsorted(self.erg_binss[index], self.ax.get_xlim(), side='right') - 1
                I0 -= 1
                I1 += 1

                I0 = max(0, I0)
                I1 = min(len(energy_binned_times), I1)
            else:
                I0 = 0
                I1 = len(energy_binned_times)

            energy_binned_times = energy_binned_times[I0: I1]

            get_erg_spec(energy_binned_times, time_range, arr=out[I0: I1])  # modifies out

            out_err = np.sqrt(out)

        if bg_subtractQ:  # purposely after out_err declaration
            out = remove_background(out, bg_window)

        if self.checkbox_make_density.get_status()[0]:
            out[:-1] /= self.erg_bin_widths[index]

        out[:-1] *= scales
        out_err[:-1] *= scales

        if append_last_value:
            out[-1] = out[-2]

        return out, out_err

    def _update_time_cut_display(self):
        def get_text():
            if self._time_integratedQ:
                t0, t1 = self.time_range
            else:
                t0, t1 = self.time_cut_min, self.time_cut_max

            return f'{t0:.1f} < t < {t1:.1f}'

        if self.time_cut_text is None:
            self.time_cut_text = self.ax.text(0.80, 0.97, get_text(),
                                              transform=self.ax.transAxes, fontsize=12)
        else:
            self.time_cut_text.set_text(get_text())

    def _update_plot(self, *args):
        for index in range(len(self)):
            y, yerr = self._calc_y(index, self.time_cut_min, self.time_cut_max, True)
            self.update_line(y, yerr, index)

        if self.checkbox_track_fits.get_status()[0]:
            self._re_perform_fits()

        if self.checkbox_make_density.get_status()[0]:
            self.ax.set_ylabel('[(keV)$^{-1}$]')
        else:
            self.ax.set_ylabel('[()]')

        self._update_time_cut_display()
        self._draw()

    def _xlim_changed(self, event=None):
        pass
        # print("_xlim_changed")
        # ax_range = self.ax.get_xlim()
        # i = self._active_spec_index
        # if self.views[i][1] - ax_range[1] < self.slider_window.val*0.1:
        #     new_v = np.searchsorted(ax_range[1], self.erg_binss[i], side='right')
        #     self.views[i][1] = min(len(self.energy_binned_timess[i]))

    @property
    def _time_integratedQ(self):
        return self.checkbox_time_integrated.lines[0][0].get_visible()

    @property
    def _active_spec_index(self) -> int:
        """
        Get the index according to which spectrum is currently selected to be active in the GUI.

        (Thanks matplotlib for making this way more complicated than it needs to be)

        Returns: int

        """
        return [t.get_text() for t in self.radiobutton_active_select.labels].index(self.radiobutton_active_select.value_selected)

    def _on_checked_time_integrated(self, label):
        """
        Handles the checkbox that displays the time-integrated spectrum and also enables/disables the time cut sliders
         when time-integrated spectrum is being viewed.
        Args:
            label: Not used.

        Returns:

        """
        for slider in [self.slider_time_center, self.slider_window]:
            slider.poly.set_visible(not self._time_integratedQ)  # grey-out slider
            slider.set_active(not self._time_integratedQ)  # Freeze slider

        if self.checkbox_track_fits.get_status()[0]:
            self._re_perform_fits()
        else:
            self._clear_fits()  # no purpose in keeping them!

        self._update_plot()
        self._set_ylims()

    def set_default_peak_text(self):
        self.selecting_peaks_text.set_text('Press spacebar to enter peak selection mode')
        self.selecting_peaks_text.set_color('black')

    def _key_press(self, event):
        if event.key == ' ':
            if self.selecting_peaks:
                self._perform_fits()
                self.set_default_peak_text()
                self.fig.canvas.draw_idle()
            else:
                self.selecting_peaks_text.set_text("PEAK SELECTION MODE ON (press spacebar again to perform fits)")
                self.selecting_peaks_text.set_color('tab:red')
                self.fig.canvas.draw_idle()

            self.selecting_peaks = not self.selecting_peaks

        try:  # call any functions bound to key by self.key_press_events
            self.key_press_events[event.key]()
        except KeyError:
            pass

    def _re_perform_fits(self):
        """
        Re-does fitsAndNotes in active spectrum after change ion slider values. Fixes the energy to that of the original fit.
        Returns:

        """
        for index in range(len(self)):
            fit_infos = self.current_fits[index]
            if not len(fit_infos):
                continue

            centers_lists = []
            sigmas_lists = []

            for fit_info in fit_infos:
                fit_result = fit_info['fit_result']
                centers = []
                sigmas = []
                for k in fit_result.params:
                    if m := re.match('_([0-9]+)_center', k):
                        centers.append(fit_result.params[k].value)
                    elif m := re.match('_([0-9]+)_sigma', k):
                        sigmas.append(fit_result.params[k].value)

                sigmas_lists.append(sigmas)
                centers_lists.append(centers)

            self._clear_fits(index=index)

            for cs, sigmas in zip(centers_lists, sigmas_lists, ):
                self._perform_fits(center_guesses=cs, sigma_guesses=sigmas, force_sigma=False, force_centers=True,
                                   index=index)

            self._set_fit_line_attribs(index)

    def verify_fit_width_input(self, *args):
        try:
            _ = float(self.fit_width_textbox.text)
        except ValueError:
            self.fit_width_textbox.set_val('5')
            self.fig.canvas.draw_idle()
            raise

    @property
    def fit_window(self):
        return float(self.fit_width_textbox.text)

    def _perform_fits(self, center_guesses=None, fixed_binsQs=None, sigma_guesses=None, force_centers=False,
                      force_sigma=False, index=None):
        fit_verbose = InteractiveSpectra.FIT_VERBOSE
        if center_guesses is None:
            center_guesses = self.fit_clicks['x_points']

        if fixed_binsQs is None:
            if len(self.fit_clicks['fixed_binsQs']):
                fixed_binsQs = self.fit_clicks['fixed_binsQs']
            else:
                fixed_binsQs = [False] * len(center_guesses)

        if index is None:
            index = self._active_spec_index

        center_guesses = np.array(center_guesses)
        center_guesses = center_guesses[np.where((center_guesses > self.erg_binss[index][0]) &
                                                 (center_guesses < self.erg_binss[index][-1]))]

        if not len(center_guesses):  # no fitsAndNotes to perform!
            return

        if not self.checkbox_make_density.get_status()[0]:
            self.checkbox_make_density.set_active(0)

        bins_slice_range, y_slice = get_fit_slice(self.erg_binss[index], center_guesses,
                                                  self.fit_window)

        y_slice = slice(*y_slice)
        bins_slice = slice(*bins_slice_range)
        bins = self.erg_binss[index][bins_slice]

        y, yerr = self.get_y(index)
        y = y[y_slice]
        yerr = yerr[y_slice]

        try:
            if fit_verbose:
                print(f"Initial peak centers: {center_guesses}")
                print(f"Fit range is {bins[0]} <= x <= {bins[-1]}")
            fit_result: GausFitResult = multi_guass_fit(bins=bins, y=y,
                                                        center_guesses=center_guesses,
                                                        fixed_in_binQ=fixed_binsQs,
                                                        yerr=yerr,
                                                        sigma_guesses=sigma_guesses,
                                                        fix_centers=force_centers,
                                                        fix_sigmas=force_sigma,
                                                        share_sigma=self.fit_settings['only_one_sigma'],
                                                        fit_buffer_window=None)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)
            warning(f"Fit failed:\n{exc_type, fname, exc_tb.tb_lineno}")
            self._reset_fit_clicks()
            return

        if self.Print_FIT_REPORT:
            print(fit_result.fit_report())

        x = fit_result.userkws[fit_result.model.independent_vars[0]]
        # y = fit_result.data

        params = fit_result.params

        line = self.handles[index]

        fit_line1 = self.ax.plot(x, fit_result.eval(x=x, params=params), c='black', ls='-', lw=2.5)[0]
        fit_line2 = self.ax.plot(x, fit_result.eval(x=x, params=params), c=line.get_color(), ls='--', lw=2.5)[0]

        fit_info = {'visibles': [fit_line1, fit_line2],
                    'fit_result': fit_result,
                    'fixed_binsQs': fixed_binsQs}

        text_trans = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)

        for i in range(len(center_guesses)):
            prefix = f"_{i}_"
            fit_center = params[f'{prefix}center'].value

            xpos = fit_center

            text = self.ax.text(xpos, 1.01,
                                f"E=${fit_result.centers(i):.2fL}$ keV\n"
                                f"N=${fit_result.amplitudes(i):.2L}$",
                                rotation=46, fontdict={'fontsize': 8.5},
                                transform=text_trans)

            vert_line = self.ax.axvline(fit_center, c='black', lw=1)

            fit_info['visibles'].append(text)
            fit_info['visibles'].append(vert_line)

        for visible in fit_info['visibles']:  # set _default_color attribute for changing things to gray ad back
            try:
                default_color = visible.get_color()
                setattr(visible, '_default_color', default_color)
            except AttributeError:
                pass

        self.current_fits[index].append(fit_info)
        self._reset_fit_clicks()  # self._draw is called here.
        self._set_ylims()

    def _set_fit_line_attribs(self, index=None):
        """
        Set fit lines of index to grey, etc. if they're not from the current spectrum.
        Args:
            index: Spectrum to process. None to do all.

        Returns: None

        """
        if index is None:
            for index in range(len(self)):
                self._set_fit_line_attribs(index)
            return

        for fit_info in self.current_fits[index]:
            for thing in fit_info['visibles']:
                try:
                    if index == self._active_spec_index:
                        thing.set_color(thing._default_color)
                        if isinstance(thing, Text):  # make text of active spectrum visible
                            thing.set_alpha(1)
                    else:
                        if isinstance(thing, Text):  # make text invisible
                            thing.set_alpha(0)
                        thing.set_color('lightgrey')

                except AttributeError:
                    pass

    def _radio_button_on_clicked(self, val):
        for index in range(len(self)):
            line = self.handles[index]

            if index == self._active_spec_index:
                line.set_alpha(1)
                line.set_zorder(10)
                line.set_linewidth(2)
            else:
                line.set_alpha(0.80)
                line.set_zorder(-10)
                line.set_linewidth(1.5)

            self._set_fit_line_attribs(index)

        self._draw()

    def _reset_fit_clicks(self):
        """
        Re-set to default data used to track fits initiated by mouse clicks while holding space bar.
        Called after self._perform_fits().
        Returns:

        """
        try:
            for thing in self.fit_clicks['axvlines']:
                thing.set_visible(0)  # no longer needed
                del thing
        except (AttributeError, KeyError):
            pass  # self.fit_clicks not initiated yet. Move along.

        self.fit_clicks = {'x_points': [], 'fixed_binsQs': [], 'axvlines': []}
        self._draw()
        return self.fit_clicks

    def _clear_fits(self, event=None, index=None, clear_all=False):
        """

        Args:
            event: Not used
            index: Which spectrum to be cleared. Default is selected spectrum.
            clear_all: Will clear fits from all spectra (instead of just active spectra)

        Notes:
            the code snippet, "(not len(self.current_fits[index]))" implements a feature where clicking the clear fit
             button twice clears all fits (not just active).

        Returns:

        """
        if index is None:
            index = self._active_spec_index

        def clear_(index_):  # clears fit data stored for spectrum index_
            fit_infos = self.current_fits[index_]

            for fit_info in fit_infos:
                for thing in fit_info['visibles']:
                    thing.set_visible(0)
                    del thing

            self.current_fits[index_] = []

        double_click_clear = (not len(self.current_fits[index])) and self._active_spec_index == index
        # double_click_clear evaluates to True if clear fits button is pressed twice.
        if clear_all or double_click_clear:  # clear all spectra fits
            for i in range(len(self)):
                clear_(i)
        else:
            clear_(index)  # just clear current spectrum fits
        self._draw()

    @staticmethod
    def _help(*args):
        """
        Print manual to std:out.
        Args:
            *args:

        Returns:

        """
        print(instructions)

    def set_erg_window(self, erg_min, erg_max):
        """
        Set x-axis limit that will be set upon opening.
        Data outside (erg_min, erg_max) is still available via zoom.
        Args:
            erg_min:
            erg_max:

        Returns: None

        """
        self.ax.set_xlim(erg_min, erg_max)

    def __init__(self, init_tmin=None, init_tmax=None, init_bg_subtractQ=False, window_max=None, delta_t=5,
                 fig_title=None):
        """

        Args:
            init_tmin: initial min time cut
            init_tmax:  Initial max time cut
            init_bg_subtractQ: If True, baseline subtract will turned on initially.
            window_max: Max width of time integration window accessible with to the slider
            delta_t: time step for the slider determining the center time of the integration time range
            fig_title: Title at top of window.

        """
        old_show = plt.show

        def new_show(*args, **wkargs):
            self._prepare()
            return old_show(*args, **wkargs)

        plt.show = new_show  # link calls to plt.show to a function that includes a call to self._prepare

        self.fit_settings = {}
        self.set_fit_settings()

        # if keyboard key specified by dict key _ is pressed, call function specified by dict value
        self.key_press_events = {}

        self._events_times_range: List[float] = [None, None]  # global time min/max of all events of all spectra

        self.handles: List[Line2D] = []
        self.erg_binss = []
        self.energy_binned_timess = []
        self.views = []  # indices for the energy range being calculated. Same len of energy_binned_timess
        self.scales = []  # scale each spectrum by a value.
        self.titles = []

        self.erg_bin_widths = []

        fig, ax = plt.subplots(figsize=(16, 9))

        if fig_title is not None:
            fig.suptitle(str(fig_title))

        a = 16/9  # aspect ratio: x : y

        self.fig = fig
        self.ax: Axes = ax

        self.maxy = 1

        plt.subplots_adjust(bottom=0.15, top=0.83)

        self.window_max = window_max
        self.delta_t = delta_t

        self.vanilla_button_axs: Dict[str, Axes] = {}
        self.vanilla_buttons = {}

        def vanilla_button(name, callback):  # make button in the upper right "toolbar".
            if len(self.vanilla_button_axs):
                x0 = self.vanilla_button_axs[list(self.vanilla_button_axs.keys())[-1]].get_position().x1
            else:
                x0 = 0.75

            x0 += 0.005  # some distance between buttons
            y0 = self.ax.get_position().y1 + 0.01

            dx = 0.04 * a / 10 * len(name) + 0.01
            ax = fig.add_axes([x0, y0, dx, 0.03])
            self.vanilla_button_axs[name] = ax

            b = Button(ax, name)
            self.vanilla_buttons[name] = b

            if callback is not None:
                b.on_clicked(callback)

        self.slider_erg_center_ax = fig.add_axes([0.1, 0.06, 0.8, 0.05])
        self.slider_window_ax = fig.add_axes([0.1, 0.01, 0.8, 0.05])
        self.checkbox_time_integrated_ax = fig.add_axes([0.91, 0.1, 0.15, 0.15 * 2])
        self.checkbox_make_density_ax = fig.add_axes([0.91, 0.05, 0.15, 0.15])
        self.checkbox_track_fits_ax = fig.add_axes([0.02, self.ax.get_position().y1 - 0.02, 0.15, 0.15 * 2])

        self.checkbox_bg_subtract_ax = fig.add_axes([0.83, 0.81, 0.1, 0.2])
        self.bg_textbox_ax = fig.add_axes([0.8,  0.9, 0.03, 0.02])

        self.fit_width_textbox_ax = fig.add_axes([0.95,  0.95, 0.03, 0.02])

        vanilla_button('Clear fits', self._clear_fits)

        vanilla_button('rescale Y (y)', self._set_ylims)
        self.key_press_events['y'] = self._set_ylims

        vanilla_button('help', self._help)

        self.radiobutton_active_select_ax = fig.add_axes([0.9, 0.35, 0.15, 0.3])

        self.slider_erg_center_ax.set_axis_off()
        self.slider_window_ax.set_axis_off()
        self.checkbox_time_integrated_ax.set_axis_off()
        self.checkbox_make_density_ax.set_axis_off()
        self.checkbox_track_fits_ax.set_axis_off()
        self.radiobutton_active_select_ax.set_axis_off()
        self.checkbox_bg_subtract_ax.set_axis_off()

        if init_tmax is None or init_tmin is None:
            self._init_window_width = self._init_slider_pos = None
        else:
            self._init_window_width = init_tmax - init_tmin
            self._init_slider_pos = 0.5 * (init_tmax + init_tmin)

        # The Widgets below will be declared once the number of spectra is being simultaneously plotted is known
        self.slider_time_center: Slider = None  # cant define until range is
        self.slider_window: Slider = None
        self.radiobutton_active_select: RadioButtons = None

        self.time_cut_text = None

        self.fig.canvas.mpl_connect('button_press_event', self._mouse_click)
        # self.ax.callbacks.connect('xlim_changed', self._set_ylims)

        # \begin GUI state variables.
        # self.fit_clicks contains info on each energy selection for upcoming fits
        # (as determined by mouse clicks while space bar is being held down)
        self.fit_clicks: Dict[str, List] = self._reset_fit_clicks()

        # self.current_fits is dict (for each spectrum) of lists of dicts (one for each fit!) containing things
        # relevant to the fits currently on canvas, e.g. Line2D objects
        self.current_fits: Dict[int, List] = None  # is initialized in self.show!

        # \end GUI state variables.

        self.checkbox_time_integrated = CheckButtons(self.checkbox_time_integrated_ax, ['All\nevents'],
                                                     actives=[init_tmin is None])
        self.checkbox_time_integrated.on_clicked(self._on_checked_time_integrated)

        self.checkbox_make_density = CheckButtons(self.checkbox_make_density_ax, ['make\ndensity'], actives=[False])
        self.checkbox_make_density.on_clicked(self._update_plot)

        self.checkbox_track_fits = CheckButtons(self.checkbox_track_fits_ax, ['Track fits'], actives=[False])

        self.checkbox_bg_subtract = CheckButtons(self.checkbox_bg_subtract_ax, ['Remove\nbaseline'],
                                                 actives=[init_bg_subtractQ])
        self.checkbox_bg_subtract.on_clicked(self._update_plot)

        self.bg_textbox = TextBox(self.bg_textbox_ax, "width (keV): ", '40', )
        self.bg_textbox.on_submit(self._update_plot)

        self.fit_width_textbox = TextBox(self.fit_width_textbox_ax, "Fit Width", '5', )
        self.fit_width_textbox.on_submit(self.verify_fit_width_input)

        self.fig.canvas.mpl_connect('key_press_event', self._key_press)

        self.selecting_peaks = False
        self.selecting_peaks_text = self.fig.text(0.3, .98, '', ha='center')
        self.set_default_peak_text()

    @cache
    def bin_widths(self, index):
        return self.erg_binss[index][1:] - self.erg_binss[index][:-1]

    @cache
    def _mean_bin_width(self, index):
        return np.mean(self.bin_widths(index))

    def set_fit_settings(self, fit_buffer_window=5, only_one_sigma=True):
        """
        Set global setting for fits.
        Args:
            fit_buffer_window: Amount to extend the fit beyond the provided energy. Should be roughly 2-4 times the FWHM

            only_one_sigma: If True, all peaks share the same sigma for each fit.

        Returns:

        """
        self.fit_settings['fit_buffer_window'] = fit_buffer_window
        self.fit_settings['only_one_sigma'] = only_one_sigma

    def __len__(self):
        """
        Number opf spectra
        Returns:

        """
        return len(self.handles)

    def _prepare(self):
        if not hasattr(self, '__has_called_prepare'):
            self.__has_called_prepare = True
        else:
            return

        if self._events_times_range[0] is None:  # means no spectra have been added.
            return
        elif self._events_times_range[0] == self._events_times_range[1]:
            self._events_times_range[1] += 0.01

        if self._init_window_width is None:
            _time_range = self.time_range
            dt = _time_range[1] - _time_range[0]
            self._init_window_width = dt / 2
            self._init_slider_pos = _time_range[0] + dt / 4

        self.slider_time_center = plt.Slider(self.slider_erg_center_ax, "Center time", valmin=self._events_times_range[0],
                                             valmax=self._events_times_range[1], valinit=self._init_slider_pos,
                                             valstep=self.delta_t)

        self.current_fits = {i: [] for i in range(len(self))}

        if self.window_max is None:
            self.window_max = self.time_range[-1] - self.time_range[0]

        self.slider_window = plt.Slider(self.slider_window_ax, 'Time\nwindow', valmin=0,
                                        valmax=self.window_max, valinit=self._init_window_width,
                                        valstep=self.delta_t/2)

        self.radiobutton_active_select = RadioButtons(self.radiobutton_active_select_ax, labels=self.titles,
                                                      activecolor='black')

        for index, circle in enumerate(self.radiobutton_active_select.circles):
            circle: Circle
            c = self.cmap(index)
            circle.set_color(c)
            circle.set_linewidth(2)

        self.slider_time_center.on_changed(self._update_plot)
        self.slider_window.on_changed(self._update_plot)

        self.radiobutton_active_select.on_clicked(self._radio_button_on_clicked)
        self.radiobutton_active_select.set_active(0)

        self._on_checked_time_integrated('')
        # self.ax.callbacks.connect('xlim_changed', self._xlim_changed)


def trest_gassian(peaks, rel_amplitudes, N, slope_ratio=3, xmin=0, xmax=100, sigma=2.5, nbins=None, non_uniform_bins=False,
                  i: InteractiveSpectra = None):
    def linear_rnd(x0, x1, ratio, n):
        """
        Generate data set following linear curve.
        Args:
            x0: min value
            x1: max value
            ratio: Ratio between number of events in minimum and maximum bins
            n: number of samples

        Returns:

        """
        y = np.random.rand(n)
        r = ratio
        out = (-np.sqrt((1 + r) ** 2) + np.sqrt(1 + (-1 + r ** 2) * y) + r * np.sqrt(1 + (-1 + r ** 2) * y)) / (
                    (-1 + r) * np.sqrt((1 + r) ** 2))
        out *= (x1 - x0)
        out += x0
        return out

    data = linear_rnd(xmin, xmax, slope_ratio, int(N/2))

    rel_amplitudes = np.array(rel_amplitudes)/np.sum(rel_amplitudes)

    amplitudes = []
    for peak, A in zip(peaks, rel_amplitudes):
        data = np.concatenate([data, np.random.normal(peak, sigma, int(A*N))])
        amplitudes.append(int(A*N))

    if nbins is None:
        nbins = 2 * (xmax - xmin)

    if non_uniform_bins:
        bins = np.cumsum(np.linspace(0, 1, 150)/2 + 1)
        bins = (bins - bins[0])/(bins[-1] - bins[0])
        bins *= xmax - xmin
        bins += xmin
    else:
        bins = np.linspace(xmin, xmax, )

    y, _ = np.histogram(data, bins)
    fit_result = multi_guass_fit(bins, y, [884.2], make_density=True)
    # fit_result.plot_fit()

    if i is None:
        i = InteractiveSpectra()

    i.add_static_spectra(bins, y, erg_min=xmin)
    # i._prepare()

    # for amp_fit, amp_true, peak in zip(fit_result.amplitudes(), amplitudes, peaks):
    #     print(f"{peak} keV: True: {amp_true:.2e}, fit: {amp_fit:.2e}")



if __name__ == '__main__':

    bins = np.arange(0, 20, 0.5)
    xs = [5, 16]
    vals = np.concatenate([np.random.normal(loc=xs[0], scale=1, size=10000),
                          np.random.normal(loc=xs[1], scale=2, size=10000)])


    y, _  = np.histogram(vals, bins=bins)
    fit = multi_guass_fit(bins, y, center_guesses=xs, share_sigma=False)

    for i in range(2):
        print(fit.sigmas(i))

    fit.plot_fit(

    )
    plt.show()

