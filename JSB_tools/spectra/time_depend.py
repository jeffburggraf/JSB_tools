import numbers
import re
import time
import warnings
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.text import Text
from JSB_tools import mpl_hist, convolve_gauss
from pathlib import Path
from matplotlib.patches import Circle
from functools import cached_property, cache
from typing import List, Dict, Union
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
from numba.typed import List as numba_list
from numba import jit, prange
from math import erf
from logging import warning


instructions = """
Instructions

Performing fits:
    While holding the space key, click the canvas one or more times to select peaks to be fit with Gaussians. 
    A multi-gaussian fit is performed for all selected peaks upon release of the space bar. When clicking peaks, 
    a single  mouse click will use the nearest local maxima for initial peak energy value while a double click will 
    constrain the value to within the bin that was clicked.

"Track fits" checkbox: 
    When selected, fits will be re-calculated as time cuts are changed, while preserving the original peak centers and 
    sigmas . 
    
"Clear Fits" button:
    Click the "Clear fits" button in the upper right to clear all fits of current selected spectrum.
    Click a second time to clear all fits.

Selecting Spectra:
    The radio buttons on the center right can be used to select the active spectrum. All other spectra (if any) besides 
    the selected spectrum will be grey, while the active spectrum blue.

Setting spectra time cuts:
    The time cuts possible are of the form
        c-w/2 <= time <= c + w/2
    where c and w are set by the sliders titled 'Center time' and 'Time window', respectively.
    To disable time cuts, select the check box in the lower left titled "All events" (this is the default).

"Rescale y" button:
    Rescale y to the min/max of selected spectrum. 
    

"""


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
def get_erg_spec(erg_binned_times: numba_list, time_range: np.ndarray, n: int=None, bg_subtract=False, bg_window=40):
    """
    Return compiled function for generating energy spectrum with time cuts.
    Args:
        erg_binned_times:

        time_range: array like [min_time, max_time]. A value of np.array([]) means no time cuts.

        n: Size of array to be filled with result (when larger than len(erg_binned_times), 0's are
            padded onto end of return array)

        bg_subtract:

        bg_window:

    Returns: array of shape len(erg_binned_times) representing counts for each energy bin

    """
    if n is None:
        out = np.zeros(len(erg_binned_times))
    else:
        out = np.zeros(n)

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

        # if scales[i] != 1:
        #     val *= scales[i]

        out[i] = val

    if bg_subtract:
        out = remove_background(out, median_window=bg_window)

    return out


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
    xmin, xmax = bins[0], bins[-1] - 1E-3
    I0, I1 = np.searchsorted(bins, [max(xmin, min(center_guesses) - fit_buffer_window),
                                    min(xmax, max(center_guesses) + fit_buffer_window)],
                             side='right') - 1
    return (I0, I1 + 2), (I0, I1 + 1)


def linear(x, slope, bg):
    """a line used for fitting background. """
    return slope * (x - x[len(x) // 2]) + bg


def multi_guass_fit(bins, y, center_guesses, fixed_in_binQ: List[bool]=None, yerr=None, share_sigma=True,
                    sigma_guesses: Union[list, float] = None, fix_sigmas: bool = False, fix_centers: bool = False,
                    fix_bg: float = None) \
        -> ModelResult:
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

        yerr: Error in y.

        share_sigma: If True, all peaks share the same sigma (as determined by fit).

        sigma_guesses: sigma guess. If number, use same number for all fits.

        fix_sigmas: Force sigmas to sigma_guesses.

        fix_centers: If True, peak centers wil be fixed at center_guesses

        fix_bg: Fix background to this value

    Returns: lmfit.ModelResult

    """

    if fix_sigmas:
        assert sigma_guesses is not None, "Use of fix_sigmas requires sigma_guesses argument. "
        if isinstance(sigma_guesses, numbers.Number):
            sigma_guesses = [sigma_guesses] * len(center_guesses)

    x = 0.5 * (bins[1:] + bins[:-1])

    assert len(center_guesses) != 0, 'Cannot perform fit!'

    if fixed_in_binQ is None:
        fixed_in_binQ = [False] * len(center_guesses)

    center_guesses = [max(bins[0], min(bins[-1] - 1E-10, x)) for x in center_guesses]

    b_width = 0.5 * ((bins[1] - bins[0]) + (bins[-1] - bins[-2]))  # ~mean bin width

    if isinstance(y[0], UFloat):
        yerr = unp.std_devs(y)
        y = unp.nominal_values(y)

    if yerr is None:
        weights = np.ones_like(y)
    else:
        weights = 1.0/np.where(yerr > 0, yerr, 1)

    model: Model = None
    params: Parameters = None

    max_slope = np.abs(max(y) - min(y)) / (x[-1] - x[0])
    quarter = int(len(y)//4)  # 1/4 length of y/x
    slope_guess = (np.mean(y[-quarter:]) - np.mean(y[:quarter]))/(x[-1] - x[0])
    bg_guess = fix_bg if fix_bg is not None else min([np.mean([y[i-1], y[i], y[i + 1]]) for i in range(1, len(y) - 1)])

    min_sigma, max_sigma = 1.75*b_width, b_width * len(y)/3

    def set_sigma(param: Parameter):
        kwargs = {'min': min_sigma, 'max': max_sigma}

        if sigma_guesses is None:
            kwargs['value'] = b_width * 3
        else:
            kwargs['value'] = sigma_guesses[i]

        if fix_sigmas:
            kwargs['vary'] = False
        else:
            kwargs['vary'] = True

        if share_sigma:
            if param.name == '_0_sigma':  # this param is the only sigma, don't change kwargs
                pass
            else:
                kwargs['expr'] = f'_0_sigma'  # param
                kwargs['value'] = None
                kwargs['min'] = None
                kwargs['max'] = None
        param.set(**kwargs)

    def set_center(param: Parameter):
        kwargs = {'vary': True, 'value': center_guess}

        if fix_centers:
            kwargs['vary'] = False

        elif fixedQ:
            kwargs['min'] = bins[i0]
            kwargs['max'] = bins[i0 + 1]
            assert bins[i0] < center_guess <= bins[i0 + 1], (bins[i0], center_guess, bins[i0 + 1])
        else:
            kwargs['value'] = x[find_local_max(y, i0)]

        param.set(**kwargs)

    def amp_guess(max_y, sigma):
        return (max_y - bg_guess) * np.sqrt(2 * np.pi) * sigma

    for i, (fixedQ, center_guess) in enumerate(zip(fixed_in_binQ, center_guesses)):
        i0 = np.searchsorted(bins, center_guess, side='right') - 1

        prefix = f'_{i}_'
        sub_model = GaussianModel(prefix=prefix)
        sub_params = sub_model.make_params()

        if model is None:
            model = Model(linear) + sub_model

            params = model.make_params()
            params['slope'].set(value=slope_guess, max=max_slope, min=-max_slope)
            params['bg'].set(value=bg_guess, vary=fix_bg is None)
        else:
            model += sub_model
            params.update(sub_params)

        set_sigma(params[f'{prefix}sigma'])

        set_center(params[f'{prefix}center'])

        _amp_guess = amp_guess(y[i0], params[f'{prefix}sigma'].value)

        params[f'{prefix}amplitude'].set(value=amp_guess(y[i0], params[f'{prefix}sigma'].value),
                                         min=_amp_guess/10)

    fit_result = model.fit(data=y, params=params, weights=weights, x=x, )

    fit_result.userkws['b_width'] = b_width

    return fit_result


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


class InteractiveSpectra:
    print_click_coords = False  # If True, print mouse click points in fractional Axes coords (used for dev)

    default_fit_window = 6  # in keV
    default_n_convolve_bins = 3  # amount to convolve signal when guessing guass fit centers.

    cmap = plt.get_cmap("tab10")

    def add_list(self, spec, scale=1, disable_eff_corr=False, title=None, erg_min=None, erg_max=None):
        """
        Same as add_spectra except can be used
        Args:
            spec:
            scale:
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
        self.add_spectra(spec.energy_binned_times, spec.erg_bins, effs=effs, scale=scale, title=title,
                         erg_min=erg_min, erg_max=erg_max)

    def add_spectra(self, energy_binned_times, energy_bins, effs=None, scale=1, title=None, erg_min=80, erg_max=None):
        """

        Args:
            energy_binned_times: See ... todo
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
        else:
            assert isinstance(scale, (float, int))
            scale = scale * np.ones(len(energy_binned_times), dtype=float)

        i0 = (-1 + np.searchsorted(energy_bins, erg_min, side='right')) if erg_min is not None else 0
        i1 = (np.searchsorted(energy_bins, erg_max, side='right')) if erg_max is not None else \
            (len(energy_bins) - 1)

        energy_binned_times = energy_binned_times[i0: i1]
        effs = effs[i0: i1] if effs is not None else None
        scale = scale[i0: i1] if isinstance(scale, np.ndarray) else scale
        energy_bins = energy_bins[i0: i1 + 1]

        if effs is not None:
            scale /= effs

        self.scales.append(scale)
        self.energy_binned_timess.append(numba_list(energy_binned_times))
        self.erg_binss.append(energy_bins)

        if self._events_times_range[0] is None:
            self._events_times_range[0] = min([ts[0] for ts in energy_binned_times if len(ts)])
        else:
            self._events_times_range[0] = min(self._events_times_range[0], min([ts[0] for ts in energy_binned_times if len(ts)]))

        if self._events_times_range[-1] is None:
            self._events_times_range[-1] = max([ts[-1] for ts in energy_binned_times if len(ts)])
        else:
            self._events_times_range[-1] = max(self._events_times_range[-1], max([ts[-1] for ts in energy_binned_times if len(ts)]))

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

        if ' ' in self.keys_currently_held_down:  # fit peak selected
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
        self.fig.canvas.draw()

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
        line._yerr = yerr  # custom attribute for now. todo: Draw error bars?

        line.stale = True
        line._invalidy = True

    @property
    def min_time(self):
        """
        Min time according to the two time cut sliders
        Returns:

        """
        return self.slider_time_center.val - self.slider_window.val / 2

    @property
    def max_time(self):
        """
        Max time according to the two time cut sliders

        Returns:

        """
        return self.slider_time_center.val + self.slider_window.val / 2

    def _set_ylims(self, *args, **kwargs):
        xlim = self.ax.get_xlim()
        ylim = np.inf, -np.inf

        x, y = self.handles[self._active_spec_index].get_data()
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

    def _calc_y(self, index, tmin, tmax, append_last_value=False):
        """

        Args:
            index: Index to select spectrum
            tmin:
            tmax:
            append_last_value: Last bin value can be duplicated in order to draw histogram.

        Returns: y, yerr

        """
        if self._time_integratedQ:
            time_range = np.array([])  # accept all events.
        else:
            time_range = np.array([tmin, tmax])  # perform time cut.

        bg_subtractQ = self.checkbox_bg_subtract.get_status()[0]
        bg_window = int(self.bg_textbox.text)

        energy_binned_times = self.energy_binned_timess[index]
        scales = self.scales[index]

        n = len(energy_binned_times)

        if append_last_value:
            n += 1  # array will have one extra value at end.

        out = get_erg_spec(energy_binned_times, time_range, n=n)

        out_err = np.sqrt(out)

        if bg_subtractQ:  # purposely after out_err declaration
            out = remove_background(out, bg_window)

        out[:-1] *= scales
        out_err[:-1] *= scales

        if append_last_value:
            out[-1] = out[-2]

        return out, out_err

    def _update_time_cut_display(self):
        def get_text():
            if self._time_integratedQ:
                return "No time cuts applied."
            else:
                t0, t1 = self.min_time, self.max_time

            return f'{t0:.1f} < t < {t1:.1f}'

        if self.time_cut_text is None:
            self.time_cut_text = self.ax.text(0.80, 0.97, get_text(),
                                              transform=self.ax.transAxes, fontsize=12)
        else:
            self.time_cut_text.set_text(get_text())

    def _update_plot(self, *args):
        for index in range(len(self)):
            y, yerr = self._calc_y(index, self.min_time, self.max_time, True)
            self.update_line(y, yerr, index)

        if self.checkbox_track_fits.get_status()[0]:
            self._re_perform_fits()

        self._update_time_cut_display()
        self._draw()

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

    def _key_press(self, event):
        self.keys_currently_held_down[event.key] = True

        try:  # call any functions bound to key by self.key_press_events
            self.key_press_events[event.key]()
        except KeyError:
            pass

    def _re_perform_fits(self):
        """
        Re-does fits in active spectrum after change ion slider values. Fixes the energy to that of the original fit.
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

    def _perform_fits(self, center_guesses=None, fixed_binsQs=None, sigma_guesses=None, force_centers=None,
                      force_sigma=False, index=None):
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

        if not len(center_guesses):  # no fits to perform!
            logging.warning("Space bar pressed but no peaks were selected!")
            return

        bins_slice, y_slice = get_fit_slice(self.erg_binss[index], center_guesses, self.fit_settings['fit_buffer_window'])

        y_slice = slice(*y_slice)
        bins_slice = slice(*bins_slice)
        bins = self.erg_binss[index][bins_slice]

        y, yerr = self.get_y(index)
        y = y[y_slice]
        yerr = yerr[y_slice]

        fit_result: ModelResult = multi_guass_fit(bins=bins, y=y,
                                                  center_guesses=center_guesses,
                                                  fixed_in_binQ=fixed_binsQs,
                                                  yerr=yerr,
                                                  sigma_guesses=sigma_guesses,
                                                  fix_centers=force_centers,
                                                  fix_sigmas=force_sigma,
                                                  share_sigma=self.fit_settings['only_one_sigma'])

        x = fit_result.userkws[fit_result.model.independent_vars[0]]
        y = fit_result.data
        b_width = fit_result.userkws['b_width']

        params = fit_result.params

        line = self.handles[index]

        fit_line1 = self.ax.plot(x, fit_result.eval(x=x, params=params), c='black', ls='-', lw=2)[0]
        fit_line2 = self.ax.plot(x, fit_result.eval(x=x, params=params), c=line.get_color(), ls='--', lw=2)[0]

        fit_info = {'visibles': [fit_line1, fit_line2],
                    'fit_result': fit_result,
                    'fixed_binsQs': fixed_binsQs}

        text_trans = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)

        for i in range(len(center_guesses)):
            prefix = f"_{i}_"
            fit_center = params[f'{prefix}center'].value

            xpos = fit_center

            text = self.ax.text(xpos, 1.01,
                                f"E=${get_param_ufloat(f'{prefix}center', params):.2fL}$ keV\n"
                                f"N=${get_param_ufloat(f'{prefix}amplitude', params) / b_width:.2L}$",
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

    def _key_release(self, event):
        del self.keys_currently_held_down[event.key]
        if event.key == ' ':
            # do fitting stuff
            self._perform_fits()

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
        plt.figure()
        fix, ax = plt.subplots()
        plt.text(0.9, 0.9, instructions)

    def __init__(self, init_window_width=35, window_max=None, delta_t=5):
        """

        Args:
            init_window_width: Initial time-integration window width
            window_max: Max width of time integration window possible to choose using the slider
            delta_t: time step for the slider determining the center time of the integration time range
        """
        self.fit_settings = {}
        self.set_fit_settings()

        # if keyboard key specified by dict key _ is pressed, call function specified by dict value
        self.key_press_events = {}

        self._events_times_range: List[float] = [None, None]  # global time min/max of all events of all spectra

        self.handles: List[Line2D] = []
        self.erg_binss = []
        self.energy_binned_timess = []
        self.scales = []  # scale each spectrum by a value.
        self.titles = []

        fig, ax = plt.subplots(figsize=(16, 9))
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
        self.checkbox_track_fits_ax = fig.add_axes([0.1, self.ax.get_position().y1 - 0.02, 0.15, 0.15 * 2])

        self.checkbox_bg_subtract_ax = fig.add_axes([0.83, 0.81, 0.1, 0.2])
        self.bg_textbox_ax = fig.add_axes([0.8,  0.9, 0.03, 0.02])

        vanilla_button('Clear fits', self._clear_fits)

        vanilla_button('rescale y (Y)', self._set_ylims)
        self.key_press_events['y'] = self._set_ylims

        vanilla_button('help', self._help)

        self.radiobutton_active_select_ax = fig.add_axes([0.9, 0.35, 0.15, 0.3])

        self.slider_erg_center_ax.set_axis_off()
        self.slider_window_ax.set_axis_off()
        self.checkbox_time_integrated_ax.set_axis_off()
        self.checkbox_track_fits_ax.set_axis_off()
        self.radiobutton_active_select_ax.set_axis_off()
        self.checkbox_bg_subtract_ax.set_axis_off()

        self.window_width = init_window_width
        self.slider_pos = init_window_width / 2

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

        self.keys_currently_held_down = {}
        # \end GUI state variables.

        # self.check_boxes = CheckButtons

        self.checkbox_time_integrated = CheckButtons(self.checkbox_time_integrated_ax, ['All\nevents'], actives=[True])
        self.checkbox_time_integrated.on_clicked(self._on_checked_time_integrated)

        self.checkbox_track_fits = CheckButtons(self.checkbox_track_fits_ax, ['Track fits'], actives=[False])

        self.checkbox_bg_subtract = CheckButtons(self.checkbox_bg_subtract_ax, ['Remove\nbaseline'], actives=[False])
        self.checkbox_bg_subtract.on_clicked(self._update_plot)

        self.bg_textbox = TextBox(self.bg_textbox_ax, "width (keV): ", '40', )
        self.bg_textbox.on_submit(self._update_plot)

        self.fig.canvas.mpl_connect('key_press_event', self._key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._key_release)

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

    def show(self):
        if not len(self):
            warning("Cannot show InteractiveSpectra plot because no spectra have been added. Use self.add_spectra().")
            return

        self.slider_time_center = plt.Slider(self.slider_erg_center_ax, "Center time", valmin=self._events_times_range[0],
                                             valmax=self._events_times_range[1], valinit=self.window_width / 2,
                                             valstep=self.delta_t)

        self.current_fits = {i: [] for i in range(len(self))}

        if self.window_max is None:
            self.window_max = 0.75 * self._events_times_range[-1]

        self.slider_window = plt.Slider(self.slider_window_ax, 'Time\nwindow', valmin=0,
                                        valmax=self.window_max, valinit=self.window_width)

        # radio_labels = [str(i + 1) for i in range(len(self))]
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


if __name__ == '__main__':
    n = int(1E6)
    is_ = np.random.randint(0, n, n)
    times = np.arange(0, n)
    erg_binned_timnes = make_energy_binned_times(3000, is_, times)
    print('hi')
    # from JSB_tools import mpl_hist_from_data, TabPlot
    # erg_binned_times = [[] for i in range(10)]
    # n = 10000
    # n_ergs = len(erg_binned_times)
    # time_bins = np.linspace(0, 30, 12)
    # ax1 = None
    #
    # for e, t in zip(np.random.normal(n_ergs/2, 1.5, n), np.random.exponential(10/np.log(2), n)):
    #     i = int(e)
    #     if not 0 <= i < n_ergs:
    #         continue
    #     erg_binned_times[i].append(t)
    #
    # for i, ts in enumerate(erg_binned_times):
    #     erg_binned_times[i] = np.array(erg_binned_times[i])
    #
    #     _, ax1 = mpl_hist_from_data(time_bins, ts, ax=ax1, label=i - n_ergs/2)
    #
    # erg_binned_times = numba_list(erg_binned_times)
    #
    # plt.figure()
    #
    # tab = TabPlot()
    # gaus_times = np.linspace(0, max(time_bins), 20)
    # result = []
    #
    # for i, erg_spec in enumerate(gaussian_weighted_cut(erg_binned_times, gaus_times, 4)):
    #     ax = tab.new_ax(f"{gaus_times[i]:.1f}")
    #     result.append(sum(erg_spec))
    #     mpl_hist(np.arange(0, n_ergs + 1), erg_spec, ax=ax)
    #
    #
    # ax1.plot(gaus_times, result, label='gaus time dep')
    # ax1.legend()
    #
    # plt.show()
    #
    # from analysis import Shot
    # # times_ = None
    # # #  todo:
    # # #   Add energy rebin input box.
    # # #   Add titles to replace default radio button lables of "1", "2", etc.
    # # #   Add button for turning off fit tracking with slider.
    # #
    # #
    # # def phelix_list():
    # #     from Germany2022 import Shot
    # #     out = None
    # #     for shotnum in range(50, 65):
    # #         out += Shot(shotnum).left_spec
    # #         out += Shot(shotnum).right_spec
    # #     return out
    # #
    # list_shot1 = Shot(134).list
    # list_shot2 = Shot(13).list
    # #
    # # InteractiveSpectra.print_click_coords = True
    # #
    # interactive = InteractiveSpectra()
    # interactive.print_click_coords = True
    #
    # interactive.add_list(list_shot1, erg_min=200, erg_max=300)
    # interactive.add_list(list_shot2, erg_min=200, erg_max=300)
    # interactive.show()
    # #
    # interactive.checkbox_track_fits.set_active(0)
    # interactive.checkbox_time_integrated.set_active(0)
    # #
    # interactive._perform_fits([218])
    # #
    # # interactive.slider_window.val = 50
    #
    #
    # plt.show()
