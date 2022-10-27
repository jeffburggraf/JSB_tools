import time
import warnings
import numpy as np
from matplotlib import pyplot as plt
from JSB_tools import mpl_hist, convolve_gauss
from pathlib import Path
from functools import cached_property, cache
from typing import List, Dict
from lmfit import Model, Parameters, Parameter
from lmfit.models import GaussianModel
from lmfit.model import ModelResult
from uncertainties import ufloat, UFloat
import uncertainties.unumpy as unp
from matplotlib.widgets import CheckButtons, Slider, Button, RadioButtons
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import logging
import matplotlib.transforms as transforms
from numba.typed import List as numba_list
from numba import jit, prange


instructions = """
Instructions

Selecting Spectra:
    The radio buttons on the center right can be used to select the active spectrum. All other spectra (if any) besides 
    the selected spectrum will be grey, while the active spectrum blue.

Setting spectra time cuts:
    The time cuts possible are of the form
        c-w/2 <= time <= c + w/2
    where c and w are set by the sliders titled 'center' and 'window', respectively.
    To disable time cuts, select the check box in the lower left titled "All events" (this is the initial state).

Performing fits:
    While holding the space key, click the canvas one or more times to select peaks to be fit with Gaussians. 
    A multi-gaussian fit is performed for all selected peaks upon release of the space bar. When clicking peaks, 
    a single  mouse click will use the nearest local maxima for initial peak energy value while a double click will 
    constrain the value to within the bin that was clicked.

    Clearing Fits:
        Click the "Clear fits" button in the upper right to clear all fits of current selected spectrum.
        Click a second time to clear all fits.


"""


@jit(nopython=True)
def get_erg_spec(n: int, erg_binned_times: numba_list, time_range: np.ndarray, scales: np.ndarray):
    """
    Return compiled function for generating energy spectrum with time cuts.
    Args:
        n: Size of array to be filled with result (when larger than len(erg_binned_times), 0's are
            padded onto end of return array)

        erg_binned_times:

        time_range: array like [min_time, max_time]. A value of np.array([]) means no time cuts.

        scales: Multiplicative array. (used for e.g. efficiency corrections)

    Returns: array of shape len(erg_binned_times) representing counts for each energy bin

    """
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

        if scales[i] != 1:
            val *= scales[i]

        out[i] = val

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
        a: Array
        i0: inital index

    Returns: int

    """
    # print(f"Starting at i = {i0}, a[i0] = {a[i0]}")
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


def multi_guass_fit(bins, y, center_guesses, fixed_in_binQ: List[bool], fit_buffer_window=5, share_sigma=True) \
        -> ModelResult:
    """
    Perform multi-gaussian fit to data.

    Notes:
        If y is a UFloat, erry = unp.std_devs(y), else erry = np.sqrt(y).

    Args:
        bins: Bin left edges. Must be len(y) + 1
        y: Bin values.
        center_guesses: Guesses for centers of each peak to be fitted.

        fixed_in_binQ: List of bool. True means energy is constrained to the energy bin according to the
            corresponding element in `center_guesses`. Must be same len as center_guesses.

        fit_buffer_window: Amount to extend the fit beyond the provided min/max of `center_guesses`.
            Should be roughly 2-4 times the FWHM

        share_sigma: If True, all peaks share the same sigma for each fit.

    Returns: lmfit.ModelResult

    """

    def prepare_data():
        x_out = 0.5 * (bins[1:] + bins[:-1])
        I0, I1 = np.searchsorted(bins, [min(center_guesses) - fit_buffer_window, max(center_guesses) + fit_buffer_window],
                                 side='right') - 1
        i0s = np.searchsorted(bins, center_guesses, side='right') - 1 - I0

        x_out = x_out[I0: I1 + 1]

        y_out = y[I0: I1 + 1]

        if isinstance(y[0], UFloat):
            yerr = unp.std_devs(y_out)
        else:
            yerr = np.sqrt(y_out)  # assume possoin errors

        weights_out = 1.0/np.where(yerr > 0, yerr, 1)

        init_is_out = []

        bins_ = bins[I0: I1 + 2]
        b_width_out = np.mean(bins_[1:] - bins_[:-1])

        vary_ranges_out = []

        init_xs_out = []

        for fixed, i0 in zip(fixed_in_binQ, i0s):
            if not fixed:
                vary_ranges_out.append(None)  # gaus center is not fixed.
                find_local_max(y_out, i0)
            else:
                vary_ranges_out.append([bins_[i0], bins_[i0 + 1]])

            init_xs_out.append(x_out[i0])
            init_is_out.append(i0)

        return (b_width_out, x_out), (y_out, weights_out), init_xs_out, init_is_out, vary_ranges_out

    def linear(x, slope, bg):
        """a line used for fitting background. """
        return slope * (x - x[len(x) // 2]) + bg

    center_guesses = [max(bins[0], min(bins[-1] - 0.001, x)) for x in center_guesses]  #

    (b_width, x), (y, weights), init_xs, init_is, vary_ranges = prepare_data()

    model: Model = None
    params: Parameters = None

    max_slope = np.abs(max(y) - min(y)) / (x[-1] - x[0])
    quarter = int(len(y)//4)  # 1/4 length of y/x
    slope_guess = (np.mean(y[-quarter:]) - np.mean(y[:quarter]))/(x[-1] - x[0])
    bg_guess = min([np.mean([y[i-1], y[i], y[i + 1]]) for i in range(1, len(y) - 1)])

    sigma_guess = b_width * 3
    min_sigma, max_sigma = b_width * len(y)/3, 1.75*b_width

    def amp_guess(max_y, sigma):
        return (max_y - bg_guess) * np.sqrt(2 * np.pi) * sigma
    # amp_guess

    for i, (x0, i0, range_) in enumerate(zip(init_xs, init_is, vary_ranges)):
        prefix = f'_{i}_'
        sub_model = GaussianModel(prefix=prefix)
        sub_params = sub_model.make_params()
        if model is None:
            model = Model(linear) + sub_model

            params = model.make_params()
            params['slope'].set(value=slope_guess, max=max_slope, min=-max_slope)
            params['bg'].set(value=bg_guess)

            if share_sigma:
                params[f'{prefix}sigma'].set(value=sigma_guess, max=min_sigma, min=max_sigma)
        else:
            model += sub_model

            params.update(sub_params)
            params[f'{prefix}sigma'].set(vary=False, expr=f'_0_sigma')

        if range_ is None:  # Fit energy is not fixed within selected bin.
            params[f'{prefix}center'].set(value=x0)
        else:
            params[f'{prefix}center'].set(value=x0, min=range_[0], max=range_[1])

        if not share_sigma:
            params[f'{prefix}sigma'].set(value=sigma_guess, max=min_sigma, min=max_sigma)

        _amp_guess = amp_guess(y[i0], params[f'{prefix}sigma'].value)

        params[f'{prefix}amplitude'].set(value=amp_guess(y[i0], params[f'{prefix}sigma'].value),
                                         min=_amp_guess/10)

    fit_result = model.fit(data=y, params=params, weights=weights, x=x, )

    plt.figure()
    fit_result.plot_fit(show_init=True)

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

    def add_list(self, spec, scale=1, ergmin=None, ergmax=None):
        assert hasattr(spec, 'energy_binned_times')
        self.add_spectra(spec.energy_binned_times, spec.erg_bins, effs=spec.effs, scale=scale, erg_min=ergmin, erg_max=ergmax)

    def add_spectra(self, energy_binned_times, energy_bins, effs=None, scale=1, erg_min=100, erg_max=None):
        """

        Args:
            energy_binned_times: See todo
            energy_bins: Energy bin edges
            effs: Array of efficiencies. len(effs) == len(energy_bins) - 1
            scale: Scale the spectrum. Can be scalar or array.
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

        handle = self.ax.plot(energy_bins, np.ones(len(energy_bins)), ds='steps-post')[0]
        self.handles.append(handle)

    def _mouse_click(self, event):
        """"""
        if self.print_click_coords:
            x1, x2 = interactive.ax.get_xlim()
            y1, y2 = interactive.ax.get_ylim()
            fx, fy = (event.xdata - x1) / (x2 - x1), (event.ydata - y1) / (y2 - y1)
            # print(x1, x2, y1, y2)
            # print(event.x, event.y)
            print(f"x,y = {fx:.2f},{fy:.2f} in fractional Axes coords.")
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

    def update_line(self, y, index):
        """
        Args:
            y: Hist values with y[-1] appended onto end (i.e. must be same length as self.erg_bins[index]).
            index: Index representing the active spectrum

        Returns:

        """
        assert len(y) == len(self.erg_binss[index]), "Bad trouble!"
        line: Line2D = self.handles[index]
        line._yorig = y

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
        for l in self.handles:  # was self.ax.lines. Is this better?
            x, y = l.get_data()
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

    def _get_y(self, index, tmin, tmax, append_last_value=False):
        global times_
        t0 = time.time()

        if self._time_integratedQ:
            time_range = np.array([])  # accept all events.
        else:
            time_range = np.array([tmin, tmax])  # perform time cut.

        energy_binned_times = self.energy_binned_timess[index]
        scales = self.scales[index]

        n = len(energy_binned_times)
        if append_last_value:
            n += 1  # array will have one extra value at end.

        out = get_erg_spec(n, energy_binned_times, time_range, scales)

        if append_last_value:
            out[-1] = out[-2]
        t1 = time.time()
        if times_ is None:
            times_ = []
        else:
            times_.append(t1 - t0)
        print(f"Time to calc spec: {t1 - t0:.2e}. mean = {np.mean(times_) if len(times_) else 0:.2e}")

        return out

    def _update_time_cut_display(self):
        def get_text():
            if self._time_integratedQ:
                t0, t1 = self._events_times_range
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
            counts = self._get_y(index, self.min_time, self.max_time, True)

            self.update_line(counts, index)
        self._update_time_cut_display()
        # self._set_ylims()
        self._draw()

    @property
    def _time_integratedQ(self):
        return self.checkbox_time_integrated.lines[0][0].get_visible()

    @property
    def _active_spec_index(self) -> int:
        """
        Get the index according to which spectrum is currently selected to be active in the GUI.
        Returns: int

        """
        return int(self.radiobutton_active_select.value_selected) - 1

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

        self._clear_fits()  # no purpose in keeping them!
        self._update_plot()

    def _key_press(self, event):
        self.keys_currently_held_down[event.key] = True

        try:  # call any functions bound to key by self.key_press_events
            self.key_press_events[event.key]()
        except KeyError:
            pass

    def _perform_fits(self):
        i = self._active_spec_index
        if not len(self.fit_clicks):  # no fits to perform!
            logging.warning("Space bar pressed but no peaks were selected!")
            return

        fit_result: ModelResult = multi_guass_fit(bins=self.erg_binss[i], y=self.handles[i].get_ydata(),
                                                  center_guesses=self.fit_clicks['x_points'],
                                                  fixed_in_binQ=self.fit_clicks['fixed_binsQs'],
                                                  fit_buffer_window=self.fit_settings['fit_buffer_window'],
                                                  share_sigma=self.fit_settings['only_one_sigma'])
        x = fit_result.userkws[fit_result.model.independent_vars[0]]
        y = fit_result.data
        b_width = fit_result.userkws['b_width']

        params = fit_result.params

        line = self.handles[i]

        fit_line1 = self.ax.plot(x, fit_result.eval(x=x, params=params), c='black', ls='-', lw=2)[0]
        fit_line2 = self.ax.plot(x, fit_result.eval(x=x, params=params), c=line.get_color(), ls='--', lw=2)[0]

        fit_info = {'visibles': [fit_line1, fit_line2]}

        text_trans = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)

        for i in range(len(self.fit_clicks['x_points'])):
            prefix = f"_{i}_"
            fit_center = params[f'{prefix}center'].value

            # dx = -0.01 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            # dy = 0.01 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            xpos = fit_center
            # ypos = y[np.searchsorted(x, params[f'{prefix}center'].value)] + 0.1 * (max(y) - min(y))
            # ypos = self.ax.get_ylim()[-1] + dy

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

        self.current_fits[self._active_spec_index].append(fit_info)
        self._reset_fit_clicks()  # self._draw is called here.

    def _key_release(self, event):
        del self.keys_currently_held_down[event.key]
        if event.key == ' ':
            # do fitting stuff
            self._perform_fits()

    def _radio_button_on_clicked(self, val):
        for index in range(len(self)):
            line = self.handles[index]

            if index == self._active_spec_index:
                line.set_color("tab:blue")
                line.set_zorder(10)
            else:
                line.set_color("lightgrey")
                line.set_zorder(-10)

            for fit_info in self.current_fits[index]:
                for thing in fit_info['visibles']:
                    try:
                        if index == self._active_spec_index:
                            thing.set_color(thing._default_color)
                        else:
                            thing.set_color('lightgrey')
                    except AttributeError:
                        pass
        self._draw()

    def _reset_fit_clicks(self):
        """
        Re-set to default data used to track fits initiated by mouse clicks while holding space bar.
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

    def _clear_fits(self, event=None, clear_all=False):
        """

        Args:
            event: Not used
            clear_all: Will clear fits from all spectra (instead of just active spectra)

        Notes:
            the code snippet, "(not len(self.current_fits[index]))" implements a feature where clicking the clear fit
             button twice clears all fits (not just active).

        Returns:

        """
        index = self._active_spec_index

        def clear_(index_):  # clears fit data stored for spectrum index_
            fit_infos = self.current_fits[index_]

            for fit_info in fit_infos:
                for thing in fit_info['visibles']:
                    thing.set_visible(0)

            self.current_fits[index_] = []

        if clear_all or (not len(self.current_fits[index])):  # clear all spectra fits
            for i in range(len(self)):
                clear_(i)
        else:
            clear_(index)  # just clear current spectrum fits

        self._draw()

    @staticmethod
    def _help(*args):
        plt.figure()
        print("helkp")
        fix, ax = plt.subplots()
        plt.text(0.9, 0.9, instructions)

    def __init__(self, init_window_width=10, window_max=None, delta_t=5):
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
            ax = plt.axes([x0, y0, dx, 0.03])
            self.vanilla_button_axs[name] = ax

            b = Button(ax, name)
            self.vanilla_buttons[name] = b

            b.on_clicked(callback)

        self.slider_erg_center_ax = plt.axes([0.1, 0.06, 0.8, 0.05])
        self.slider_window_ax = plt.axes([0.1, 0.01, 0.8, 0.05])
        self.checkbox_time_integrated_ax = plt.axes([0.91, 0.1, 0.15, 0.15 * 2])

        vanilla_button('Clear fits', self._clear_fits)

        vanilla_button('rescale y (Y)', self._set_ylims)
        self.key_press_events['y'] = self._set_ylims

        vanilla_button('help', self._help)

        self.radiobutton_active_select_ax = plt.axes([0.9, 0.35, 0.15, 0.3])

        self.slider_erg_center_ax.set_axis_off()
        self.slider_window_ax.set_axis_off()
        self.checkbox_time_integrated_ax.set_axis_off()
        self.radiobutton_active_select_ax.set_axis_off()

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
        # self.fit_clicks contains info on each energy selection for upcoming fit
        # (as determined by mouse clicks while space bar is being held down)
        self.fit_clicks: Dict[str, List] = self._reset_fit_clicks()

        # self.current_fits is dict (for each spectrum) of lists of dicts (one for each fit!) containing things
        # relevant to the fits currently on canvas, e.g. Line2D objects
        self.current_fits: Dict[int, List] = None  # is set in self.show!

        self.keys_currently_held_down = {}
        # \end GUI state variables.

        self.checkbox_time_integrated = CheckButtons(self.checkbox_time_integrated_ax, ['All\nevents'], actives=[True])
        self.checkbox_time_integrated.on_clicked(self._on_checked_time_integrated)

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
        self.slider_time_center = plt.Slider(self.slider_erg_center_ax, "Center time", valmin=self._events_times_range[0],
                                             valmax=self._events_times_range[1], valinit=self.window_width / 2,
                                             valstep=self.delta_t)

        self.current_fits = {i: [] for i in range(len(self))}

        if self.window_max is None:
            self.window_max = 0.75 * self._events_times_range[-1]

        self.slider_window = plt.Slider(self.slider_window_ax, 'Time\nwindow', valmin=0,
                                        valmax=self.window_max, valinit=self.window_width)

        radio_labels = [str(i + 1) for i in range(len(self))]
        self.radiobutton_active_select = RadioButtons(self.radiobutton_active_select_ax, labels=radio_labels)

        self.slider_time_center.on_changed(self._update_plot)
        self.slider_window.on_changed(self._update_plot)

        self.radiobutton_active_select.on_clicked(self._radio_button_on_clicked)
        self.radiobutton_active_select.set_active(0)

        self._on_checked_time_integrated('')


if __name__ == '__main__':
    times_ = None
    #  todo:
    #   Add an scale option ot add_spectra/add_list.
    #   Add titles to replace default radio button lables.

    from analysis import Shot
    list_shot1 = Shot(134).list
    list_shot2 = Shot(13).list
    InteractiveSpectra.print_click_coords = True
    interactive = InteractiveSpectra()

    interactive.add_spectra(list_shot1.energy_binned_times, list_shot1.erg_bins, list_shot1.effs)
    interactive.add_spectra(list_shot2.energy_binned_times, list_shot2.erg_bins, list_shot2.effs)

    # guass_fit(list_shot1.erg_bins, list_shot1.get_erg_spectrum(nominal_values=True), [535.34, 539.4], [False, False], 5)

    interactive.show()

    plt.show()
