import warnings

import numpy as np
from matplotlib import pyplot as plt
from JSB_tools import mpl_hist, convolve_gauss
from pathlib import Path
from functools import cached_property, cache
from typing import List, Dict
from mpl_interactions.widgets import RangeSlider
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from lmfit import Model, Parameters, Parameter
from lmfit.models import GaussianModel
from lmfit.model import ModelResult
from uncertainties import ufloat
from matplotlib.widgets import CheckButtons, Slider, Button, RadioButtons
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import logging

# Parameter.set()
def erg_cut(bins, erg_binned_times, effs, emin=None, emax=None):
    i0 = (-1 + np.searchsorted(bins, emin, side='right')) if emin is not None else 0
    i1 = (-1 + np.searchsorted(bins, emax, side='right')) if emax is not None else len(bins) - 2
    return erg_binned_times[i0: i1 + 1], bins[i0: i1 + 2], effs[i0: i1 + 1] if effs is not None else None


def guass_fit(bins, y, click_points, fixed_in_binQ: List[bool], fit_buffer_window=5, only_one_sigma=True) -> ModelResult:
    """

    Args:
        bins:
        y:
        click_points:

        fixed_in_binQ: List of bool. True means energy is constrainted to the energy bin according to the
            corresponding click_point. Must be same len as click_points.

        fit_buffer_window: Amount to extend the fit beyond the provided energy. Should be roughly 2-4 times the FWHM

        only_one_sigma: If True, all peaks share the same sigma for each fit.
    Returns:

    """
    def prepare_data():
        x_out = 0.5 * (bins[1:] + bins[:-1])
        I0, I1 = np.searchsorted(bins, [min(click_points) - fit_buffer_window, max(click_points) + fit_buffer_window],
                                 side='right') - 1
        i0s = np.searchsorted(bins, click_points, side='right') - 1 - I0

        # I1 += 1
        x_out = x_out[I0: I1 + 1]

        y_out = y[I0: I1 + 1]

        yerr = np.sqrt(y_out)

        weights_out = 1.0/np.where(yerr > 0, yerr, 1)

        init_is_out = []

        bins_ = bins[I0: I1 + 2]
        b_width_out = np.mean(bins_[1:] - bins_[:-1])

        vary_ranges_out = []

        init_xs_out = []

        for fixed, i0 in zip(fixed_in_binQ, i0s):
            if not fixed:
                vary_ranges_out.append(None)

                for s in [-1, 1]:
                    if (i0 == 0) or (i0 == len(x_out) - 1):
                        break
                    while y_out[i0 + s * 1] > y_out[i0]:
                        i0 += s
            else:
                vary_ranges_out.append([bins_[i0], bins_[i0 + 1]])

            init_xs_out.append(x_out[i0])
            init_is_out.append(i0)

        return (b_width_out, x_out), (y_out, weights_out), init_xs_out, init_is_out, vary_ranges_out

    def linear(x, slope, bg):
        """a line used for fitting background. """
        return slope * (x - x[len(x) // 2]) + bg

    click_points = [max(bins[0], min(bins[-1] - 0.001, x)) for x in click_points]  #

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
        print(i, y[i0-1], y[i0], y[i0 + 1])

        prefix = f'_{i}_'
        sub_model = GaussianModel(prefix=prefix)
        sub_params = sub_model.make_params()
        if model is None:
            model = Model(linear) + sub_model

            params = model.make_params()
            params['slope'].set(value=slope_guess, max=max_slope, min=-max_slope)
            params['bg'].set(value=bg_guess)

            if only_one_sigma:
                params[f'{prefix}sigma'].set(value=sigma_guess, max=min_sigma, min=max_sigma)
        else:
            model += sub_model

            params.update(sub_params)
            params[f'{prefix}sigma'].set(vary=False, expr=f'_0_sigma')

        if range_ is None:  # Fit energy is not fixed within selected bin.
            params[f'{prefix}center'].set(value=x0)
        else:
            params[f'{prefix}center'].set(value=x0, min=range_[0], max=range_[1])

        if not only_one_sigma:
            params[f'{prefix}sigma'].set(value=sigma_guess, max=min_sigma, min=max_sigma)

        _amp_guess = amp_guess(y[i0], params[f'{prefix}sigma'].value)

        params[f'{prefix}amplitude'].set(value=amp_guess(y[i0], params[f'{prefix}sigma'].value),
                                         min=_amp_guess/10)

    # for k, v in params.items():
    #     print(k, v)

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
    default_fit_window = 6  # in keV
    default_n_convolve_bins = 3  # amount to convolve signal when guessing guass fit centers.

    def add_spectra(self, energy_binned_times, energy_bins, effs=None, erg_min=100, erg_max=None):
        energy_binned_times, energy_bins, effs = \
            erg_cut(energy_bins, energy_binned_times, effs, emin=erg_min, emax=erg_max)

        self.energy_binned_timess.append(energy_binned_times)
        self.erg_binss.append(energy_bins)

        if self.times_range[0] is None:
            self.times_range[0] = min([ts[0] for ts in energy_binned_times if len(ts)])
        else:
            self.times_range[0] = min(self.times_range[0], min([ts[0] for ts in energy_binned_times if len(ts)]))

        if self.times_range[-1] is None:
            self.times_range[-1] = max([ts[-1] for ts in energy_binned_times if len(ts)])
        else:
            self.times_range[-1] = max(self.times_range[-1], max([ts[-1] for ts in energy_binned_times if len(ts)]))

        self.effss.append(effs)

        handle = self.ax.plot(energy_bins, np.ones(len(energy_bins)), ds='steps-post')[0]
        self.handles.append(handle)

    def _mouse_click(self, event):
        """"""
        print(event)
        x_now = event.xdata

        if x_now is None:
            return

        if ' ' in self.keys_currently_held_down:
            x_points_array = self.fit_clicks['x_points']  # Previous fit energies selected during current fit.
            fixed_bins_array = self.fit_clicks['fixed_binsQs']

            others = np.array(x_points_array)

            if len(x_points_array) and x_points_array[-1] == x_now and event.dblclick:
                # double click means user wants to perform fixed energy bin fit. Change entry from previous click.
                fixed_bins_array[-1] = True
                self.fit_clicks['axvlines'][-1].set_ls('-')
                self._draw()
                return  # dont add another fit energy

            if np.all(np.abs(x_now - others) > self._mean_bin_width(self._active_spec_index) ):
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
        return self.slider_time_center.val - self.slider_window.val / 2

    @property
    def max_time(self):
        return self.slider_time_center.val + self.slider_window.val / 2

    def _set_ylims(self, *args, **kwargs):
        xlim = self.ax.get_xlim()
        ylim = np.inf, -np.inf
        for l in self.ax.lines:
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

    def _get_y(self, index, tmin, tmax, append_last_value=False):
        v = [tmin, tmax]

        erg_bins = self.erg_binss[index]
        energy_binned_times = self.energy_binned_timess[index]
        effs = self.effss[index]

        if append_last_value:
            counts = np.zeros(len(erg_bins), dtype=float)
        else:
            counts = np.zeros(len(erg_bins) - 1, dtype=float)

        for index in range(len(energy_binned_times)):
            if len(energy_binned_times[index]) == 0:
                continue

            if self._full_spec_Q:
                counts[index] = len(energy_binned_times[index])
            else:
                i1, i2 = np.searchsorted(energy_binned_times[index], v)
                counts[index] = (i2 - i1)

            if effs is not None:
                counts[index] /= effs[index]

        if append_last_value:
            counts[-1] = counts[-2]

            return counts

    def _update_plot(self, *args):
        for index in range(len(self.energy_binned_timess)):
            counts = self._get_y(index, self.min_time, self.max_time, True)

            self.update_line(counts, index)

            line = self.handles[index]

            if index == self._active_spec_index:
                line.set_color("tab:blue")
                line.set_zorder(10)
            else:
                line.set_color("lightgrey")
                line.set_zorder(-10)

        self._set_ylims()
        self.fig.canvas.draw()

    @property
    def _full_spec_Q(self):
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
            slider.poly.set_visible(not self._full_spec_Q)  # grey-out slider
            slider.set_active(not self._full_spec_Q)  # Freeze slider

        self._update_plot()

    def _key_press(self, event):
        self.keys_currently_held_down[event.key] = True
        print(f'_key_press:\n\t{event.key}')
        print(f"\t{self.keys_currently_held_down}")

    def _key_release(self, event):
        print(f'_key_release:\n\t{event.key}')
        del self.keys_currently_held_down[event.key]
        print(f"\t{self.keys_currently_held_down}")
        if event.key == ' ':
            print(f"Doing fits for {self.fit_clicks}")
            # do fitting stuff
            i = self._active_spec_index
            fit_result: ModelResult = guass_fit(bins=self.erg_binss[i], y=self.handles[i].get_ydata(),
                                                click_points=self.fit_clicks['x_points'],
                                                fixed_in_binQ=self.fit_clicks['fixed_binsQs'],
                                                fit_buffer_window=self.fit_settings['fit_buffer_window'],
                                                only_one_sigma=self.fit_settings['only_one_sigma'])
            x = fit_result.userkws[fit_result.model.independent_vars[0]]
            y = fit_result.data
            b_width = fit_result.userkws['b_width']

            params = fit_result.params

            line = self.handles[i]

            fit_line1 = self.ax.plot(x, fit_result.eval(x=x, params=params), c='black', ls='-', lw=2)[0]
            fit_line2 = self.ax.plot(x, fit_result.eval(x=x, params=params), c=line.get_color(), ls='--', lw=2)[0]

            fit_info = {'visibles': [fit_line1, fit_line2]}

            for i in range(len(self.fit_clicks['x_points'])):
                prefix = f"_{i}_"
                fit_center = params[f'{prefix}center'].value
                xpos = fit_center + 0.1 * (x[-1] - x[0])

                text = self.ax.text(xpos, max(y),
                                    f"E = {get_param_ufloat(f'{prefix}center', params):.2f}\n"
                                    f"N = {get_param_ufloat(f'{prefix}amplitude', params) / b_width:.2e}",
                                    c=line.get_color())

                # vert_line = self.ax.axvline(fit_center, c='black', lw=1)

                fit_info['visibles'].append(text)
                # fit_info['visibles'].append(vert_line)

            self.current_fits.append(fit_info)
            self._reset_fit_clicks()

    def _reset_fit_clicks(self):
        self.fit_clicks = {'x_points': [], 'fixed_binsQs': [], 'axvlines': []}
        return self.fit_clicks

    def _clear_fits(self, event=None):
        for fit_info in self.current_fits:
            for thing in fit_info['visibles']:
                thing.set_visible(0)

        for thing in self.fit_clicks['axvlines']:
            thing.set_visible(0)

        self.current_fits = []
        # self.fit_clicks = {'x_points': [], 'fixed_binsQs': []}
        self.fig.canvas.draw()

    def __init__(self, init_window_width=10, window_max=None, delta_t=5):
        """

        Args:
            init_window_width: Initial time-integration window width
            window_max: Max width of time integration window possible to choose using the slider
            delta_t: time step for the slider determining the center time of the integration time range
        """
        self.fit_settings = {}
        self.set_fit_settings()

        self.times_range: List[float] = [None, None]  # global time min/max

        self.handles: List[Line2D] = []
        self.erg_binss = []
        self.energy_binned_timess = []
        self.effss = []

        fig, ax = plt.subplots(figsize=(15, 7))
        self.fig = fig
        self.ax: Axes = ax

        self.maxy = 1

        plt.subplots_adjust(bottom=0.2)

        self.window_max = window_max
        self.delta_t = delta_t

        self.slider_erg_center_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
        self.slider_window_ax = plt.axes([0.1, 0.01, 0.8, 0.05])
        self.checkbox_time_integrated_ax = plt.axes([0.91, 0.15, 0.12, 0.12])
        self.clear_button_ax = plt.axes([0.88, 0.88, 0.1, 0.1])
        self.radiobutton_active_select_ax = plt.axes([0.9, 0.35, 0.15, 0.3])

        self.slider_erg_center_ax.set_axis_off()
        self.slider_window_ax.set_axis_off()
        self.checkbox_time_integrated_ax.set_axis_off()
        self.radiobutton_active_select_ax.set_axis_off()

        self.window_width = init_window_width
        self.slider_pos = init_window_width / 2

        # below will be declared once the number of spectra is being simultaneously plotted is known
        self.slider_time_center: Slider = None  # cant define until range is
        self.slider_window: Slider = None
        self.radiobutton_active_select: RadioButtons = None

        self.fig.canvas.mpl_connect('button_press_event', self._mouse_click)
        self.ax.callbacks.connect('xlim_changed', self._set_ylims)

        # \begin GUI state variables.

        # self.fit_clicks contains info on each energy selection for upcoming fit
        # (as determined by mouse clicks while space bar is being held down)
        self.fit_clicks: Dict[str, List] = self._reset_fit_clicks()
        self.keys_currently_held_down = {}
        self.current_fits = []  # list of dicts containing things relevant to the current fits, e.g. Line2D objects

        # \end GUI state variables.

        self.checkbox_time_integrated = CheckButtons(self.checkbox_time_integrated_ax, ['Full\nspec'], actives=[True])
        self.checkbox_time_integrated.on_clicked(self._on_checked_time_integrated)

        self.clear_button = Button(self.clear_button_ax, 'Clear fits')
        self.clear_button.on_clicked(self._clear_fits)

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
        self.slider_time_center = plt.Slider(self.slider_erg_center_ax, "Center", valmin=self.times_range[0],
                                             valmax=self.times_range[1], valinit=self.window_width/2,
                                             valstep=self.delta_t)
        if self.window_max is None:
            self.window_max = 0.75 * self.times_range[-1]

        self.slider_window = plt.Slider(self.slider_window_ax, 'Width', valmin=0,
                                        valmax=self.window_max, valinit=self.window_width)

        radio_labels = [str(i + 1) for i in range(len(self))]
        self.radiobutton_active_select = RadioButtons(self.radiobutton_active_select_ax, labels=radio_labels)

        self.slider_time_center.on_changed(self._update_plot)
        self.slider_window.on_changed(self._update_plot)
        self.radiobutton_active_select.on_clicked(self._update_plot)

        self.slider_time_center.set_val(self.slider_time_center.val)

        self._on_checked_time_integrated('')


if __name__ == '__main__':
    #  todo: Make axvlines go away, make persistent fits between selecting active spectrum, make text readable for multi fits. 

    from analysis import Shot
    list_shot1 = Shot(134).list
    list_shot2 = Shot(13).list

    interactive = InteractiveSpectra()

    interactive.add_spectra(list_shot1.energy_binned_times, list_shot1.erg_bins, list_shot1.effs)
    interactive.add_spectra(list_shot2.energy_binned_times, list_shot2.erg_bins, list_shot2.effs)

    guass_fit(list_shot1.erg_bins, list_shot1.get_erg_spectrum(nominal_values=True), [535.34, 539.4], [False, False], 5)

    interactive.show()

    plt.show()
