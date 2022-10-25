import warnings

import numpy as np
from matplotlib import pyplot as plt
from JSB_tools import mpl_hist, convolve_gauss
from pathlib import Path
from functools import cached_property
from typing import List
from mpl_interactions.widgets import RangeSlider
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from lmfit import Model
from uncertainties import ufloat
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons, Slider


def erg_cut(bins, erg_binned_times, effs, emin=None, emax=None):
    i0 = (-1 + np.searchsorted(bins, emin, side='right')) if emin is not None else 0
    i1 = (-1 + np.searchsorted(bins, emax, side='right')) if emax is not None else len(bins) - 2
    return erg_binned_times[i0: i1 + 1], bins[i0: i1 + 2], effs[i0: i1 + 1] if effs is not None else None

def guass_fit():

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

    def onclick(self, event):
        # print(('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata), 'selecting_peaks: ', self.selecting_peaks)
        if event.xdata is None:
            return

        if not self.selecting_peaks:  # space bar is not pressed.
            return

        def get(bins, y):
            x = 0.5 * (bins[1:] + bins[:-1])
            dl = InteractiveSpectra.default_fit_window/2
            i0 = np.searchsorted(x, event.xdata)

            ilow, ihigh = np.searchsorted(x, [x[i0] - dl, x[i0] + dl])
            i0 = i0 - ilow
            x = x[ilow: ihigh]
            y = y[ilow: ihigh]
            bwidth = np.average(x[1:] - x[:-1])
            yerr = np.sqrt(y)

            weights = 1.0/np.where(yerr > 0, yerr, 1)

            for s in [-1, 1]:
                if (i0 == 0) or (i0 == len(x) - 1):
                    break
                while y[i0 + s * 1] > y[i0]:
                    i0 += s
                print(f"Global max at {x[i0]}")

            return i0, bwidth, x, y, weights

        def gaussian(x, amp, cen, wid):
            """1-d gaussian: gaussian(x, amp, cen, wid)"""
            return (amp / (np.sqrt(2 * np.pi) * wid)) * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

        def linear(x, slope, bg):
            """a line"""
            return slope * (x - x[len(x)//2]) + bg

        for bins, line in zip(self.erg_binss, self.handles):
            i0, bwidth, x, y, weights = get(bins, line.get_ydata())

            t0 = self.slider_erg_center.val

            if (x[i0], t0) in [(infos['x0'], infos['t0']) for infos in self.current_fits]:
                warnings.warn(f"Fit at x = {x[i0]} and t = {t0} already plotted. No fit performed.")
                continue

            mod = Model(gaussian) + Model(linear)

            max_slope = (max(y) - min(y)) / (x[-1] - x[0])

            params = mod.make_params()
            params['bg'].set(value=min([np.mean([y[i-1], y[i], y[i + 1]]) for i in range(1, len(y) - 1)]))
            params['amp'].set(value=(y[i0] - min(y)) * np.sqrt(2*np.pi))
            params['cen'].set(value=x[i0])
            params['wid'].set(value=3)
            params['slope'].set(value=0, min=-max_slope, max=max_slope)
            print("max_slope: ", max_slope)

            fit = mod.fit(data=y, params=params, x=x, weights=weights)
            params = fit.params

            def get_param_ufloat(name):
                err = params[name].stderr
                if err is None:
                    err = np.nan
                return ufloat(params[name].value, err)

            xpos = params['cen'].value + 0.1 * (x[-1] - x[0])

            text = self.ax.text(xpos, max(y), f"E = {get_param_ufloat('cen'):.2f}\nN = {get_param_ufloat('amp')/bwidth:.2e}",
                                c=line.get_color())

            fit_line1 = self.ax.plot(x, fit.eval(x=x, params=params), c='black', ls='-', lw=2)[0]
            fit_line2 = self.ax.plot(x, fit.eval(x=x, params=params), c=line.get_color(), ls='--', lw=2)[0]
            vert_line = self.ax.axvline(params['cen'].value)

            fit_info = {'visibles': [fit_line1, fit_line2, vert_line, text], 'x0': x[i0], 't0': t0}
            self.current_fits.append(fit_info)

            print(fit.fit_report())

    @staticmethod
    def update_line(y, line):
        """

        Args:
            y: Hist values with y[-1] appended onto end.
            line:

        Returns:

        """
        line._yorig = y

        line.stale = True
        line._invalidy = True

    @property
    def min_time(self):
        return self.slider_erg_center.val - self.slider_window.val/2

    @property
    def max_time(self):
        return self.slider_erg_center.val + self.slider_window.val / 2

    def update_plot(self):
        # if self.full_spec_Q:
        #     v = self.global_time_range
        # else:
        v = np.array([self.min_time, self.max_time])

        print(f"update_plot called with self.full_spec_Q = {self.full_spec_Q}")
        ymax = 0

        for (erg_bins, handle, energy_binned_times, effs) in zip(self.erg_binss, self.handles,
                                                                 self.energy_binned_timess, self.effss):
            counts = np.zeros(len(erg_bins))

            for index in range(len(energy_binned_times)):
                if len(energy_binned_times[index]) == 0:
                    continue

                if self.full_spec_Q:
                    counts[index] = len(energy_binned_times[index])
                else:
                    i1, i2 = np.searchsorted(energy_binned_times[index], v)
                    counts[index] = (i2 - i1)

                if effs is not None:
                    counts[index] /= effs[index]

            counts[-1] = counts[-2]

            if max(counts) > ymax:
                ymax = max(counts)

            self.update_line(counts, handle)

        self.ax.set_ylim(0, 1.15 * ymax)
        self.fig.canvas.draw()

    @cached_property
    def global_time_range(self):  # return lowest and highest times out of all spectra
        min_t = max_t = None

        for erg_binned_times in self.energy_binned_timess:
            for l in erg_binned_times:

                if len(l):
                    _max = max(l)
                    _min = min(l)
                else:
                    continue

                if min_t is None:
                    max_t = _max
                    min_t = _min
                    continue

                if _max > max_t:
                    max_t = _max

                if _min < min_t:
                    min_t = _min
        return min_t, max_t

    @property
    def full_spec_Q(self):
        return self.full_spec_checkbox.lines[0][0].get_visible()

    def on_checked_erg_spec(self, label):
        # see self.full_spec_Q, which is automatically changed by button click.
        self.slider_window.poly.set_visible(not self.full_spec_Q)
        for slider in [self.slider_erg_center, self.slider_window]:
            slider.poly.set_visible(not self.full_spec_Q)
            slider.set_active(not self.full_spec_Q)

        self.update_plot()

    def __init__(self, init_window_width=10, window_max=100, delta_t=5):
        self.times_range: List[float] = [None, None]

        self.handles = []
        self.erg_binss = []
        self.energy_binned_timess = []
        self.effss = []

        self.fig, self.ax = plt.subplots()

        self.maxy = 1

        plt.subplots_adjust(bottom=0.2)

        self.window_max = window_max
        self.delta_t = delta_t

        self.slider_erg_center_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
        self.slider_window_ax = plt.axes([0.1, 0.01, 0.8, 0.05])
        self.checkbox_ax = plt.axes([0.91, 0.15, 0.12, 0.12])
        self.clear_button_ax = plt.axes([0.88, 0.88, 0.1, 0.1])

        self.slider_erg_center_ax.set_axis_off()
        self.slider_window_ax.set_axis_off()
        self.checkbox_ax.set_axis_off()

        self.window_width = init_window_width
        self.slider_pos = init_window_width / 2

        self.slider_erg_center: plt.Slider = None
        self.slider_window: plt.Slider = None

        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # GUI state variables.
        self.selecting_peaks = False
        # self.full_spec_Q = False

        self.current_fits = []

        def g_press(event):
            if event.key == ' ':
                self.selecting_peaks = True

        def g_release(event):
            if event.key == ' ':
                self.selecting_peaks = False

        def clear_fits(event):
            for fit_info in self.current_fits:
                for thing in fit_info['visibles']:
                    thing.set_visible(0)

            self.current_fits = []
            self.fig.canvas.draw()

        self.full_spec_checkbox = CheckButtons(self.checkbox_ax, ['Full\nspec'], actives=[True])
        self.full_spec_checkbox.on_clicked(self.on_checked_erg_spec)

        self.clear_button = Button(self.clear_button_ax, 'Clear')
        self.clear_button.on_clicked(clear_fits)

        self.fig.canvas.mpl_connect('key_press_event', g_press)
        self.fig.canvas.mpl_connect('key_release_event', g_release)

    def _slider_erg_center(self, val):
        self.update_plot()

    def _slider_window(self, val):
        self.update_plot()
        pass

    def show(self):
        self.slider_erg_center = plt.Slider(self.slider_erg_center_ax, "Center", valmin=self.times_range[0],
                                            valmax=self.times_range[1], valinit=self.window_width/2,
                                            valstep=self.delta_t)
        self.slider_window = plt.Slider(self.slider_window_ax, 'Width', valmin=0,
                                        valmax=self.window_max, valinit=self.window_width)

        self.slider_erg_center.on_changed(self._slider_erg_center)
        self.slider_window.on_changed(self._slider_window)

        self.slider_erg_center.set_val(self.slider_erg_center.val)

        self.on_checked_erg_spec('')



if __name__ == '__main__':
    # line = plt.plot([0, 1, 2, 3], [1,2,3, 3], ds='steps-post')
    # print()
    # from JSB_tools.spectra import ListSpectra
    from analysis import Shot
    l = Shot(134).list
    i = InteractiveSpectra()
    i.add_spectra(l.energy_binned_times, l.erg_bins, l.effs)
    i.show()

    plt.show()
