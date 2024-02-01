from __future__ import annotations
import warnings
from uncertainties import UFloat
from JSB_tools.spectra.time_depend import multi_guass_fit, GausFitResult
from matplotlib import pyplot as plt
from JSB_tools.hist import mpl_hist
import numpy as np
import matplotlib
from typing import List
from matplotlib.widgets import TextBox, CheckButtons, RadioButtons
matplotlib.use('Qt5agg')


class Click:
    def __init__(self, event, ax):
        self.event = event
        self.axvline = ax.axvline(event.xdata, ls='-' if self.dblclick else '--', lw=0.7, c='black')

    @property
    def dblclick(self):
        return self.event.dblclick

    def remove(self):
        self.axvline.remove()

    @property
    def x(self):
        return self.event.xdata

    def __repr__(self):
        return f'{"" if not self.dblclick else "double "}click  x={self.x:.4g}'


class InteractivePlot:
    color_cycle = ['blue', 'red', 'green', 'black', 'gray']

    def __init__(self, bins, y, fit_window=3, ax=None, button_ax=None, make_density=False):
        self.bins = bins
        if make_density:
            bw = bins[1:] - bins[:-1]
            self.y = y / bw
        else:
            self.y = y

        self.fit_clicks: List[Click] = []
        self.fit_visuals = []

        self.holding_space = False

        if ax is None:
            self.fig, (self.ax, self.button_ax) = plt.subplots(1, 2, width_ratios=[10, 1], figsize=(14, 8))
        else:
            self.fig = ax.figure
            self.ax = ax

        self.button_ax.set_axis_off()

        self.sigma_textbox = self.window_textbox = self.sigma_share_check_button = self.bg_radio_button = None

        self.setup_buttons()

        mpl_hist(bins, self.y, ax=self.ax)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.fit_window = fit_window

    def setup_buttons(self):
        sigma_t_ax = self.fig.add_axes([0.90, 0.1, 0.03, 0.03])
        window_t_ax = self.fig.add_axes([0.90, 0.15, 0.03, 0.03])
        bg_check_ax = self.fig.add_axes([0.90, 0.3, 0.07, 0.2])
        sigma_check_ax = self.fig.add_axes([0.9, 0.05, 0.055, 0.04])

        self.sigma_textbox = TextBox(sigma_t_ax, r'$\sigma_{0}$')
        self.window_textbox = TextBox(window_t_ax, 'Fit\nwindow', initial='3')
        self.sigma_share_check_button = CheckButtons(sigma_check_ax, ['Share $\sigma$'])

        self.bg_radio_button = RadioButtons(bg_check_ax, ['Const', 'Lin', 'None'], )
        self.fig.text(0.93, 0.515, 'Background\nterm', ha='center')

    def update(self):
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if self.holding_space:
            click = Click(event, self.ax)

            if len(self.fit_clicks) and click.dblclick:
                self.fit_clicks[-1].remove()
                del self.fit_clicks[-1]

            print(click)
            self.fit_clicks.append(click)

            self.update()

    def clear(self):
        for t in (self.fit_visuals + self.fit_clicks):
            t.remove()

        self.fit_visuals = []

        self.update()

    def on_key_press(self, event):
        if event.key == ' ':
            self.holding_space = True

        if event.key == 'escape':
            self.clear()

    def on_key_release(self, event):
        if event.key == ' ':
            if self.holding_space:
                try:
                    self.fit()
                except Exception as e:
                    warnings.warn(f"Exception: {e}")
                    raise

                self.fit_clicks = []
                self.holding_space = False

    def label(self, fit: GausFitResult):
        for i in range(len(fit)):
            efit = fit.centers(i)
            A = fit.amplitudes(i)
            i0 = np.searchsorted(self.bins, efit.n, side='right') - 1

            y0 = self.y[i0]
            if isinstance(y0, UFloat):
                y0 = y0.n

            dy = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])/12
            t = self.ax.text(efit.n, y0 + dy, fr"$\mu$={efit:.3g}" "\n" fr"$\sigma={fit.sigmas(i):.3g}$" "\n" f"A={A:.3g}", va='center',
                             fontsize=6.5)

            self.fit_visuals.append(t)

    @property
    def fit_click_centers(self):
        return [c.x for c in self.fit_clicks]

    @property
    def fix_in_binQ(self):
        return [c.dblclick for c in self.fit_clicks]

    def fit(self):
        if not len(self.fit_clicks):
            return

        share_sigma = self.sigma_share_check_button.get_status()[0]
        if self.sigma_textbox.text.strip():
            sigma_guess = float(self.sigma_textbox.text)
        else:
            sigma_guess = None

        if self.window_textbox.text.strip():
            fit_window = float(self.window_textbox.text)
        else:
            fit_window = 3

        bg = self.bg_radio_button.value_selected.lower()
        fix_bg = None
        if bg == 'none':
            fix_bg = 0
            bg = 'const'

        fits = multi_guass_fit(self.bins, self.y, fixed_in_binQ=self.fix_in_binQ,
                               center_guesses=self.fit_click_centers,
                               share_sigma=share_sigma,
                               sigma_guesses=sigma_guess,
                               fit_buffer_window=fit_window, fix_bg=fix_bg, bg=bg)

        x = np.linspace(fits.fit_x[0], fits.fit_x[-1], 5000)
        y = fits.eval(x=x)

        plt, = self.ax.plot(x, y, c='tab:orange')
        self.fit_visuals.append(plt)

        self.label(fits)

        for i, click in enumerate(self.fit_clicks):
            click.remove()

            _x = fits.centers(i).nominal_value
            axvline = self.ax.axvline(_x,  ls='-' if click.dblclick else '--', lw=0.7, c='black')
            self.fit_visuals.append(axvline)

        self.update()


if __name__ == '__main__':
    s = 100000

    mus = [1, 3, 5.75, 7]
    sigmas = [0.1, 0.3,  0.45, 0.4]
    As = s * np.array([3, 5, 2.6, 7])

    data = []
    bins = np.linspace(-5, max(mus) + 5, 2000)

    for x, sig, a in zip(mus, sigmas, As):
        data.extend(np.random.normal(loc=x, scale=sig, size=int(a)))

    y, _ = np.histogram(data, bins=bins)

    i = InteractivePlot(bins, y, make_density=True)

    # mpl_hist(bins, y)

    plt.show()


