from uncertainties import UFloat
from JSB_tools.spectra.time_depend import multi_guass_fit, GausFitResult
from matplotlib import pyplot as plt
from JSB_tools.hist import mpl_hist
import numpy as np


class InteractivePlot:
    color_cycle = ['blue', 'red', 'green', 'black', 'gray']

    def __init__(self, bins, y, fit_window=3, ax=None):
        self.bins = bins
        self.y = y

        self.click_points = []
        self.holding_space = False

        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = ax.figure
            self.ax = ax

        mpl_hist(bins, y, ax=self.ax)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.axvlines = []
        self.fit_plots = []
        self.texts = []

        self.fit_window = fit_window

        # self.fit()

    def on_click(self, event):
        print(event)
        if self.holding_space:

            self.click_points.append(event.xdata)
            self.axvlines.append(self.ax.axvline(event.xdata, ls='--', lw=1, c='black'))

    def on_key_press(self, event):
        if event.key == ' ':
            self.holding_space = True

    def on_key_release(self, event):
        if event.key == ' ':
            if self.holding_space:
                self.fit()
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
            t = self.ax.text(efit.n, y0 + dy, fr"x={efit:.2f}" "\n" fr"$\sigma={fit.sigmas(i):.2g}$" "\n" f"{A:.3e}", va='center')

            self.texts.append(t)

    def fit(self):
        fits = multi_guass_fit(self.bins, self.y, center_guesses=self.click_points, fit_buffer_window=self.fit_window)
        x = np.linspace(fits.fit_x[0], fits.fit_x[-1], 5000)
        y = fits.eval(x=x)

        plt, = self.ax.plot(x, y, c='tab:orange')
        self.fit_plots.append(plt)

        self.click_points = []

        self.label(fits)

        for t in self.axvlines:
            t.remove()

        self.axvlines = []

        self.fig.canvas.draw_idle()
