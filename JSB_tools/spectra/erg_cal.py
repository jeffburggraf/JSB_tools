import re
import warnings
import numpy as np
from JSB_tools.spe_reader import SPEFile, InteractivePlot
from JSB_tools.nuke_data_tools import Nuclide
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from typing import Union, List
from matplotlib.widgets import TextBox
from uncertainties import UFloat
from lmfit.models import PolynomialModel
from uncertainties import ufloat

matplotlib.use('Qt5agg')
cwd = Path(__file__).parent


check_sources = ['Ba133', 'Co60', 'Co57', 'Eu152', 'Cs137', 'Na22', 'Sr90', 'Zn65']


def lin_fit(chs, ergs, errs, deg=1):
    errs = np.array(errs)
    _m = np.mean(errs[np.where(~np.isnan(errs))])
    errs[np.where(np.isnan(errs))] = _m
    weights = 1.0/errs

    model = PolynomialModel(degree=deg)
    params = model.guess(x=chs,data=ergs)

    fit = model.fit(x=chs, data=ergs, params=params, weights=weights)

    return fit, [ufloat(fit.params[f'c{i}'].value, fit.params[f'c{i}'].stderr) for i in range(deg + 1)]


def fwhm_lin_fit(ergs, fwhms, errs, deg=1):
    errs = np.array(errs)
    _m = np.mean(errs[np.where(~np.isnan(errs))])
    errs[np.where(np.isnan(errs))] = _m
    weights = 1.0 / errs

    model = PolynomialModel(degree=deg)
    params = model.guess(x=ergs, data=fwhms)

    fit = model.fit(x=ergs, data=fwhms, params=params, weights=weights)

    return fit, [ufloat(fit.params[f'c{i}'].value, fit.params[f'c{i}'].stderr) for i in range(deg + 1)]


class PltGammaLine:
    def __init__(self, erg, intensity, nuclide=None):
        self.erg = erg
        self.intensity = intensity
        self.nuclide_name = nuclide.strip() if nuclide is not None else nuclide

        self.axvline: Union[None, Line2D] = None

        self.annotation_text = None

    def __repr__(self):
        return f'{self.nuclide_name} @ {self.erg} keV'


GAMMA_LINES = {'background': [],
               'check_sources': []}

with open(cwd / 'Background.csv') as f:
    [f.readline() for i in range(3)]
    while line := f.readline():
        erg, intensity, nuclide, _ = line.split(',')
        erg = float(erg)
        try:
            intensity = float(intensity)
        except ValueError:
            intensity = 1

        gline = PltGammaLine(erg, intensity, nuclide)
        GAMMA_LINES['background'].append(gline)


for nuclide_name in check_sources:
        nuclide = Nuclide(nuclide_name)
        for gline in nuclide.get_gammas():
            gline = PltGammaLine(gline.erg.n, gline.intensity.n, nuclide.name)
            GAMMA_LINES['check_sources'].append(gline)


def get_gamma_lines(check_sourcesQ=False, backgroundQ=False, nuclides=None):
    gamma_lines: List[PltGammaLine] = []

    if backgroundQ:
        gamma_lines.extend(GAMMA_LINES['background'])

    if check_sourcesQ:
        gamma_lines.extend(GAMMA_LINES['check_sources'])

    gamma_lines = list(sorted(gamma_lines, key=lambda x: x.erg))

    if nuclide is None:
        pass  # todo
    return gamma_lines


def print_location(event):
    print(np.array([event.x, event.y]) / erg.fig.bbox.size)
    # print(event.x, event.y)
    # print(event.xdata, event.ydata)
    print(dir(event))


def plt_init(spe: SPEFile, ax_erg, ax_shape):
    ax_erg.plot(spe.channels, spe.energies, color='red', lw=1)
    ax_shape.plot(spe.energies, spe.get_fwhm(spe.energies), color='red', lw=1)

class ErgCal(InteractivePlot):
    """
    todo: button to change degree between 1 and 2.
        Change current spectrum to new energy cal
        Also fit the FWHMs

    """
    def __init__(self, spe_path: Union[Path, str]):
        self.spe: SPEFile = SPEFile(spe_path)
        self.erg_calibration = self.spe.erg_calibration

        self.data_point_fig, self.data_point_axs = plt.subplots(1, 2, figsize=(12, 8))
        plt_init(self.spe, *self.data_point_axs)

        self.data_point_axs[0].set_title("Energy fit")
        self.data_point_axs[1].set_title("Shape fit")

        self.data_point_axs[0].set_xlabel("Ch.")
        self.data_point_axs[0].set_ylabel("Energy")

        self.data_point_axs[1].set_xlabel("Energy")
        self.data_point_axs[1].set_ylabel("FWHM")

        self.data_point_axs = {'erg_fit': self.data_point_axs[0], 'shape_fit': self.data_point_axs[1]}
        self.data_point_lines = {'erg_fit': None, 'shape_fit': None}

        bins = self.spe.erg_bins
        counts = self.spe.get_counts(make_density=True, nominal_values=True)
        super().__init__(bins, counts)

        self.gamma_lines: List[PltGammaLine] = get_gamma_lines(backgroundQ=True, check_sourcesQ=True)

        intensity_txt_box_ax = self.fig.add_axes([0.8, 0.88, 0.15, 0.04])

        self._mouse_over_gline: Union[PltGammaLine, None] = None  # PltGammaLine that the mouse is currently over
        self._selected_gline: Union[PltGammaLine, None] = None

        self.intensity_textbox = TextBox(intensity_txt_box_ax, 'min.\nintensity(%)', initial='10%')
        self.intensity_textbox.on_submit(self.plot_glines)
        self.plot_glines()

        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

        self.fit_chs = []
        self.fit_ergs = []
        self.fit_erg_errs = []

        self.fit_fwhms = []
        self.fit_fwhms_errs = []

        self.fig.suptitle(self.spe.path.name)

    def erg_2_channel(self, erg):
        if len(self.spe.erg_calibration) == 2:
            a, b = self.spe.erg_calibration
            c = 0
        else:
            a, b, c = self.spe.erg_calibration

        if c != 0:
            a, b, c = self.spe.erg_calibration
            sqrt_term = np.sqrt(b**2 - 4*a*c + 4*c*erg)
            sols = 2/c * np.array([sqrt_term - b, sqrt_term - b])
            return max(sols)
        else:
            return (erg - a)/b

    def clear_fit(self):
        self.fit_fwhms = []
        self.fit_fwhms_errs = []
        self.fit_chs = []
        self.fit_ergs = []
        self.fit_erg_errs = []

        for _, things in self.data_point_lines.items():
            if things is not None:
                for thing in things:
                        thing.set_visible(0)

        self.data_point_fig.canvas.draw_idle()

        print("Cleared fit!")

    def on_key_press(self, event):
        super(ErgCal, self).on_key_press(event)

        if event.key == 'escape':
            self.clear_fit()

    def on_key_release(self, event):
        fit = super(ErgCal, self).on_key_release(event)
        if fit is not None and self.selected_gline is not None:

            i0 = np.argmin([abs(self.selected_gline.erg - fit.centers(i)) for i in range(len(fit))])

            if abs(fit.centers(i0).n - self.selected_gline.erg) > 15:
                warnings.warn("Selected gamma line and fitted center are not within acceptable range!")
                return

            fit_erg = fit.centers(i0).n
            fit_erg_error = fit.centers(i0).std_dev

            fit_fwhm = fit.fwhms(i0).n
            fit_ch = self.erg_2_channel(fit_erg)

            # fit_fwhm = self.erg_2_channel(fit_erg + fit_fwhm/2) - self.erg_2_channel(fit_erg - fit_fwhm/2)
            fit_fwhm_err = fit_fwhm * fit.fwhms(i0).std_dev/fit.fwhms(i0).n

            self.fit_chs.append(fit_ch)
            self.fit_ergs.append(self.selected_gline.erg)

            self.fit_fwhms.append(fit_fwhm)

            self.fit_erg_errs.append(fit_erg_error)
            self.fit_fwhms_errs.append(fit_fwhm_err)

            print(f"Added fit @ {self.selected_gline.erg:.1f} keV\n")
            deg = 1  # todo: make button

            if len(self.fit_chs) > 1:
                erg_fit_result, new_erg_cal = lin_fit(self.fit_chs, self.fit_ergs, self.fit_erg_errs, deg=deg)
                self.erg_calibration = new_erg_cal

                shape_fit_result, new_shape_cal = fwhm_lin_fit(self.fit_ergs, self.fit_fwhms, self.fit_fwhms_errs, deg=deg)

                self._plot_fit(erg_fit_result, shape_fit_result)

                if len(new_erg_cal) == 2:
                    new_erg_cal = self.erg_calibration + [0.0]
                    new_shape_cal = new_shape_cal + [0.0]

                print("Current energy cal.:")
                print('\t' + ' '.join([f'{x.n if isinstance(x, UFloat) else x:.5e}' for x in new_erg_cal]))
                print("Current shape cal.:")
                print('\t' + ' '.join([f'{x.n if isinstance(x, UFloat) else x:.5e}' for x in new_shape_cal]))

    def _plot_fit(self, erg_fit, shape_fit):
        def plot(x, y, yerr, fit, name):
            if self.data_point_lines[name] is not None:
                first = False
                for thing in self.data_point_lines[name]:
                    thing.set_visible(0)
            else:
                first = True

            self.data_point_lines[name] = []

            ax = self.data_point_axs[name]
            line, _, (errbars,) = ax.errorbar(x, y, yerr=yerr, label='Data', color='black', marker='o', ls='None')

            self.data_point_lines[name].extend([line, errbars])

            _x = np.linspace(min(x), max(x), 1000)
            _y = fit.eval(x=_x)

            l, = ax.plot(_x, _y, label='fit', color='black', ls='--')
            self.data_point_lines[name].append(l)

            if first:
                ax.legend()

        chs = erg_fit.userkws['x']
        energies = erg_fit.data
        yerr = 1/erg_fit.weights

        plot(chs, energies, yerr, erg_fit, 'erg_fit')

        energies = shape_fit.userkws['x']
        fwhms = shape_fit.data
        yerr = 1/shape_fit.weights

        plot(energies, fwhms, yerr, shape_fit, 'shape_fit')

        self.data_point_fig.canvas.draw_idle()

    @property
    def erg_bins(self):
        """
        Erg bins according to current calibration.
        Returns:

        """
        chs = np.arange(len(self.spe.counts) + 1) - 0.5
        return np.sum([coeff * chs ** i for i, coeff in enumerate(self.erg_calibration)], axis=0)

    @property
    def energies(self):
        """
        Erg centers according to current calibration.
        Returns:

        """
        chs = np.arange(len(self.spe.counts))
        return np.sum([coeff * chs ** i for i, coeff in enumerate(self.erg_calibration)], axis=0)

    def on_click(self, event):
        """
        If inside a self.Gline,
        Args:
            event:

        Returns:

        """
        if self.holding_space:
            super(ErgCal, self).on_click(event)
        else:
            if self.mouse_over_gline and self.mouse_over_gline.axvline.contains(event): # set currently selected Gline to current line if mouse is inside one
                self.selected_gline = self.mouse_over_gline

    def annotate(self):
        x = self.mouse_over_gline.erg
        y = self.ax.get_ylim()[-1]

        self.mouse_over_gline.annotation_text = self.ax.annotate(self.mouse_over_gline.nuclide_name, (x, y))

        self.update()

    def clear_annotate(self):
        if self.mouse_over_gline is not None and self.mouse_over_gline.annotation_text is not None:
            self.mouse_over_gline.annotation_text.set_visible(False)
            self.mouse_over_gline.annotation_text = None
            self.update()

    @property
    def selected_gline(self):
        return self._selected_gline

    @selected_gline.setter
    def selected_gline(self, val: Union[None, PltGammaLine]):
        if self._selected_gline is not None:  # set old selected line back to default
            self._selected_gline.axvline.set_color('black')
            self._selected_gline.axvline.set_linestyle('--')

        else:  # None to deselect
            pass

        self._selected_gline = val

        if val is not None:
            self._selected_gline.axvline.set_color('red')
            self._selected_gline.axvline.set_linestyle('-')

    @property
    def mouse_over_gline(self):
        return self._mouse_over_gline

    @mouse_over_gline.setter
    def mouse_over_gline(self, val: Union[None, PltGammaLine]):
        self.clear_annotate()
        self._mouse_over_gline = val

        if val is not None:
            self.annotate()
        pass

    def hover(self, event):
        if self.mouse_over_gline is not None:
            if not self.mouse_over_gline.axvline.contains(event)[0]:
                self.mouse_over_gline = None
                return

        for gline in self.gamma_lines:
            if not gline.axvline.get_visible():
                continue

            if gline.axvline.contains(event)[0]:
                self.mouse_over_gline = gline  # does annotation
                break
        else:
            self.mouse_over_gline = None  # Removes annotation
            return

    def plot_glines(self, event=None):
        if self.gamma_lines[0].axvline is None:  # set axvlines for each gamma line
            for g in self.gamma_lines:
                g.axvline = self.ax.axvline(g.erg, c='black', ls='--')

        for g in self.gamma_lines:
            if g.intensity > self.min_intensity:
                g.axvline.set_visible(True)
            else:
                g.axvline.set_visible(False)

        self.update()

    @property
    def min_intensity(self):
        m = re.match('([0-9.Ee+-]+)%?', self.intensity_textbox.text)
        assert m
        return float(m.groups()[0])/100


if __name__ == '__main__':
    m = PolynomialModel(degree=2)
    x = np.linspace(0.1, 10, 10)
    y = x + 1
    params = m.guess(x=x, data=y)
    fit = m.fit(x=x, data=y, params=params, weights=y)
    fig, axs = plt.subplots()
    a = axs.errorbar(x, y, yerr=y)
    print()

    erg = ErgCal('/detectorModels/GRETA0/cal/Co60_50cm.Spe')

    #  4.16926e+00 5.10001e-04
    plt.show()