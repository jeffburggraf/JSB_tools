import pickle
import re
import warnings
import numpy as np
from JSB_tools.spe_reader import SPEFile, InteractivePlot
from JSB_tools.nuke_data_tools import Nuclide
from JSB_tools import BijectiveMap
from pathlib import Path
from matplotlib import transforms
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from typing import Union, List, Set
from matplotlib.widgets import TextBox, CheckButtons, RadioButtons
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
    ALL_LINES = {}

    @staticmethod
    def load_lines():
        PltGammaLine.ALL_LINES['BG'] = []
        with open(cwd / 'Background.csv') as f:
            [f.readline() for i in range(3)]
            i=0
            while line := f.readline():
                erg, intensity, nuclide, _ = line.split(',')
                erg = float(erg)
                try:
                    intensity = float(intensity)
                except ValueError:
                    intensity = 1.

                gline = PltGammaLine(erg, intensity, nuclide, source="BG", color="green")
                PltGammaLine.ALL_LINES['BG'].append(gline)
                i += 1

        PltGammaLine.ALL_LINES['Check sources'] = []
        for nuclide_name in check_sources:
            nuclide = Nuclide(nuclide_name)
            for gline in nuclide.get_gammas():
                gline = PltGammaLine(gline.erg.n, gline.intensity.n, nuclide.name)
                PltGammaLine.ALL_LINES['Check sources'].append(gline)

        with open(cwd/'gammas') as f:
            f.readline()
            for line in f.readlines():
                source_name, decaying_nucleus, gamma_erg, rel_rate, gammma_intensity = line.split(';')
                source_name = source_name.strip()
                gammma_intensity = float(gammma_intensity)
                gamma_erg = float(gamma_erg)
                gline = PltGammaLine(gamma_erg, gammma_intensity, decaying_nucleus, source=source_name)

                if source_name not in PltGammaLine.ALL_LINES:
                    PltGammaLine.ALL_LINES[source_name] = [gline]
                else:
                    PltGammaLine.ALL_LINES[source_name].append(gline)

        with open(cwd/'neutron_activation_lines.pickle', 'rb') as f:
            lines = pickle.load(f)
            for symbol, d in lines.items():
                source_name = f'{symbol}(n,*g)'
                PltGammaLine.ALL_LINES[source_name] = []

                for A, lines in d.items():
                    for line in lines:
                        gamma_erg = line['energy']
                        gammma_intensity = line['intensity'] * line['abundance']
                        decaying_nucleus = line['decaying_daughter']
                        # source_name = line['target']
                        gline = PltGammaLine(gamma_erg, gammma_intensity, decaying_nucleus, source=source_name)
                        PltGammaLine.ALL_LINES[source_name].append(gline)

    def plot(self, ax):
        self.axvline = ax.axvline(self.erg, c=self.color, ls='--')
        return self.axvline

    def __hash__(self):
        return hash((self.erg, self.nuclide_name))

    def __eq__(self, other):
        if self.__hash__() != other.__hash__:
            return False
        # if other.erg != self.erg or self.nuclide_name != other.nuclide_name:
        #     return False
        return True

    def __init__(self, erg, intensity, nuclide=None, source=None, color='black'):
        self.erg = erg
        assert isinstance(intensity, float)

        self.intensity = intensity
        self.nuclide_name = nuclide.strip() if nuclide is not None else nuclide

        self.axvline: Union[None, Line2D] = None

        self.source = source.strip() if source is not None else None

        self.annotation_text = None
        self.color = color

    def get_label(self):
        out = f"{self.nuclide_name}"
        if self.source is not None and self.source != self.nuclide_name:
            out += f" ({self.source})"

        if not np.isnan(self.intensity):
            intensity = 100 * self.intensity
            if intensity < 1:
                intensity = f'{intensity:.2e}'
            else:
                intensity = f'{int(intensity)}'

            out += f'\n{intensity}%'

        return out

    def __repr__(self):
        return f'{self.nuclide_name} @ {self.erg} keV'


PltGammaLine.load_lines()



def print_location(event):
    print(np.array([event.x, event.y]) / erg.fig.bbox.size)
    # print(event.x, event.y)
    # print(event.xdata, event.ydata)
    print(dir(event))


def plt_init(spe: SPEFile, ax_erg, ax_shape,):
    ax_erg.plot(spe.channels, spe.energies, color='red', lw=1, label="Prior fit")
    ax_shape.plot(spe.energies, spe.get_fwhm(spe.energies), color='red', lw=1, label="Prior fit")
    ax_shape.legend()
    ax_erg.legend()


BG_auto_fit = [
    ("Tl-208", 2614.453),
    ("Tl-208", 583.191),
    ("S.E.", 2103.5),
    ("Bi-214", 1764.494),
    ("Bi-214", 1120.287),
    ("Bi-214", 609.312),
    ("K-40", 1460.83),
    ("Ac-228", 911.204),
    ("Pb-214", 351.932),
    ("Pb-214", 295.224),
               ]


class MyMouseEvent:
    def __init__(self, xdata, ax):
        self.xdata = xdata
        self.ax = ax
        self.dblclick = False


class MyKeyEvent:
    def __init__(self, key):
        self.key = key


class ErgCal(InteractivePlot):
    """
    todo: button to change degree between 1 and 2.
        Change current spectrum to new energy cal
        Also fit the FWHMs

    """
    def __init__(self, spe_path: Union[Path, str]):
        self.spe: SPEFile = SPEFile(spe_path)
        self.erg_calibration = self.spe.erg_calibration

        self.data_point_fig, self.data_point_axs = plt.subplots(2, 2, figsize=(12, 8))
        plt_init(self.spe, self.data_point_axs[0, 0], self.data_point_axs[0, 1])
        self.data_point_axs[1, 0].set_xlabel("Energy [keV]")
        self.data_point_axs[1, 0].set_ylabel("Fit residue [keV]")

        erg_ax = self.data_point_axs[0, 0]
        shape_ax = self.data_point_axs[0, 1]

        erg_ax.set_title("Energy fit")
        shape_ax.set_title("Shape fit")

        erg_ax.set_xlabel("Ch.")
        erg_ax.set_ylabel("Energy")

        shape_ax.set_xlabel("Energy")
        shape_ax.set_ylabel("FWHM")

        self.data_point_axs = {'erg_fit': erg_ax, 'shape_fit': shape_ax, "erg_fit_residue": self.data_point_axs[1, 0]}
        self.data_point_lines = {'erg_fit': None, 'shape_fit': None}

        bins = self.spe.erg_bins
        counts = self.spe.get_counts(make_density=True, nominal_values=True)
        super().__init__(bins, counts)
        self.ax.set_yscale('log')
        self.ax.set_xlabel("Energy [MeV]")
        self.ax.set_ylabel("Counts")

        # intensity_txt_box_ax = self.fig.add_axes([0.8, 0.88, 0.15, 0.04])
        intensity_txt_box_ax = self.fig.add_axes([0.01, 0.2, 0.062, .03])
        self.intensity_textbox = TextBox(intensity_txt_box_ax, '', initial='10%')
        self.fig.text(0.01, 0.24, 'min.\nintensity(%)')
        self.intensity_textbox.on_submit(self.on_intensity_change)

        self._mouse_over_gline: Union[PltGammaLine, None] = None  # PltGammaLine that the mouse is currently over
        self._selected_gline: Union[PltGammaLine, None] = None

        self.visible_gamma_lines: Set[PltGammaLine] = set()

        self._visible_decay_sources = set()

        clickable_lines = 'BG', 'Check sources', 'Fiss. Prod'
        check_button_ax = self.fig.add_axes([0.01, 0.3, 0.062, .55])
        self.gline_labels_map = BijectiveMap({x: f"{'\n'.join(x.split())}" for x in clickable_lines})
        self.gamma_line_check_buttons = CheckButtons(check_button_ax, self.gline_labels_map.values())
        self.gamma_line_check_buttons.on_clicked(self.on_source_check_button)

        submit_nuclide_textbox_ax = self.fig.add_axes([0.1, 0.95, 0.062, .03])
        self.submit_nuclide_textbox = TextBox(submit_nuclide_textbox_ax, 'Add nucleus', initial='N(n,*g)')
        self.submit_nuclide_textbox.on_submit(self.add_subtract_nucleus_textbox)

        fit_deg_text_box_ax = self.fig.add_axes([0.9, 0.6, 0.03, 0.1])
        self.fitdeg_radio_buttons = RadioButtons(fit_deg_text_box_ax, ['1', '2'])

        self._visible_decay_sources_text = self.fig.text(0.175, 0.96, "")
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

        self.fit_chs = []
        self.fit_ergs = []
        self.fit_erg_errs = []

        self.fit_fwhms = []
        self.fit_fwhms_errs = []

        self.fig.suptitle(self.spe.path.name)

        self.residue_line = None

        self.add_subtract_nucleus_textbox("N(n,*g)")
        self.on_intensity_change(99)
        plt.subplots_adjust(top=0.905,
                            bottom=0.11,
                            left=0.135,
                            right=0.995)

    @property
    def fit_degree(self):
        return int(self.fitdeg_radio_buttons.value_selected)

    def _set_display_nuclides_text(self):
        t = ', '.join(self._visible_decay_sources)
        self._visible_decay_sources_text.set_text(t)

    def check_intensity(self, val):
        """
        If True, intensity of val means line should bne plotted
        Args:
            val:

        Returns:

        """
        if np.isnan(val):
            return True
        return val > self.min_intensity

    def add_subtract_nucleus_textbox(self, value):
        try:
            lines = PltGammaLine.ALL_LINES[value]

            removeQ = False
            if value in self._visible_decay_sources:
                removeQ = True
                self._visible_decay_sources.remove(value)
            else:
                self._visible_decay_sources.add(value)

            self._set_display_nuclides_text()
            for line in lines:
                if removeQ:
                    self.remove_line(line)
                else:
                    if self.check_intensity(line.intensity):
                        self.add_line(line)

            self.update()
        except KeyError:
            print(f'Invalid nucleus, "{value}".\noptions are:\n{'\n\t'.join(PltGammaLine.ALL_LINES.keys())}'
                  '\n See valid decay sources above!')

    def erg_2_channel(self, erg):
        if len(self.spe.erg_calibration) == 2:
            a, b = self.spe.erg_calibration
            c = 0
        else:
            a, b, c = self.spe.erg_calibration

        if c != 0:
            a, b, c = self.spe.erg_calibration
            sqrt_term = np.sqrt(b**2 - 4*a*c + 4*c*erg)
            sols = 1/(2*c) * np.array([sqrt_term - b])
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

    def add_line(self, line: PltGammaLine):
        if line not in self.visible_gamma_lines:
            if line.axvline is not None:
                line.axvline.set_visible(True)
            else:
                line.plot(self.ax)

            self.visible_gamma_lines.add(line)

    def remove_line(self, line: PltGammaLine):
        if line in self.visible_gamma_lines:
            line.axvline.set_visible(False)
            self.visible_gamma_lines.remove(line)

    def on_intensity_change(self, text):
        for label in self._visible_decay_sources:
            for line in  PltGammaLine.ALL_LINES[label]:
                if self.check_intensity(line.intensity):
                    self.add_line(line)
                else:
                    self.remove_line(line)
        # for line in list(self.visible_gamma_lines):
        #     if not self.check_intensity(line.intensity):
        #         self.remove_line(line)
        #     else:

        # for status, label in zip(self.gamma_line_check_buttons.get_status(),
        #                          self.gamma_line_check_buttons.labels):
        #     if not status:
        #         continue
        #
        #     label = self.gline_labels_map[label.get_text()]
        #
        #     for line in PltGammaLine.ALL_LINES[label]:
        #         if self.check_intensity(line.intensity):
        #             self.add_line(line)
        #         else:
        #             self.remove_line(line)

    def on_source_check_button(self, clicked_label):
        """
        Updates visible gamma lines

        Args:
            *args:

        Returns:

        """
        label = self.gline_labels_map[clicked_label]
        _label_index = ([x.get_text() for x in self.gamma_line_check_buttons.labels]).index(clicked_label)
        just_turned_on = self.gamma_line_check_buttons.get_status()[_label_index]

        lines = PltGammaLine.ALL_LINES[label]

        if just_turned_on:
            self._visible_decay_sources.add(label)
            for line in lines:
                try:
                    if self.check_intensity(line.intensity):
                        self.add_line(line)
                except TypeError:
                    print(f'line.intensity: {line.intensity} {type(line.intensity)}\nself.min_intensity: {self.min_intensity}')
                    raise
        else:
            self._visible_decay_sources.remove(label)

            for line in lines:
                self.remove_line(line)

        self.update()

    def on_key_press(self, event):
        super(ErgCal, self).on_key_press(event)

        if event.key == 'escape':
            self.clear_fit()

    def on_key_release(self, event):
        fit = super(ErgCal, self).on_key_release(event)
        if fit is not None and self.selected_gline is not None:

            i0 = np.argmin([abs(self.selected_gline.erg - fit.centers(i)) for i in range(len(fit))])

            if abs(fit.centers(i0).n - self.selected_gline.erg) > 25:
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

            print(f"Added fit @ {self.selected_gline.erg:.1f} keV (ch = {fit_ch:.1f}) len= {len(self.fit_chs)}\n")

            print(f"Ergs: {self.fit_ergs}")
            print(f"FWHMs: {self.fit_fwhms}")

            if len(self.fit_chs) > self.fit_degree:
                erg_fit_result, new_erg_cal = lin_fit(self.fit_chs, self.fit_ergs, self.fit_erg_errs, deg=self.fit_degree)
                self.erg_calibration = new_erg_cal

                shape_fit_result, new_shape_cal = fwhm_lin_fit(self.fit_ergs, self.fit_fwhms, self.fit_fwhms_errs, deg=self.fit_degree)

                self._plot_fit(erg_fit_result, shape_fit_result)

                if self.fit_degree == 1:
                    new_erg_cal = self.erg_calibration + [0.0]
                    new_shape_cal = new_shape_cal + [0.0]

                print("Current energy cal.:")
                print('\t' + ' '.join([f'{x.n if isinstance(x, UFloat) else x:.5e}' for x in new_erg_cal]))
                print("Current shape cal.:")
                print('\t' + ' '.join([f'{x.n if isinstance(x, UFloat) else x:.5e}' for x in new_shape_cal]))

    def auto_fit(self, nuclide_names_ergs: List, fit_window=4):
        line_lookup = {}
        self.set_fit_window(fit_window)

        for name, lines in PltGammaLine.ALL_LINES.items():
            for line in lines:
                line: PltGammaLine
                line_lookup[(line.nuclide_name, line.erg)] = line

        def find_gline():
            return line_lookup[(nuclide_name, erg)]

        for nuclide_name, erg in nuclide_names_ergs:
            self._selected_gline = find_gline()
            event = MyMouseEvent(erg, self.ax)
            self.holding_space = True
            self.on_click(event)
            self.on_key_release(MyKeyEvent(' '))

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

        if self.residue_line is not None:
            self.residue_line.set_visible(0)

        self.residue_line, = self.data_point_axs['erg_fit_residue'].plot(energies, erg_fit.eval(x=chs) - energies, color='black', marker='o', ls='None')

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
            if self.mouse_over_gline:  # set currently selected GLine to current line if mouse is inside one
                if isinstance(event, MyMouseEvent):  # fake mouse event
                    contains = True
                else:
                    contains = self.mouse_over_gline.axvline.contains(event)
                if contains:
                    self.selected_gline = self.mouse_over_gline

    def annotate(self):
        if self.mouse_over_gline.annotation_text is not None:
            self.mouse_over_gline.annotation_text.set_visible(True)
            return
        trans = transforms.blended_transform_factory(self.ax.transData, self.ax.transAxes)
        x = self.mouse_over_gline.erg
        self.mouse_over_gline.annotation_text = self.fig.text(x, 1, self.mouse_over_gline.get_label(), transform=trans, va='bottom')

        self.update()

    def clear_annotate(self):
        if self.mouse_over_gline is not None and self.mouse_over_gline.annotation_text is not None:
            self.mouse_over_gline.annotation_text.set_visible(False)
            self.update()

    @property
    def selected_gline(self):
        return self._selected_gline

    @selected_gline.setter
    def selected_gline(self, val: Union[None, PltGammaLine]):
        if self._selected_gline is not None:  # set old selected line back to default
            self._selected_gline.axvline.set_color(self._selected_gline.color)
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

    # @property
    # def get_gamma_lines(self):
    #
    #     pass
    #     todo: Return list of activated gamma lines

    def hover(self, event):
        if self.mouse_over_gline is not None:
            if not self.mouse_over_gline.axvline.contains(event)[0]:
                self.mouse_over_gline = None
                return

        for gline in self.visible_gamma_lines:
            if not gline.axvline.get_visible():
                continue

            if gline.axvline.contains(event)[0]:
                self.mouse_over_gline = gline  # does annotation
                break
            else:
                pass
        else:
            self.mouse_over_gline = None  # Removes annotation
            return

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