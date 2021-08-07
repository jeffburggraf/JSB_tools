from pathlib import Path
from datetime import datetime
import re
from matplotlib import pyplot as plt
import numpy as np
from JSB_tools.TH1 import TH1F, convolve_gaus
import uncertainties.unumpy as unp
from uncertainties import UFloat
from JSB_tools.regression import PeakFit
from scipy.signal import find_peaks
from JSB_tools.regression import LogPolyFit, PeakFit
from JSB_tools import Nuclide
from typing import List
import ROOT

class SPEFile:
    def __init__(self, path):
        path = Path(path)
        assert path.exists()
        with open(path) as f:
            lines = f.readlines()
        index = lines.index('$SPEC_REM:\n')+1
        self.detector_id = int(lines[index][5:])
        index += 1

        if m := re.match(r".+# ([a-zA-Z0-9-]+)", lines[index]):
            self.device_serial_number = (m.groups()[0])
        else:
            self.device_serial_number = None

        index += 1
        if m := re.match(r".+Maestro Version ([0-9.]+)", lines[index]):
            self.maestro_version = m.groups()[0]
        else:
            self.maestro_version = None

        self.description = lines[lines.index('$SPEC_ID:\n')+1].rstrip()
        self.start_time = datetime.strptime(lines[lines.index('$DATE_MEA:\n')+1].rstrip(), '%m/%d/%Y %H:%M:%S')

        self.livetime, self.realtime = map(int, lines[lines.index('$MEAS_TIM:\n')+1].split())
        index = lines.index('$DATA:\n')
        init_channel, final_channel = map(int, lines[index+1].split())
        self.counts = np.zeros(final_channel-init_channel+1)
        self.channels = np.arange(init_channel, final_channel+1)
        start_index = index + 2
        re_counts = re.compile(' +([0-9]+)')
        i = 0
        while m := re_counts.match(lines[start_index + i]):
            self.counts[i] = int(m.groups()[0])
            i += 1
        self.counts = unp.uarray(self.counts, np.sqrt(self.counts))
        lines = lines[start_index + i:]
        mca_cal = lines[lines.index('$MCA_CAL:\n')+2].split()
        self.erg_fit = list(map(float, mca_cal[:-1]))
        self.erg_units = mca_cal[-1]
        self.energies = self.channel_2_erg(self.channels)

        self.shape_cal = list(map(float, lines[lines.index('$SHAPE_CAL:\n')+2].split()))

    def channel_2_erg(self, a):
        return np.sum([coeff*a**i for i, coeff in enumerate(self.erg_fit)], axis=0)

    def erg_2_fwhm(self, erg):
        channel_width = self.erg_bins[self.erg_2_channel(erg):]
        channel_width = channel_width[1] - channel_width[0]

        return channel_width*np.sum([coeff*erg**i for i, coeff in enumerate(self.shape_cal)], axis=0)

    def erg_2_peakwidth(self, erg):
        return 2.2*self.erg_2_fwhm(erg)

    def erg_2_channel(self, erg):
        if isinstance(erg, UFloat):
            erg = erg.n
        a, b, c = self.erg_fit
        if c != 0:
            out = (-b + np.sqrt(b**2 - 4*a*c + 4*c*erg))/(2.*c)
        else:
            out = (erg - a)/b
        return int(out)

    @property
    def erg_bins(self):
        """
        Bin left edges for use in histograms.
        Returns:

        """
        # print((self.channel_2_erg(self.channels) - 0.5)[:10])
        # print(self.energies[:10])
        ch = np.concatenate([self.channels, [self.channels[-1] + 1]])
        return self.channel_2_erg(ch-0.5)

    def get_spectrum_hist(self):
        hist = TH1F(bin_left_edges=self.erg_bins)
        hist.__set_bin_values__(self.counts)
        return hist

    def get_background(self, num_iterations=20, clipping_window_order=2, smoothening_order=5):
        assert clipping_window_order in [2,4,6,8]
        assert smoothening_order in [3, 5, 7, 9, 11, 13, 15]
        spec = ROOT.TSpectrum()
        result = unp.nominal_values(self.counts)
        cliping_window = getattr(ROOT.TSpectrum, f'kBackOrder{clipping_window_order}')
        smoothening = getattr(ROOT.TSpectrum, f'kBackSmoothing{smoothening_order}')
        spec.Background(result, len(result), num_iterations, ROOT.TSpectrum.kBackDecreasingWindow,
                        cliping_window, ROOT.kTRUE,
                        smoothening, ROOT.kTRUE)
        return result

    def add_eff_calibration_peaks(self, nuclide: Nuclide,
                                  peaks: List[float],
                                  counting_window_width,
                                  ref_activity: float,
                                  activity_ref_date: datetime,
                                  activity_unit='uCi',
                                  **bg_kwargs):
        # todo make a stand alone class for when there is no SPE file. Call it from here
        baseline_subtracted = self.counts - self.get_background(**bg_kwargs)
        counting_window_width /= self.erg_fit[1]
        counting_window_width = int(counting_window_width)
        for g in nuclide.decay_gamma_lines:

            if any(np.isclose(g.erg.n, p_erg, atol=0.1) for p_erg in peaks):
                ngammas = g.get_n_gammas(ref_activity=ref_activity, activity_ref_date=activity_ref_date,
                                         tot_acquisition_time=self.livetime, acquisition_ti=self.start_time,
                                         activity_unit=activity_unit)
                peak_index = np.searchsorted(self.energies, g.erg.n)
                counts = sum(baseline_subtracted[peak_index-counting_window_width//2: peak_index+counting_window_width//2])
                print(f"Gamma @ {g.erg}. Eff: {counts/ngammas}, counts: {counts}")


class EfficiencyCal:
    def __init__(self):
        self.erg_bin_centers = None
        self._cal_ergs = []
        self._cal_meas_counts = []
        self._cal_true_counts = []
        self._cal_sources = []
        self.window_widths = []

    @property
    def cal_meas_counts(self):
        return np.array(self._cal_meas_counts)[np.argsort(self._cal_ergs)]

    @property
    def cal_true_counts(self):
        return np.array(self._cal_true_counts)[np.argsort(self._cal_ergs)]

    @property
    def cal_sources(self):
        return np.array(self._cal_sources)[np.argsort(self._cal_ergs)]


    @property
    def cal_ergs(self):
        return np.array(self._cal_ergs)[np.argsort(self._cal_ergs)]

    def add_cal_peak_with_spe(self, spe_file: SPEFile,
                              nuclide: Nuclide,
                              peak_energies: List[float],
                              counting_window_width,
                              ref_activity: float,
                              activity_ref_date: datetime,
                              activity_unit='uCi',
                              include_decay_chain=False,
                              **bg_kwargs):
        baseline_removed = spe_file.counts - spe_file.get_background(**bg_kwargs)
        if self.erg_bin_centers is None:
            self.erg_bin_centers = spe_file.energies
        else:
            assert all(spe_file.energies == self.erg_bin_centers),\
                "Incompatible SPE files used! (they have different erg bins)"
        if include_decay_chain:
            all_gamma_lines = nuclide.decay_gamma_lines + nuclide.decay_chain_gamma_lines()
        else:
            all_gamma_lines = nuclide.decay_gamma_lines
        for erg in peak_energies:
            g = all_gamma_lines[np.argmin([abs(erg - _g.erg) for _g in all_gamma_lines])]
            if abs(g.erg - erg)<0.1:
                self.window_widths.append(counting_window_width)
                self._cal_ergs.append(erg)

                ngammas = g.get_n_gammas(ref_activity=ref_activity, activity_ref_date=activity_ref_date,
                                         tot_acquisition_time=spe_file.livetime, acquisition_ti=spe_file.start_time,
                                         activity_unit=activity_unit)
                selector = np.where((spe_file.energies >= erg-counting_window_width//2) &
                                                        (spe_file.energies <= erg+counting_window_width//2))
                peak_counts = baseline_removed[selector]
                peak_counts = sum(peak_counts)
                self._cal_meas_counts.append(peak_counts)
                self._cal_true_counts.append(ngammas)
                self._cal_sources.append(nuclide.name)

    @property
    def eff_cal_points(self):
        out = np.array([m/t for m, t in zip(self.cal_meas_counts, self.cal_true_counts)])
        return out

    @property
    def nominal_eff_cal_points(self):
        out = np.array([x.n for x in self.eff_cal_points])
        return out

    @property
    def eff_cal_points_error(self):
        out = np.array([x.std_dev for x in self.eff_cal_points])
        return out

    def plot_eff(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        for n_name in set(self.cal_sources):
            select = np.where(self.cal_sources == n_name)
            x = self.cal_ergs[select]
            y = self.nominal_eff_cal_points[select]
            yerr = self.eff_cal_points_error[select]

            ax.errorbar(x, y, yerr=yerr, ls='None', marker='d', label=n_name)
        ax.legend()
        return ax

    def plot_counts(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.plot(self.cal_ergs, [x.n for x in self.cal_meas_counts], ls='None', marker='d')
        return ax



if __name__ == '__main__':
    from JSB_tools import Nuclide

    import ROOT

    # from JSB_tools.nuke_data_tools.gamma_spec import PrepareGammaSpec
    spe_eu152 = SPEFile('/Users/burggraf1/Desktop/HPGE_temp/Eu152EffCal_center.Spe')
    spe_Y88 = SPEFile('/Users/burggraf1/Desktop/HPGE_temp/Y88EffCal_center.Spe')
    spe_Co60 = SPEFile('/Users/burggraf1/Desktop/HPGE_temp/Co60EffCal_center.Spe')
    spe_Na22 = SPEFile('/Users/burggraf1/Desktop/HPGE_temp/Na22EffCal_center.Spe')
    spe = spe_eu152
    n = Nuclide.from_symbol('Y88')
    for g in n.decay_gamma_lines:
        print(g)
    eff = EfficiencyCal()

    eff.add_cal_peak_with_spe(spe_eu152, Nuclide.from_symbol('Eu152'), [867.4, 121.8, 1408, 964.1, 1112.1, 778.9,444.0,411.1, 344.3], 10, 1.06,
                                  datetime(year=2008, month=7, day=1))
    eff.add_cal_peak_with_spe(spe_Y88, Nuclide.from_symbol('Y88'),
                              [1836.07, 898], 10, 433,  datetime(year=2019, month=7, day=1), activity_unit='kBq')
    # eff.add_cal_peak_with_spe(spe_Co60, Nuclide.from_symbol('Co60'),
    #                           [1332.49, 1173.23], 10, 1.082, datetime(year=2008, month=7, day=1), activity_unit='uCi')
    #
    # eff.add_cal_peak_with_spe(spe_Na22, Nuclide.from_symbol('Na22'),
    #                           [1274.54], 10, 1.146, datetime(year=2008, month=7, day=1), activity_unit='uCi')
    #
    # eff_fit = LogPolyFit(eff.cal_ergs, eff.eff_cal_points, order=1)
    # print(eff_fit)
    # eff_fit.plot_fit()
    # eff.plot_eff()
    print(list(unp.nominal_values(eff.eff_cal_points)))
    print(list(unp.nominal_values(eff.cal_ergs)))

    h = spe_Y88.get_spectrum_hist()
    h2 = spe_Y88.get_spectrum_hist()-spe_Y88.get_background()
    ax = h.plot()
    h2.plot(ax)

    eff.plot_counts()


    #
    plt.show()