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
        return self.channel_2_erg(self.channels-0.5)

    def get_spectrum_hist(self):
        hist = TH1F(bin_left_edges=self.erg_bins)
        hist.__set_bin_values__(self.counts)
        return hist


if __name__ == '__main__':
    from JSB_tools import Nuclide
    from JSB_tools.nuke_data_tools.gamma_spec import PrepareGammaSpec
    s_eu152 = SPEFile('/Users/burggraf1/Desktop/HPGE_temp/Eu152EffCal_center.Spe')
    # s_y88 = SPEFile('/Users/burggraf1/Desktop/HPGE_temp/Y88EffCal_center.Spe')

    eu152 = Nuclide.from_symbol('Eu152')
    for g in eu152.decay_gamma_lines[:10]:
        print(g)
    peak_center = 1408

    hist = s_eu152.get_spectrum_hist()
    hist /= hist.bin_widths
    fit = hist.peak_fit(peak_center)
    fit.plot_fit()
    print(fit)
    hist *= hist.bin_widths
    hist_windowed = hist.remove_bins_outside_range(peak_center-50, peak_center+50)
    baseline = hist_windowed.convolve_median(50/hist_windowed.bin_widths[0], True)
    signal = hist_windowed - baseline

    ax = baseline.plot(leg_label='baseline')

    print('Simple Amp: ', sum(signal.remove_bins_outside_range(peak_center-6, peak_center+6).bin_values))
    print("Shape: ", s_eu152.erg_2_peakwidth(peak_center))
    print(s_eu152.shape_cal)
    noms = hist.nominal_bin_values
    peak_idx, peak_info = find_peaks(convolve_gaus(10, noms), prominence=3*convolve_gaus(10, hist.std_errs), height=100)
    peak_xs = []
    peak_ys = []
    for i in peak_idx:
        peak_xs.append(hist.bin_centers[i])
        peak_ys.append(noms[i])

    ax = hist.plot()
    ax.plot(peak_xs, peak_ys, ls='None', marker='o')

    # hist_windowed.plot(ax, leg_label='measured')
    # signal.plot(ax, leg_label='Signal')
    # plt.legend()
    #
    plt.show()