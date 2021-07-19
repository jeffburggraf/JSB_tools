from pathlib import Path
from datetime import datetime
import re
from matplotlib import pyplot as plt
import numpy as np
from JSB_tools.TH1 import TH1F
import uncertainties.unumpy as unp
from uncertainties import UFloat

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

    def channel_2_erg(self, a):
        return np.sum([coeff*a**i for i, coeff in enumerate(self.erg_fit)], axis=0)

    def erg_2_channel(self, erg):
        if isinstance(erg, UFloat):
            erg = erg.n
        a, b, c = self.erg_fit
        if c != 0:
            out = (-b + np.sqrt(b**2 - 4*a*c + 4*c*erg))/(2.*c)
        else:
            out = (erg - a)/b
        return out

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
    s_y88 = SPEFile('/Users/burggraf1/Desktop/HPGE_temp/Y88EffCal_center.Spe')

    eu152 = Nuclide.from_symbol('Eu152')
    p = PrepareGammaSpec(len(s_eu152.channels))
    channels = []
    ergs_true = []
    true_counts = []
    for g in eu152.decay_gamma_lines[:6]:
        true_counts.append(g.get_n_gammas(1.060, activity_ref_date=datetime(2008, 7, 1), tot_acquisition_time=s_eu152.livetime))
        ergs_true.append(g.erg)
        channels.append(s_eu152.erg_2_channel(g.erg))

    p.add_peaks_4_calibration(s_eu152.counts, channels, ergs_true, true_counts)

    eu152 = Nuclide.from_symbol('Eu152')
    Y88 = Nuclide.from_symbol('Y88')
    p = PrepareGammaSpec(len(s_eu152.channels))
    channels = []
    ergs_true = []
    true_counts = []
    s_y88.get_spectrum_hist().plot()
    for g in Y88.decay_gamma_lines[:4]:
        print(g)
        true_counts.append(
            g.get_n_gammas(370, activity_ref_date=datetime(2008, 7, 1), tot_acquisition_time=s_y88.livetime,
                           activity_unit='kBq'))
        ergs_true.append(g.erg)
        channels.append(s_eu152.erg_2_channel(g.erg))
    p.add_peaks_4_calibration(s_y88.counts, channels, ergs_true, true_counts)

    # p.compute_calibration()
    #
    # h = s_eu152.get_spectrum_hist()
    # h.plot()
    #
    plt.show()