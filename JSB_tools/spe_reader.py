from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime
import re
from matplotlib import pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import UFloat
from uncertainties.core import AffineScalarFunc
from JSB_tools import Nuclide, mpl_hist, calc_background
from typing import List, Tuple
import ROOT
import marshal
from JSB_tools.regression import FitBase
import marshal


class SPEFile:
    time_format = '%m/%d/%Y %H:%M:%S'

    def __init__(self, path):
        self.path = Path(path)
        assert self.path.exists(), self.path
        with open(self.path) as f:
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
        self.system_start_time = datetime.strptime(lines[lines.index('$DATE_MEA:\n') + 1].rstrip(), SPEFile.time_format)

        self.livetime, self.realtime = map(float, lines[lines.index('$MEAS_TIM:\n')+1].split())
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
        self.erg_coefs = list(map(float, mca_cal[:-1]))
        self.erg_units = mca_cal[-1]
        self.energies = self.channel_2_erg(self.channels)

        self.shape_cal = list(map(float, lines[lines.index('$SHAPE_CAL:\n')+2].split()))

    def pickle(self, f_name=None, directory=None):
        if f_name is None:
            f_name = self.path
        if directory is None:
            directory = self.path.parent
        else:
            directory = Path(directory)
            assert directory.exists(), f"Specified directory doesn't exist:\n{directory}t"

        f_path = (directory/f_name).with_suffix('.marshalSpe')  # add suffix
        d_simple = {'shape_cal': self.shape_cal, 'energies': self.energies, 'erg_units': self.erg_units,
                    'erg_coefs': self.erg_coefs, 'channels': self.channels, 'livetime': self.livetime,
                    'realtime': self.realtime, 'system_start_time': self.system_start_time.strftime(SPEFile.time_format)
                    ,'description': self.description, 'maestro_version': self.maestro_version,
                    'device_serial_number': self.device_serial_number, 'detector_id': self.detector_id,
                    'path': str(self.path)}
        d_unp = {'counts': (unp.nominal_values(self.counts), unp.std_devs(self.counts))}
        np_dtypes = {'channels': 'int'}
        with open(f_path, 'wb') as f:
            marshal.dump(d_simple, f)
            marshal.dump(d_unp, f)
            marshal.dump(np_dtypes, f)

    @classmethod
    def from_pickle(cls, f_path) -> SPEFile:
        f_path = Path(f_path).with_suffix('.marshalSpe')
        self = cls.__new__(cls)
        with open(f_path, 'rb') as f:
            d_simple = marshal.load(f)
            d_unp = marshal.load(f)
            np_dtypes = marshal.load(f)
        for k, v in d_simple.items():
            if isinstance(v, bytes):
                if k in np_dtypes:
                    t = getattr(np, np_dtypes[k])
                else:
                    t = np.float
                v = np.frombuffer(v, t)
            setattr(self, k, v)
        for k, v in d_unp.items():
            v = unp.uarray(np.frombuffer(v[0], np.float), np.frombuffer(v[1], np.float))
            setattr(self, k, v)
        self.path = Path(self.path)
        self.system_start_time = datetime.strptime(self.system_start_time, SPEFile.time_format)
        return self

    def set_eff_cal(self, fit: FitBase):
        pass

    def set_erg_cal(self, *coeffs):
        self.erg_coefs = coeffs
        self.energies = self.channel_2_erg(self.channels)

    def channel_2_erg(self, a):
        return np.sum([coeff * a ** i for i, coeff in enumerate(self.erg_coefs)], axis=0)

    def erg_2_fwhm(self, erg):
        if hasattr(erg, '__iter__'):
            channels = self.erg_2_channel(erg)
            i_s = np.where(channels<len(self.erg_bin_widths), channels, len(channels)-1 )
            channel_width = self.erg_bin_widths[i_s]
        # print([self.erg_2_channel(erg), len(self.erg_bin_widths)-1])
        else:
            i = np.min([self.erg_2_channel(erg), len(self.erg_bin_widths)-1])
            channel_width = self.erg_bin_widths[i]

        return channel_width*np.sum([coeff*erg**i for i, coeff in enumerate(self.shape_cal)], axis=0)

    def erg_2_peakwidth(self, erg):
        return 2.2*self.erg_2_fwhm(erg)

    def erg_2_channel(self, erg):
        if isinstance(erg, UFloat):
            erg = erg.n
        a, b, c = self.erg_coefs
        if c != 0:
            out = (-b + np.sqrt(b**2 - 4*a*c + 4*c*erg))/(2.*c)
        else:
            out = (erg - a)/b
        if hasattr(out, '__iter__'):
            return np.array(out, dtype=int)
        else:
            return int(out)

    def erg_bin_index(self, erg):
        if hasattr(erg, '__iter__'):
            return np.array([self.erg_bin_index(e) for e in erg])
        if isinstance(erg, AffineScalarFunc):
            erg = erg.n
        return np.searchsorted(self.erg_bins, erg, side='right') - 1

    @property
    def erg_bin_widths(self):
        bins = self.erg_bins
        return np.array([b1-b0 for b0, b1 in zip(bins[:-1], bins[1:])])

    @property
    def erg_bins(self):
        """
        Bin left edges for use in histograms.
        Returns:

        """
        ch = np.concatenate([self.channels, [self.channels[-1] + 1]])
        return self.channel_2_erg(ch-0.5)

    def get_counts(self, erg_min: float = None, erg_max: float = None, make_rate=False, nominal_values=False)\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Get energy spectrum within (optionally) a specified energy range.
        Args:
            erg_min:
            erg_max:
            make_rate: If True, divide by livetime
            nominal_values: If false, return uncertain array, else don't

        Returns: counts, bin_edges


        """
        if any(x is None for x in [erg_min, erg_min]):
            if erg_min is None:
                erg_min = self.erg_bins[0]
            if erg_max is None:
                erg_max = self.erg_bins[-1]
            cut = np.where((self.erg_bins <= erg_max) & (self.erg_bins >= erg_min))
            bins = self.erg_bins[cut]
            counts = self.counts[cut]
        else:
            bins = self.erg_bins
            counts = self.counts
        if make_rate:
            counts /= self.livetime
        return unp.nominal_values(counts) if nominal_values else counts, bins

    def plot_erg_spectrum(self, erg_min: float = None, erg_max: float = None, ax=None,
                          leg_label=None, make_rate=False, **ax_kwargs):
        """
        Plot energy spectrum within (optionally) a specified energy range.
        Args:
            erg_min:
            erg_max:
            leg_label:
            ax_kwargs:
            make_rate: If True, divide by livetime
            ax:

        Returns:

        """
        if ax is None:
            plt.figure()
            ax = plt.gca()

        counts, bins = self.get_counts(erg_min=erg_min, erg_max=erg_max, make_rate=make_rate)
        mpl_hist(bins, counts, ax=ax, label=leg_label, **ax_kwargs)
        ax.set_xlabel('Energy [KeV]')
        ax.set_ylabel('Counts')
        return ax

    @property
    def nominal_counts(self) -> np.ndarray:
        return unp.nominal_values(self.counts)

    def get_baseline(self, num_iterations=20, clipping_window_order=2, smoothening_order=5) -> np.ndarray:
        return calc_background(self.counts, num_iterations=num_iterations, clipping_window_order=clipping_window_order,
                               smoothening_order=smoothening_order)

    def get_baseline_removed(self, num_iterations=20, clipping_window_order=2, smoothening_order=5) -> np.ndarray:
        return self.counts - self.get_baseline(num_iterations=num_iterations,
                                               clipping_window_order=clipping_window_order,
                                               smoothening_order=smoothening_order)


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
        baseline_removed = spe_file.counts - spe_file.get_baseline(**bg_kwargs)
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
                                         tot_acquisition_time=spe_file.livetime, acquisition_ti=spe_file.system_start_time,
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
    import time
    import cProfile

    p = '/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot119.Spe'

    def f():
        for i in range(100):
            SPEFile(p)
    cProfile.run('f()')

    # t = time.time()
    # for i in range(1000):
    #     s = SPEFile(p)
    # print((time.time() - t)/100)
    #
    # # s.pickle()
    # t = time.time()
    # for i in range(1000):
    #     s = SPEFile.from_pickle(p)
    # print((time.time() - t)/100)
    # print(dir(s))
    # s.get_spectrum_hist().plot()
    #
    #
    # plt.show()