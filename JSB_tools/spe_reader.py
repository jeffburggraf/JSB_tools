from __future__ import annotations
import time
import warnings
from pathlib import Path
from datetime import datetime
import re
from matplotlib import pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import UFloat
from uncertainties.core import AffineScalarFunc
from JSB_tools import Nuclide, mpl_hist, calc_background, human_friendly_time, rolling_median
from typing import List, Tuple, Union
import marshal
from lmfit.models import GaussianModel
from scipy.signal import find_peaks


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
        re_counts = re.compile(' *([0-9]+)')
        i = 0
        while m := re_counts.match(lines[start_index + i]):
            self.counts[i] = int(m.groups()[0])
            i += 1
        self.counts = unp.uarray(self.counts, np.sqrt(self.counts))
        lines = lines[start_index + i:]
        mca_cal = lines[lines.index('$MCA_CAL:\n')+2].split()
        self._erg_calibration = list(map(float, mca_cal[:-1]))
        self.__energies__ = None  # if None, this signals to the energies property that energies must be calculated.
        self.erg_units = mca_cal[-1]
        # self.energies = self.channel_2_erg(self.channels)

        self.shape_cal = list(map(float, lines[lines.index('$SHAPE_CAL:\n')+2].split()))
        self.pretty_realtime = human_friendly_time(self.realtime)
        self.pretty_livetime = human_friendly_time(self.livetime)

    @property
    def deadtime_corr(self):
        return self.realtime/self.livetime

    @property
    def erg_calibration(self):
        return self._erg_calibration

    @erg_calibration.setter
    def erg_calibration(self, value):
        self._erg_calibration = value
        self.__energies__ = None

    @property
    def energies(self):
        if self.__energies__ is None:
            self.__energies__ = self.channel_2_erg(self.channels)
        return self.__energies__

    def pickle(self, f_name=None, directory=None):
        if f_name is None:
            f_name = self.path
        if directory is None:
            directory = self.path.parent
        else:
            directory = Path(directory)
            assert directory.exists(), f"Specified directory doesn't exist:\n{directory}t"

        f_path = (directory/f_name).with_suffix('.marshalSpe')  # add suffix
        d_simple = {'shape_cal': self.shape_cal, 'erg_units': self.erg_units,
                    'erg_calibration': self.erg_calibration, 'channels': self.channels, 'livetime': self.livetime,
                    'realtime': self.realtime, 'system_start_time': self.system_start_time.strftime(SPEFile.time_format)
                    , 'description': self.description, 'maestro_version': self.maestro_version,
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

    @classmethod
    def from_lis(cls, path):
        from JSB_tools.list_reader import MaestroListFile
        l = MaestroListFile(path)
        return l.list2spe()

    def set_energy_cal(self, *coeffs, update_file=False):
        self.erg_calibration = np.array(coeffs)
        # self.energies = self.channel_2_erg(self.channels)
        if len(coeffs) == 2:
            coeffs = list(coeffs) + [0.]

        coeffs_string = f'{" ".join(map(lambda x: f"{x:.6E}", coeffs))} {self.erg_units}\n'

        if update_file:
            with open(self.path) as f:
                lines = f.readlines()
                try:
                    i1 = lines.index('$ENER_FIT:\n') + 1
                    i2 = lines[i1:].index('$MCA_CAL:\n') + i1 + 1
                except IndexError as e:
                    raise Exception('Invalid SPE file. ') from e

                lines[i1] = ' '.join(map(str, coeffs[:2])) + '\n'
                lines[i2] = f'3\n'
                lines[i2+1] = coeffs_string
            temp_file_path = self.path.with_suffix(f'{self.path.suffix}._tmp')
            with open(temp_file_path, 'w') as f:
                f.writelines(lines)
            temp_file_path.rename(self.path)

    def channel_2_erg(self, chs) -> np.ndarray:
        return np.sum([coeff * chs ** i for i, coeff in enumerate(self.erg_calibration)], axis=0)

    def erg_2_fwhm(self, erg):
        """
        Get the Full Width Half Max from the shape calibration.
        Todo: This doesnt tend to give good results. Figure this out.
        Args:
            erg:

        Returns:

        """
        iter_flag = True
        if not hasattr(erg, '__iter__'):
            erg = [erg]
            iter_flag = False

        channels = self.erg_2_channel(erg)
        fwhm_in_chs = np.sum([coeff*channels**i for i, coeff in enumerate(self.shape_cal)], axis=0)
        fwhm = fwhm_in_chs*self.erg_bin_widths
        return fwhm if iter_flag else fwhm[0]

    def erg_2_peakwidth(self, erg):
        """
        Uses self.erg_2_fwhm, but converts to sigma.
        """
        return 2.2*self.erg_2_fwhm(erg)

    def erg_2_channel(self, erg):
        return self.__erg_index__(erg)

    def __erg_index__(self, erg):
        """
        Get the index which corresponds to the correct energy bin(s).
        Examples:
            The result can be used to find the number of counts in the bin for 511 KeV:
                self.counts[self.__erg_index__(511)]
        """
        if isinstance(erg, AffineScalarFunc):
            erg = erg.n
        if erg >= self.erg_bins[-1]:
            return len(self.counts) - 1
        if erg < self.erg_bins[0]:
            return 0
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
        # ch = np.arange(len(self.counts) + 1, dtype=float)
        chs = np.concatenate([self.channels, [self.channels[-1]+1]])
        return self.channel_2_erg(chs-0.5)

    @property
    def rates(self):
        return self.counts/self.livetime

    def get_counts(self, erg_min: float = None, erg_max: float = None, make_rate=False, remove_baseline=False,
                   make_density=False,
                   nominal_values=False,
                   return_bin_edges=False,
                   deadtime_corr=False,
                   debug_plot=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get energy spectrum within (optionally) a specified energy range.
        Args:
            erg_min: Min energy cut
            erg_max: Max energy cut
            make_rate: If True, divide by livetime
            remove_baseline: Whether or not to remove baseline.
            make_density: If True, result is divided by bin width
            nominal_values: If false, return uncertain array, else don't
            return_bin_edges: If True, return bin edges
            deadtime_corr: If True, correct for deatime
            debug_plot: Plot counts

        Returns: counts, bin_edges


        """
        if remove_baseline:
            counts = self.counts - calc_background(self.counts)
        else:
            counts = self.counts
        if not all(x is None for x in [erg_min, erg_min]):
            if erg_min is None:
                erg_min = self.erg_bins[0]
            if erg_max is None:
                erg_max = self.erg_bins[-1]
            # erg_max = min([self.energies[-1], erg_max])
            # erg_min = max([self.energies[0], erg_min])
            imin = self.__erg_index__(erg_min)
            imax = self.__erg_index__(erg_max)
            bins = self.erg_bins[imin: imax+1]
            counts = counts[imin: imax]
            b_widths = self.erg_bin_widths[imin: imax]
        else:
            bins = self.erg_bins
            b_widths = self.erg_bin_widths

        if make_rate:
            counts /= self.livetime
        if make_density:
            counts /= b_widths
        if deadtime_corr:
            counts *= self.deadtime_corr

        out = unp.nominal_values(counts) if nominal_values else counts
        if debug_plot:
            mpl_hist(bins, out, title='"get_counts()" debug plot')
        if return_bin_edges:
            return out, bins
        else:
            return out

    def plot_erg_spectrum(self, erg_min: float = None, erg_max: float = None, ax=None,
                          leg_label=None, make_rate=False, remove_baseline=False, make_density=False, **ax_kwargs):
        """
        Plot energy spectrum within (optionally) a specified energy range.
        Args:
            erg_min:
            erg_max:
            leg_label:
            ax_kwargs:
            make_rate: If True, divide by livetime
            remove_baseline: Is True, remove baseline
            make_density: y units will be counts/unit energy
            ax:

        Returns:

        """
        if ax is None:
            plt.figure()
            ax = plt.gca()

        counts, bins = self.get_counts(erg_min=erg_min, erg_max=erg_max, make_rate=make_rate,
                                       remove_baseline=remove_baseline, make_density=make_density,
                                       return_bin_edges=True)

        mpl_hist(bins, counts, ax=ax, label=leg_label, **ax_kwargs)
        ylabel = 'Counts'
        if make_rate:
            ylabel += '/s'
        if make_density:
            ylabel += '/KeV'

        ax.set_ylabel(ylabel)

        ax.set_xlabel('Energy [KeV]')

        return ax

    @property
    def nominal_counts(self) -> np.ndarray:
        return unp.nominal_values(self.counts)

    def get_baseline(self, num_iterations=20, clipping_window_order=2, smoothening_order=5) -> np.ndarray:
        return calc_background(self.counts, num_iterations=num_iterations, clipping_window_order=clipping_window_order,
                               smoothening_order=smoothening_order)

    def get_baseline_median(self,  erg_min: float = None, erg_max: float = None, window_width_kev=30):
        """
        Calculate the median withing a rolling window of width `window_width`. Generally good at estimating baseline.
        Args:
            erg_min: Min energy cut
            erg_max: Max energy cut
            window_width_kev: Size of rolling median window in KeV

        Returns:

        """
        if erg_min is None:
            erg_min = self.erg_bins[0]
        if erg_max is None:
            erg_max = self.erg_bins[-1]
        center = (erg_min + erg_max)/2
        window_width = self.__erg_index__(center+window_width_kev/2) - self.__erg_index__(center-window_width_kev/2)
        _slice = slice(self.__erg_index__(erg_min), self.__erg_index__(erg_max))
        bg_est = rolling_median(window_width=window_width, values=self.counts)[_slice]
        return bg_est

    def get_baseline_removed(self, num_iterations=20, clipping_window_order=2, smoothening_order=5) -> np.ndarray:
        return self.counts - self.get_baseline(num_iterations=num_iterations,
                                               clipping_window_order=clipping_window_order,
                                               smoothening_order=smoothening_order)

    def multi_peak_fit(self, centers: List[float], sigma_guess=3, fit_window=100):
        model = None
        params = None
        y = self.get_baseline_removed()
        _center = int(np.median(list(map(self.__erg_index__, centers))))
        _slice = slice(max([0, _center - fit_window//2]), min([len(y)-1, _center+fit_window//2]))
        y = y[_slice]
        x = self.energies[_slice]
        plt.figure()
        y /= self.erg_bin_widths[_slice]  # make density
        mpl_hist(self.erg_bins[_slice.start: _slice.stop + 1], y)

        # y /= np.mean(x[1:] - x[:-1])
        peaks, peak_infos = find_peaks(unp.nominal_values(y), height=unp.std_devs(y), width=0)
        # plt.plot(peaks, peak_infos['prominences'], ls='None', marker='o')
        select_peak_ixs = np.argmin(np.array([np.abs(c-np.searchsorted(x, centers)) for c in peaks]).T, axis=1)
        peak_widths = peak_infos['widths'][select_peak_ixs]*self.erg_bin_widths[_center]
        amplitude_guesses = peak_infos['peak_heights'][select_peak_ixs]*peak_widths
        sigma_guesses = peak_widths/2.355

        for i, erg in enumerate(centers):
            m = GaussianModel(prefix=f'_{i}')
            if model is None:
                params = m.make_params()
                params[f'_{i}center'].set(value=erg)
                model = m
            else:
                model += m
                params.update(m.make_params())
            # bin_index = self.__erg_index__(erg)
            params[f'_{i}amplitude'].set(value=amplitude_guesses[i], min=0)
            params[f'_{i}center'].set(value=erg, min=erg-0.1, max=erg+0.1)
            params[f'_{i}sigma'].set(value=sigma_guesses[i])
        weights = unp.std_devs(y)
        weights = np.where(weights>0, weights, 1)
        weights = 1.0/weights
        plt.plot(x, model.eval(params=params, x=x))
        fit_result = model.fit(data=unp.nominal_values(y), x=x, weights=weights, params=params)
        fit_result.plot()
        return fit_result

    @classmethod
    def build(cls, path, counts, erg_calibration: List[float], live_time, realtime, channels=None, erg_units='KeV',
              shape_cal=None, description=None, system_start_time=None) -> SPEFile:
        self = SPEFile.__new__(cls)
        self.__energies__ = None

        if isinstance(counts[0], UFloat):
            self.counts = counts
        else:
            self.counts = unp.uarray(counts, np.sqrt(counts))

        if channels is None:
            self.channels = np.arange(len(self.counts))
        else:
            self.channels = channels

        self.erg_calibration = list(erg_calibration)
        self.erg_units = erg_units

        if shape_cal is None:
            self.shape_cal = [0, 1, 0]
        else:
            self.shape_cal = shape_cal

        self.path = path

        self.livetime = live_time
        self.realtime = realtime

        if system_start_time is None:
            system_start_time = datetime(year=1, month=1, day=1)
        self.system_start_time = system_start_time

        if description is None:
            description = ''
        self.description = description

        self.device_serial_number = 0
        self.detector_id = 0

        return self


if __name__ == '__main__':
    # from
    pass
    spe = SPEFile('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot119.Spe')
    spe.plot_erg_spectrum()
    plt.show()