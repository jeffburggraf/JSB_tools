from __future__ import annotations
from JSB_tools.interactive_plot import InteractivePlot
import re
import warnings
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import shutil
import uncertainties.unumpy as unp
from uncertainties import UFloat, ufloat
from uncertainties.core import AffineScalarFunc
from JSB_tools import mpl_hist, calc_background, human_friendly_time, rolling_median, shade_plot, rebin
from JSB_tools.nuke_data_tools.nuclide import GammaLine
from typing import List, Union
import marshal
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
from matplotlib.axes import Axes
from JSB_tools.spectra import EfficiencyCalMixin
from JSB_tools.spectra.time_depend import multi_guass_fit, GausFitResult
from JSB_tools.tab_plot import TabPlot
from lmfit.models import PolynomialModel


def _rebin(rebin, values, bins):
    if rebin != 1:
        l = len(values)
        out = np.sum([values[rebin * i: rebin * (i + 1)] for i in range(l // rebin)], axis=1)
        bins = np.array([bins[rebin * i] for i in range(l // rebin + 1)])
    else:
        out = values

    return out, bins


def _backup_file(path: Path):
    path = Path(path)
    directory = path.parent / 'backup'
    directory.mkdir(exist_ok=True)
    newpath = directory / path.name
    shutil.copy(path, newpath)


def save_spe(spe: SPEFile, path):
    lines = ['$SPEC_ID:', spe.description,
             '$DATE_MEA:', datetime.strftime(spe.system_start_time, spe.time_format),
             '$MEAS_TIM:', f'{int(spe.livetime)} {int(spe.realtime)}', '$DATA', f'0 {len(spe.energies)}',
             '\n'.join(map(lambda x:f"       {int(x)}", unp.nominal_values(spe.counts))),
             '$ENER_FIT:', ' '.join(map(lambda x: f"{x:.7E}", spe.erg_calibration[:2])),
             '$MCA_CAL:',
             len(spe.erg_calibration), ' '.join(map(lambda x: f"{x:.7E}", spe.erg_calibration)) + ' keV']
    if spe.shape_cal is not None:
        lines.extend(['$SHAPE_CAL:',
                      len(spe.shape_cal),
                      ' '.join(map(lambda x: f"{x:.7E}", spe.shape_cal))])
    out = '\n'.join(map(str, lines))
    path = Path(path).with_suffix('.Spe')
    with open(path, 'w') as f:
        f.write(out)


def _get_SPE_data(path):
    with open(path) as f:
        lines = f.readlines()

    top_header = lines[:lines.index('$DATA:\n') + 2]
    out = {}
    i = 0

    while i < len(top_header):
        try:
            if lines[i] == '$SPEC_ID:\n':
                i += 1
                out['description'] = lines[i].rstrip()
                # i += 1
            elif lines[i] == '$SPEC_REM:\n':
                i += 1
                out['detector_id'] = int(lines[i][5:])
                # i += 1
            elif lines[i][:8] == 'DETDESC#':
                out['device_serial_number'] = lines[i][8:].rstrip().lstrip()
                # i += 1
            elif 'Maestro Version' in lines[i]:
                out['maestro_version'] = lines[i].split()[-1]
                # i += 1
            elif lines[i] == '$DATE_MEA:\n':
                i += 1
                try:
                    m = re.match(r'([0-9]+)\/([0-9]+)\/([0-9]+) ([0-9]+):([0-9]+):([0-9]+)\.?[0-9]*', lines[i].strip())
                    if m:
                        mo, day, yr, hr, min, sec = map(int, m.groups())
                        out['system_start_time'] = datetime(year=yr, month=mo, day=day,hour=hr, minute=min, second=sec, )
                    # out['system_start_time'] = datetime.strptime(lines[i].rstrip(),
                    #                                          SPEFile.time_format)
                    else:
                        raise ValueError
                except ValueError:
                    warnings.warn(f"Spe file, '{path}' contains invalid date format:\n{lines[i]}")
                    out['system_start_time'] = datetime(1, 1, 1)
                # i += 1
            elif lines[i] == '$MEAS_TIM:\n':
                i += 1
                try:
                    out['livetime'], out['realtime'] = map(float, lines[i].split())
                except ValueError:
                    raise ValueError(
                        f"Livetime and realtime did not appear in correct format after '$MEAS_TIM:'. in '{path}'\n"
                          "What I expected, e.g.,\n"
                          "\t$MEAS_TIM:\n\t220 230\n"
                          "What I got,\n"
                          f"\t$MEAS_TIM:\n\t{lines[i]}\n"
                          "There is a problem in Spe file.")
                # i += 1
            elif lines[i] == '$DATA:\n':
                i += 1
                try:
                    out['n_channels'] = int(lines[i].split()[-1]) + 1
                except ValueError:
                    raise ValueError(
                        f"Number of channels did not appear in correct format after '$DATA:' in '{path}'\n"
                        "What I expected, e.g.,\n"
                        "\t$DATA:\n\t0 16383\n"
                        "What I got,\n"
                        f"\t$DATA:\n\t{lines[i]}\n"
                        "There is a problem in Spe file.")

        except (IndexError, ValueError) as e:
            continue
        finally:
            i += 1

    for key in ['description', 'detector_id', 'device_serial_number', 'maestro_version']:
        if key not in out:
            out[key] = None
    counts = []

    while True:
        try:
            counts.append(float(lines[i]))
            i += 1
        except ValueError:
            break

    counts = unp.uarray(counts, np.sqrt(counts))
    out['counts'] = counts

    bot_header = lines[i:]
    try:
        data = bot_header[bot_header.index('$MCA_CAL:\n') + 2].split()
        erg_cal = []
        for val in data:
            try:
                erg_cal.append(float(val))
            except ValueError:
                break
        out['_erg_calibration'] = erg_cal
        out['erg_units'] = data[-1]
    except ValueError as e:
        raise ValueError(f"InvLid energy calibration in Spe file '{path}'\n{e}")

    try:
        data = bot_header[bot_header.index('$SHAPE_CAL:\n') + 2].split()
        out['shape_cal'] = list(map(float, data))
    except (ValueError, IndexError) as e:
        # warnings.warn(f"Invalid shape calibration in Spe file '{path}'\n{e}")
        out['shape_cal'] = None

    return out


class SPEFile(EfficiencyCalMixin):
    """
    Used for processing and saving ORTEC ASCII SPE files. Can also build from other data using SPEFile.build(...).

    """
    time_format = '%m/%d/%Y %H:%M:%S'

    def __init__(self, path, eff_path=None):
        """
        Analyse ORTEC .Spe ASCII spectra files.

        Args:
            path:
            eff_path:

        """
        self.path = Path(path)
        assert self.path.exists(), self.path

        super(SPEFile, self).__init__()

        self._energies = None  # if None, this signals to the energies property that energies must be calculated.
        self._erg_bins = None  # same as above
        self.eff_path = Path(eff_path) if eff_path is not None else None

        data = _get_SPE_data(path)
        self.description = data['description']
        self.detector_id = data['detector_id']
        self.device_serial_number = data['device_serial_number']
        self.maestro_version = data['maestro_version']
        self.system_start_time = data['system_start_time']
        self.livetime = data['livetime']
        self.realtime = data['realtime']
        self.counts = data['counts']
        self.counts.flags.writeable = False
        self.channels = np.arange(len(self.counts))

        self._erg_calibration = data['_erg_calibration']
        self.erg_units = data['erg_units']
        self.shape_cal = data['shape_cal']

        try:
            self.unpickle_efficiency(eff_path)
        except FileNotFoundError:
            pass

    def get_sigma(self, erg):
        """
        Gets sigma from linear Eq. for FWHM: a + bx + cx^2
        Args:
            erg:
            a: linear y intercept for FWHM
            b: Scaling constant for FWHM

        Returns:

        """

        a, b, c = self.shape_cal
        if c == 0:
            return (1 / 2.35482) * (a + b * erg)
        else:
            return (1 / 2.35482) * (a + b * erg + c * erg ** 2)

    @staticmethod
    def save_erg_call_all(path, erg_cal=None, shape_cal=None):
        """

        Args:
            path:
            erg_cal:
            shape_cal:

        Returns:

        """
        path = Path(path)
        assert path.is_dir()
        msg = ''
        if erg_cal is not None:
            msg += 'erg'
        if shape_cal is not None:
            if len(msg):
                msg += ' & shape'
            else:
                msg += 'shape'

        for p in path.iterdir():
            if p.suffix.lower() == '.spe':
                spe = SPEFile(p)
                spe.save_erg_cal(erg_cal, shape_cal)

                print(f"Updated {msg} calibration for {p}")

    def save_erg_cal(self, erg_cal=None, shape_cal=None):
        def get_coeffs(a):
            if len(a) == 2:
                c0, c1 = a
                c2 = 0
            elif len(a) == 3:
                c0, c1, c2 = a
            else:
                raise ValueError(f"erg_cal/shape_cal not of correct length '{a}'")

            return c0, c1, c2

        assert self.path.suffix != '.pickle'

        if erg_cal is not None:
            c0, c1, c2 = get_coeffs(erg_cal)
            self._erg_calibration = [c0, c1, c2]
            erg_fit_line = f"{c0:.5e} {c1:.5e} {c2:.5e}".upper() + " keV\n"
        else:
            erg_fit_line = None

        if shape_cal is not None:
            c0, c1, c2 = get_coeffs(shape_cal)
            self.shape_cal = [c0, c1, c2]
            shape_fit_line = f"{c0:.5e} {c1:.5e} {c2:.5e}\n".upper()
        else:
            shape_fit_line = None

        found_shape_cal = False
        if self.path.suffix != '.pickle':
            with open(self.path) as f:
                lines = f.readlines()

            i = 0

            while i < len(lines):
                line = lines[i]
                if erg_cal is not None:
                    if "$ENER_FIT:" in line:
                        i += 1
                        lines[i] = erg_fit_line
                    elif "$MCA_CAL:" in line:
                        i += 1
                        lines[i] = '3\n'
                        i += 1
                        lines[i] = erg_fit_line

                if shape_cal is not None:
                    if "$SHAPE_CAL" in line:
                        found_shape_cal = True
                        i += 1
                        lines[i] = '3\n'
                        i += 1
                        lines[i] = shape_fit_line

                i += 1

            if (not found_shape_cal) and (shape_cal is not None):
                shape_lines = ["$SHAPE_CAL:\n", '3\n', shape_fit_line]
                if '$ENDRECORD:' in lines[-1]:
                    lines = lines[:-1] + shape_lines + ["$ENDRECORD:"]
                else:
                    lines = lines + shape_lines

            new_text = "".join(lines)

            _backup_file(self.path)

            with open(self.path, 'w') as f:
                f.write(new_text)

    def set_useful_energy_range(self, erg_min=None, erg_max=None):
        if erg_min is None:
            i0 = 0
        else:
            i0 = self.__erg_index__(erg_min)

        if erg_max is None:
            i1 = len(self.energies)
        else:
            i1 = self.__erg_index__(erg_max) + 1

        self.counts = self.counts[i0: i1]
        self.counts.flags.writeable = False

        self._energies = self.energies[i0: i1]
        self.channels = self.channels[i0: i1]
        self._erg_bins = None

        if self._effs is not None:
            self._effs = self.effs[i0: i1]

    def rebin(self, new_bins):
        """
        Rebins self.counts into `new_bins`.

        Args:
            new_bins:

        Returns:

        """
        new_counts = rebin(self.erg_bins, self.counts, new_bins)
        self.counts = new_counts
        self._erg_bins = new_bins
        self._energies = 0.5 * (new_bins[1:] + new_bins[:-1])
        return new_counts

    @property
    def pretty_realtime(self):
        return human_friendly_time(self.realtime)

    @property
    def pretty_livetime(self):
        return human_friendly_time(self.livetime)

    @property
    def deadtime_corr(self):
        return self.realtime/self.livetime

    @property
    def erg_calibration(self):
        return self._erg_calibration

    @erg_calibration.setter
    def erg_calibration(self, values):
        self._erg_calibration = values
        # if self._energies is not None:
        #     old_energies = self.energies
        # else:
        #     old_energies = None
        self._energies = None
        self._erg_bins = None
        self.recalc_effs()

    @property
    def erg_centers(self):
        return self.energies

    @property
    def energies(self):
        if self._energies is None:
            self._energies = self.channel_2_erg(self.channels + 0.5)
        return self._energies

    def pickle(self, f_path: Union[str, Path] = None):

        if f_path is None:
            f_path = self.path

        f_path = Path(f_path).with_suffix('.marshalSPe')

        d_simple = {'counts': list(map(float, unp.nominal_values(self.counts))), 'shape_cal': self.shape_cal,
                    'erg_units': self.erg_units,
                    '_erg_calibration': self._erg_calibration, 'channels': self.channels, 'livetime': self.livetime,
                    'realtime': self.realtime, 'system_start_time': self.system_start_time.strftime(SPEFile.time_format)
                    ,'description': self.description, 'maestro_version': self.maestro_version,
                    'device_serial_number': self.device_serial_number, 'detector_id': self.detector_id,
                    'path': str(self.path)}

        np_dtypes = {'channels': 'int'}

        with open(f_path, 'wb') as f:
            marshal.dump(d_simple, f)
            marshal.dump(np_dtypes, f)

    @classmethod
    def from_pickle(cls, f_path, load_erg_cal: Union[Path, bool] = None) -> SPEFile:
        f_path = Path(f_path).with_suffix('.marshalSpe')
        self = SPEFile.__new__(cls)
        super(SPEFile, self).__init__()  # run EfficiencyMixin init

        with open(f_path, 'rb') as f:
            d_simple = marshal.load(f)
            np_dtypes = marshal.load(f)
        for k, v in d_simple.items():
            if k == 'counts':
                v = unp.uarray(v, np.sqrt(v))
            if isinstance(v, bytes):
                if k in np_dtypes:
                    t = getattr(np, np_dtypes[k])
                else:
                    t = np.float
                v = np.frombuffer(v, t)
            setattr(self, k, v)

        self.path = Path(self.path)
        self.system_start_time = datetime.strptime(self.system_start_time, SPEFile.time_format)
        self.counts.flags.writeable = False
        self._energies = None
        self._erg_bins = None

        try:
            self.unpickle_efficiency(self.eff_path)
        except FileNotFoundError:
            pass

        return self

    @classmethod
    def build(cls, path, counts, erg_calibration: List[float], livetime, realtime, channels=None, erg_units='KeV',
              shape_cal=None, description=None, system_start_time=None, eff_path=None,
              load_eff_cal=True) -> SPEFile:
        """
        Build Spe file from arguments.
        Args:
            path:
            counts:
            erg_calibration:
            livetime:
            realtime:
            channels:
            erg_units:
            shape_cal:
            description:
            system_start_time:
            load_eff_cal:

        Returns:

        """
        self = SPEFile.__new__(cls)
        self.eff_path = eff_path
        super(SPEFile, self).__init__()  # run EfficiencyMixin init

        self._energies = None
        self._erg_bins = None

        if isinstance(counts[0], UFloat):
            self.counts = counts
        else:
            self.counts = unp.uarray(counts, np.sqrt(counts))
        self.counts.flags.writeable = False

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

        self.livetime = livetime
        self.realtime = realtime

        if system_start_time is None:
            system_start_time = datetime(year=1, month=1, day=1)
        self.system_start_time = system_start_time

        if description is None:
            description = ''
        self.description = description

        self.device_serial_number = 0
        self.detector_id = 0
        self.maestro_version = None

        if load_eff_cal:
            try:
                self.unpickle_efficiency(self.eff_path)
            except FileNotFoundError:
                pass

        return self

    @classmethod
    def from_lis(cls, path):
        from JSB_tools.maestro_reader import MaestroListFile
        l = MaestroListFile.from_pickle(path)
        return l.SPE

    def channel_2_erg(self, chs) -> np.ndarray:
        # x = chs - 0.5  # makes spectra agree with PeakEasy
        return np.sum([coeff * chs ** i for i, coeff in enumerate(self.erg_calibration)], axis=0)

    def erg_2_fwhm(self, erg):
        """
        Get the Full Width Half Max from the shape calibration.
        Todo: This doesnt tend to give good results. Figure this out.
        Args:
            erg:

        Returns:

        """
        if self.shape_cal is None:
            raise AttributeError(f'Spefile, "{self.path.name}" does not have `shape_cal` instance set.'
                                 'This is likely due to either there being no shape cal info in SPE ASCII file, '
                                 'or that this SPEfile instance was not built from a SPE ASCII text file.')
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
            return len(self.counts)
        if erg < self.erg_bins[0]:
            return 0
        return np.searchsorted(self.erg_bins, erg, side='right') - 1

    @property
    def erg_bin_widths(self):
        bins = self.erg_bins
        return np.array([b1-b0 for b0, b1 in zip(bins[:-1], bins[1:])])

    @property
    def chs_bins(self):
        return np.arange(len(self) + 1) - 0.5  # channel "bins" such that first bin is centered around channel=0

    @property
    def erg_bins(self):
        """
        Bin left edges for use in histograms.
        Returns:

        """
        if self._erg_bins is not None:
            return self._erg_bins
        else:
            chs_bins = self.chs_bins
            out = self.channel_2_erg(chs_bins)
            self._erg_bins = out
            return out

    def erg_bins_cut(self, erg_min, erg_max):
        """
        Get array of energy bins in range specified by arguments.
        Args:
            erg_min:
            erg_max:

        Returns:

        """
        i0 = self.__erg_index__(erg_min)
        i1 = self.__erg_index__(erg_max) + 1
        return self.erg_bins[i0: i1]

    @property
    def rates(self):
        return self.counts/self.livetime

    def get_counts(self, erg_min: float = None, erg_max: float = None, eff_corr=False, make_rate=False, remove_baseline=False,
                   make_density=False,
                   nominal_values=False,
                   deadtime_corr=False,
                   rebin=1,
                   baseline_method='root',
                   baseline_kwargs=None,
                   return_bin_edges=False,
                   return_background=False,
                   debug_plot=False):
        """
        Get energy spectrum within (optionally) a specified energy range.
        Args:
            erg_min: Min energy cut
            erg_max: Max energy cut
            eff_corr: If True, correct for efficiency.
            make_rate: If True, divide by livetime
            remove_baseline: Whether or not to remove baseline.
            make_density: If True, result is divided by bin width
            nominal_values: If false, return uncertain array, else don't
            deadtime_corr: If True, correct for deatime
            rebin: Combine bins into n bins
            baseline_method: Either 'root' or 'median'. If 'median', use rolling median technique. Else use
                ROOT.TSpectrum
            baseline_kwargs: kwargs passed to baseline remove function.
            return_bin_edges: If True, return bin edges (changes return signature)
            return_background: If True, return background (changes return signature)
            debug_plot: If not False, plot counts for debugging purposes. If axis instance, plot on that axis.


        Returns:
            counts # if return_background == return_bin_edges == False
            counts, bin_edges  # if return_background == False and return_bin_edges == True
            counts, back_ground  # if return_background == True and return_bin_edges == False
            counts, back_ground, bin_edges  # if return_background == return_bin_edges == True


        """
        bg = None
        if remove_baseline:
            if baseline_kwargs is None:
                baseline_kwargs = {}
            if baseline_method.lower() == 'root':
                bg = calc_background(self.counts, **baseline_kwargs)
            elif baseline_method.lower() == 'median':
                bg = self.get_baseline_median(**baseline_kwargs)
            else:
                assert False, f'Invalid `baseline_method` argument, "{baseline_method}"'
            counts = self.counts - bg
        else:
            if any(([make_rate, make_density, deadtime_corr, eff_corr])):  # don't modify self.counts
                counts = self.counts.copy()
            else:
                counts = self.counts

        if not all(x is None for x in [erg_min, erg_min]):
            if erg_min is None:
                erg_min = self.erg_bins[0]
            if erg_max is None:
                erg_max = self.erg_bins[-1]
            imin = self.__erg_index__(erg_min)
            imax = self.__erg_index__(erg_max)
            bins = self.erg_bins[imin: imax+1]
            counts = counts[imin: imax]
            if bg is not None:
                bg = bg[imin: imax]
            b_widths = self.erg_bin_widths[imin: imax]
        else:
            imin = 0
            imax = len(self.counts)
            bins = self.erg_bins
            b_widths = self.erg_bin_widths

        if nominal_values:
            out = unp.nominal_values(counts)
        else:
            out = counts.copy()

        if rebin != 1:
            if bg is not None:
                bg, _ = _rebin(rebin, bg, bins)
            out, bins = _rebin(rebin, out, bins)
            b_widths = bins[1:] - bins[:-1]

        if eff_corr:
            assert self.effs is not None, f'No efficiency information set for\n{self.path}'
            ar = self.effs[imin: imax]

            if nominal_values:
                ar = unp.nominal_values(ar)
            # print(f"E = {0.5*(erg_min + erg_max)}, raw counts: {sum(counts)}")
            # print(f"eff = {np.mean(unp.nominal_values(ar))} +/- {np.std(unp.nominal_values(ar))}")
            # print(f"Before: {sum(out)}")
            if out.dtype != ar.dtype:
                ar = ar.astype(out.dtype)

            out /= np.where(ar > 0, ar, 1)
            # print(f"After: {sum(out)}\n")

        if make_rate:
            out /= self.livetime
        if make_density:
            out /= b_widths
        if deadtime_corr:
            out *= self.deadtime_corr

        if debug_plot is not False:
            debug_ax2 = None
            if isinstance(debug_plot, Axes):
                debug_ax1 = debug_plot
                fig = plt.gcf()
            else:
                if bg is not None:
                    fig, debug_axs = plt.subplots(1, 2, sharex='all')
                    debug_ax1, debug_ax2 = debug_axs
                else:
                    fig, debug_axs = plt.subplots(1, 1)
                    debug_ax1 = debug_axs

            extra_range = 20
            _label = 'counts'
            if make_rate:
                _label += '/s'
            if make_density:
                _label += '/KeV'

            debug_counts, debug_bg, debug_bins = \
                self.get_counts(erg_min=erg_min-extra_range, erg_max=erg_max+extra_range,
                                make_rate=make_rate,
                                remove_baseline=remove_baseline, make_density=make_density,
                                nominal_values=nominal_values, deadtime_corr=deadtime_corr,
                                rebin=rebin,
                                baseline_method=baseline_method,
                                baseline_kwargs=baseline_kwargs,
                                return_bin_edges=True,
                                return_background=True,
                                eff_corr=eff_corr)
            # debug_bins = self.erg_bins_cut(erg_min - extra_range, erg_max + extra_range)
            mpl_hist(debug_bins, debug_counts, ax=debug_ax1, label='Sig.')
            if debug_ax2 is not None:
                mpl_hist(debug_bins, debug_bg, label='Bg.', ax=debug_ax2)
                mpl_hist(debug_bins, debug_bg + debug_counts, label='Bg. + Sig.', ax=debug_ax2)
                debug_ax2.legend()

            shade_plot(debug_ax1, [erg_min, erg_max], label='Counts range', alpha=0.2)
            debug_ax1.set_xlabel('[KeV]')
            debug_ax1.set_ylabel(f'[{_label}]')
            fig.suptitle(f'"({self.path.name}).get_counts(); integral= {sum(out)}" debug plot')
            debug_ax1.legend()

        if return_bin_edges or return_background:
            out = [out]
            if return_background:
                out += [bg]
            if return_bin_edges:
                out += [bins]
            return tuple(out)
        else:
            return out

    def plot_erg_spectrum(self, erg_min: float = None, erg_max: float = None, ax=None, rebin=1, eff_corr=False,
                          leg_label=None, make_rate=False, remove_baseline=False, make_density=False,
                          scale=1, nominal_values=False, **ax_kwargs):
        """
        Plot energy spectrum within (optionally) a specified energy range.
        Args:
            erg_min:
            erg_max:
            ax:
            rebin: Combine bins into groups of n for better stats.
            eff_corr:
            leg_label:
            make_rate: If True, divide by livetime
            remove_baseline: Is True, remove baseline
            make_density: y units will be counts/unit energy
            scale: Arbitrary scaling constant

        Returns:

        """
        if ax is None:
            plt.figure()
            ax = plt.gca()

        counts, bins = self.get_counts(erg_min=erg_min, erg_max=erg_max, make_rate=make_rate, eff_corr=eff_corr,
                                       remove_baseline=remove_baseline, rebin=rebin, make_density=make_density,
                                       return_bin_edges=True, nominal_values=nominal_values)

        if not isinstance(scale, (int, float)) or scale != 1:
            counts *= scale

        label=ax_kwargs.pop('label', leg_label)
        mpl_hist(bins, counts, ax=ax, label=label, **ax_kwargs)
        ylabel = 'Counts'
        if make_rate:
            ylabel += '/s'
        if make_density:
            ylabel += '/KeV'

        ax.set_ylabel(ylabel)

        ax.set_xlabel('Energy [KeV]')
        ax.set_title(self.path.name)

        return ax

    @property
    def nominal_counts(self) -> np.ndarray:
        return unp.nominal_values(self.counts)

    # def get_baseline_ROOT(self, num_iterations=20, clipping_window_order=2, smoothening_order=5) -> np.ndarray:
    #     return calc_background(self.counts, num_iterations=num_iterations, clipping_window_order=clipping_window_order,
    #                            smoothening_order=smoothening_order)

    def get_baseline_median(self, erg_min: float = None, erg_max: float = None, window_kev=30):
        """
        Calculate the median withing a rolling window of width `window_width`. Generally good at estimating baseline.
        Args:
            erg_min: Min energy cut
            erg_max: Max energy cut
            window_kev: Size of rolling median window in KeV

        Returns:

        """
        if erg_min is None:
            erg_min = self.erg_bins[0]
        if erg_max is None:
            erg_max = self.erg_bins[-1]
        center = (erg_min + erg_max)/2
        window_width = self.__erg_index__(center + window_kev / 2) - self.__erg_index__(center - window_kev / 2)
        _slice = slice(self.__erg_index__(erg_min), self.__erg_index__(erg_max))
        bg_est = rolling_median(window_width=window_width, values=self.counts)[_slice]
        return bg_est

    def get_baseline_removed(self, method='ROOT', **kwargs) -> np.ndarray:
        """

        Args:
            method: Either 'ROOT' or 'median'.
                The former used ROOT.TSpectrum, with the following optional kwargs:
                    num_iterations=20
                    clipping_window_order=2
                    smoothening_order=5
                The latter will calculate the rolling median with the following keyword argumets:
                    window_kev=30

            kwargs: Keyword arguments to pass to baseline estimation method. See above.


        Returns:

        """
        if method.lower() == 'root':
            _valid_kwargs = ['num_iterations', 'clipping_window_order', 'smoothening_order']
            assert all(x in _valid_kwargs for x in kwargs), \
                f'Invalid kwargs for method "ROOT": {[x for x in kwargs if x not in _valid_kwargs]}'
            return self.counts - self.get_baseline_ROOT(**kwargs)
        elif method.lower() == 'median':
            _valid_kwargs = ['window_kev']
            assert all(x in _valid_kwargs for x in kwargs), \
                f'Invalid kwargs for method "ROOT": {[x for x in kwargs if x not in _valid_kwargs]}'
            return self.counts - self.get_baseline_median(**kwargs)
        else:
            raise TypeError(f'Invalid method, "{method}". Valid methods are "ROOT" or "median"')

    def multi_peak_fit(self, centers: List[float], baseline_method='ROOT', baseline_kwargs=None,
                       fit_window: float = None, eff_corr=False, deadtime_corr=True, debug_plot=False):
        """
        Fit one or more peaks in a close vicinity.
        Args:
            centers: List of centers of peaks of interest. Doesn't have to be exact, but the closer the better.
            baseline_method: either 'ROOT' or 'median'
            baseline_kwargs: Arguments send to calc_background() or rolling_median() (See JSB_tools.__init__)
            fit_window: A window that should at least encompass the peaks (single number in KeV).
            eff_corr: If True, correct for efficiency
            deadtime_corr: If True, correct for deadtime.
            debug_plot: Produce an informative plot.

        Returns:

        """
        model = None
        params = None

        y = self.counts.copy()
        if deadtime_corr:
            y = y * self.deadtime_corr

        if eff_corr is True:
            y /= np.where(self.effs > 0, self.effs, 1)

        if baseline_kwargs is None:
            baseline_kwargs = {}
        baseline_method = baseline_method.lower()

        if baseline_method == 'root':
            baseline = calc_background(y, **baseline_kwargs)
        elif baseline_method == 'median':
            if 'window_width_kev' in baseline_kwargs:
                _window_width = baseline_kwargs['window_width_kev']
            else:
                _window_width = 30
            _window_width /= self.erg_bin_widths[len(self.erg_bins)//2]
            baseline = rolling_median(values=y, window_width=_window_width)
        else:
            raise TypeError(f"Invalid `baseline_method`: '{baseline_method}'")

        y -= baseline

        centers_idx = list(sorted(map(self.__erg_index__, centers)))
        _center = int((centers_idx[0] + centers_idx[-1])/2)
        _bin_width = self.erg_bin_widths[_center]

        if fit_window is None:
            if len(centers) > 1:
                fit_window = 1.5*max([max(centers)-min(centers)])
                if fit_window*_bin_width < 10:
                    fit_window = 10/_bin_width
            else:
                fit_window = 10/_bin_width

        _slice = slice(int(max([0, _center - fit_window//2])), int(min([len(y)-1, _center+fit_window//2])))
        y = y[_slice]
        x = self.energies[_slice]
        plt.figure()
        density_sale = self.erg_bin_widths[_slice]  # array to divide by bin widths.
        y /= density_sale  # make density

        peaks, peak_infos = find_peaks(unp.nominal_values(y), height=unp.std_devs(y), width=0)

        select_peak_ixs = np.argmin(np.array([np.abs(c-np.searchsorted(x, centers)) for c in peaks]).T, axis=1)
        peak_widths = peak_infos['widths'][select_peak_ixs]*self.erg_bin_widths[_center]
        amplitude_guesses = peak_infos['peak_heights'][select_peak_ixs]*peak_widths
        sigma_guesses = peak_widths/2.355

        for i, erg in enumerate(centers):
            m = GaussianModel(prefix=f'_{i}')
            # erg = extrema_centers[np.argmin(np.abs(erg-extrema_centers))]
            if model is None:
                params = m.make_params()
                params[f'_{i}center'].set(value=erg)
                model = m
            else:
                model += m
                params.update(m.make_params())

            params[f'_{i}amplitude'].set(value=amplitude_guesses[i], min=0)
            params[f'_{i}center'].set(value=erg)
            params[f'_{i}sigma'].set(value=sigma_guesses[i])

        weights = unp.std_devs(y)
        weights = np.where(weights>0, weights, 1)
        weights = 1.0/weights

        fit_result = model.fit(data=unp.nominal_values(y), x=x, weights=weights, params=params)

        if debug_plot:
            ax = mpl_hist(self.erg_bins[_slice.start: _slice.stop + 1], y*density_sale, label='Observed')
            _xs_upsampled = np.linspace(x[0], x[-1], 5*len(x))
            density_sale_upsampled = density_sale[np.searchsorted(x, _xs_upsampled)]
            model_ys = fit_result.eval(x=_xs_upsampled, params=fit_result.params)*density_sale_upsampled
            model_errors = fit_result.eval_uncertainty(x=_xs_upsampled, params=fit_result.params)*density_sale_upsampled
            ax.plot(_xs_upsampled, model_ys, label='Model')
            ax.fill_between(_xs_upsampled, model_ys-model_errors, model_ys+model_errors, alpha=0.5, label='Model error')
            ax.legend()
            ax.set_ylabel("Counts")
            ax.set_xlabel("Energy")
            for i in range(len(centers)):
                amp = ufloat(fit_result.params[f'_{i}amplitude'].value, fit_result.params[f'_{i}amplitude'].stderr)
                _x = fit_result.params[f'_{i}center'].value
                _y = model_ys[np.searchsorted(_xs_upsampled, _x)]
                ax.text(_x, _y*1.05, f'N={amp:.2e}')
            ax.set_title(self.path.name)

        return fit_result

    def __len__(self):
        return len(self.counts)

    def interactive_plot(self, nominal_values=True, channels=False, debug=False, scale=1):
        if channels:
            bins = self.chs_bins
        else:
            bins = self.erg_bins
        counts = self.get_counts(make_density=True, nominal_values=nominal_values) * scale
        iplot = InteractivePlot(bins, counts, debug=debug)

        iplot.fig.suptitle(self.path.name)

        if not hasattr(self, 'iplots'):
            setattr(self, 'iplots', [])

        getattr(self, 'iplots').append(iplot)

        return iplot

    def erg_cal_plot(self, fit_window=3):
        def format_coord(ch, yp):
            erg = self.channel_2_erg(ch)
            return f"ch={ch:.1f} erg={erg:.1f};  y={yp:.2g}"

        fig, ax = plt.subplots(1, 1, sharex="all")
        ax.format_coord = format_coord

        chs_bins = np.arange(len(self.erg_bins))
        # chs = 0.5 * (chs_bins[1:] + chs_bins[:-1])

        counts = self.get_counts(make_density=True)
        iplot = InteractivePlot(chs_bins, counts, ax=ax, fit_window=fit_window)
        ax2 = ax.twiny()

        ax2.set_zorder(ax.get_zorder() - 1)

        ax2.plot(self.energies, unp.nominal_values(counts), alpha=1)

        ax2.set_xlabel("Energy")
        ax.set_xlabel("Channel")

        if not hasattr(self, 'iplots'):
            setattr(self, 'iplots', [])

        getattr(self, 'iplots').append(iplot)

        return iplot


class ErgCalHelper:
    def __init__(self, spe: SPEFile, cal_nuclide_name=None, list_of_gammas=None, list_of_ergs=None, intensity_limit=0.05, min_peak_dist=8):
        """

        Args:
            cal_nuclide_name:
            list_of_gammas:
            spe:
            intensity_limit:
            min_peak_dist:
        """
        if list_of_gammas is None and cal_nuclide_name is None and list_of_ergs is None:
            raise ValueError("Need either a list of GammaLine instances, a calibration nuclide name, or list of energies")

        if list_of_gammas is None and list_of_ergs is not None:
            list_of_gammas = []
            for erg in list_of_ergs:
                g = GammaLine.__new__(GammaLine)
                g.erg = ufloat(erg, 0)
                g.intensity = ufloat(1, 0)
                list_of_gammas.append(g)

        if cal_nuclide_name is None:
            gamma_lines = list_of_gammas
            intensity_limit = None
            self.nuclide = None
        else:
            gamma_lines = Nuclide(cal_nuclide_name).decay_gamma_lines
            self.nuclide = Nuclide(cal_nuclide_name)

        self.spe = spe

        self.fits: List[GausFitResult] = []

        self.fit_centers = []
        self.adjacent_peaks = []
        self.gamma_lines: List[GammaLine] = []

        self.run_fits(gamma_lines, intensity_limit=intensity_limit, min_peak_dist=min_peak_dist)

        self.erg_cal = [0, 0, 0]
        self.shape_cal = [0, 0, 0]

    def run_fits(self, gamma_lines, intensity_limit=0.05, min_peak_dist=8):
        def is_interfere(e, intnsty):
            if e == erg:
                return False

            if abs(e - erg) < min_peak_dist:
                if intnsty / intensity < 0.05:
                    return False
                return True

        if isinstance(gamma_lines[0], GammaLine):
            gamma_lines = [g for g in gamma_lines]
            gamma_intensities = [g.intensity.n for g in gamma_lines]
            gamma_ergs = [g.erg.n for g in gamma_lines]
        else:
            gamma_ergs = gamma_lines
            gamma_intensities = gamma_lines = [None]*len(gamma_ergs)

        for gline, erg, intensity in zip(gamma_lines, gamma_ergs, gamma_intensities):
            if intensity_limit is not None and intensity < intensity_limit:
                continue

            self.fit_centers.append(erg)
            self.gamma_lines.append(gline)

            adjacent_peaks = []

            for ee, ii in zip(gamma_ergs, gamma_intensities):
                if is_interfere(ee, ii):
                    adjacent_peaks.append(ee)

            center_guesses = [erg] + adjacent_peaks
            fix_centers = [False] + [True] * len(adjacent_peaks)

            # if channelsQ:
            #     center_guesses = [spe.erg_2_channel(c) for c in center_guesses]
            #     bins = np.arange(len(spe.erg_bins))
            # else:
            bins = self.spe.erg_bins

            fit = multi_guass_fit(bins, self.spe.counts / self.spe.erg_bin_widths,
                                  center_guesses=center_guesses,
                                  fixed_in_binQ=fix_centers,
                                  fit_buffer_window=15)

            self.fits.append(fit)

        self.erg_cal = [0, 0, 0]
        self.shape_cal = [0, 0, 0]

    def plot(self,):
        if not sum(self.erg_cal) == 0:
            self.spe._erg_calibration = self.erg_cal

        iplot = self.spe.interactive_plot()
        ys = self.spe.get_counts(make_density=True, nominal_values=True)
        for g in self.gamma_lines:
            x = g.erg.n
            iplot.ax.axvline(x, color='black', lw=1.5, alpha=0.5)

            y = ys[self.spe.__erg_index__(g.erg.n)]
            iplot.ax.text(x, y*1.05, f'{g.parent_nuclide.name} - {g.erg.n:.2f} keV', rotation=90)

    def __len__(self):
        return len(self.fits)

    def get_counts(self):
        return [fit.amplitudes(0) for fit in self.fits]

    def plot_fits(self):
        tab = TabPlot()
        for i in range(len(self)):
            fit = self.fits[i]
            center = fit.centers(0)
            erg = self.gamma_lines[i].erg.n
            ax = tab.new_ax(i)
            fit.plot(ax=ax)
            ax.text(0.8, 0.8, f'E={erg:.2f}\nCh.={center:.2f}\n'
                              f'fwhm={fit.sigmas(0):.2g}', transform=ax.transAxes)

    def get_ch(self, erg):
        return np.searchsorted(self.spe.erg_bins, erg, side='right') - 1

    def get_fit(self, indices=None, degree=1):
        ergs = []
        chs = []
        fwhms = []
        fig, fit_plt_axs, = plt.subplots(2, 2)
        fit_plt_axs = fit_plt_axs.flatten()

        fit_plt_axs[0].set_xlabel('Channel')
        fit_plt_axs[1].set_xlabel('~ Energy [keV]')
        fit_plt_axs[1].set_ylabel('Fwhm [keV]')
        fit_plt_axs[0].set_ylabel('Energy [keV]')
        fit_plt_axs[2].set_xlabel('Energy')
        fit_plt_axs[2].set_ylabel('Energy error')
        fit_plt_axs[3].set_axis_off()

        handles, labels = [], []

        for i in range(len(self)):
            if indices is not None and i not in indices:
                continue

            fit = self.fits[i]
            sigma = fit.sigmas(0)

            center_erg = fit.centers()[0]
            rel_err = center_erg.std_dev/center_erg.n

            center_ch = self.get_ch(center_erg.n)
            center_ch = ufloat(center_ch, rel_err * center_ch)

            erg = self.gamma_lines[i].erg.n

            fwhms.append(2.355 * sigma)
            ergs.append(erg)
            chs.append(center_ch)

            handle = fit_plt_axs[0].errorbar([center_ch.n], [erg], xerr=[center_ch.std_dev], ls='None', marker='o')

            handles.append(handle)
            labels.append(f'{i}')

        fig.legend(handles, labels)

        model = PolynomialModel(degree=degree)

        y = ergs
        x = unp.nominal_values(chs)
        params = model.guess(data=y, x=x)
        fit_erg = model.fit(data=y, x=x, params=params)

        self.erg_cal[0] = fit_erg.params['c0'].value
        self.erg_cal[1] = fit_erg.params['c1'].value

        plt_x = np.linspace(min(x), max(x), 1000)
        fit_plt_axs[0].plot(plt_x, fit_erg.eval(x=plt_x))

        fit_plt_axs[0].text(0.1, 0.6, fr"E = ch $\times$ {fit_erg.params['c1'].value:.5e} + {fit_erg.params['c1'].value:4e}")

        fit_plt_axs[2].plot(y, y - fit_erg.eval(x=x), ls='None', marker='o')

        for i in range(len(ergs)):
            erg = ergs[i]
            fwhm = fwhms[i] * fit_erg.params['c1'].value
            fit_plt_axs[1].errorbar([erg], [fwhm.n], yerr=[fwhm.std_dev], ls='None', marker='o')

        weights = 1/np.where(unp.std_devs(fwhms) > 0, unp.std_devs(fwhms), 1)
        x = ergs
        # print(fit_erg.params['c1'].value)
        y = unp.nominal_values(fwhms) * fit_erg.params['c1'].value
        params = model.guess(y, x=x)
        fit_fwhm = model.fit(data=y, x=x, weights=weights, params=params)

        # plt_x = np.linspace(min(x), max(x), 1000)
        fit_plt_axs[1].plot(plt_x, fit_fwhm.eval(x=plt_x))

        self.shape_cal[0] = fit_fwhm.params['c0'].value
        self.shape_cal[1] = fit_fwhm.params['c1'].value

    def save_cal(self, entire_directory=False):
        if not entire_directory:
            self.spe.save_erg_cal(self.erg_cal, self.shape_cal)
        else:
            self.spe.save_erg_call_all(self.spe.path.parent, self.erg_cal, self.shape_cal)

if __name__ == '__main__':
    from scipy.stats.mstats import winsorize
    from JSB_tools.MCNP_helper.outp_reader import OutP
    from JSB_tools.nuke_data_tools import Nuclide


    spe = SPEFile("/Users/burgjs/PycharmProjects/miscMCNP/detectorModels/GRETA0/cal_2024/BG_long.Spe")

    peaks = [237.2, 350.4, 608.68, 2614.5]

    spe.plot_erg_spectrum()
    spe.erg_bins
    spe.energies
    cal = ErgCalHelper(spe, list_of_gammas=peaks)

    cal.get_fit()

    spe.interactive_plot()
    plt.show()
