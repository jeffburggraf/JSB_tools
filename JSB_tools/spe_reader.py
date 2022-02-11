from __future__ import annotations
import pickle
import time
import warnings
from pathlib import Path
from datetime import datetime
import re

import lmfit
from matplotlib import pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import UFloat, ufloat
from uncertainties.core import AffineScalarFunc
from JSB_tools import Nuclide, mpl_hist, calc_background, human_friendly_time, rolling_median, shade_plot
from typing import List, Tuple, Union, Dict
import marshal
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
from matplotlib.axes import Axes
from scipy.signal import argrelextrema


def _rebin(rebin, values, bins):
    if rebin != 1:
        l = len(values)
        out = np.sum([values[rebin * i: rebin * (i + 1)] for i in range(l // rebin)], axis=1)
        bins = np.array([bins[rebin * i] for i in range(l // rebin + 1)])
    else:
        out = values

    return out, bins


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
                i += 1
            if lines[i] == '$SPEC_REM:\n':
                i += 1
                out['detector_id'] = int(lines[i][5:])
                i += 1
            if lines[i][:8] == 'DETDESC#':
                out['device_serial_number'] = lines[i][8:].rstrip().lstrip()
                i += 1
            if 'Maestro Version' in lines[i]:
                out['maestro_version'] = lines[i].split()[-1]
                i += 1
            if lines[i] == '$DATE_MEA:\n':
                i += 1
                try:
                    out['system_start_time'] = datetime.strptime(lines[i].rstrip(),
                                                             SPEFile.time_format)
                except ValueError:
                    warnings.warn(f"Spe file, '{path}' contains invalid date format:\n{lines[i]}")
                    out['system_start_time'] = datetime(1, 1, 1)
                i += 1
            if lines[i] == '$MEAS_TIM:\n':
                i += 1
                try:
                    out['livetime'], out['realtime'] = map(int, lines[i].split())
                except ValueError:
                    raise ValueError(
                        f"Livetime and realtime did not appear in correct format after '$MEAS_TIM:'. in '{path}'\n"
                          "What I expected, e.g.,\n"
                          "\t$MEAS_TIM:\n\t220 230\n"
                          "What I got,\n"
                          f"\t$MEAS_TIM:\n\t{lines[i]}\n"
                          "There is a problem in Spe file.")
                i += 1
            if lines[i] == '$DATA:\n':
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
            counts.append(int(lines[i]))
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
        warnings.warn(f"Invalid shape calibration in Spe file '{path}'\n{e}")
        out['shape_cal'] = None

    return out


class EnergyCalMixin:
    """
    Mixin for MaestroListFile and SPEFile that allows to override energy calibration by loading pickled values.
    For this to work, you must first do self.save_erg_cal

    This is useful to make sure that enhanced calibration coeffs don't get over-ridden from self.pickle
    """
    def __init__(self, erg_calibration, load_erg_cal=None):
        """
        If load_erg_cal is None, look for energy calibration automatically with appropriate name and override
            `erg_calibration` argument.
        if load_erg_cal is a Path or str, look for erg cal in path
        If load_erg_cal is False, use erg cal from file.
        Args:
            erg_calibration:
            load_erg_cal:
        """

        self._erg_calibration = erg_calibration
        if load_erg_cal is not False:
            if isinstance(load_erg_cal, (str, Path)):
                load_erg_cal = Path(load_erg_cal)
            else:
                load_erg_cal = None
            self.load_erg_cal(load_erg_cal)

        self._load_erg_cal = load_erg_cal

    def __erg_cal_path__(self, other_path=None) -> Path:
        """
        Returns energy calibration path
        Args:
            other_path:

        Returns:

        """
        if other_path is not None:
            path = other_path
        else:
            path = getattr(self, 'path', None)
            if path is None:
                raise FileNotFoundError

        eff_dir = path.parent / 'cal_files'
        eff_dir.mkdir(exist_ok=True)
        return eff_dir / path.with_suffix('.erg').name

    def load_erg_cal(self, path=None):
        try:
            with open(self.__erg_cal_path__(path), 'rb') as f:
                self._erg_calibration = pickle.load(f)
        except FileNotFoundError:
            return

    def save_erg_cal(self, path=None):
        with open(self.__erg_cal_path__(path), 'wb') as f:
            pickle.dump(self._erg_calibration, f)


class EfficiencyCalMixin:
    """
    Mixin class for storing and maneging spectra's efficiency calibrations.

    Each Spe/List file has one associated with it. By default, all fields are None (efficiency is unity).

    There are two representations of efficiency.
        1. An lmfit.model.ModelResult object
            (stored in self.eff_model)
        2. A list of efficiencies with length equal to the number of channels
            (stored in self.effs)

    Only one representation can be used at a time, and this is enforced in the code.

    When using representation 1, calls to self.effs will evaluate self.eff_model.
    When using representation 2, self.eff_model is None and all efficiency information is accessible
        via self.effs

    There is an option to set a constant scale of the efficiency by changing
        the value of self.eff_scale to something other than 1.0 (this works when using either representation).

    Efficiency information can be pickled and unpickled. This is handled automatically when pickleing either a
        list file or an Spe file.

    """
    def __init__(self):
        self._effs = None
        self._eff_model: lmfit.model.ModelResult = None
        self.eff_scale = 1

    def interp_eff(self, new_energies):
        eff = np.interp(new_energies, self.erg_centers, unp.nominal_values(self.effs))
        eff_errs = np.interp(new_energies, self.erg_centers, unp.std_devs(self.effs))
        return unp.uarray(eff, eff_errs)

    def __eff_path__(self, other_path=None) -> Path:
        if other_path is not None:
            path = other_path
        else:
            assert hasattr(self, 'path') and self.path is not None, 'No self.path instance! Cannot pickle the ' \
                                                                    'efficiency.'
            path = self.path
            if path is None:
                raise FileNotFoundError

        eff_dir = path.parent / 'cal_files'
        if not eff_dir.exists():
            eff_dir.mkdir(exist_ok=True)
        return eff_dir / path.with_suffix('.eff').name

    def unpickle_eff(self):
        """
        Unpickle from auto-determined path. If no file exists, set attribs to defaults.
        Returns:

        """
        try:
            path = self.__eff_path__()
            with open(path, 'rb') as f:
                # out = EfficiencyCalMixin.__new__(EfficiencyCalMixin)
                d = pickle.load(f)
                for k, v in d.items():
                    setattr(self, k, v)

        except FileNotFoundError:
            pass
        except Exception:
            warnings.warn("Could not unpickle efficiency")
            raise

    def pickle_eff(self, path=None):
        path = self.__eff_path__(path)
        if self.effs is self.eff_model is None:  # no efficiency data. Delete old if it exists.
            path.unlink(missing_ok=True)
        else:
            d = {'effs': self.effs, 'eff_model': self.eff_model, 'eff_scale': self.eff_scale}
            with open(path, 'wb') as f:
                pickle.dump(d, f)

    @property
    def effs(self):
        if self._effs is not None:
            return self.eff_scale*self._effs
        else:
            return None

    @effs.setter
    def effs(self, value):
        self.eff_model = None
        self._effs = value

    @property
    def eff_model(self):
        return self._eff_model

    @eff_model.setter
    def eff_model(self, value: lmfit.model.ModelResult):
        self._eff_model = value
        if isinstance(value, lmfit.model.ModelResult):
            self.recalc_effs()

    def recalc_effs(self, old_erg_centers=None, new_erg_centers=None):
        """
        Recalculate for new energies of parent class, *or* if old_energies and new_energies are specified, recalculate
          efficiency points (and delete the model if it exists).
        Args:
            old_erg_centers:
            new_erg_centers:

        Returns:

        """
        # if not hasattr(self, '_eff_model'):
        #     self._eff_model = None
        # if not hasattr(self, 'effs'):
        #     self.effs = None
        if self.eff_model is not None:
            eff = self.eff_model.eval(x=self.erg_centers)
            eff = np.where(eff > 1E-6, eff, 1E-6)
            eff_err = self.eff_model.eval_uncertainty(x=self.erg_centers)
            self.effs = unp.uarray(eff, eff_err)
        else:
            if self.effs is not None:
                assert old_erg_centers is not new_erg_centers is not None, "Must supply these args"
                if isinstance(self.effs[0], UFloat):
                    self.effs = unp.uarray(np.interp(new_erg_centers, old_erg_centers, unp.nominal_values(self.effs)),
                                           np.interp(new_erg_centers, old_erg_centers, unp.std_devs(self.effs)))
                else:
                    self.effs = np.interp(new_erg_centers, old_erg_centers, self.effs)

    def print_eff(self):
        if self.eff_model is not None:
            return self.eff_model.fit_report()
        else:
            return self.effs.__repr__()

    def plot_efficiency(self, ax=None, **mpl_kwargs):
        assert self.effs is not None, 'No efficiency to plot. '

        if ax is None:
            fig, ax = plt.subplots()

        ls = mpl_kwargs.pop('ls', None)
        alpha = mpl_kwargs.pop('alpha', 0.35)

        _y = unp.nominal_values(self.effs)
        yerr = unp.std_devs(self.effs)
        label = mpl_kwargs.pop('label', None)

        lines = ax.plot(self.erg_centers, _y, ls=ls, **mpl_kwargs, label=label)
        c = lines[0].get_color()
        ax.fill_between(self.erg_centers, _y - yerr, _y + yerr, alpha=alpha, **mpl_kwargs)

        if self.eff_model is not None:
            x_points = self.eff_scale*self.eff_model.userkws[self.eff_model.model.independent_vars[0]]
            y_points = self.eff_scale*self.eff_model.data

            if self.eff_model.weights is not None:
                yerr = 1.0 / self.eff_model.weights
            else:
                yerr = np.zeros_like(x_points)
            yerr = self.eff_scale*yerr

            ax.errorbar(x_points, y_points, yerr, ls='None', marker='o', c=c)
        ax.set_xlabel("Energy KeV]")
        ax.set_ylabel("Efficiency")
        if label is not None:
            ax.legend()
        return ax


class SPEFile(EfficiencyCalMixin, EnergyCalMixin):
    """
    Used for processing and saving ORTEC ASCII SPE files. Can also build from other data using SPEFile.build(...).

    """
    time_format = '%m/%d/%Y %H:%M:%S'

    def __init__(self, path):
        """
        Analyse ORTEC .Spe ASCII spectra files.

        Args:
            path: Path to Spe file.
        """
        self.path = Path(path)
        assert self.path.exists(), self.path
        super(SPEFile, self).__init__()

        self._energies = None  # if None, this signals to the energies property that energies must be calculated.
        self._erg_bins = None  # same as above

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

        # self._erg_calibration = data['_erg_calibration']
        super(EfficiencyCalMixin, self).__init__(data['_erg_calibration'], load_erg_cal=False)
        self.erg_units = data['erg_units']
        self.shape_cal = data['shape_cal']

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
        if self._energies is not None:
            old_energies = self.energies
        else:
            old_energies = None
        self._energies = None
        self._erg_bins = None
        self.recalc_effs(old_energies, self.energies)

    @property
    def erg_centers(self):
        return self.energies

    @property
    def energies(self):
        if self._energies is None:
            self._energies = self.channel_2_erg(self.channels)
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
        self.unpickle_eff()
        # if erg_cal_path is not None:
        #     self.load_erg_cal(erg_cal_path)
        super(EfficiencyCalMixin, self).__init__(self.erg_calibration, load_erg_cal)

        return self

    @classmethod
    def build(cls, path, counts, erg_calibration: List[float], livetime, realtime, channels=None, erg_units='KeV',
              shape_cal=None, description=None, system_start_time=None, eff_model=None, effs=None, eff_scale=1,
              load_erg_cal: Union[Path, bool] = None, load_eff_cal=True) -> SPEFile:
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
            eff_model:
            effs:
            eff_scale:
            load_erg_cal:
            load_eff_cal:

        Returns:

        """
        self = SPEFile.__new__(cls)
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

        # self.erg_calibration = list(erg_calibration)
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

        self.effs = effs
        self.eff_model = eff_model
        self.eff_scale = eff_scale

        super(EfficiencyCalMixin, self).__init__(erg_calibration, load_erg_cal)
        if load_eff_cal:
            if self.path is not None:
                self.unpickle_eff()

        return self

    @classmethod
    def from_lis(cls, path):
        from JSB_tools.list_reader import MaestroListFile
        l = MaestroListFile.from_pickle(path)
        return l.SPE

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
    def erg_bins(self):
        """
        Bin left edges for use in histograms.
        Returns:

        """
        if self._erg_bins is not None:
            return self._erg_bins
        else:
            chs = np.concatenate([self.channels, [self.channels[-1]+1]])
            out = self.channel_2_erg(chs-0.5)
            self._erg_bins = out
            return out

    @property
    def eff_weights(self):
        assert self.effs is not None
        return 1.0/self.effs

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
            out = counts

        if rebin != 1:
            if bg is not None:
                bg, _ = _rebin(rebin, bg, bins)
            out, bins = _rebin(rebin, out, bins)
            b_widths = bins[1:] - bins[:-1]

        if eff_corr:
            assert self.effs is not None, f'No efficiency information set for\n{self.path}'
            out /= self.effs[imin: imax]
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
                          scale=1, **ax_kwargs):
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
                                       return_bin_edges=True)

        if not isinstance(scale, (int, float)) or scale != 1:
            counts *= scale

        mpl_hist(bins, counts, ax=ax, label=leg_label, **ax_kwargs)
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

    def get_baseline_ROOT(self, num_iterations=20, clipping_window_order=2, smoothening_order=5) -> np.ndarray:
        return calc_background(self.counts, num_iterations=num_iterations, clipping_window_order=clipping_window_order,
                               smoothening_order=smoothening_order)

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
            y /= self.effs

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




if __name__ == '__main__':
    from scipy.stats.mstats import winsorize
    from JSB_tools.MCNP_helper.outp_reader import OutP
    outp = OutP('/Users/burggraf1/PycharmProjects/IACExperiment/mcnp/sims/du_shot131/outp')
    tally = outp.get_f4_tally('Active up')

    products = []

    for n in [Nuclide.from_symbol('Ni58'), Nuclide.from_symbol('Ni60')]:
        for k, v in n.get_incident_gamma_daughters('all').items():
            y = np.average(v.xs.interp(tally.energies), weights=tally.fluxes)
            y *= 0.5**(10/v.half_life) - 0.5**(300/v.half_life)
            products.append([y, v])
    products = list(sorted(products, key=lambda x: -x[0]))
    yields = np.array([p[0] for p in products])
    yields /= max(yields)
    products = [p[1] for p in products]

    for y, p in zip(yields, products):
        if len(p.decay_gamma_lines):
            print(y, p.name, p)
            for g in p.decay_gamma_lines[:3]:
                print(f"\t{g}")

    spe = SPEFile('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot140.Spe')
    spe.plot_erg_spectrum(eff_corr=False, make_rate=True)
    plt.show()
    # save_spe(spe, '/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/_test.spe')
    # _set_SPE_data('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/Nickel/Nickel.Spe')
    # pass
    # spe = SPEFile('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot119.Spe')

    # def win(a, w):
    #     w = int(w/np.mean(spe.erg_bin_widths))
    #     _hw = w//2
    #     g = (a[i - _hw if _hw < i else 0: i + _hw if i + _hw < len(a) else len(a) - 1] for i in range(len(a)))
    #
    #     def _win(x):
    #         return np.mean(winsorize(x, limits=(0.25, 0.5)))
    #
    #     out = np.fromiter(map(_win, g), dtype=float)
    #     # slices = [slice(max([0, i-w//2]), min([len(a)-1, i+w//2])) for i in range(len(a))]  # works
    #     # out = [np.mean(winsorize(a[s], limits=(0.25, 0.5))) for s in slices]  # works
    #     return out
    # ar = spe.get_counts(nominal_values=True)
    # ax = spe.plot_erg_spectrum()
    # w = win(ar, 30)
    # mpl_hist(spe.erg_bins, w, ax=ax, poisson_errors=False)
    # mpl_hist(spe.erg_bins, calc_background(ar), ax=ax, label='convent.')
    # ax.legend()
    #
    #
    # plt.show()