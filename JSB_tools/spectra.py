from __future__ import annotations

import datetime
import pickle
import warnings
from pathlib import Path
from typing import List

import lmfit
import numpy as np
import sortednp as snp
from matplotlib import pyplot as plt
from numpy.core._exceptions import UFuncTypeError
from uncertainties import unumpy as unp
from uncertainties import UFloat
from JSB_tools import mpl_hist, calc_background, rolling_median, _float
from copy import deepcopy

def cached_decorator(func):
    """
    Like cached property. Creates an attribute named _var_name.
    When using this decorator, just calculate and return the value as usual in the function body. The return value will
    only be calculated once. To force re-evaluation, delete self._func_name

    Args:
        func:

    Returns:

    """
    var_name = f'_{func.__name__}'

    @property
    def out(self):
        try:
            return getattr(self, var_name)
        except AttributeError:
            val = func(self)
            setattr(self, var_name, val)
            return val
    return out


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
            path = Path(other_path)
        else:
            assert self.path is not None, 'self.path is None! User must supply other_path arg.'
            path = self.path

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
            self._effs = unp.uarray(eff, eff_err)
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


class ListSpectra(EfficiencyCalMixin):
    # all attributes that should be ndarrays.
    ndarray_attribs = 'energies', 'times', 'erg_bins', 'fraction_live', 'fraction_live_times'

    cached_properties = 'energies', 'energy_binned_times' # used to reset all such properties when needed

    pickle_attribs = 'energies', 'times', 'erg_bins', 'path', 'fraction_live', 'start_time'

    def __init__(self, channels_list, times, erg_bins, path=None,
                 fraction_live=None, fraction_live_times=None,
                 start_time: datetime.datetime = None):
        """

        Args:
            channels_list: A list of channels corresponding to each event.
            times: The associated times for each event in channels_list
            erg_bins: Used to define mapping between channels_list and energies.
            path: Used to tie this instance to a file path. Used for pickling/unpickling.

        """
        self.times = times
        self.channels_list = channels_list

        assert max(channels_list) < len(erg_bins) - 1, \
            "A channel in channels_list is beyond the number of energy bins provided!"

        self.erg_bins = erg_bins
        self.path = path

        if self.path is not None:
            self.path = Path(self.path)

        self.fraction_live = fraction_live
        if fraction_live is not None:
            assert fraction_live_times is not None
            assert len(fraction_live_times) == len(fraction_live)
        self.fraction_live_times = fraction_live_times

        self.start_time = start_time

        for name in ListSpectra.ndarray_attribs:
            val = getattr(self, name)
            if not isinstance(val, np.ndarray):
                if val is not None:
                    setattr(self, name, np.array(val, dtype=float))

        super().__init__()
        self.unpickle_eff()

    @cached_decorator
    def energies(self):
        return self.erg_centers[self.channels_list]

    # @energies.setter
    # def energies(self, val):
    #     self._energies = val

    @property
    def erg_centers(self):
        return 0.5*(self.erg_bins[1:] + self.erg_bins[:-1])

    @property
    def bin_widths(self):
        return self.erg_bins[1:] - self.erg_bins[:-1]

    @property
    def n_channels(self):
        return len(self.erg_bins) - 1

    @cached_decorator
    def energy_binned_times(self) -> List[np.ndarray]:
        """
        The times of all events are segregated into the available energy bins.
        Returns:

        """
        out = [[] for _ in range(len(self.erg_bins) - 1)]

        for i, t in zip(self.__erg_index__(self.energies), self.times):
            if i >= self.n_channels:
                continue
            out[i].append(t)
        out = [np.array(ts) for ts in out]
        # self._energy_spec = None  # Force _energy_spec to update on next call
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

    def __erg_index__(self, energies):
        """
        Get the index which corresponds to the correct energy bin(s).

        Examples:
            The result can be used to find the number of counts in the bin for 511 KeV:
                self.counts[self.erg_bin_index(511)]

        Args:
            energies:

        Returns:

        """
        out = np.searchsorted(self.erg_bins, energies, side='right') - 1

        return out

    def get_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = None,
                         time_max: float = None, rebin=1, eff_corr=False, make_density=False, nominal_values=False,
                         return_bin_edges=False,
                         remove_baseline=False, baseline_method='ROOT', baseline_kwargs=None):
        """
        Get energy spectrum according to provided condition.
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            rebin: Combine bins into rebin groups
            eff_corr: If True, correct for efficiency to get absolute counts.
            make_density: If True, divide by bin widths.
            nominal_values: If False, return unp.uarray
            return_bin_edges: Return the energy bins, e.g. for use in a histogram.
            remove_baseline: If True, remove baseline according to the following arguments.
            baseline_method: If "ROOT" then use JSB_tools.calc_background
                             If "median" then use JSB_tools.rolling_median
            baseline_kwargs: Kwargs ot pass to either JSB_tools.calc_background or JSB_tools.rolling_median

        Returns:

        """
        if erg_min is None:
            erg_min = self.erg_bins[0]
        if erg_max is None:
            erg_max = self.erg_bins[-1]
        assert erg_min < erg_max, f"`erg_min` must be less than `erg_max`, not erg_min={erg_min} and erg_max={erg_max}"

        time_min, time_max, erg_min, erg_max = tuple(map(_float, [time_min, time_max, erg_min, erg_max]))

        b = (time_min is not None, time_max is not None)

        if b == (0, 0):
            get_n_events = len
        elif b == (1, 0):
            def get_n_events(x):
                return len(x) - np.searchsorted(x, time_min, side='left')
        elif b == (0, 1):
            def get_n_events(x):
                return np.searchsorted(x, time_max, side='right')
        else:
            def get_n_events(x):
                return np.searchsorted(x, time_max, side='right') - np.searchsorted(x, time_min,
                                                                                    side='left')

        erg_index0 = self.__erg_index__(erg_min)
        erg_index1 = self.__erg_index__(erg_max)

        time_arrays = self.energy_binned_times[erg_index0: erg_index1]

        out = np.fromiter(map(get_n_events, time_arrays), dtype=int)

        bins = self.erg_bins_cut(erg_min, erg_max)

        if not nominal_values:
            out = unp.uarray(out, np.sqrt(out))

        if remove_baseline:
            full_y = np.fromiter(map(get_n_events, self.energy_binned_times), dtype=int)

            if baseline_kwargs is None:
                baseline_kwargs = {}
            baseline_method = baseline_method.lower()
            if baseline_method == 'root':
                bg = calc_background(full_y, **baseline_kwargs)
                # out = out - calc_background(full_y, **baseline_kwargs)
            elif baseline_method == 'median':
                if 'window' not in baseline_kwargs:
                    baseline_kwargs['window_width'] = 75  # size of rolling window in keV
                # out = out - rolling_median(values=full_y, **baseline_kwargs)
                bg = rolling_median(values=full_y, **baseline_kwargs)
            else:
                raise TypeError(f"Invalid `baseline_method`, '{baseline_method}'")

            bg = bg[erg_index0: erg_index1]

            try:
                out -= bg
            except UFuncTypeError:
                out = out - bg

        if make_density:
            out = out/(bins[1:] - bins[:-1])

        if eff_corr:
            assert self.effs is not None, 'Cannot perform efficiency correction. No efficiency data. '
            effs = self.effs[erg_index0: erg_index1]
            if nominal_values:
                effs = unp.nominal_values(effs)
                if not out.dtype == 'float64':
                    out = out.astype('float64')
            out /= effs

        # if rebin != 1:
        #     out, bins = _rebin(rebin, out, bins)

        if return_bin_edges:
            return out, bins
        else:
            return out

    def plot_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = 0,
                          time_max: float = None, rebin=1, eff_corr=False, remove_baseline=False, make_density=False,
                          ax=None,
                          label=None,
                          return_bin_values=False):
        """
        Plot energy spectrum with time and energy cuts.
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            rebin: Combine n bins
            eff_corr: If True, correct for efficiency to get absolute counts.
            remove_baseline: If True, remove baseline.
            make_density:
            ax:
            label:
            return_bin_values:

        Returns:

        """
        title = ''
        if self.path is not None:
            title += self.path.name

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        else:
            fig = plt.gcf()
        fig.suptitle(title)

        bin_values, bins = self.get_erg_spectrum(erg_min, erg_max, time_min, time_max, make_density=make_density,
                                                 return_bin_edges=True, rebin=rebin, eff_corr=eff_corr)
        if remove_baseline:
            bl = calc_background(bin_values)
            bin_values = -bl + bin_values

        mpl_hist(bins, bin_values, ax=ax, label=label)

        if not (time_min == 0 and time_max is None):
            ax.set_title(f"{time_min:.1f} <= t <= {time_max:.1f}")

        ax.set_xlabel('Energy [KeV]')
        ax.set_ylabel('Counts')
        if not return_bin_values:
            return ax
        else:
            return ax, bin_values

    def copy(self, deep_copy=False) -> ListSpectra:
        """
        Returns an identical instance with distinct identity.

        Args:
            deep_copy: If True, copy all mutable objects.

        Returns:

        """
        out = ListSpectra.__new__(ListSpectra)
        attribs = 'channels_list', 'times', 'erg_bins', 'path', 'fraction_live', 'fraction_live_times', 'start_time',\
                  '_effs', '_eff_model', 'eff_scale', '_erg_centers', '_energies', '_effs'

        for a in attribs:
            try:
                val = getattr(self, a)
            except AttributeError:
                continue

            if deep_copy:
                if isinstance(val, (np.ndarray, list)):
                    val = deepcopy(val)

            setattr(out, a, val)

        return out

    def __len__(self):
        return len(self.erg_centers)

    def rebin(self, new_bins):
        if not isinstance(new_bins, np.ndarray):
            new_bins = np.array(new_bins)

        sel = np.ones(len(self.energies), dtype=bool)
        if self.erg_bins[-1] > new_bins[-1] or self.erg_bins[0] < new_bins[0]:
            max_i = self.__erg_index__(new_bins[-1])
            min_i = self.__erg_index__(new_bins[0])
            if min_i > 0:
                sel &= self.channels_list >= min_i

            if max_i < len(self) - 1:
                sel &= self.channels_list < max_i

        if sel is True:
            sel = Ellipsis

        old_erg_centers = self.erg_centers[:]

        old_energies = self.energies[sel]

        new_energies = old_energies + \
                       self.bin_widths[self.channels_list[sel]]*(np.random.random(len(old_energies)) - 0.5)

        self.erg_bins = new_bins
        self.channels_list = self.__erg_index__(new_energies)

        self._energies = self.erg_centers[self.channels_list]

        try:
            del self._energy_binned_times
        except AttributeError:
            pass

        self.energy_binned_times

        if self.effs is not None:
            self.recalc_effs(old_erg_centers=old_erg_centers, new_erg_centers=self.erg_centers)

    def __add__(self, other: ListSpectra):
        self = self.copy()
        other = other.copy()

        other.rebin(self.erg_bins)

        self.times, (indices_self, indices_other) = snp.merge(self.times, other.times, indices=True)

        new_energies = np.zeros_like(self.times)
        new_energies[indices_self] = self.energies
        new_energies[indices_other] = other.energies

        if self.effs is not None and other.effs is not None:
            if len(self.effs) == len(other.effs):
                if not all(np.isclose(unp.nominal_values(self.effs), unp.nominal_values(other.effs), rtol=0.01)):
                    erg_spec_self = self.get_erg_spectrum(nominal_values=True, eff_corr=False)
                    erg_spec_other = other.get_erg_spectrum(nominal_values=True, eff_corr=False)
                    effs_self = unp.nominal_values(self.effs)
                    effs_other = unp.nominal_values(other.effs)
                    rel_errors = (unp.std_devs(effs_other) + unp.std_devs(effs_other))/(effs_self + effs_other)
                    new_effs = (effs_self*erg_spec_self + effs_other*erg_spec_other)/(erg_spec_self + erg_spec_other)
                    new_effs = unp.uarray(new_effs, rel_errors*new_effs)
                    self.effs = new_effs
                else:
                    pass  # We're good.
            else:
                raise NotImplementedError("No way to combine different length Spectra yet. Todo")

        # todo: fraction live











