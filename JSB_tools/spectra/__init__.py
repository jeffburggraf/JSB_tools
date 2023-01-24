from __future__ import annotations
import sys
import datetime
import marshal
import pickle
import time
import warnings
from pathlib import Path
from typing import List, Callable, Union, Tuple, Iterable
import lmfit
import numpy as np
import sortednp as snp
from matplotlib import pyplot as plt
from numpy.core._exceptions import UFuncTypeError
from uncertainties import unumpy as unp
from uncertainties import UFloat, ufloat
import JSB_tools
from JSB_tools import mpl_hist, calc_background, rolling_median, _float, discrete_interpolated_median, shade_plot
from JSB_tools import convolve_gauss, calc_background, InteractivePlot
from copy import deepcopy
from matplotlib.axes import Axes
from lmfit.model import save_modelresult
from pickle import UnpicklingError
from JSB_tools.spectra.time_depend import InteractiveSpectra
from functools import cached_property
from numba.typed import List as numba_list
from numba import jit
from JSB_tools.spectra.time_depend import get_erg_spec, remove_background
np.random.seed(69)


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
            self._cached.add(var_name)
            return val
    return out


class EfficiencyCalMixin:
    """
    Mixin class for storing and maneging spectra's efficiency calibrations.


    """
    _pickle_attribs = '_effs', '_effs_ergs'

    # @staticmethod
    # def pickle_attribes
    def __init__(self):
        self._effs = None  # efficiencies at _effs_ergs

        if not hasattr(self, 'eff_path'):
            self.eff_path: Path = None  # Can be over_written by subclasses!

        self._effs_ergs = None  # Energies at which the efficiencies were given

    def set_efficiency(self, effs, ergs=None):
        if ergs is None:
            ergs = getattr(self, 'erg_centers', None)
            if ergs is None:
                raise ValueError("Must supply `ergs` argument since no subclass present. ")
        self._effs_ergs = ergs
        self._effs = effs
        self.recalc_effs()

    def _get_pickle_eff_path(self, eff_path):
        """
        Gets eff_path.
        Args:
            eff_path:

        Returns:

        """
        if eff_path is None:
            eff_path = self.eff_path
        else:
            self.eff_path = Path(eff_path).with_suffix('.eff')  # update self.eff_path

        if eff_path is None:  # search for path in subclass
            assert hasattr(self, "erg_centers"), "Must provide `eff_path` arg since there is no Spectrum subclass."
            path = getattr(self, 'path')
            if path is None:
                return None

            eff_path = Path(path)

        eff_path = Path(eff_path).with_suffix('.eff')

        return eff_path

    def unpickle_efficiency(self, eff_path: Union[str, Path] = None):
        """
        Load efficiency. If eff_path is None, use self.eff_path

        Args:
            eff_path:

        Returns:

        """
        assert isinstance(self, EfficiencyCalMixin), "EfficiencyCalMixin.unpickle_efficiency is not a class method. " \
                                                     "Create an EfficiencyCalMixin object first,e.g.\n" \
                                                     "eff = EfficiencyCalMixin()\neff.unpickle_efficiency()"

        eff_path = self._get_pickle_eff_path(eff_path)

        if eff_path is None:
            raise FileNotFoundError

        if not eff_path.is_absolute():
            assert hasattr(self, 'path'), f"Relative eff_path, '{eff_path}', with no subclass to anchor to. "
            eff_path = getattr(self, 'path').parent/eff_path

        try:
            with open(eff_path, 'rb') as f:
                attribs = pickle.load(f)  # self._effs and self.eff_scale

        except FileNotFoundError:
            raise FileNotFoundError(f"No efficiency saved at path {self.eff_path}")

        except UnpicklingError:
            warnings.warn(f"Failed to unpickle {eff_path}")
            return

        for name, attrib in attribs.items():
            setattr(self, name, attrib)
        self.recalc_effs()

    def eval_efficiency(self, erg, nominal=False):
        scalar_flag = False
        if not hasattr(erg, '__iter__'):
            scalar_flag = True
            erg = [erg]

        out = np.interp(erg, self._effs_ergs, self.effs)

        if scalar_flag:
            return out[0]
        else:
            return out

    def __call__(self, ergs,):
        return self.eval_efficiency(ergs)

    def pickle_efficiency(self, eff_path=None):
        """
        Save efficiency cal. First uses eff_path argument, then self.eff_path, then self.path (is attribute exists)
        Args:
            eff_path: Full path. .eff (or also .eff_model) suffix will be added

        Returns:

        """
        eff_path = self._get_pickle_eff_path(eff_path)

        with open(eff_path, 'wb') as f:
            data = {name: getattr(self, name) for name in EfficiencyCalMixin._pickle_attribs}
            pickle.dump(data, f)

    @property
    def nominal_effs(self):
        return unp.nominal_values(self.effs)

    @cached_property
    def effs(self):
        if self._effs is None:
            return None

        self._effs = np.array(self._effs)

        if hasattr(self, 'erg_centers'):
            if self._effs.dtype == object:
                nominal_effs = unp.nominal_values(self._effs)
                stds = unp.std_devs(self._effs)
                ystd = np.interp(getattr(self, 'erg_centers'), self._effs_ergs, stds)
                ynom = np.interp(getattr(self, 'erg_centers'), self._effs_ergs, nominal_effs)
                return unp.uarray(ynom, ystd)
            else:

                return np.interp(getattr(self, 'erg_centers'), self._effs_ergs, self._effs)

        return self._effs

    def recalc_effs(self):
        """
        Use when subclass's self.erg_centers changes
        Returns:

        """
        try:
            delattr(self, 'effs')
        except AttributeError:
            pass
        return self.effs

    def plot_efficiency(self, ax=None, **mpl_kwargs):
        assert self.effs is not None, 'No efficiency to plot. '

        if ax is None:
            fig, ax = plt.subplots()

            if self.eff_path is not None:
                self.eff_path = Path(self.eff_path)
                try:
                    fig.suptitle(f"{self.eff_path.relative_to(self.eff_path.parents[1]).name}")
                except IndexError:
                    pass

        if hasattr(self, 'erg_centers'):
            ergs = getattr(self, 'erg_centers')
        else:
            ergs = self._effs_ergs

        if self.eff_path is not None:
            self.eff_path = Path(self.eff_path)
            label = self.eff_path.relative_to(self.eff_path.parents[1])
        else:
            label = None

        ax.plot(ergs, self.effs, label=label, **mpl_kwargs)
        return ax


class ListSpectra(EfficiencyCalMixin):
    REFS = []  # references to prevent garbage collection

    datetime_format = '%m/%d/%Y %H:%M:%S'

    _cached = set()  # a set of the names of cached attributes (automatically poopulated).

    # all attributes that should be ndarrays, along with each dtype
    ndarray_attribs = {'adc_channels': int,
                       '_energies': float,
                       'times': float,
                       'erg_bins': float,
                       'fraction_live': float,
                       'fraction_live_times': float}

    _reset_properties = 'energies', 'energy_binned_times'  # used to reset all such properties when needed

    def reset_cach(self):
        for name in ListSpectra._reset_properties:
            name = f'_{name}'
            try:
                delattr(self, name)
            except AttributeError:
                pass
        # self._effs = None

    # var_name: (function to apply before marshaling, function to apply after un-marshaling). None means lambda x:x
    # Function is only applied if un-marshaled value is not None.
    pickle_attribs = {'_energies': (None, None),
                      'adc_channels': (None, None),
                      'times': (None, None),
                      'erg_bins': (None, None),
                      'path': (str, Path),
                      'fraction_live': (None, None),
                      'fraction_live_times': (None, None),
                      'start_time': (lambda x: x.strftime(ListSpectra.datetime_format),
                                     lambda x: datetime.datetime.strptime(x, ListSpectra.datetime_format)),
                      'eff_path': (str, Path)
                      }

    def __init__(self, adc_channels, times, erg_bins, path=None,
                 fraction_live=None, fraction_live_times=None,
                 start_time: datetime.datetime = None):
        """

        Args:
            adc_channels: A list of ADC channels corresponding to each event. Max value allowed is len(erg_bins) - 2
                as it is assumed to start from channel 0.

            times: The associated times for each event in channels_list

            erg_bins: Used to define mapping between channels_list and energies. len(erg_bins) = 1 + len(adc_channels)
                e.g., a adc_channels[0] corresponds to an event within energy bin defined by [erg_bins[0], erg_bins[1])

            path: Used to tie this instance to a file path. Used for pickling/unpickling.

        """
        self.times = times
        self.adc_channels = adc_channels

        self.eff_path = None  # for linking to efficiency calibration

        assert max(adc_channels) < len(erg_bins) - 1, \
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

        if start_time is not None:
            assert isinstance(start_time, datetime.datetime)
        self.start_time = start_time

        for name, dtype in ListSpectra.ndarray_attribs.items():
            try:
                val = getattr(self, name)
            except AttributeError:
                continue
            if not isinstance(val, np.ndarray):
                if val is not None:
                    setattr(self, name, np.array(val, dtype=dtype))
            elif not val.dtype == dtype:
                val = val.astype(dtype)
                setattr(self, name, val)

        super(ListSpectra, self).__init__()

        try:
            self.unpickle_efficiency()
        except FileNotFoundError:
            pass

    def time_cut(self, tmin=None, tmax=None) -> ListSpectra:
        """
        Remove all event with time outside of range.
        Args:
            tmin:
            tmax:

        Returns:

        """
        if tmax is None:
            tmax = self.times[-1]
        if tmin is None:
            tmin = self.times[0]

        i0, i1 = np.searchsorted(self.times, [tmin, tmax])
        self.times = self.times[i0: i1]
        self.adc_channels = self.adc_channels[i0: i1]

        try:
            del self.energies
        except AttributeError:
            pass

        try:
            del self.energy_binned_times
        except AttributeError:
            pass
        return self

    def plot(self, interactive_spec_instance: InteractiveSpectra = None, init_tmin=None, init_tmax=None,
             init_bg_subtractQ=False, window_max=None, delta_t=5, fig_title=None,
             scale=1, eff_corr=True, erg_min=None, erg_max=None) -> InteractiveSpectra:
        """
        Example:
            # Plot two spectra:
                i = l1.plot()
                l2.plot(i)

        Args:
            interactive_spec_instance:
            init_tmin:
            init_tmax:
            init_bg_subtractQ:
            window_max:
            delta_t:
            fig_title:
            scale:
            eff_corr:
            erg_min:
            erg_max:

        Returns:

        """

        if fig_title is None and self.path is not None:
            fig_title = Path(self.path).name

        if interactive_spec_instance is None:
            interactive_spec_instance = InteractiveSpectra(init_tmin=init_tmin, init_tmax=init_tmax,
                                                           init_bg_subtractQ=init_bg_subtractQ, window_max=window_max,
                                                           delta_t=delta_t, fig_title=fig_title)

            ListSpectra.REFS.append(interactive_spec_instance)

        interactive_spec_instance.add_list(self, scale=scale, disable_eff_corr=not eff_corr, title=fig_title,
                                           erg_min=erg_min, erg_max=erg_max)

        return interactive_spec_instance

    @cached_decorator
    def energies(self):
        return self.erg_centers[self.adc_channels]

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
        if hasattr(self, '_energy_binned_times'):  # for direct pickling/unpickling. See __set_state__
            out = getattr(self, '_energy_binned_times')
            delattr(self, '_energy_binned_times')
            return out

        n = len(self.erg_bins) - 1
        out = [[] for _ in range(n)]

        assert len(self.adc_channels) == len(self.times), "This is not allowed. There is a bug."

        for i, t in zip(self.adc_channels, self.times):
            if i >= self.n_channels:
                raise IndexError("Bug. ")

            out[i].append(t)

        out = numba_list([np.array(ts) for ts in out])  # numba_list helps in cases where numba.jit is used.
        # out = [np.array(ts) for ts in out]  #
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

    def get_time_dependence(self, energy,
                            bins: Union[str, int, np.ndarray, list] = 'auto',
                            signal_window_kev: float = 3,
                            bg_window_kev=None,
                            bg_offsets: Union[None, Tuple, float] = None,
                            make_rate=False,
                            eff_corr=False,
                            scale: Union[float, Callable, np.ndarray] = 1.,
                            nominal_values=True,
                            convolve: Union[float, int] = None,
                            debug_plot: Union[bool, str] = False,
                            debug_title:str=None):
        """
        Get the time dependence around erg +/- signal_window_kev/2. Baseline is estimated and subtracted.
        Estimation of baseline is done by taking the median rate ( with an energy of `energy` +/- `bg_window_kev`/2,
        excluding signal window) of events in each time bin. The median is then normalized to the width of the signal
        window (in KeV), and finally subtracted from the time dependence of events in the signal window.

        Args:
            energy: Energy in keV at which time dependence will be analysed

            bins: Time bins. Will be used for: np.histogram(bins=bins).

            signal_window_kev: Width around `energy` that will be considered the signal.

            bg_window_kev: Size of the window used for baseline estimation.

            bg_offsets: Offset the baseline window from the center (default will avoid signal window by distance of
                half `signal_window_kev`)

            make_rate: Makes units in Hz instead of counts.

            eff_corr: If True, account for efficiency

            scale: User supplied float or array of floats (len(bins) - 1) used to scale rates.
                If a Callable, the bins wil be passed to the supplied function which must return array of normalization
                values of same length as bins.

            nominal_values: If False, return unp.uarray (i.e. include Poissonian errors).

            convolve: If not None, perform gaussian convolution with sigma according to this value
                (sigma units are array indicies).

            debug_plot: If False, do nothing.
                        If True, plot signal and background (energy vs counts) for every time bin.
                        If "simple", plot one plot for all bins.

            debug_title: Title for debug figure.

        Returns: Tuple[signal window rate, baseline rate estimation, bins used]
        """
        if isinstance(energy, UFloat):
            energy = energy.n

        if bg_offsets is None:
            bg_offsets = [2 * signal_window_kev] * 2
        elif isinstance(bg_offsets, Iterable):
            bg_offsets = tuple(map(lambda x: 2 * signal_window_kev if x is None else x, bg_offsets))
            if not len(bg_offsets) == 2:
                raise ValueError('Too many elements in arg `bg_offsets')
        elif isinstance(bg_offsets, (float, int)):
            bg_offsets = tuple([bg_offsets] * 2)
        else:
            raise ValueError(f'Bad value for `bg_offsets`: {bg_offsets}')

        if bg_window_kev is None:
            bg_window_kev = 6 * signal_window_kev

        bg_offsets = np.abs(bg_offsets)

        bg_window_left_bounds = np.array([-bg_window_kev / 2, 0]) + energy - bg_offsets[0]
        bg_window_right_bounds = np.array([0, bg_window_kev / 2]) + energy + bg_offsets[1]
        sig_window_bounds = [energy - signal_window_kev / 2, energy + signal_window_kev / 2]

        bg_window_left_is = self.__erg_index__(bg_window_left_bounds)
        bg_window_right_is = self.__erg_index__(bg_window_right_bounds)
        # bg_window_right_is[-1] += 1  # + 1 for python slicing rules
        # bg_window_left_is[-1] += 1  # + 1 for python slicing rules

        sig_window_is = self.__erg_index__(sig_window_bounds)
        # sig_window_is[-1] += 1  # + 1 for python slicing rules
        n_sig_erg_bins = sig_window_is[-1] - sig_window_is[0]

        if eff_corr:
            effs = unp.nominal_values(self.effs)
            effs = np.where(effs > 0, effs, 1)
            bg_weights = np.concatenate([1.0 / effs[slice(*bg_window_left_is)],
                                         1.0 / effs[slice(*bg_window_right_is)]])

            sig_weight = np.average((1.0 / self.effs[slice(*sig_window_is)]),
                                    weights=np.fromiter(map(len, self.energy_binned_times[slice(*sig_window_is)]),
                                                        dtype=int))
        else:
            bg_weights = 1
            sig_weight = ufloat(1, 0)

        bg_times_list = self.energy_binned_times[slice(*bg_window_left_is)]
        bg_times_list += self.energy_binned_times[slice(*bg_window_right_is)]

        sig_times_list = self.energy_binned_times[slice(*sig_window_is)]
        sig_times = np.concatenate(sig_times_list)

        # sig_times, n_sig_erg_bins = self.energy_slice(*sig_window_bounds, return_num_bins=True)

        if isinstance(bins, str):
            _, time_bins = np.histogram(sig_times, bins=bins)
        elif isinstance(bins, int):
            time_bins = np.linspace(sig_times[0], sig_times[-1], bins)
        else:
            assert hasattr(bins, '__iter__')
            time_bins = bins

        assert len(time_bins) != 1, "Only one bin specified. This is not allowed. "

        def _median(_x):
            """

            Args:
                _x: A list of the number of events in each energy bin (for a given time bin)

            Returns:

            """
            # _x = convolve_gauss(_x, min([len(_x)//2, 5]))
            out = discrete_interpolated_median(_x * bg_weights)

            if nominal_values:
                return out
            else:
                s = sum(_x)
                if s == 0:
                    return ufloat(out, 0)
                else:
                    return ufloat(out, out * 1 / (np.sqrt(sum(_x))))

        bg = np.array([_median(x) for x in
                       np.array([np.histogram(times, bins=time_bins)[0] for times in bg_times_list]).transpose()])
        bg *= n_sig_erg_bins  # from per_bin to per width of signal window

        sig = np.histogram(sig_times, bins=time_bins)[0]
        sig = sig.astype(float)

        if not nominal_values:
            sig = unp.uarray(sig, np.sqrt(sig))

        sig *= sig_weight.n if (nominal_values and isinstance(sig_weight, UFloat)) else sig_weight

        if debug_plot is not False:  # todo: move this somewhere else.
            if isinstance(debug_plot, str) and debug_plot.lower() == 'simple':
                ax, y = self.plot_erg_spectrum(bg_window_left_bounds[0] - signal_window_kev,
                                               bg_window_right_bounds[-1] + signal_window_kev,
                                               time_min=time_bins[0],
                                               time_max=time_bins[-1],
                                               return_bin_values=True,
                                               eff_corr=eff_corr,
                                               )
                shade_plot(ax, bg_window_left_bounds, color='red')
                shade_plot(ax, bg_window_right_bounds, color='red', label='Bg. window')
                shade_plot(ax, sig_window_bounds, color='blue', label='Sig. window')
                ax.legend()
            else:
                tab_plot = JSB_tools.TabPlot()
                tab_plot.fig.suptitle(self.path.name if debug_title is None else debug_title)
                for bin_index, (b1, b2) in enumerate(zip(time_bins[:-1], time_bins[1:])):
                    plot_legend = False
                    # if bin_index % 4 == 0:
                    #     fig, axs = plt.subplots(2, 2, figsize=(12, 5))
                    #     fig.suptitle(self.path.name)
                    #     axs = axs.flatten()
                    #     plt.subplots_adjust(hspace=0.4, wspace=0.1, left=0.04, right=0.97)
                    #     plot_legend = True

                    lines = []
                    legends = []
                    # ax = axs[bin_index % 4]
                    button_label = f"{b1:.1f}-{b2:.1f} s"
                    try:
                        ax = tab_plot.new_ax(button_label=button_label)
                    except OverflowError:
                        tab_plot = JSB_tools.TabPlot()
                        ax = tab_plot.new_ax(button_label=button_label)

                    ax, y = self.plot_erg_spectrum(bg_window_left_bounds[0] - signal_window_kev,
                                                   bg_window_right_bounds[-1] + signal_window_kev,
                                                   time_min=b1,
                                                   time_max=b2,
                                                   return_bin_values=True,
                                                   ax=ax,
                                                   eff_corr=eff_corr)

                    l = shade_plot(ax, sig_window_bounds)
                    lines.append(l)
                    legends.append('Sig. window')

                    l = shade_plot(ax, bg_window_right_bounds, color='red')
                    _ = shade_plot(ax, bg_window_left_bounds, color='red')
                    lines.append(l)
                    legends.append('Bg. window')

                    bg_est = bg[bin_index] / n_sig_erg_bins
                    sig_est = sig[bin_index] / n_sig_erg_bins
                    if isinstance(bg_est, UFloat):
                        bg_est = bg_est.n
                        sig_est = sig_est.n

                    l = ax.plot(bg_window_right_bounds, [bg_est] * 2, color='red', ls='--')[0]
                    _ = ax.plot(bg_window_left_bounds, [bg_est] * 2, color='red', ls='--')
                    lines.append(l)
                    legends.append('Bg. est.')

                    l = ax.plot(sig_window_bounds, [sig_est] * 2, color='blue', ls='--')[0]
                    lines.append(l)
                    legends.append('sig. est.')

                    if plot_legend:
                        plt.figlegend(lines, legends, 'upper right')

                    ax.text(np.mean(bg_window_left_bounds), ax.get_ylim()[-1] * 0.82, f'{bg_est * n_sig_erg_bins:.2e}',
                            horizontalalignment='center',
                            verticalalignment='center', color='red', size='small', rotation='vertical')
                    ax.text(np.mean(sig_window_bounds), ax.get_ylim()[-1] * 0.82, f'{sig_est * n_sig_erg_bins:.2e}',
                            horizontalalignment='center',
                            verticalalignment='center', color='blue', size='small', rotation='vertical')

        if make_rate:
            if not isinstance(time_bins, np.ndarray):
                time_bins = np.array(time_bins)
            b_widths = time_bins[1:] - time_bins[:-1]
            sig /= b_widths
            bg /= b_widths

        if convolve is not None:
            sig, bg = convolve_gauss(sig, convolve), convolve_gauss(bg, convolve)

        # if offset_sample_ready:
        #     time_bins -= self.sample_ready_median

        if hasattr(scale, '__call__'):
            c = scale(time_bins)
        else:
            if hasattr(scale, '__len__'):
                assert len(scale) == len(time_bins) - 1, '`normalization` argument of incorrect length.'
            c = scale

        return (sig - bg) * c, bg * c, time_bins

    def plot_time_dependence(self, energy, bins: Union[str, int, np.ndarray] = 'auto', signal_window_kev: float = 3,
                             bg_window_kev=None, bg_offsets: Union[None, Tuple, float] = None, make_rate=False,
                             eff_corr=False,
                             plot_background=False, ax=None, offset_sample_ready=False,
                             convolve=None,
                             debug_plot=False, **mpl_kwargs):
        """
        Todo: finish this doc string
        Args:
            energy:
            bins:
            signal_window_kev:
            bg_window_kev:
            bg_offsets:
            make_rate:
            eff_corr:
            plot_background:
            ax:
            offset_sample_ready:
            convolve:
            debug_plot:
            mpl_kwargs:

        Returns:

        """
        sig, bg, bins = \
            self.get_time_dependence(energy=energy, bins=bins, signal_window_kev=signal_window_kev,
                                     bg_window_kev=bg_window_kev, bg_offsets=bg_offsets, make_rate=make_rate,
                                     eff_corr=eff_corr, nominal_values=False,
                                     convolve=convolve, debug_plot=debug_plot)

        label = ''
        if ax is None:
            fig, ax = plt.subplots()
            if self.path is not None:
                ax.set_title(self.path.name)
                label = mpl_kwargs.pop("label", self.path.name)

            elims = energy - signal_window_kev/2, energy + signal_window_kev/2
            fig.suptitle(f"{elims[0]:.1f} < erg < {elims[-1]:.1f}")
        label = mpl_kwargs.get('label', label)

        _, c = mpl_hist(bins, sig, return_line_color=True, ax=ax,
                        label=label + "(signal)" if plot_background else "", **mpl_kwargs)

        if plot_background:
            mpl_kwargs['ls'] = '--'
            mpl_kwargs.pop('c', None)
            mpl_kwargs.pop('color', None)
            mpl_kwargs['c'] = c
            mpl_hist(bins, bg, return_line_color=True, ax=ax, label=label + "(baseline)", **mpl_kwargs)

        y_label = "Raw" if not eff_corr else ""
        y_label += " Counts"

        if make_rate:
            y_label += '/s'

        ax.set_ylabel(y_label)
        ax.set_xlabel("time [s]")

        return ax

    def get_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = None,
                         time_max: float = None, eff_corr=False, scale=1, make_density=False, nominal_values=False,
                         return_bin_edges=False,
                         remove_baseline=False, median_window=40,
                         debug_plot=False):
        """
        Get energy spectrum according to provided condition.
        Todo: Expand energy range to be >= 5 * median_window for spectrum passed to remove_baseline()
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            eff_corr: If True, correct for efficiency to get absolute counts.
            make_density: If True, divide by bin widths.
            scale: Multiplicative constant
            nominal_values: If False, return unp.uarray
            return_bin_edges: Return the energy bins, e.g. for use in a histogram.
            remove_baseline: If True, remove baseline according to the following arguments.

            median_window: Window width for baseline removal routine. Should be larger than three times full peak width.

            debug_plot:

        Returns:
            y if not return_bin_edges else (y, bins)

        """

        if erg_min is None:
            erg_min = self.erg_bins[0]
        if erg_max is None:
            erg_max = self.erg_bins[-1]
        assert erg_min < erg_max, f"`erg_min` must be less than `erg_max`, not erg_min={erg_min} and erg_max={erg_max}"

        if time_min is None:
            time_min = self.times[0]
        if time_max is None:
            time_max = self.times[-1]

        erg_index0 = self.__erg_index__(erg_min)
        erg_index1 = self.__erg_index__(erg_max)

        time_arrays = self.energy_binned_times[erg_index0: erg_index1]

        time_range = np.array([time_min, time_max])

        out = get_erg_spec(time_arrays, time_range, )

        if not nominal_values:
            yerr = np.sqrt(out)
        else:
            yerr = None

        bins = self.erg_bins_cut(erg_min, erg_max)

        if remove_baseline:
            out = remove_background(out, median_window=median_window)

        if not nominal_values:
            out = unp.uarray(out, yerr)

        if make_density:
            out = out/(bins[1:] - bins[:-1])

        if eff_corr:
            assert self.effs is not None, 'Cannot perform efficiency correction. No efficiency data. '
            effs = self.effs[erg_index0: erg_index1]

            if nominal_values:
                effs = unp.nominal_values(effs)

            effs = np.where(effs > 0, effs, 1)
            out /= effs

        if scale != 1:
            out *= scale

        if debug_plot:
            if isinstance(debug_plot, Axes):
                _ax = debug_plot
            else:
                _ax = None

            erg_width = erg_max - erg_min
            window = [erg_min-0.75*erg_width, erg_max+0.75*erg_width]

            _ax = self.plot_erg_spectrum(erg_min=window[0], erg_max=window[1], ax=_ax, time_min=time_min,
                                         time_max=time_max, remove_baseline=remove_baseline,
                                         make_density=make_density, eff_corr=eff_corr)

            _ax.set_title(f"{_ax.get_title()}\n" + rf"$\Sigma$ = {sum(out)}")
            shade_plot(ax=_ax, window=[erg_min, erg_max], color='black')

        if return_bin_edges:
            return out, bins
        else:
            return out

    def plot_erg_spectrum(self, erg_min: float = None, erg_max: float = None, time_min: float = None,
                          time_max: float = None, eff_corr=False, make_density=False, remove_baseline=False,
                          median_window=40,
                          ax=None,
                          label=None,
                          return_bin_values=False,
                          scale=1, **mpl_kwargs):
        """
        Plot energy spectrum with time and energy cuts.
        Args:
            erg_min:
            erg_max:
            time_min:
            time_max:
            eff_corr: If True, correct for efficiency to get absolute counts.

            remove_baseline: If True, remove baseline.
            remove_baseline: If True, remove baseline according to the following arguments.

            make_density: Divide by bin widths.
            ax:
            label:
            return_bin_values:
            scale:

        Returns:

        """

        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        else:
            fig = plt.gcf()

        if fig._suptitle is None:
            fig.suptitle(self.path.name)

        bin_values, bins = self.get_erg_spectrum(erg_min, erg_max, time_min, time_max, eff_corr=eff_corr,
                                                 make_density=make_density, return_bin_edges=True,
                                                 remove_baseline=remove_baseline, median_window=median_window)
        if scale != 1:
            bin_values *= scale

        # if remove_baseline:
        #     bl = calc_background(bin_values)
        #     bin_values = -bl + bin_values

        mpl_kwargs['label'] = label
        mpl_hist(bins, bin_values, ax=ax, **mpl_kwargs)

        if not (time_min is time_max is None):
            if time_max is None:
                time_max = self.times[-1]
            if time_min is None:
                time_min = 0
            title0 = f"{ax.get_title()}; " if ax.get_title() else ""
            ax.set_title(f"{title0}{time_min:.1f} <= t <= {time_max:.1f}")

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
        # attribs = 'adc_channels', 'times', 'erg_bins', 'path', 'fraction_live', 'fraction_live_times', 'start_time',\
        #           '_effs', 'eff_model', 'eff_model_scale', '_erg_centers', '_energies'

        attribs = set(ListSpectra.ndarray_attribs.keys())
        attribs.update(ListSpectra.pickle_attribs.keys())
        attribs.update(['_effs', 'eff_model', 'eff_model_scale'])
        # self._effs = None
        # self.eff_model: lmfit.model.ModelResult = None
        # self.eff_model_scale = 1

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

    def __getstate__(self):  # numba_list object cannot be pickled.
        out = {k: v for k, v in self.__dict__.items()}
        out['_energy_binned_times'] = list(self.energy_binned_times)  # avoids issues ith setting property attribute.
        return out

    def __setstate__(self, state):
        self.__dict__ = state
        energy_binned_times = getattr(self, '_energy_binned_times')
        setattr(self, '_energy_binned_times', numba_list(energy_binned_times))

    def pickle(self, path=None, meta_data: dict = None):
        """

        Args:
            path: path_to_pickle_file. None will use self.path.
            meta_data: dict of attribute_name:attribute pairs. will be pickled into seperate
            file and unpickled as expected.


        Returns:

        """
        assert not (self.path is path is None), "A `path` argument must be supplied since self.path is not None"

        if path is None:
            path = self.path

        if not hasattr(self, '_energies'):
            _ = self.energies  # force evaluation

        path = Path(path)

        path = path.with_suffix('.pylist')
        # data = {name: None for name in ListSpectra.pickle_attribs}
        data = {}
        for name, (func, _) in ListSpectra.pickle_attribs.items():
            if not hasattr(self, name):
                continue
            else:
                val = getattr(self, name)

            if val is None:
                data[name] = val
                continue

            if func is not None:  # Apply a function to make obj marshalable.
                val = func(val)

            try:  # make sure dtypes are correct.
                dtype = ListSpectra.ndarray_attribs[name]
                if not val.dtype == dtype:
                    val = val.astype(dtype)
            except KeyError:
                pass

            data[name] = val

        with open(path, 'wb') as f:
            marshal.dump(data, f)

        if meta_data is not None:
            meta_data_path = path.with_suffix('.list_meta')

            with open(meta_data_path, 'wb') as f:
                pickle.dump(meta_data, f)

    @classmethod
    def from_pickle(cls, path: Path, ignore_missing_data=False):
        self = cls.__new__(cls)

        path = Path(path)
        path = Path(path).with_suffix('.pylist')

        with open(path, 'rb') as f:
            data = marshal.load(f)

        #  ================ backwards compatibility =================
        #  None
        #  ================ End backwards compatibility =================

        for name, (_, func) in ListSpectra.pickle_attribs.items():
            try:
                val = data[name]
            except KeyError as e:
                if not ignore_missing_data:
                    raise Exception(f"Pickle'd data missing attribute '{name}' from {path}") from e
                continue

            if isinstance(val, bytes):

                assert name in ListSpectra.ndarray_attribs
                val = np.frombuffer(val, dtype=ListSpectra.ndarray_attribs[name])
            else:
                if func is not None and val is not None:
                    val = func(val)
            setattr(self, name, val)

        super(ListSpectra, self).__init__()  # EfficiencyCalMixin

        meta_data_path = path.with_suffix('.list_meta')

        try:
            with open(meta_data_path, 'rb') as f:
                meta_data = pickle.load(f)

            for name, val in meta_data.items():
                setattr(self, name, val)
        except FileNotFoundError:
            pass

        try:
            self.unpickle_efficiency(self.eff_path)
        except FileNotFoundError:
            pass

        return self

    def __len__(self):
        return len(self.erg_centers)

    def energy_cut(self, erg_min=None, erg_max=None):
        """Remove all events with energy outside range"""
        sel = np.ones(len(self.energies), dtype=bool)

        if erg_min is not None:
            sel[self.energies < erg_min] = False

        if erg_max is not None:
            sel[self.energies > erg_max] = False

        self.adc_channels = self.adc_channels[sel]
        self.times = self.adc_channels[sel]

        self.reset_cach()

    def merge_bins(self, n):
        self.rebin(self.erg_bins[::int(n)])

    def rebin(self, new_bins):
        if not isinstance(new_bins, np.ndarray):
            new_bins = np.array(new_bins)

        sel = np.ones(len(self.energies), dtype=bool)

        if self.erg_bins[-1] > new_bins[-1] or self.erg_bins[0] < new_bins[0]:
            max_i = self.__erg_index__(new_bins[-1])
            min_i = self.__erg_index__(new_bins[0])
            if min_i > 0:
                sel &= self.adc_channels >= min_i

            if max_i < len(self) - 1:
                sel &= self.adc_channels < max_i

        if sel is True:
            sel = Ellipsis

        old_erg_centers = self.erg_centers[:]

        old_energies = self.energies[sel]

        new_energies = old_energies + \
                       self.bin_widths[self.adc_channels[sel]] * (np.random.random(len(old_energies)) - 0.5)

        self.erg_bins = new_bins
        self.adc_channels = self.__erg_index__(new_energies)

        self.times = self.times[sel]  # remove from times what we did from energies

        self.reset_cach()

        if self.effs is not None:
            self.recalc_effs()

    def __repr__(self):
        return f"ListSpectra: len={len(self.erg_centers)}; n_events:{len(self.times)}"

    def __radd__(self, other):
        if other is None:
            return self
        else:
            return other + self

    def __iadd__(self, other, recalc_effs=True, truncate_time=False):
        """
        See __add__.
        Args:
            other:
            recalc_effs:
            truncate_time:

        Returns:

        """
        return self.__add__(other, copy=False, recalc_effs=recalc_effs, truncate_time=truncate_time)

    def __add__(self, other: ListSpectra, copy=True, recalc_effs=True, truncate_time=False):
        """

        Args:
            other:
            copy:
            recalc_effs:
            truncate_time: If True, the duration of both spectra are trimmed to the duration of the shortest

        Returns:

        """
        if copy:
            self = self.copy()

        if np.any(other.erg_bins != self.erg_bins):
            other.rebin(self.erg_bins)

        if truncate_time:
            time_max = min(self.times[-1], other.times[-1])
            iself = np.searchsorted(self.times, time_max)
            iother = np.searchsorted(other.times, time_max)
            for thing, i in zip([self, other], [iself, iother]):
                thing.adc_channels = thing.adc_channels[:i]
                thing.times = thing.times[:i]

        self.times, (indices_self, indices_other) = snp.merge(self.times, other.times, indices=True)

        new_adc_channels = np.zeros(len(self.times), dtype=int)
        new_adc_channels[indices_self] = self.adc_channels
        new_adc_channels[indices_other] = other.adc_channels

        self.adc_channels = new_adc_channels

        self.reset_cach()

        if (self.effs is not None) and (other.effs is not None) and recalc_effs:
            if len(self.effs) == len(other.effs):
                if not all(np.isclose(unp.nominal_values(self.effs), unp.nominal_values(other.effs), rtol=0.01)):
                    erg_spec_self = self.get_erg_spectrum(nominal_values=True, eff_corr=False)
                    erg_spec_other = other.get_erg_spectrum(nominal_values=True, eff_corr=False)
                    effs_self = unp.nominal_values(self.effs)
                    effs_other = unp.nominal_values(other.effs)

                    # Todo: The rel error of the sum is not the correct end point
                    rel_errors = (unp.std_devs(effs_other) + unp.std_devs(effs_other))/(effs_self + effs_other)

                    # don't divide by zero below. Setting zero entries to one results in taking the average between
                    # efficiencies
                    zero_ids = np.where((erg_spec_self == erg_spec_other) & (erg_spec_other == 0))[0]
                    erg_spec_other[zero_ids] = 1
                    erg_spec_self[zero_ids] = 1

                    new_effs = (effs_self*erg_spec_self + effs_other*erg_spec_other)/(erg_spec_self + erg_spec_other)
                    new_effs = unp.uarray(new_effs, rel_errors*new_effs)
                    self.effs = new_effs
                else:
                    pass  # We're good.
            else:
                raise NotImplementedError("No way to combine different length Spectra yet. Todo")
        # todo: fraction live

        return self

    @staticmethod
    def multi_plotly(list_objs: List[ListSpectra], scales=None, leg_labels=None, erg_min=40, erg_max=None,
                     eff_corr=True,
                     time_bins=None, time_bin_width=15,
                     time_step: int = 5, time_min=None, time_max=None, remove_baseline=False,
                     nominal_values=True):
        """
        Plots energy spectra at different time intervals according to an interactive slider.
        Args:
            list_objs: series of MaestroListFile objects
            scales: List of normalizations of each list_obj
            leg_labels: None,
            erg_min:
            erg_max:
            eff_corr:
            time_bins:
            time_bin_width:
            time_step:
            time_min:
            time_max:
            remove_baseline:
            nominal_values:

        Returns:

        """
        _plt: InteractivePlot = None
        if scales is None:
            scales = np.ones(len(list_objs))

        assert all(isinstance(x, ListSpectra) for x in list_objs), \
            "`list_objs` must all be MaestroListFile instances"

        if leg_labels is None:
            leg_labels = [None] * len(list_objs)

        assert len(leg_labels) == len(list_objs)
        kwargs = None

        for l, label, s in zip(list_objs, leg_labels, scales):
            if kwargs is None:
                kwargs = l.plotly(erg_min=erg_min, erg_max=erg_max, eff_corr=eff_corr, time_bins=time_bins,
                                  time_bin_width=time_bin_width, time_step=time_step, time_min=time_min,
                                  time_max=time_max, remove_baseline=remove_baseline, nominal_values=nominal_values,
                                  leg_label=label, dont_plot=True,
                                  scale=s)
            else:
                l.plotly(**kwargs, eff_corr=eff_corr, time_bin_width=time_bin_width, leg_label=label, scale=s,
                         dont_plot=True)

        kwargs['interactive_plot'].plot()

    def plotly(self, erg_min=40, erg_max=None, eff_corr=False, time_bins=None, time_bin_width=15,
               time_step: float = 5., time_min=None, time_max=None, remove_baseline=False,
               interactive_plot: InteractivePlot = None, nominal_values=True, leg_label=None, scale=1,
               dont_plot=False, convolve_overlay_sigma=None,
               time_scale_func: Callable = None):
        """

        Args:
            erg_min:
            erg_max:
            eff_corr:
            time_bins:
            time_bin_width: time range for each "frame"
            time_step: time step between each "frame".
            time_min:
            time_max:
            remove_baseline:
            interactive_plot:
            nominal_values:
            leg_label:

            scale: Multiply y axis by this number.

            dont_plot: Used internally do overlay multiple plots.

            convolve_overlay_sigma: If a not None, but instead a number (in units of KeV), then overlay a plot that is
                a gauss conv with sigma equal to `convolve_overlay_sigma' .

            time_scale_func: function which takes two args, left_time_bin and right_time_bin, and returns a float.
                The returned float will be used to scale the result accordingly.

        Returns:
            dictionary of keys/values of some arguments passed to self.plotly. e.g.:
                {"interactive_plot": interactive_plot: JSB_tools.InteractivePlot,
                #  The rest of the key/values are arguments provided to plotly (they may be different if None wsa used).
                "remove_baseline": `remove_baseline`,
                "time_bins": time_bins,
                "erg_max": erg_max, "erg_min": erg_min,
                "nominal_values": nominal_values}

        """
        if time_bins is not None:
            time_bins = np.array(time_bins, copy=False)
            assert time_bins.ndim == 2 and time_bins.shape[1] == 2
            # assert None is time_bin_width is time_step is time_min is time_max,
        else:
            if time_max is None:
                time_max = self.times[-1]
            if time_min is None:
                time_min = -time_bin_width / 2

            assert None not in [time_min, time_bin_width, time_step]
            time_max = time_min + time_step * ((time_max - time_min) // time_step)
            time_centers = np.arange(time_min, time_max + time_step, time_step)
            lbins = time_centers - time_bin_width / 2
            rbins = time_centers + time_bin_width / 2
            time_bins = list(
                zip(np.where(lbins >= time_min, lbins, time_min), np.where(rbins <= time_max, rbins, time_max)))

        time_bins = np.array(time_bins, copy=False)

        ys = []

        if convolve_overlay_sigma is not None:
            ys_convolved = []
            assert isinstance(convolve_overlay_sigma, (int, float))
            convolve_sigma = int(convolve_overlay_sigma / np.mean(self.bin_widths))

        else:
            ys_convolved = None
            convolve_sigma = None

        def convolve(__y__):
            if not nominal_values:
                __y__ = unp.nominal_values(__y__)
            return convolve_gauss(__y__, convolve_sigma)

        labels4frames = []
        assert len(time_bins) > 0
        bin_edges = None

        assert len(time_bins), "No time bins! Mistakes were made (probably by you just now)."

        for (b0, b1) in time_bins:
            _y, bin_edges = self.get_erg_spectrum(erg_min, erg_max, b0, b1, eff_corr=eff_corr,
                                                  nominal_values=nominal_values,
                                                  return_bin_edges=True)

            _y /= (b1 - b0)
            _y *= scale

            if time_scale_func is not None:
                _y *= time_scale_func(b0, b1)

            if remove_baseline:
                _y -= rolling_median(45, _y)
            ys.append(_y)
            if ys_convolved is not None:
                ys_convolved.append(convolve(_y))
            labels4frames.append(f"{b0} <= t < {b1} ({0.5 * (b0 + b1)})")

        if interactive_plot is None:
            interactive_plot = InteractivePlot(labels4frames, "time ")

        x = (bin_edges[1:] + bin_edges[:-1]) / 2

        if leg_label is None:
            leg_label = self.path.name if self.path is not None else ""

        color = interactive_plot.add_ys(x, ys, leg_label=leg_label, line_type='hist', return_color=True)
        if ys_convolved is not None:
            interactive_plot.add_ys(x, ys_convolved, line_type=None, opacity=0.65,
                                    leg_label=f'{leg_label}*{convolve_sigma}')

        tot_time = (time_bins[-1][-1] - time_bins[0][0])

        if nominal_values:
            tot_y = np.mean(ys, axis=0)
            error_y = None
        else:
            tot = np.mean(ys, axis=0)
            tot_y = unp.nominal_values(tot)
            error_y = unp.std_devs(tot)
        # tot_y = self.get_erg_spectrum(erg_min=erg_min, erg_max=erg_max, time_min=time_min, time_max=time_max,
        #                               remove_baseline=remove_baseline, nomin)

        interactive_plot.add_persistent(x, tot_y, yerr=error_y, leg_label='All time', opacity=0.5, line_type='hist')

        # interactive_plot.fig.update_layout(
        #     scene=dict(
        #         yaxis=dict(range=[0, interactive_plot.max_y])))

        if not dont_plot:
            interactive_plot.plot()

        return {"interactive_plot": interactive_plot, "remove_baseline": remove_baseline, "time_bins": time_bins,
                "erg_max": erg_max, "erg_min": erg_min,
                "nominal_values": nominal_values,
                'convolve_overlay_sigma': convolve_overlay_sigma,
                'time_scale_func': time_scale_func}


def debug_list() -> ListSpectra:
    n = 25000
    erg_bins = np.arange(0, 2000)
    times = (np.random.random(n) + 0.5)*100
    times = times[np.argsort(times)]
    energies = np.random.uniform(erg_bins[0], erg_bins[-1], n)
    channels_list = np.floor(energies)
    return ListSpectra(channels_list, times, erg_bins, Path().cwd()/'test', start_time=datetime.datetime.now())


def get_test_spec1():
    return ListSpectra.from_pickle(Path(__file__).parent/'misc_data'/'test_spec1.pylist')


def get_test_spec2():
    return ListSpectra.from_pickle(Path(__file__).parent/'misc_data'/'test_spec2.pylist')


if __name__ == '__main__':
    # import  cProfile
    # from line_profiler import line_profiler
    s1 = get_test_spec1()
    s2 = get_test_spec2()

    def f():
        ListSpectra.__add__(s1, s2)


    from line_profiler import LineProfiler

    lprofiler = LineProfiler()

    lprofiler.add_function(ListSpectra.__add__)
    lp_wrapper = lprofiler(f)

    lprofiler.print_stats()

