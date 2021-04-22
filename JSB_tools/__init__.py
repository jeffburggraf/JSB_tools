"""
Core functions like ROOT_Loop, as well as functions that I didn't know where else to put
"""
import os
import sys
# from .outp_reader import OutP
import warnings
from abc import abstractmethod, ABCMeta
from openmc.data import atomic_weight
import re
from typing import List, Dict
import numpy as np
import time
from numbers import Number
from itertools import islice
from sortedcontainers import SortedDict
from pathlib import Path
from typing import Union, Sequence
import pickle
from atexit import register
from dataclasses import dataclass
cwd = Path(__file__).parent
from scipy.interpolate import interp1d
from uncertainties import unumpy as unp
from uncertainties import UFloat, ufloat
import time
from matplotlib import pyplot as plt
from JSB_tools.TH1 import TH1F


try:
    import ROOT
    root_exists = True
except ModuleNotFoundError:
    root_exists = False


class ProgressReport:
    def __init__(self, i_final, sec_per_print=2, i_init=0):
        self.__i_final__ = i_final
        self.__i_init__ = i_init
        self.__sec_per_print__ = sec_per_print
        # self.__i_current__ = i_init
        self.__next_print_time__ = time.time() + sec_per_print
        self.__init_time__ = time.time()
        self.__rolling_average__ = []

    def __report__(self, t_now, i):
        evt_per_sec = (i-self.__i_init__)/(t_now - self.__init_time__)
        self.__rolling_average__.append(evt_per_sec)
        evt_per_sec = np.mean(self.__rolling_average__)
        if len(self.__rolling_average__) >= 5:
            self.__rolling_average__ = self.__rolling_average__[:5]
        evt_remaining = self.__i_final__ - i
        sec_remaining = evt_remaining/evt_per_sec
        sec_per_day = 60**2*24
        days = sec_remaining//sec_per_day
        hours = (sec_remaining % sec_per_day)//60**2
        minutes = (sec_remaining % 60**2)//60
        sec = (sec_remaining % 60)
        msg = " {0} seconds".format(int(sec))
        if minutes:
            msg = " {0} minutes,".format(minutes) + msg
        if hours:
            msg = " {0} hours,".format(hours) + msg
        if days:
            msg = "{0} days,".format(days) + msg
        print(msg + " remaining.", i/self.__i_final__)

    def log(self, i):
        t_now = time.time()
        if t_now > self.__next_print_time__:
            self.__report__(t_now, i)
            self.__next_print_time__ += self.__sec_per_print__


def closest(sorted_dict: SortedDict, key):
    """Return closest key in `sorted_dict` to given `key`."""
    assert isinstance(sorted_dict, SortedDict)
    assert len(sorted_dict) > 0
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))


class TBrowser:
    def __init__(self):
        assert root_exists, 'Must install ROOT to use TBRowser'
        tb = ROOT.TBrowser()
        while type(tb.GetBrowserImp()) is not ROOT.TBrowserImp:
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
        del tb


def ROOT_loop():
    try:
        import time
        while True:
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
    except ModuleNotFoundError:
        warnings.warn('ROOT not installed. Cannot run ROOT_loop')


class FileManager:
    def __init__(self, path_to_root_dir: Union[str, Path], recreate=False):
        """
        todo: Make it so files in the current/any sub dir are valid. The 'root_dir' is just the dir that containes the
            __file_info__.pickle.
            Make  "__file_info__.pickle" a hidden file
            This is a good place to use doctests

        Args:
            path_to_root_dir:
            recreate:
        Examples:


        """

        self.root_directory = Path(path_to_root_dir)
        assert self.root_directory.parent.exists() and self.root_directory.parent.is_dir(),\
            f'Supplied root directory, "{self.root_directory}", is not a valid directory'
        if not self.root_directory.exists():
            print(f'Creating directory for FileContainer:\n{self.root_directory}')
            self.root_directory.mkdir()
        self.file_lookup_data: Dict[Path, dict] = {}

        self.lookup_path = self.root_directory/"__file_lookup__.pickle"
        try:
            with open(self.lookup_path, 'rb') as f:
                self.file_lookup_data = pickle.load(f)
        except FileNotFoundError:
            pass

        if recreate:
            self.file_lookup_data = {}

        for path in self.file_lookup_data.copy():
            if not path.exists():
                warnings.warn(f'\nFile, "{path}", was expected, but is missing.')

        register(self.__at_exit__)

    def __save_lookup_data__(self):
        with open(self.lookup_path, 'wb') as f:
            pickle.dump(self.file_lookup_data, f)

    def add_path(self, path, missing_ok=False, **lookup_attributes):
        path = self.root_directory/Path(path)
        if not missing_ok:
            assert path.exists(), f'The path, "{path}", does not exist. Cannot add this path to FileManager'
        assert not path.is_dir(), f'The path, "{path}", is a directory.'
        assert path not in self.file_lookup_data, f'Cannot add path, "{path}", to FileManager twice.'
        assert lookup_attributes not in self.file_lookup_data.values(),\
            f'FileManger requires a unique set of attributes for each file added.\n' \
            f'"{lookup_attributes}" has already been used.'
        self.file_lookup_data[path] = lookup_attributes
        self.__save_lookup_data__()

    def get_path(self, **lookup_kwargs) -> Union[None, Path]:
        for path, attribs in self.file_lookup_data.items():
            if lookup_kwargs == attribs:
                return path

    def get_paths(self, **lookup_kwargs) -> Dict[Path, dict]:
        """
        Return list of all paths for which every key/value in lookup_kwargs appears in file's keys/values.
        Args:
            **lookup_kwargs: key/values

        Returns:

        """
        lookup_kwargs = set(lookup_kwargs.items())
        matches = {}
        for path, attribs in self.file_lookup_data.items():
            attribs_set = set(attribs.items())
            if len(lookup_kwargs - attribs_set) == 0:
                matches[path] = attribs
        return matches

    def pickle_data(self, data, path=None, **lookup_attributes):
        if path is None:
            i = 0
            while path := (self.root_directory/f"file_{i}.pickle"):
                i += 1
                if path not in self.file_lookup_data:
                    break
        path = self.root_directory/path

        with open(path, 'wb') as f:
            pickle.dump(data, f)
        self.add_path(path, **lookup_attributes)

    def unpickle_data(self, **lookup_kwargs):
        path = self.get_path(**lookup_kwargs)

        with open(path, 'rb') as f:
            return pickle.load(f)

    def __at_exit__(self):
        self.__save_lookup_data__()

    @property
    def available_files(self):
        outs = []
        for path, keys_values in self.file_lookup_data.items():

            outs.append(f'{keys_values}   {path}  [{"exists" if path.exists() else "missing"}]')
        return '\n'.join(outs)

    def __repr__(self):
        return "FileManager\nAvailable files:\nAttribs\tPaths\n{}".format(self.available_files)

    def clean(self):
        for path in self.file_lookup_data.keys():
            path = Path(path)
            path.unlink(missing_ok=True)
        self.lookup_path.unlink(missing_ok=True)


def interp1d_errors(x: Sequence[float], y: Sequence[UFloat], x_new: Sequence[float], order=2):
    """
    Extends interpolation to data with errors
    Args:
        x:
        y: uncertain array
        x_new: Values to interpolate.
        order:

    Returns: unp.uarray

    """
    orders = {0: 'zero', 1:'linear', 2: 'quadratic', 3: 'cubic'}
    assert isinstance(order, int)
    assert order in orders, f'Invalid order, "{order}". Valid are:\n\t{list(orders.keys())}'
    order = orders[order]
    assert hasattr(y, '__iter__')
    assert hasattr(x, '__iter__')
    x = np.array(x)
    if all(x[np.argsort(x)] == x):
        assume_sorted = True
    else:
        assume_sorted = False

    assert hasattr(x_new, '__iter__')
    if not isinstance(y[0], UFloat):
        y = unp.uarray(y, np.zeros_like(y))
    if isinstance(x[0], UFloat,):
        raise NotImplementedError('Errors in x not implemented yet. Maybe someday')
    y_errors = unp.std_devs(y)
    y_nominal = unp.nominal_values(y)
    print(x, x_new)
    new_nominal_ys = interp1d(x, y_nominal, kind=order, copy=False, bounds_error=False, fill_value=(0, 0), assume_sorted=assume_sorted)(x_new)
    new_stddev_ys = interp1d(x, y_errors, kind=order, copy=False, bounds_error=False,  fill_value=(0, 0), assume_sorted=assume_sorted)(x_new)
    return unp.uarray(new_nominal_ys, new_stddev_ys)


class ROOTFitBase:
    def __init__(self, x, y, x_err, y_err):
        assert hasattr(x, '__iter__')
        assert hasattr(y, '__iter__')

        if isinstance(x[0], UFloat):
            assert x_err is not None
            x_err = np.array([i.std_dev for i in x])
            x = np.array([i.n for i in x])
        if isinstance(y[0], UFloat):
            assert y_err is not None
            y_err = np.array([i.std_dev for i in y])
            y = np.array([i.n for i in y])

        if x_err is None:
            x_err = np.zeros_like(x)
        if y_err is None:
            y_err = np.zeros_like(x)

        self.x = np.array(x, dtype=np.float)
        self.y = np.array(y, dtype=np.float)
        self.x_err = np.array(x_err, dtype=np.float)
        self.y_err = np.array(y_err, dtype=np.float)

    def eval_fit(self, x: Union[None, Number, Sequence] = None) -> Sequence[UFloat]:
        """
        Eval the fit result at x.
        Args:
            x:

        Returns:unp.uarray
        """
        assert hasattr(self, '__tf1__'), 'Subclass must have a __tf1__ attribute'
        assert hasattr(self, '__ROOT_fit_result__'), 'Subclass must have a __ROOT_fit_result__ attribute'
        if x is None:
            x = self.x
        if not hasattr(x, '__iter__'):
            x = [x]

        out_nominal = np.array([self.__tf1__.Eval(_x) for _x in x])
        _x = np.array(x, dtype=np.float)
        out_error = np.zeros_like(_x)
        self.__ROOT_fit_result__.GetConfidenceIntervals(len(_x), 1, 1, _x, out_error, 0.68, False)
        if len(out_error) == 1:
            return ufloat(out_nominal[0], out_error[0])
        return unp.uarray(out_nominal, out_error)

    def plot_fit(self, ax=None, title=None):
        if ax is None:
            ax = plt.gca()
            plt.figure()
        ax.errorbar(self.x, self.y, yerr=self.y_err, xerr=self.x_err, label="Data", ls='None', marker='o', zorder=0)
        fit_errs = unp.std_devs(self.eval_fit())
        fit_ys = unp.nominal_values(self.eval_fit())
        ax.fill_between(self.x, fit_ys-fit_errs, fit_ys+fit_errs, color='grey', alpha=0.5, label='Fit error')
        ax.plot(self.x, unp.nominal_values(self.eval_fit()), label='Fit', ls='--')
        ax.legend()
        if title is not None:
            ax.set_title(title)
        return ax


class PeakFit(ROOTFitBase):
    def __init__(self, hist: TH1F, center_guess, min_x=None, max_x=None, sigma_guess=1):
        if isinstance(center_guess, UFloat):
            center_guess = center_guess.n
        assert isinstance(center_guess, (float, int))
        if not hist.is_density:
            warnings.warn('Histogram passed to fit_peak may not have density as bin values!'
                          ' To fix this, do hist /= hist.binvalues before passing to peak_fit')
        # Set background guess before range cut is applied. This could be useful when one wants the background
        # estimate to be over a larger range that may include other interfering peaks
        bg_guess = hist.median_y.n

        if min_x is not None or max_x is not None:
            # cut hist
            hist = hist.remove_bins_outside_range(min_x, max_x)
        super(PeakFit, self).__init__(hist.bin_centers, unp.nominal_values(hist.bin_values), hist.bin_widths,
                                      unp.std_devs(hist.bin_values))

        func = self.__tf1__ = ROOT.TF1('peak_fit', '[0]*TMath::Gaus(x,[1],[2], kTRUE) + [3]')
        amp_guess = np.sum(unp.nominal_values(hist.bin_widths * (hist.bin_values - hist.median_y)))
        func.SetParameter(0, amp_guess)

        func.SetParameter(1, center_guess)
        func.SetParameter(2, sigma_guess)
        func.SetParameter(3, bg_guess)
        self.__ROOT_fit_result__ = hist.__ROOT_hist__.Fit('peak_fit', "SN")
        self.amp = ufloat(func.GetParameter(0), func.GetParError(0))
        self.center = ufloat(func.GetParameter(1), func.GetParError(1))
        self.sigma = ufloat(func.GetParameter(2), func.GetParError(2))
        self.bg = ufloat(func.GetParameter(3), func.GetParError(3))


class PolyFit(ROOTFitBase):
    def __init__(self, x, y, x_err=None, y_err=None, order=1):
        super(PolyFit, self).__init__(x, y, x_err, y_err)
        self.tgraph = ROOT.TGraphErrors(len(x), self.x, self.y, self.x_err, self.y_err)

        f_name = f"pol{order}"
        # self.__tf1__ = ROOT.TF1(f_name, f_name)
        self.__ROOT_fit_result__ = self.tgraph.Fit(f_name, "S")
        self.__tf1__ = self.tgraph.GetListOfFunctions().FindObject(f_name)
        self.coeffs = [ufloat(self.__tf1__.GetParameter(i), self.__tf1__.GetParError(i)) for i in range(order+1)]

