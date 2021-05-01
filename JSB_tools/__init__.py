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
from lmfit.models import GaussianModel


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
    new_nominal_ys = interp1d(x, y_nominal, kind=order, copy=False, bounds_error=False, fill_value=(0, 0), assume_sorted=assume_sorted)(x_new)
    new_stddev_ys = interp1d(x, y_errors, kind=order, copy=False, bounds_error=False,  fill_value=(0, 0), assume_sorted=assume_sorted)(x_new)
    return unp.uarray(new_nominal_ys, new_stddev_ys)


class ROOTFitBase:
    """
    Base class that handles the leg work of doing a fit using ROOT.
    The base class must have an attribute named __ROOT_fit_result__ and __tf1__, explained below.
        __ROOT_fit_result__
            A ROOT.TFitResultPtr instance. For example,these are created by the following code:
                hist.Fit(funcname, 'S')
                or,
                tgraph.Fit(funcname, 's')
        __tf1__
            A ROOT.TF1 instance. For example,these are created by the following code:
                tgraph.Fit("pol3", 'S')
                self.__tf1__ = tgraph.GetListOfObjects().FindObject("pol3")
                or,
                self.__tf1__ = ROOT.Tf1(funcname, "[0] + [1]*TMath::Log(x) + [1]*TMath::Log(x)^2")

    Call super(sub_cass, self).__init__(x, y, x_err, y_err) to initialize `x`, `y`, `x_err`, and `y_err`.
    `x` and `y` may be a unp.array. If not, then uncertainties can may be optionally specified by `x_err` and `y_err`

    """
    def __init__(self, x: Sequence[Union[UFloat, Number]], y: Sequence[Union[UFloat, Number]],
                 x_err: Sequence[Number], y_err: Sequence[Number], max_calls=10000):
        """
        Initializes data for fitting.
        Args:
            x:
            y:
            x_err:
            y_err:
        """
        ROOT.TVirtualFitter.SetMaxIterations(max_calls*10)
        # TVirtualFitter::SetMaxIterations(n)
        assert hasattr(x, '__iter__')
        assert hasattr(y, '__iter__')

        if isinstance(x[0], UFloat):
            assert x_err is None, "`x` is a unp.uarray, so x_err` cannot be specified!"
            x_err = np.array([i.std_dev for i in x])
            x = np.array([i.n for i in x])
        if isinstance(y[0], UFloat):
            assert y_err is None, "`y` is a unp.uarray, so y_err` cannot be specified!"
            y_err = np.array([i.std_dev for i in y])
            y = np.array([i.n for i in y])

        if x_err is None:
            x_err = np.zeros_like(x)
        if y_err is None:
            y_err = np.zeros_like(x)

        arg_sort = np.argsort(x)

        self.x = np.array(x, dtype=np.float)[arg_sort]
        self.y = np.array(y, dtype=np.float)[arg_sort]
        self.x_err = np.array(x_err, dtype=np.float)[arg_sort]
        self.y_err = np.array(y_err, dtype=np.float)[arg_sort]

    def eval_fit(self, x: Union[None, Number, Sequence] = None) -> Union[Sequence[UFloat], ufloat]:
        """
        Eval the fit result at points specified by `x`.
        Args:
            x: data points to evaluate function at

        Returns:unp.uarray
        """
        assert hasattr(self, '__tf1__'), 'Subclass must have a __tf1__ attribute. Example: __tf1__ = ' \
                                         'tgraph.GetListOfFunctions().FindObject(f_name) '
        assert hasattr(self, '__ROOT_fit_result__'), "Subclass must have a __ROOT_fit_result__ attribute. " \
                                                     "\nExample: self.__ROOT_fit_result__ = " \
                                                     "hist.__ROOT_hist__.Fit('peak_fit', 'SN')"
        __ROOT_fit_result__ = getattr(self, '__ROOT_fit_result__')
        __tf1__ = getattr(self, '__tf1__')
        assert isinstance(__tf1__, ROOT.TF1), f"Invalid type of ROOTFitBase subclass attribute, '__tf1__'. " \
                                              f"Must be of type ROOT.TF1, not '{type(__tf1__)}'"
        assert isinstance(__ROOT_fit_result__, ROOT.TFitResultPtr),\
            f"Invalid type of ROOTFitBase subclass attribute, '__ROOT_fit_result__'. " \
            f"Type must be ROOT.TFitResultPtr, not '{type(__ROOT_fit_result__)}'"
        if x is None:
            x = self.x
        if not hasattr(x, '__iter__'):
            x = [x]
        if isinstance(x[0], UFloat):
            x = unp.nominal_values(x)

        out_nominal = np.array([__tf1__.Eval(_x) for _x in x])
        _x = np.array(x, dtype=np.float)
        out_error = np.zeros_like(_x)
        __ROOT_fit_result__.GetConfidenceIntervals(len(_x), 1, 1, _x, out_error, 0.68, False)  # fills out_error
        if len(out_error) == 1:  # if `x` is a scalar value
            return ufloat(out_nominal[0], out_error[0])
        return unp.uarray(out_nominal, out_error)

    def plot_fit(self, ax=None, x=None, x_label=None, y_label=None, title=None):
        """
        Plots the fit result.
        Args:
            ax:
            x: Points to plot fit result at.
            x_label: axis labels
            y_label:
            title:

        Returns: ax

        """
        if ax is None:
            plt.figure()
            ax = plt.gca()

        if x is None:
            x = np.linspace(min(self.x), self.x[-1], 3*len(self.x))
        ax.errorbar(self.x, self.y, yerr=self.y_err, xerr=self.x_err, label="Data", ls='None', marker='o', zorder=0)
        fit_errs = unp.std_devs(self.eval_fit(x))
        fit_ys = unp.nominal_values(self.eval_fit(x))
        ax.fill_between(x, fit_ys-fit_errs, fit_ys+fit_errs, color='grey', alpha=0.5, label='Fit error')
        ax.plot(x, fit_ys, label='Fit', ls='--')
        ax.legend()
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if title is not None:
            ax.set_title(title)
        return ax


class PeakFit(ROOTFitBase):
    def guess(self, x, y, y_err):
        model = GaussianModel()
        weights = 1.0/np.where(y_err == 0, 1, y_err)
        params = model.guess(y, x=x)
        fit_result = model.fit(data=(y - np.median(y)), x=x, weights=weights, params=params)
        print(fit_result.plot_fit())
        plt.show()

    @classmethod
    def from_points(cls, x, y, center_guess=None, y_err=None, sigma_guess=1):
        assert len(x) == len(y)
        assert not isinstance(x[0], UFloat)

        if y_err is None:
            assert not isinstance(y[0], UFloat)
        else:
            y = unp.uarray(y, y_err)
        # hist = TH1F.points_to_bins(x)
        hist = TH1F.from_x_and_y(x, y, y_err=y_err)
        # hist += y
        return cls(hist, center_guess=center_guess, sigma_guess=sigma_guess)

    def __init__(self, hist: TH1F, center_guess=None, min_x=None, max_x=None, sigma_guess=None):
        if isinstance(center_guess, UFloat):
            center_guess = center_guess.n
        if center_guess is None:
            center_guess = np.median(hist.bin_centers)

        assert isinstance(center_guess, Number), f'Center guess must be a number, not {type(center_guess)}'
        if not hist.is_density:
            warnings.warn('Histogram passed to fit_peak may not have density as bin values!'
                          ' To fix this, do hist /= hist.binvalues before passing to peak_fit')
        # Set background guess before range cut is applied. This could be useful when one wants the background
        # estimate to be over a larger range that may include other interfering peaks
        bg_guess = hist.median_y.n
        # self.guess(hist.bin_centers, hist.nominal_bin_values, hist.bin_std_devs)


        if min_x is not None or max_x is not None:
            # cut hist
            hist = hist.remove_bins_outside_range(min_x, max_x)
        super(PeakFit, self).__init__(hist.bin_centers, unp.nominal_values(hist.bin_values), hist.bin_widths,
                                      unp.std_devs(hist.bin_values))

        _temp_hist = hist - bg_guess
        _temp_hist.__ROOT_hist__.Fit('gaus', '0Q')
        _g_tf1 = _temp_hist.__ROOT_hist__.GetListOfFunctions().FindObject('gaus')
        if sigma_guess is None:
            sigma_guess = _g_tf1.GetParameter(2)
        amp_guess = _g_tf1.GetParameter(0)*_g_tf1.GetParameter(2)*np.sqrt(2*np.pi)
        # print('Gauss fit par[0]: ', _g_tf1.GetParameter(0)*_g_tf1.GetParameter(2)*np.sqrt(2*np.pi))


        func = self.__tf1__ = ROOT.TF1('peak_fit', '[0]*TMath::Gaus(x,[1],[2], kTRUE) + [3]')
        # amp_guess = np.sum(unp.nominal_values(hist.bin_widths * (hist.bin_values - hist.median_y)))
        func.SetParameter(0, amp_guess)

        func.SetParameter(1, center_guess)
        func.SetParLimits(2, 1E-10, 100)
        func.SetParameter(2, abs(sigma_guess))
        # func.SetParLimits(3, 0, 1E10)
        func.SetParameter(3, bg_guess)

        self.__ROOT_fit_result__ = hist.__ROOT_hist__.Fit('peak_fit', "SNMBFE")
        self.amp = ufloat(func.GetParameter(0), func.GetParError(0))
        self.center = ufloat(func.GetParameter(1), func.GetParError(1))
        self.sigma = ufloat(func.GetParameter(2), func.GetParError(2))
        self.bg = ufloat(func.GetParameter(3), func.GetParError(3))


class PolyFit(ROOTFitBase):
    """Basic polynomial fit."""
    def __init__(self, x, y, x_err=None, y_err=None, order=1):
        super(PolyFit, self).__init__(x, y, x_err, y_err)
        self.tgraph = ROOT.TGraphErrors(len(x), self.x, self.y, self.x_err, self.y_err)

        f_name = f"pol{order}"
        # self.__tf1__ = ROOT.TF1(f_name, f_name)
        self.__ROOT_fit_result__ = self.tgraph.Fit(f_name, "SME")
        self.__tf1__ = self.tgraph.GetListOfFunctions().FindObject(f_name)
        self.coeffs = [ufloat(self.__tf1__.GetParameter(i), self.__tf1__.GetParError(i)) for i in range(order+1)]


class LogPolyFit(ROOTFitBase):
    """
    A fit of the form e^(c0 + c1 log(x) + c2 log(x)^2 + ... + cn log(x)^n).
    This expression is good for fitting efficiency curves from HPGe detectors.

    """
    def __init__(self,  x, y, x_err=None, y_err=None, order=2, fix_coeffs_to_guess=None):
        """
        Log poly fit.
        Args:
            x: array or unp.uarray
            y: array or unp.uarray
            x_err:
            y_err:
            order: Order of polynomial
            fix_coeffs_to_guess: Sometimes it is helpful to fix a coeff to the value determined from the initial
                fit in Log-Log space, since the fitting in linear here (this is how the parameters are "guessed").
                This can help reduce issues during the final non-linear fit. Recommended to fix the highest order params
                first.
        """
        if fix_coeffs_to_guess is None:
            fix_coeffs_to_guess = []
        assert hasattr(fix_coeffs_to_guess, '__iter__')
        assert all(isinstance(i, int) for i in fix_coeffs_to_guess)
        assert all([i in range(order+1) for i in fix_coeffs_to_guess]),\
            "Values in `fix_coeffs_to_guess` must be within range [0, order] "
        assert len(fix_coeffs_to_guess) < order, "Too many parameters are fixed."

        super(LogPolyFit, self).__init__(x, y, x_err, y_err)
        log_x = unp.log(unp.uarray(self.x, self.x_err))
        log_y = unp.log(unp.uarray(self.y, self.y_err))
        log_y_err = unp.std_devs(log_y)
        log_x_err = unp.std_devs(log_x)
        log_y = unp.nominal_values(log_y)
        log_x = unp.nominal_values(log_x)
        tgraph = ROOT.TGraphErrors(len(x), log_x, log_y, log_x_err, log_y_err)
        f_name = f'pol{order}'
        tgraph.Fit(f_name, 'S')
        __tf1__ = tgraph.GetListOfFunctions().FindObject(f_name)
        coeffs = [__tf1__.GetParameter(i) for i in range(order+1)]
        formula = "+".join([f'TMath::Log(x)**{i}*[{i}]' for i in range(order+1)])
        formula = f"2.71828**({formula})"

        self.__tf1__ = self.__tf1__ = ROOT.TF1("log_fit", formula)

        self.tgraph = ROOT.TGraphErrors(len(x), self.x, self.y, self.x_err, self.y_err)

        for i in range(order + 1):
            if i in fix_coeffs_to_guess:
                self.__tf1__.FixParameter(i, coeffs[i])
            else:
                self.__tf1__.SetParameter(i, coeffs[i])
        self.__ROOT_fit_result__ = self.tgraph.Fit('log_fit', 'S')
        self.coeffs = [ufloat(self.__tf1__.GetParameter(i), self.__tf1__.GetParError(i)) for i in range(order+1)]

