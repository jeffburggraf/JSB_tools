import os
import sys
# from .outp_reader import OutP
import warnings
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
from uncertainties import UFloat
import time
from matplotlib import pyplot as plt


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
        # assert len(matches) != 0, f'No file with matching lookup keys/values: {lookup_kwargs}\n' \
        #                           f'Available files:\n{self.available_files}'
        # assert len(matches) == 1, 'Multiple matching keys/values:\n{}' \
        #     .format("\n".join(["{}   {}".format(self.file_lookup_data[p], Path(p).relative_to(self.root_directory))
        #                        for p in matches]))
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


def interp1d_errors(x: Sequence[float], y: Sequence[UFloat], x_new: Sequence[float], order=2, n_samples=30):
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
    new_nominal_ys = interp1d(x, y_nominal, kind=order, copy=False, assume_sorted=assume_sorted)(x_new)
    new_stddev_ys = interp1d(x, y_errors, kind=order, copy=False, assume_sorted=assume_sorted)(x_new)
    # plt.errorbar(x, y_nominal, y_errors, label='Actual', marker='o')
    # plt.errorbar(x_new, new_nominal_ys, new_stddev_ys, label='Interped')
    # plt.legend()
    # plt.show()
    return unp.uarray(new_nominal_ys, new_stddev_ys)




    print(time.time() - t0)
    print('len of y: ', len(y))
    print(dys.shape, ys.shape)


# ergs = np.array([0, 59.9, 88.4, 122, 166, 392, 514, 661, 898, 1173, 1332, 1835], dtype=np.float)
# effs = np.array([0, 0.06, 0.1, 0.144, 0.157, 0.1, 0.07, 0.05, 0.04, 0.03, 0.027, 0.018])
#
# x = ergs
#
# rel_errors = np.arange(len(x))
# rel_errors = rel_errors/max(rel_errors)
# rel_errors *= 0.25
# y = unp.uarray(effs, effs*rel_errors)
#
# # plt.plot(x, rel_errors)
#
# interp1d_errors(x, y, np.linspace(ergs[0], ergs[-1], 4*len(ergs)), order=1, n_samples=100)
# plt.show()
