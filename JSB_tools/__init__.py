"""
Core functions like ROOT_Loop, as well as functions that I didn't know where else to put
"""

import warnings
from typing import List, Dict
import numpy as np
from itertools import islice
from sortedcontainers import SortedDict
from pathlib import Path
from typing import Union, Sequence
import pickle
from atexit import register
from scipy.interpolate import interp1d
from uncertainties import unumpy as unp
from uncertainties import UFloat, ufloat
import time
from matplotlib import pyplot as plt
from JSB_tools.TH1 import TH1F
import sys
import traceback

cwd = Path(__file__).parent

try:
    import ROOT
    root_exists = True
except ModuleNotFoundError:
    root_exists = False


class __TracePrints(object):

    def __init__(self):
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)

    def flush(self): pass


def trace_prints():
    """
    When there is a pesky print statement somewhere, use this to find it.
    Run this function at beginning of script
    """
    sys.stdout = __TracePrints()


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
    root_files:Dict[Path, ROOT.TFile] = {}

    def __init__(self, path_to_root_dir: Union[str, Path], recreate=False):
        """
        Creates a human friendly link between file and a dictionary of descriptive attributes that make it easy to
            access files created in a previous script.

        Args:
            path_to_root_dir: Path to the top directory.
            recreate: If you are loading an existing FileManager, then this must be False, else it will override the
                previous data.

        Examples:
            When a series of files (of any type) are created, they can be loaded later without the need to use a
             regex to lookup the file. e.g. many files are created from a series of MCNP simulations for which the
             energy and position of the particle source varies.

                cwd = Path(__file__).parent  # top directory of the simulations.
                f_man = FileManager(cwd, recreate=True)

                for pos in positions:
                    for energy in energies:

                        input_deck_name = f"{pos}_{energy}"

                         # The following will create a directory for the simulation where src energy=energy and source
                         #  position=pos. The function returns the path of the created input deck
                         #  (See JSB_tools.MCNP_helper.inputDeck)
                         f_path = i.write_inp_in_scope(input_deck_name) # this
                         outp_path = f_path.parent/'outp'  # this is the name of the outp file MCNP will create

                         # Even though `outp_path` doesn't exists yet, I can make a quick lookup with using FileManager
                         #  as follows:
                         f_man.add_path(outp_path, src_energy=energy, source_pos=position)
                # upon exiting the script, FileManager will save the association between the files and the key/values
                # in a pickle file named __file_lookup__.pickle in the top directory specified by `path_to_root_dir` in
                # the FileManager instantiation.

                In another script, say, that analyses the outp files, one could do the following (almost identical to
                the rotine for initially creating the FIleManeger.
                cwd = Path(__file__).parent  # top directory of the simulations.
                f_man = FileManager(cwd, recreate=False)  # NOTE THAT IS False



        todo: Make it so files in the current/any sub dir are valid. The 'root_dir' is just the dir that containes the
            __file_info__.pickle.
            Make  "__file_info__.pickle" a hidden file
            This is a good place to use doctests



        """

        self.root_directory = Path(path_to_root_dir)
        assert self.root_directory.parent.exists() and self.root_directory.parent.is_dir(),\
            f'Supplied root directory, "{self.root_directory}", is not a valid directory'
        if not self.root_directory.exists():
            print(f'Creating directory for FileContainer:\n{self.root_directory}')
            self.root_directory.mkdir()
        self.file_lookup_data: Dict[Path, dict] = {}

        # path to file that stores association information
        self.__save_path = self.root_directory / "__file_lookup__.pickle"
        try:
            with open(self.__save_path, 'rb') as f:
                self.file_lookup_data = pickle.load(f)
        except FileNotFoundError:
            pass

        if recreate:
            self.file_lookup_data = {}

        # for path in self.file_lookup_data.copy():
        #     if not path.exists():
        #         warnings.warn(f'\nLink found to non-existing file, "{path}".')

        register(self.__at_exit__)

    def __save_lookup_data__(self):
        with open(self.__save_path, 'wb') as f:
            pickle.dump(self.file_lookup_data, f)

    def add_path(self, path, missing_ok=False, **lookup_attributes):
        path = self.root_directory/Path(path)
        if not missing_ok:
            assert path.exists(), f'The path, "{path}", does not exist. Use missing_ok=True to bypass this error'
        assert not path.is_dir(), f'The path, "{path}", is a directory.'
        assert path not in self.file_lookup_data, f'Cannot add path, "{path}", to FileManager twice.'
        assert lookup_attributes not in self.file_lookup_data.values(),\
            f'FileManger requires a unique set of attributes for each file added.\n' \
            f'"{lookup_attributes}" has already been used.'
        self.file_lookup_data[path] = lookup_attributes
        self.__save_lookup_data__()

    def find_path(self, missing_ok=False, **lookup_attributes) -> Union[None, Path]:
        """
        Return the path to a file who's keys/values **exactly** match `lookup_kwargs`. There can only be one. If non
        Args:
            missing_ok: whether to raise an error if file not found
            **lookup_attributes:

        Returns:

        """
        for path, attribs in self.file_lookup_data.items():
            if lookup_attributes == attribs:
                return path
        available_files_string = '\n'.join(map(str, self.file_lookup_data.values()))
        if not missing_ok:
            raise FileNotFoundError(f"No file with the following matching keys/values:\n {lookup_attributes}\n"
                                    f"Currently linked files are:\n{available_files_string}")

    def find_paths(self, **lookup_attributes) -> Dict[Path, dict]:
        """
        Find of all file paths for which `lookup_attributes` is a subset of the files attributes.
        Return a dictionary who's keys are file paths, and values are the corresponding
            lookup attributes (all of them for the given file, not just the ones the user searched for)
        Args:
            **lookup_attributes: key/values

        Examples:
            A FileManeger exists that links files containing the following attributes:
                f1 -> {"energy": 10, "position": 3, "particle": "neutron"}
                f2 -> {"energy": 12, "position": 3, "particle": "proton"}
                f2 -> {"energy": 19, "position": 3, "particle": "proton"}
                lookup_kwargs = (position=3) will return all file paths
                lookup_kwargs = (position=3, particle=proton) will return  file paths f2 and f3
                lookup_kwargs = (energy=10) will return  file path f1
            will match with

        Returns: Dictionary,  {Path1: file_attributes1, Path2: file_attributes2, ...}

        """
        lookup_kwargs = set(lookup_attributes.items())
        matches = {}
        for path, attribs in self.file_lookup_data.items():
            attribs_set = set(attribs.items())
            if len(lookup_kwargs - attribs_set) == 0:
                matches[path] = attribs
        return matches

    def find_tree(self, tree_name="tree", **lookup_attributes) -> ROOT.TTree:
        path = self.find_path(**lookup_attributes)
        if (path is None) or not path.exists():
            raise FileNotFoundError(f"Attempted to load ROOT tree on non-existent file. Attributes:{lookup_attributes}")
        return self.__load_tree_from_path__(path=path, tree_name=tree_name)

    @staticmethod
    def __load_tree_from_path__(path, tree_name='tree'):
        if not path.exists():
            raise FileNotFoundError(f"Attempted to load ROOT tree on non-existent file, '{path}'")
        f = ROOT.TFile(str(path))
        FileManager.root_files[path] = f

        assert tree_name in map(lambda x:x.GetName(), f.GetListOfKeys()), \
            f'Invalid `tree_name`, "{tree_name}". ROOT file, "{path}", does not contain a key named "{tree_name}"'
        tree = f.Get(tree_name)
        return tree

    def find_trees(self, tree_name="tree", **lookup_attributes) -> Dict[ROOT.TTree, dict]:
        """
        Same concept of find_paths, except the dictionary keys are ROOT trees.
        Args:
            tree_name:
            **lookup_attributes:

        Returns:

        """
        matches = {}
        for path, attribs in self.find_paths(**lookup_attributes).items():
            tree = self.__load_tree_from_path__(path=path, tree_name=tree_name)

            matches[tree] = attribs
        return matches

    def pickle_data(self, data, file_name=None, **lookup_attributes):
        """
        Save `data` to pickle file with the provided `lookup_attributes`
        Args:
            data: Data to be saved
            file_name: Name of pickle file. If not provided, then pick name automatically.
            **lookup_attributes:

        Returns:

        """
        if file_name is None:
            i = 0
            while file_name := (self.root_directory / f"file_{i}.pickle"):
                i += 1
                if file_name not in self.file_lookup_data:
                    break
        file_name = self.root_directory / file_name

        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
        self.add_path(file_name, **lookup_attributes)

    def unpickle_data(self, **lookup_kwargs):
        """
        Unpickle and return the file who's keys/values match exactly
        Args:
            **lookup_kwargs:

        Returns:

        """
        path = self.find_path(**lookup_kwargs)

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
        outs = ['-'*80]
        for path, keys_values in self.file_lookup_data.items():
            outs.append(f"{keys_values}\n\t{path}\n")
        outs[-1] = outs[-1][:-1]
        outs.append(outs[0] + '\n')
        out = "\n".join(outs)
        out = f"Files in FileManeger at '{self.__save_path}:'\n" + out
        return out
        # return "FileManager\nAvailable files:\nAttribs\tPaths\n{}".format(self.available_files)

    def clean(self):
        for path in self.file_lookup_data.keys():
            path = Path(path)
            path.unlink(missing_ok=True)
        self.__save_path.unlink(missing_ok=True)


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


if __name__ == '__main__':
   pass