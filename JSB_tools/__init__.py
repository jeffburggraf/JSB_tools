"""
Core functions like ROOT_Loop, as well as functions that I didn't know where else to put
"""
from __future__ import annotations
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
# from JSB_tools.TH1 import TH1F
import sys
from scipy.stats import norm
import matplotlib as mpl
import traceback
from JSB_tools.nuke_data_tools import Nuclide, FissionYields
import matplotlib.ticker as ticker
from uncertainties import UFloat
from scipy import ndimage
try:
    import ROOT
    root_exists = True
except ModuleNotFoundError:
    root_exists = False


cwd = Path(__file__).parent

style_path = cwd/'mpl_style.txt'


def human_friendly_time(time_in_seconds, unit_precision=2):
    """

    Args:
        time_in_seconds:
        unit_precision: Number of units to print, e.g. for 3 months and 2 days and 10 minutes
            If 1: 3 months
            If 2: 3 months 2 days
            If 3: 3 months 2 days  # num hours of 0 is omitted
            If 4: 3 months 2 days 10 minutes

    Returns:

    """

    rel_error = None
    time = time_in_seconds
    assert unit_precision >= 1
    if isinstance(time_in_seconds, UFloat):
        time = time_in_seconds.n
        rel_error = time_in_seconds.std_dev/time_in_seconds.n

    if time == np.inf or time == np.nan:
        return str(time)

    if time < 1:
        out = "{:.2e} seconds ".format(time.n)
        if rel_error is not None:
            out += f'+/- {100*rel_error:.1f}%'
        return out
    elif time < 60:
        out = "{:.1f} seconds ".format(time.n)
        if rel_error is not None:
            out += f'+/- {100 * rel_error:.1f}%'
        return out

    seconds_in_a_minute = 60
    seconds_in_a_hour = 60 * seconds_in_a_minute
    seconds_in_a_day = seconds_in_a_hour * 24
    seconds_in_a_month = seconds_in_a_day * 30
    seconds_in_a_year = 12 * seconds_in_a_month

    n_seconds = time % seconds_in_a_minute
    n_minutes = (time % seconds_in_a_hour) / seconds_in_a_minute
    n_hours = (time % seconds_in_a_day) / seconds_in_a_hour
    n_days = (time % seconds_in_a_month) / seconds_in_a_day
    n_months = (time % seconds_in_a_year) / seconds_in_a_month
    n_years = (time / seconds_in_a_year)
    units = np.array(['years', 'months', 'days', 'hours', 'minutes', 'seconds'])
    values = np.array([n_years, n_months, n_days, n_hours, n_minutes, n_seconds])
    outs = []

    printables = np.where(values>1)[0]
    printables = printables[np.where(printables - printables[0] < unit_precision)]
    value, unit = None, None
    for unit, value in zip(units[printables], values[printables]):
        outs.append(f"{int(value)} {unit}")
    if unit == 'seconds':
        outs[-1] = f'{value:.2e} {unit}'
    else:
        outs[-1] = f'{value:.2f} {unit}'
    out = ' '.join(outs)
    if rel_error is not None:
        out += f' (+/- {100*rel_error:.1f}$)'

    return out


def interpolated_median(list_):
    """
    Find the median assuming data points are pulled from a unknown continuous frequency distribution.
    Args:
        list_:

    Returns:

    """
    if len(list_) == 0:
        return None
    values, freqs = np.unique(list_, return_counts=True)
    cdf = np.cumsum(freqs)
    return np.interp(cdf[-1]/2., cdf, values)


def rolling_median(window_width, values):
    """
    Rolling median (in the y direction) over a uniform window. Window is clipped at the edges.
    Args:
        window_width: Size of independent arrays for median calculations.
        values: array of values

    Returns:

    """
    if isinstance(values[0], UFloat):
        _v = unp.nominal_values(values)
        rel_errors = unp.std_devs(values)/np.where(_v != 0, _v, 1)
        values = _v
    else:
        rel_errors = None

    window_width = int(window_width)
    n = min([window_width, len(values)])
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    window_indicies = (range(max([0, i - n // 2]), min([len(values) - 1, i + n // 2])) for i in range(len(values)))

    medians = np.array([np.median(values[idx]) for idx in window_indicies])
    if rel_errors is None:
        return medians
    else:
        return unp.uarray(medians, np.abs(rel_errors*medians))


def shade_plot(ax, window, color='blue', alpha=0.5, label=None):
    _ylims = ax.get_ylim()
    y1, y2 = [ax.get_ylim()[0]] * 2, [ax.get_ylim()[1]] * 2
    out = ax.fill_between(window, y1, y2, color=color, alpha=alpha, label=label)
    ax.set_ylim(*_ylims)
    return out


def calc_background(counts, num_iterations=20, clipping_window_order=2, smoothening_order=5, median_window=None):
    """

    Args:
        counts: Signal to operate on. Can be uarray
        num_iterations:
        clipping_window_order:
        smoothening_order:
        median_window: Only valid if ROOT isn't installed and thus background is calculated via a rolling median of
            width `median_window`

    Returns:

    """
    assert clipping_window_order in [2, 4, 6, 8]
    assert smoothening_order in [3, 5, 7, 9, 11, 13, 15]
    spec = ROOT.TSpectrum()
    result = unp.nominal_values(counts)
    if isinstance(counts[0], UFloat):
        rel_errors = unp.std_devs(counts)/np.where(result != 0, result, 1)
    else:
        rel_errors = None

    if not root_exists:
        warnings.warn("No ROOT. Background estimation accomplished by rolling median")
        if median_window is None:
            median_window = int(len(counts)//10)
        result = rolling_median(median_window, result)
        if rel_errors is not None:
            return unp.uarray(result, result*rel_errors)
    else:
        assert median_window is None, '`median_window` is not needed when ROOT is installed. '

    clipping_window = getattr(ROOT.TSpectrum, f'kBackOrder{clipping_window_order}')
    smoothening = getattr(ROOT.TSpectrum, f'kBackSmoothing{smoothening_order}')
    spec.Background(result, len(result), num_iterations, ROOT.TSpectrum.kBackDecreasingWindow,
                    clipping_window, ROOT.kTRUE,
                    smoothening, ROOT.kTRUE)
    if rel_errors is None:
        return result
    else:
        return unp.uarray(result, np.abs(rel_errors*result))


def mpl_style():
    plt.style.use(style_path)


def norm2d_kernel(length_x, sigma_x, length_y=None, sigma_y=None):
    if sigma_y is None:
        sigma_y = sigma_x

    if length_y is None:
        length_y = length_x
    xs = norm(scale=sigma_x).pdf(np.linspace(-length_x//2, length_x//2, length_x))
    ys = norm(scale=sigma_y).pdf(np.linspace(-length_y//2, length_y//2, length_y))
    out = [[x*y for x in xs] for y in ys]
    out = out/np.sum(out)
    return out


def convolve_gauss2d(a, sigma_x, kernel_sigma_window: int = 8, sigma_y=None):
    if sigma_y is None:
        sigma_y = sigma_x

    kernel = norm2d_kernel(length_x=int(kernel_sigma_window*sigma_x), length_y=int(kernel_sigma_window*sigma_y),
                           sigma_x=sigma_x, sigma_y=sigma_y)
    plt.imshow(kernel)
    plt.figure()
    out = ndimage.convolve(a, kernel)
    # out = convolve2d(a, kernel, mode='same', boundary='symm')
    # out = np.fft.irfft2(np.fft.rfft2(a) * np.fft.rfft2(kernel, a.shape))
    return out


def convolve_gauss(a, sigma: Union[float, int], kernel_sigma_window: int = 10, mode='same'):
    """
    Simple gaussian convolution.
    Args:
        a: The array to be convolved
        sigma: The width of the convolution (in units of array incicies)
        kernel_sigma_window: It's not efficient to make the window larger that a few sigma, so cut off at this value
        mode: See np.convolve

    Returns:

    """
    sigma = int(sigma)
    if sigma == 0:
        return a
    kernel_size = min([kernel_sigma_window * sigma, len(a)//2])
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel = norm(loc=0, scale=sigma).pdf(kernel_x)
    kernel /= np.sum(kernel)
    return np.convolve(a, kernel, mode=mode)


def mpl_hist(bin_Edges, y, yerr=None, ax=None, label=None, fig_kwargs=None, title=None, poisson_errors=True,
             return_line_color=False, **mpl_kwargs):
    """

    Args:
        bin_Edges: Left edges of bins (must be of length len(y) + 1)
        y:
        yerr:
        ax:
        label: For legend
        fig_kwargs: kwargs for mpl.figure
        title:
        poisson_errors: If True and yerr is not provided, assume Poissonian errors.
        return_line_color: If True, return mpl color of line.
        **mpl_kwargs:

    Returns:

    """
    if not len(bin_Edges) == len(y) + 1:
        raise ValueError(f'`bin_edges` must be of length: len(y) + 1, not {len(bin_Edges)} and {len(y)} ')
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        plt.figure(**fig_kwargs)
        ax = plt.gca()
    if isinstance(y[0], UFloat):
        yerr = unp.std_devs(y)
        y = unp.nominal_values(y)
    if yerr is None and poisson_errors:
        yerr = np.sqrt(np.where(y < 0, 0, y))

    if title is not None:
        ax.set_title(title)

    bin_centers = [(bin_Edges[i+1]+bin_Edges[i])/2 for i in range(len(bin_Edges)-1)]
    yp = np.concatenate([y, [y[-1]]])

    lines = ax.plot(bin_Edges, yp, label=label, ds='steps-post', **mpl_kwargs)
    c = lines[0].get_color()
    mpl_kwargs['c'] = c
    mpl_kwargs.pop('ls', None)
    mpl_kwargs.pop('color', None)

    lines.append(ax.errorbar(bin_centers, y, yerr,
                             ls='None',  **mpl_kwargs))
    if label is not None:
        ax.legend()
    if return_line_color:
        return ax, c
    else:
        return ax


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

    @property
    def elapsed_time(self):
        return time.time()-self.__init_time__

    def __report__(self, t_now, i, added_msg):
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
        sec = int(sec_remaining % 60)
        msg = " {0} seconds".format(sec)
        if minutes:
            msg = " {0} minute{1},".format(minutes, 's' if minutes > 1 else '') + msg
        if hours:
            msg = " {0} hour{1},".format(hours, 's' if hours > 1 else '') + msg
        if days:
            msg = "{0} day{1},".format(days, 's' if days > 1 else '') + msg
        print(f"{added_msg}... {msg} remaining {100*i/self.__i_final__:.2f}% complete")

    def log(self, i, msg=""):
        t_now = time.time()
        if t_now > self.__next_print_time__:
            self.__report__(t_now, i, msg)
            self.__next_print_time__ += self.__sec_per_print__
            return True
        return False


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


def cm_2_best_unit(list_or_number):
    """
    Find a good units for a number expressed in centimeters.
    e.g. 1.24E-4 cm -> 1.24 um
    Args:
        list_or_number:

    Returns: (Number in new units, new units (str))

    """
    if hasattr(list_or_number, '__iter__'):
        y = np.max(list_or_number)
        list_or_number = np.array(list_or_number)
    else:
        y = list_or_number
    unit_names = ["nm", "um", "mm", "cm", "m", "km"]
    orders = np.array([-7, -4, -1, 0, 2, 5])
    test_value = np.max(y)
    i = np.searchsorted(orders, np.log10(test_value), side='right') - 1
    i = max([0, i])
    units = unit_names[i]
    unit_conversion = 10. ** -orders[i]
    return list_or_number*unit_conversion, units


def ROOT_loop():
    try:
        import time
        while True:
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
    except ModuleNotFoundError:
        warnings.warn('ROOT not installed. Cannot run ROOT_loop')


class FileManager:
    root_files: Dict[Path, ROOT.TFile] = {}
    # todo: make gui for deleting files
    #  todo: male read only option

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
        self.__file_lookup_data: Dict[Path, dict] = {}

        # path to file that stores association information
        self.__save_path = self.root_directory / "__file_lookup__.pickle"

        if recreate:
            self.__file_lookup_data: Dict[Path, Dict] = {}
            self.__save_path.unlink(missing_ok=True)
        else:
            try:
                with open(self.__save_path, 'rb') as f:
                    self.__file_lookup_data: Dict[Path, Dict] = pickle.load(f)
            except (EOFError, FileNotFoundError) as e:
                raise Exception(f"No FileManager at {self.root_directory}. "
                                f"Maybe you meant to set `recreate` arg to True") from e

        register(self.__at_exit__)

    def __save_lookup_data__(self):
        with open(self.__save_path, 'wb') as f:
            pickle.dump(self.__file_lookup_data, f)

    @staticmethod
    def auto_gen_path(attribs: Dict, root_path, extension='') -> Path:
        """
        Generate a simple (obscure) path, and save the attribs to a text file for reference.
        Args:
            attribs:
            root_path: Root path will be prepended to name. If None, then no path is prepended
            extension:

        Returns: Absolute path

        """
        existing_paths = list(Path(root_path).iterdir())
        root_path = Path(root_path)

        def get_new_path(i):
            out = (root_path/f"{i}").with_suffix(extension)
            return out

        i = 0
        while (new_path := get_new_path(i)) in existing_paths:
            i += 1

        return new_path

    @staticmethod
    def __verify_attribs__(attribs: Dict):  #why was this here again?
        pass
        # for kv in attribs.items():
        #     try:
        #         _ = {kv}
        #     except TypeError as e:
        #         assert False, f"Type error for the following value: {kv}\n" \
        #                       f"Make sure all attribs are hashable.\nThe error:\n" \
        #                       f"\t{e}"

    def add_path(self, rel_path_or_abs_path=None, missing_ok=False, overwrite_ok=False, **lookup_attributes) -> Path:
        """
        Add a path and lookup attributes to the list of saved files.
        Args:
            rel_path_or_abs_path:  Either a path relative to the self.root_directory, or an absolute path rel. to
                sys root
            missing_ok:  Raise error if missing?
            **lookup_attributes: kwargs used for easy lookup later.
        :return: Returns path to file

        Returns: Returns path to file.

        """
        FileManager.__verify_attribs__(lookup_attributes)
        assert len(lookup_attributes) != 0, \
            "If you're not going to provide any attributes then this tool is no for you."
        if rel_path_or_abs_path is None:
            rel_path_or_abs_path = self.auto_gen_path(lookup_attributes, self.root_directory)
        rel_path_or_abs_path = Path(rel_path_or_abs_path)
        if str(rel_path_or_abs_path.anchor) != '/':
            rel_path_or_abs_path = self.root_directory / Path(rel_path_or_abs_path)
        abs_path = rel_path_or_abs_path
        if not missing_ok:
            assert abs_path.exists(), f'The path, "{abs_path}", does not exist. Use missing_ok=True to bypass this error'
        assert not abs_path.is_dir(), f'The path, "{abs_path}", is a directory.'
        if abs_path in self.__file_lookup_data:
            if lookup_attributes in self.__file_lookup_data.values():  # path and attrib identical. May overwrite
                # if overwrite_ok:  # overwrite
                #     warnings.warn(f"Overwriting FileManager reference to {abs_path}")
                if not overwrite_ok:
                    assert False, f"Cannot overwrite reference {abs_path}. Set parameter `overwrite_ok` to True"
            else:
                warnings.warn(f"Path {abs_path} used twice. Overwriting!")

        else:
            # if paths aren't identical, no identical attribs are allowed.
            assert lookup_attributes not in self.__file_lookup_data.values(), \
                f'FileManger requires a unique set of attributes for each file added.\n' \
                f'"{lookup_attributes}" has already been used.'

        self.__file_lookup_data[abs_path] = lookup_attributes
        self.__save_lookup_data__()
        return rel_path_or_abs_path

    def find_path(self, missing_ok=False, **lookup_attributes) -> Union[None, Path]:
        """
        Return the path to a file who's keys/values **exactly** match `lookup_kwargs`. There can only be one. If non
        Args:
            missing_ok: whether to raise an error if file not found
            **lookup_attributes:

        Returns:

        """
        for path, attribs in self.__file_lookup_data.items():
            if lookup_attributes == attribs:
                return path
        available_files_string = '\n'.join(map(str, self.__file_lookup_data.values()))
        if not missing_ok:
            raise FileNotFoundError(f"No file with the following matching keys/values:\n {lookup_attributes}\n"
                                    f"Currently linked files are:\n{available_files_string}")

    def find_paths(self, **lookup_attributes) -> Dict[Path, dict]:
        """
        Find of all file paths for which the set of `lookup_attributes` is a subset of the files attributes.
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
        Todo: Find a way to make debugging not found easier.
        """
        lookup_kwargs = lookup_attributes.items()
        matches = {}
        for path, attribs in self.__file_lookup_data.items():
            all_attribs_list = list(attribs.items())
            if all(a in all_attribs_list for a in lookup_kwargs):
                matches[path] = {k: v for k, v in attribs.items()}
        if len(matches) == 0:
            warnings.warn(f"No files fiund containing the following attribs: {lookup_attributes}")
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
                if file_name not in self.__file_lookup_data:
                    break
        file_name = self.root_directory / file_name

        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
        self.add_path(file_name, **lookup_attributes)

    @property
    def all_files(self) -> Dict[Path, Dict[str, str]]:
        return {k: v for k, v in self.__file_lookup_data.items()}

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

    # def __del__(self):
    #     self.__at_exit__()

    @property
    def available_files(self):
        outs = []
        for path, keys_values in self.__file_lookup_data.items():

            outs.append(f'{keys_values}   {path}  [{"exists" if path.exists() else "missing"}]')
        return '\n'.join(outs)

    def __repr__(self):
        outs = ['-'*80]
        for path, keys_values in self.__file_lookup_data.items():
            outs.append(f"{keys_values}\n\t{path}\n")
        outs[-1] = outs[-1][:-1]
        outs.append(outs[0] + '\n')
        out = "\n".join(outs)
        out = f"Files in FileManeger at '{self.__save_path}:'\n" + out
        return out
        # return "FileManager\nAvailable files:\nAttribs\tPaths\n{}".format(self.available_files)

    def clean(self):
        for path in self.__file_lookup_data.keys():
            path = Path(path)
            path.unlink(missing_ok=True)
        self.__save_path.unlink(missing_ok=True)

    def __iadd__(self, other: FileManager):
        for path, attribs in other.all_files.items():
            if path in self.__file_lookup_data:
                assert attribs == self.__file_lookup_data[path], f"Encountered two files with identical paths and " \
                                                                 "different attribs during merge. This is not allowed.\n" \
                                                                 f"{attribs}"
            self.__file_lookup_data[path] = attribs
        return self


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
    h = human_friendly_time(24*60*60*32 + 3*60+3.234455, 2)
    print(h)
    # plt.show()


