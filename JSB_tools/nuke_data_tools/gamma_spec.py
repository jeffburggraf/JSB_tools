import re
try:
    import ROOT
except ModuleNotFoundError:
    assert False, 'ROOT must be installed to use gamma spec. capability.'
import pickle
import numpy as np
from JSB_tools.nuke_data_tools import Nuclide
from JSB_tools.TH1 import TH1F
from warnings import warn
from GlobalValues import shot_groups
from pathlib import Path
from matplotlib import pyplot as plt
from PHELIXDataTTree import get_global_energy_bins
from numbers import Number
from scipy.signal import find_peaks
from typing import Union, Iterable, List, Dict, Tuple
from abc import ABC, abstractmethod
from matplotlib.patches import Rectangle



class HalfLifeSignature:
    def __init__(self, tree, discrete_ergs, min_time):
        self.erg_hist = TH1F(bin_left_edges=discrete_ergs)
        self.erg_hist.Project(tree, 'erg', 't>{0}'.format(min_time))



def exp_decay_maximum_likely_hood(times_measured, times_background):
    if times_background is None:
        return len(times_measured)/np.sum(times_measured)
    else:
        if len(times_measured) == len(times_background):
            return np.nan
        return (len(times_measured) - len(times_background)) / (np.sum(times_measured) - np.sum(times_background))


class PeakSignalBackgroundCut:
    def __init__(self, peak_center, signal_window_width, bg_window_width, left_bg_window_offset=0,
                 right_bg_window_offset=0):
        for var in [peak_center, signal_window_width, bg_window_width, left_bg_window_offset,
                    right_bg_window_offset]:
            assert isinstance(var, Number), 'All args passed to PeakSignalBackgroundCut must be Numbers.'

        self.peak_center = peak_center
        self.__signal_window_width__ = signal_window_width
        self.__bg_window_width__ = bg_window_width
        self.__left_bg_window_offset__ = left_bg_window_offset
        self.__right__bg_window_offset__ = right_bg_window_offset

        self.signal_window_bounds = [self.peak_center - self.__signal_window_width__/2.,
                                     self.peak_center + self.__signal_window_width__/2.]
        self.back_ground_window_bounds = np.array([[self.signal_window_bounds[0]-bg_window_width/2,
                                                    self.signal_window_bounds[0]],
                                                   [self.signal_window_bounds[1],
                                                    self.signal_window_bounds[1] + bg_window_width/2]])
        self.back_ground_window_bounds[0] -= abs(left_bg_window_offset)
        self.back_ground_window_bounds[1] += abs(right_bg_window_offset)
        print(self.back_ground_window_bounds)
        print(self.signal_window_bounds)

        # self.signal_width = self.__signal_window_width__
        # self.back_ground_width = 2*self.__bg_window_width__

    @property
    def signal_cut(self):
        return '{} <= erg && erg <= {}'.format(*self.signal_window_bounds)

    @property
    def back_ground_cut(self):
        return '({bg[0][0]}<=erg && erg <= {bg[0][1]}) || ({bg[1][0]}<=erg && erg <= {bg[1][1]})'\
            .format(bg=self.back_ground_window_bounds)

    def plot(self, ax, signal_color='red', bg_color='black', signal_label='signal window', bg_label='background window'):
        ax = ax.gca()
        ylims = ax.get_ylim()

        def make_rectangle(x0, w,  **kwargs):
            return Rectangle((x0, ylims[0]-10), width=w, height=10+ylims[1] - ylims[0], edgecolor=None, **kwargs)

        rectangles = []
        signal_rectangle = make_rectangle(self.signal_window_bounds[0], self.__signal_window_width__,
                                          color=signal_color, label=signal_label, alpha=0.7)
        rectangles.append(signal_rectangle)
        for index, bk_x_min in enumerate(self.back_ground_window_bounds):
            bk_x_min = bk_x_min[0]
            rectangles.append(make_rectangle(bk_x_min, self.__bg_window_width__/2, color=bg_color,
                                             label=bg_label if index == 0 else None, alpha=0.7))
        for r in rectangles:
            ax.add_patch(r)
        ax.set_ylim(*ylims)


class Spectrum:
    def __init__(self, **kwargs):
        """
        Only meant to be used internally. Use factory methods to create Spectrum instance.
        """
        if '__factory_method__' not in kwargs:
            warn('You should probably use one of the following factory methods to create a Spectrum instance:\n\t'
                 'Spectrum.load,Spectrum.from_tree\ninstead of calling Spectrum() directly.')
        self.tree = kwargs['tree']
        self.half_life_signature: HalfLifeSignature = kwargs['half_life_signature']
        self.discrete_ergs: Iterable[float] = kwargs['discrete_ergs']
        self.is_time_dependent: bool = kwargs['is_time_dependent']

    @classmethod
    def from_tree(cls, tree: ROOT.TTree, discrete_ergs: Union[List[Number], None], min_t_for_hl_calc: Number):
        """
        Args
            tree (TTree): ROOT TTree instance
            discrete_ergs: Iterable of all possible energies
            b_width_for_hl_calc: Bin width for half life calculations
            min_t_for_hl_calc: Starting time for half life calculations

        returns
        """
        assert isinstance(tree, ROOT.TTree), '"tree" must be a TTree instance.'
        ROOT.gROOT.LoadMacro(str(Path(__file__).parent / 'gamma_spec.c'))

        ROOT.set_up(tree)

        existing_branches = set([b.GetName() for b in tree.GetListOfBranches()])
        if 't' in existing_branches:
            is_time_dependent = True
        else:
            is_time_dependent = False

        required_branches = ['erg', 'sigma_erg', 'shot', 't', 'eff', 'eff_err', 'J']
        missing_branches = []
        for b_name in required_branches:
            if b_name not in existing_branches:
                missing_branches.append(b_name)
        if len(missing_branches) != 0:
            assert False, 'The tree passed to Spectrum.from_tree must contain all the following branches:\n' \
                          '{}\nMissing branches:\n{}'.format(required_branches, missing_branches)

        assert isinstance(discrete_ergs, Iterable), '`discrete_ergs` must be an iterable (of floats)'
        assert isinstance(min_t_for_hl_calc, Number)

        half_life_signature = _HalfLifeSignature(tree, discrete_ergs, min_t_for_hl_calc)
        out = cls(tree=tree, half_life_signature=half_life_signature, discrete_ergs=discrete_ergs,
                  is_time_dependent=is_time_dependent, __internal__=True)

        return out

    def plot_energy_spectrum(self, ax=None):
        hist = TH1F(bin_left_edges=self.discrete_ergs)
        hist.Project(self.tree, 'erg', )
        return hist.plot(ax)


    def save(self, title: str, path_to_dir: Union[str, Path]):
        """
        Save spectrum to disk for quick future use.
        Args:
            title: Unique title of spectrum
            path_to_dir: Path to directory in which to save spectrum.

        Returns:
            None

        """
        assert isinstance(title, str), '`title` argument must be a string.'
        assert isinstance(path_to_dir, (str, Path)) and Path(path_to_dir).exists(),\
            '`path_to_dir` must be a directory (str or Path instance) that exists'
        # todo

    def get_erg_calibration(self, nuclides_in_spectrum: List[Nuclide]) -> List[float]:
        """
        Calculate calibration coefficients from a set of nuclides expected to be present in spectrum.
        Args:
            nuclides_in_spectrum: List of nuclides expected to be present in spectrum.

        Returns:
            Calibration coefficients
        """
        _msg = '`nuclides_in_spectrum` arg must be an iterable of nuclides'
        assert isinstance(nuclides_in_spectrum, Iterable), _msg
        assert all(isinstance(n, Nuclide) for n in nuclides_in_spectrum), _msg
        # Todo

    def nuclide_search_MLLH(self):
        pass
    # todo


if __name__ == '__main__':
    # from GlobalValues import shot_groups
    # tree = shot_groups['He+Ar'].tree
    # # tree.MakeClass('gamma_spec')
    #
    # s = Spectrum.from_tree(tree, get_global_energy_bins(150), 20)
    # s.plot_energy_spectrum()
    # plt.show()
    x = np.linspace(-5,5,20)

    plt.plot(x+3, (x-3)**2)
    cut = PeakSignalBackgroundCut(3, 1, 1, left_bg_window_offset=-3, right_bg_window_offset=2.5)
    print(cut.signal_cut)
    print(cut.back_ground_cut)
    cut.plot(plt)
    plt.legend()
    plt.show()



