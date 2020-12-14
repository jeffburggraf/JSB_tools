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
from typing import List, Iterable
from pathlib import Path
from matplotlib import pyplot as plt
from PHELIXDataTTree import get_global_energy_bins
from numbers import Number
from scipy.signal import find_peaks
Nuclide.from_symbol('').proton_induced_fiss_xs

def __get_hl_signature__(tree, discrete_ergs, min_time):
    # if max_time is None:
    #     max_time = ROOT.get_max_time()
    hist = TH1F(bin_left_edges=discrete_ergs)
    hist.Project(tree, 'erg', 't>{0}'.format(min_time))
    peaks, peak_info =
    # hist.plot()


class Spectrum:

    def __init__(self, **kwargs):
        assert '__internal__' in kwargs, 'Use one of the following factory methods to create a Spectrum instance:\n' \
                                         'Spectrum.load,Spectrum.from_tree'

        try:
            self.tree = kwargs['tree']
            self.half_life_signature = kwargs['half_life_signature']
            self.discrete_ergs = kwargs['discrete_ergs']

        except KeyError as e:
            assert False, ('\nMissing keyword argument `{}` in call to Spectrum.__init__(**kwargs)'.format(e.args[0]))

    @classmethod
    def from_tree(cls, tree, discrete_ergs, min_t_for_hl_calc, b_width_for_hl_calc=0.1):
        """
        :tree  ROOT TTree instance
        :discrete_ergs iterable of all possible energies
        :b_width_for_hl_calc Bin width for half life calulations
        :min_t_for_hl_calc Starting time for half life calculations
        :max_t_for_hl_calc Maximum time for half life calculations
        """
        assert isinstance(tree, ROOT.TTree), '"tree" must be a TTree instance.'
        ROOT.gROOT.LoadMacro(str(Path(__file__).parent / 'gamma_spec.c'))

        ROOT.set_up(tree)

        existing_branches = set([b.GetName() for b in tree.GetListOfBranches()])
        required_branches = ['erg', 'sigma_erg', 't_min', 't_max', 'shot', 't', 'eff', 'eff_err', 'J']
        missing_branches = []
        for b_name in required_branches:
            if b_name not in existing_branches:
                missing_branches.append(b_name)
        if len(missing_branches) != 0:
            assert False, 'The tree passed to Spectrum.from_tree must contain all the following branches:\n' \
                          '{}\nMissing branches:\n{}'.format(required_branches, missing_branches)

        assert isinstance(discrete_ergs, Iterable), '`discrete_ergs` must be an iterable (of floats)'
        assert isinstance(min_t_for_hl_calc, Number)

        half_life_signature = __get_hl_signature__(tree, discrete_ergs, min_t_for_hl_calc)
        out = cls(tree=tree, half_life_signature=half_life_signature, discrete_ergs=discrete_ergs, __internal__=True)

        return out

    def save(self, title, path_to_dir):
        assert isinstance(title, str), '`title` argument must be a string.'
        assert isinstance(path_to_dir, (str, Path)) and Path(path_to_dir).exists(),\
            '`path_to_dir` must be a directory (str or Path instance) that exists'
        # todo

    def get_erg_calibration(self, nuclides_in_spectrum: List[Nuclide]):
        _msg = '`nuclides_in_spectrum` arg must be an iterable of nuclides'
        assert isinstance(nuclides_in_spectrum, Iterable), _msg
        assert all(isinstance(n, Nuclide) for n in nuclides_in_spectrum), _msg
        # Todo


if __name__ == '__main__':
    from GlobalValues import shot_groups
    tree = shot_groups['He+Ar'].tree
    # tree.MakeClass('gamma_spec')

    s = Spectrum.from_tree(tree, get_global_energy_bins(150), 20)
    plt.show()



