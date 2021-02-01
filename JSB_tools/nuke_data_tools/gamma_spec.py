# from JSB_tools.nuke_data_tools.gamma_spec import exp_decay_maximum_likely_hood
import numpy as np
from JSB_tools.TH1 import TH1F
from matplotlib import pyplot as plt
from JSB_tools.nuke_data_tools import Nuclide
from JSB_tools.nuke_data_tools.__init__ import DECAY_PICKLE_DIR
import marshal
from JSB_tools import closest
from sortedcontainers import SortedDict
from PHELIXDataTTree import get_global_energy_bins, get_time_bins
from numbers import Number
from GlobalValues import shot_groups
from typing import Collection, Union


data_dir = DECAY_PICKLE_DIR/'__fast__gamma_dict__.marshal'


DATA = None


class ValueStruct:
    def __init__(self, name, intensity, hl, erg):
        self.name = name
        self.hl = hl
        self.intensity = intensity
        self.erg = erg
        self.__rank__ = 0

    @staticmethod
    def from_values(key, *values):
        outs = []
        for name, i, hl in zip(*values):
            outs.append(ValueStruct(name, i, hl, key))
        return outs

    def __repr__(self):
        return f"{self.name}: erg: {self.erg}, intensity: {self.intensity}, hl: {self.hl} rank: {self.__rank__}"


def gamma_search(erg_center: float, e_sigma: float = 1, delta_t: Union[Number, None] = None, half_life_min=None,
                 half_life_max=None):
    """
    Search for nuclides that produce gamma decays in the neighborhood of `erg_center`
    Args:
        erg_center: Center of energy window
        e_sigma: half width of energy window
        delta_t: If the spectroscopy measurement takes place over, say, 180 seconds, then the number of decays a
         nuclide would undergo in this duration is used as a multiplicative factor during sorting.
        half_life_min:
        half_life_max:

    Returns:

    """
    assert not isinstance(delta_t, Collection)
    assert isinstance(delta_t, (type(None), Number))
    assert isinstance(erg_center, Number)
    assert isinstance(e_sigma, Number)
    erg_range = erg_center - e_sigma, erg_center + e_sigma

    global DATA
    if DATA is None:
        with open(data_dir, 'rb') as f:
            DATA = SortedDict(marshal.load(f))

    def hl_filter(cls: ValueStruct):
        if half_life_min is not None:
            if not half_life_min < cls.hl:
                return False
        if half_life_min is not None:
            if not cls.hl < half_life_max:
                return False
        return True

    def sorter(cls: ValueStruct):
        if delta_t is None:
            out = -cls.intensity
        else:
            out = -cls.intensity * (1-0.5**(delta_t/cls.hl))
        cls.__rank__ = abs(out)
        return out

    out = []
    for closest_key in DATA.irange(*erg_range):
        out.extend(ValueStruct.from_values(closest_key, *DATA[closest_key]))

    if half_life_min is not None or half_life_max is not None:
        out = list(filter(hl_filter, out))

    out = sorted(out, key=sorter)
    return out


if __name__ == '__main__':
    Gdelta_t = get_time_bins()[-1]
    data_dir = DECAY_PICKLE_DIR / '__fast__gamma_dict__.marshal'

    tree = shot_groups['He+Ar'].tree
    hist = TH1F(bin_left_edges=get_global_energy_bins())
    hist.Project(tree, 'erg', weight='1.0/eff')
    hist.plot()
    print(Nuclide.from_symbol('U238').independent_gamma_fission_yield_gef()[1])
    # for n, yield_ in Nuclide.from_symbol('U238').independent_gamma_fission_yield_gef().items():
    #     print(n, yield_)
    for i in gamma_search(2612, e_sigma=4, delta_t=Gdelta_t):
        print(i)
    plt.show()