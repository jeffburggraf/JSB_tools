from __future__ import annotations
import pickle
import numpy as np
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
from openmc.data import Reaction, Decay
from typing import Dict, List
from matplotlib import pyplot as plt
from warnings import warn
import re
from pathlib import Path
from warnings import warn
from uncertainties import ufloat
from uncertainties.umath import isinf, isnan
import uncertainties
pwd = Path(__file__).parent

NUCLIDE_INSTANCES = {}
DECAY_PICKLE_DIR = pwd/'data'/'nuclides'
NUCLIDE_NAME_MATCH = re.compile("([A-Za-z]{1,2})([0-9]{1,3})(?:_m([0-9]+))?")

class GammaLine:
    def __init__(self, nuclide: Nuclide, erg, intensity, intensity_thu_mode, from_mode):
        """
        Attributes
    ----------
    erg : uncertainties.UFloat
        Energy of gamma in KeV
    intensity : uncertainties.UFloat
        mean number of gammas with energy self.erg emitted per parent nucleus decay (through any decay channel)
    mode : DecayMode
        DecayMode instance. Contains decay channel, branching ratio, among other information
    intensity_thu_mode : uncertainties.UFloat
        mean number of gammas with energy self.erg emitted per parent nucleus decay (through self.from_mode
        decay channel)
    absolute_rate : uncertainties.UFloat
        Rate [s^-1] of gammas with energy self.erg emitted per nuclide.
    """
        self.erg: uncertainties.UFloat = erg
        # mean number of gammas with energy self.erg emitted per parent nucleus decay (through any decay channel)
        self.intensity: uncertainties.UFloat = intensity
        self.from_mode: DecayMode = from_mode
        self.intensity_thu_mode: uncertainties.UFloat = intensity_thu_mode
        self.absolute_rate: uncertainties.UFloat = nuclide.decay_constant*self.intensity

    def __repr__(self):
        return "Gamma line at {0:.1f} KeV; true_intensity = {1:.2e}; decay: {2} ".format(self.erg, self.intensity, self.from_mode)


class CrossSection1D:
    def __init__(self, ergs, xss):
        self.ergs = ergs
        self.xss = xss

    def plot(self, ax=None, fig_title=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            if fig_title is not None:
                fig.set_title(fig_title)

        ax.plot(self.ergs, self.xss)

        return ax


class DecayMode:
    def __init__(self, openmc_decay_mode):
        self.modes = openmc_decay_mode.modes
        self.daughter_name = openmc_decay_mode.daughter
        self.parent_name = openmc_decay_mode.parent
        self.branching_ratio = openmc_decay_mode.branching_ratio
        # print("parent", openmc_decay_mode.parent)
        # print("daughter", self.daughter_name)

    def is_mode(self, mode_str):
        return mode_str in self.modes

    def __repr__(self):

        return "{0} -> {1} via {2} with BR of {3}".format(self.parent_name, self.daughter_name, self.modes, self.branching_ratio)


class Nuclide:
    def __init__(self, name, **kwargs):
        assert isinstance(name, str)
        orig_name = name
        if '-' in name:
            name = name.replace('-', '')
            name = name.replace('Nat', '0')
            if name.endswith('m'):
                name = name[:-1] + '_m1'

            msg = 'OpenMC nuclides follow the GND naming convention. Nuclide ' \
                  '"{}" is being renamed as "{}".'.format(orig_name, name)
            warn(msg)
        self.name = name

        _m = NUCLIDE_NAME_MATCH.match(self.name)

        assert _m, "\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
                   "e.g. Cl38_m1, n1, Ar40".format(name)

        try:
            self.Z = ATOMIC_NUMBER[_m.groups()[0]]
        except KeyError:
            if self.name == "Nn1":
                self.Z = 0
            else:
                warn("invalid name: {0}".format(self.name))
                self.Z = None
        self.A = _m.groups()[1]
        isometric_state = _m.groups()[2]
        if isometric_state is None:
            isometric_state = 0
        self.isometric_state = isometric_state

        self.half_life = kwargs.get("half_life", None)
        self.spin = kwargs.get("spin", None)
        self.mean_energies = kwargs.get("mean_energies", None)  # look into this
        self.is_stable = kwargs.get("is_stable", None)  # maybe default to something else

        self.__decay_daughters_str__: List[str] = kwargs.get("__decay_daughters__", [])  # self.decay_daughters -> List[Nuclide] in corresponding order as self.decay_modes
        self.decay_gamma_lines: List[GammaLine] = kwargs.get("decay_gamma_lines", [])
        self.decay_modes: List[DecayMode] = kwargs.get("decay_modes", [])
        self.__decay_parents_str__: List[str] = kwargs.get("__decay_parents__", [])  # self.decay_parents -> List[Nuclide]

        self.__decay_mode_for_print__ = None

    @classmethod
    def from_symbol(cls, symbol):
        if '-' in symbol:
            symbol = symbol.replace('-', '')
            if symbol.endswith('m'):
                symbol = symbol[:-1] + '_m1'

        pickle_file = DECAY_PICKLE_DIR/(symbol + '.pickle')
        _m = NUCLIDE_NAME_MATCH.match(symbol)
        assert _m, "\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
                   "e.g. Cl38_m1, n1, Ar40".format(symbol)
        if _m not in NUCLIDE_INSTANCES:
            assert pickle_file.exists(), "Cannot find data for Nuclide {0}".format(symbol)
            with open(pickle_file, "rb") as pickle_file:
                instance = pickle.load(pickle_file)
        else:
            instance = NUCLIDE_INSTANCES[symbol]
        assert isinstance(instance, Nuclide)
        return instance

    def __repr__(self):
        out = "<Nuclide: {}; t_1/2 = {}> ".format(self.name, self.half_life)
        if self.__decay_mode_for_print__ is not None:
            out += self.__decay_mode_for_print__.__repr__()
        return out

    def daughter_decay_mode(self, daughter_nuclide_instance: Nuclide):
        raise NotImplementedError

    def parent_decay_mode(self, daughter_nuclide_instance: Nuclide):
        raise NotImplementedError

    @property
    def decay_constant(self):
        return np.log(2)/self.half_life

    @property
    def decay_parents(self):
        return list([self.from_symbol(name) for name in self.__decay_parents_str__])

    @property
    def decay_daughters(self):
        out = list([self.from_symbol(name) for name in self.__decay_daughters_str__])
        for nuclide in out:
            for decay_mode in self.decay_modes:
                if decay_mode.daughter_name == nuclide.name:
                    nuclide.__decay_mode_for_print__ = decay_mode

        return out

    def get_incident_proton_parents(self):  # todo
        raise NotImplementedError

    def get_incident_proton_daughters(self):  # todo
        raise NotImplementedError

    def __set_data_from_open_mc__(self, open_mc_decay):
        self.half_life = open_mc_decay.half_life
        self.mean_energies = open_mc_decay.average_energies
        if isinf(self.half_life) or open_mc_decay.nuclide["stable"]:
            self.is_stable = True
        else:
            self.is_stable = False

        for mode in open_mc_decay.modes:
            decay_mode = DecayMode(mode)
            self.decay_modes.append(decay_mode)

        try:
            gamma_decay_info = open_mc_decay.spectra["gamma"]
            discrete_normalization = gamma_decay_info["discrete_normalization"]
            if gamma_decay_info["continuous_flag"] != "discrete":
                return
            for gamma_line_info in gamma_decay_info["discrete"]:
                erg = gamma_line_info["energy"]
                from_mode = gamma_line_info["from_mode"]
                for mode in self.decay_modes:
                    if mode.is_mode(from_mode[0]):
                        break
                else:
                    assert False, "{0} {1}".format(self.decay_modes, gamma_line_info)
                branching_ratio = mode.branching_ratio
                intensity_thu_mode = discrete_normalization*gamma_line_info["intensity"]
                intensity = intensity_thu_mode * branching_ratio
                g = GammaLine(self, erg/1000, intensity, intensity_thu_mode, mode)
                self.decay_gamma_lines.append(g)
            self.decay_gamma_lines = list(sorted(self.decay_gamma_lines, key=lambda x: -x.intensity))
        except KeyError:
            pass
        # Todo: Add decay channels other than "gamma"

class InducedDaughter(Nuclide):
    pass


class InducedParent(Nuclide):
    pass


def pickle_decay_data(directory):
    directory = Path(directory)
    # open_mc_decays = {}
    assert directory.exists()
    i=0
    for file_path in directory.iterdir():
        i += 1
        file_name = file_path.name
        _m = re.match(r"dec-[0-9]{3}_(?P<S>[A-Za-z]{1,2})_(?P<A>[0-9]+)(?:m(?P<M>[0-9]+))?\.endf", file_name)
        if _m:
            a = int(_m.group("A"))
            _s = _m.group("S")  # nuclide symbol, e.g. Cl, Xe, Ar
            m = _m.group("M")
            if m is not None:
                m = int(m)

            parent_nuclide_name = "{0}{1}{2}".format(_s, a, "" if m is None else "_m{0}".format(m))
        else:
            continue

        if parent_nuclide_name in NUCLIDE_INSTANCES:
            parent_nuclide = NUCLIDE_INSTANCES[parent_nuclide_name]
        else:
            parent_nuclide = Nuclide(parent_nuclide_name)
            NUCLIDE_INSTANCES[parent_nuclide_name] = parent_nuclide

        openmc_decay = Decay(Evaluation(file_path))
        daughter_names = [mode.daughter for mode in openmc_decay.modes]
        for daughter_nuclide_name in daughter_names:
            if daughter_nuclide_name in NUCLIDE_INSTANCES:
                daughter_nuclide = NUCLIDE_INSTANCES[daughter_nuclide_name]
            else:
                daughter_nuclide = Nuclide(daughter_nuclide_name)
                NUCLIDE_INSTANCES[daughter_nuclide_name] = daughter_nuclide

            if daughter_nuclide_name != parent_nuclide_name:
                daughter_nuclide.__decay_parents_str__.append(parent_nuclide_name)
                parent_nuclide.__decay_daughters_str__.append(daughter_nuclide_name)

        parent_nuclide.__set_data_from_open_mc__(openmc_decay)
        print("Preparing data for {0}".format(parent_nuclide_name))

    for nuclide_name in NUCLIDE_INSTANCES.keys():
        with open(DECAY_PICKLE_DIR/(nuclide_name + '.pickle'), "wb") as pickle_file:
            print("Writing data for {0}".format(nuclide_name))
            pickle.dump(NUCLIDE_INSTANCES[nuclide_name], pickle_file)


    # for parent_nuclide_name, nuclide in NUCLIDE_INSTANCES.items():
    #     nuclide.__set_data_from_open_mc__(open_mc_decays[parent_nuclide_name])
    #     for daughter_nuclide_name in nuclide.__decay_daughters_str__:
    #         try:
            # daughter_nuclide = NUCLIDE_INSTANCES[daughter_nuclide_name]
            # except KeyError:
            #     warn("Could not find daughter nuclide {0} from {1} decy ")
            # daughter_nuclide.__decay_parents_str__.append(parent_nuclide_name)


#  Download PADF proton induced reaction files from https://www-nds.iaea.org/padf/
#  Download decay data from https://www.nndc.bnl.gov/endf/b7.1/download.html
proton_data_dir = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/PADF_2007"
decay_data_dir = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay"

if __name__ == "__main__":
    assert Path(decay_data_dir).exists, "Cannot find decay data files. " \
                                         "Download decay files from https://www.nndc.bnl.gov/endf/b7.1/download.html" \
                                         " and set the <decay_data_dir> " \
                                         "variable to the location of the unzipped directory"
    # from openmc.data.decay import FissionProductYields

    #  How to get SF yields:
    #  Download from https://www.cenbg.in2p3.fr/GEFY-GEF-based-fission-fragment,780
    # y = FissionProductYields("/Users/jeffreyburggraf/Downloads/gefy81_s/GEFY_92_238_s.dat")
    # print(y.independent[0])

    # Nuclide("Cl38_m1")
    pickle_decay_data(decay_data_dir)
    n = Nuclide.from_symbol("Cf-252")
    print(n.decay_parents)
    print(n.decay_daughters)

    assert Path(proton_data_dir).exists, "Cannot find proton data files. " \
                                         "Download proton files from https://www-nds.iaea.org/padf/ and set the " \
                                         "<proton_dir> variable to the location of the unzipped directory"
    # pickle_proton_data(proton_data_dir)

    pass






