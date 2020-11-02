from __future__ import annotations
import pickle
import numpy as np
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
from openmc.data import Reaction, Decay
from typing import Dict, List
from matplotlib import pyplot as plt
import re
from pathlib import Path
from warnings import warn
from uncertainties import ufloat
from uncertainties.umath import isinf, isnan
import uncertainties
from functools import cached_property
from openmc.data.data import NATURAL_ABUNDANCE
pwd = Path(__file__).parent

#  Todo:
#   * Add a custom exception, not AssertionError, when an unknown symbol is passed to Nuclide.
#   * Investigate why some half-lives, such as Te-123, are 0 when they are in actuality very very long.
#     Is this an artifact of openmc?
#   * Add doc strings.
#   * Add documentation, and exception messages, to explain where the data can be downloaded and how to regenerate
#     the pickle files.
#   * Add more  comments to document code.
#   * Implement openmc.FissionProductYields
#   * Any speed enhancements would be nice.
#   * Use pyne to read latest ENDSFs and fill in some missing halk lives via a pickle file.
#     Then, read results into <additional_nuclide_data> variable.


NUCLIDE_INSTANCES = {}
DECAY_PICKLE_DIR = pwd/'data'/'nuclides'
PROTON_PICKLE_DIR = pwd / "data" / "incident_proton"
NUCLIDE_NAME_MATCH = re.compile("([A-Za-z]{1,2})([0-9]{1,3})(?:_m([0-9]+))?")

additional_nuclide_data = {"In101_m1": {"half_life": ufloat(10, 5)},
                           "Lu159_m1": {"half_life": ufloat(10, 5)},
                           "Rh114_m1": {"half_life": ufloat(1.85, 0.05), "__decay_daughters_str__": "Pd114"},
                           "Pr132_m1": {"half_life": ufloat(20, 5), "__decay_daughters_str__": "Ce132"}}

XS_BIN_WIDTH_INTERPOLATION = 0.1


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'GammaLine':
            return GammaLine
        elif name == 'DecayMode':
            return DecayMode
        elif name == '__Reaction__':
            return __Reaction__
        elif name == 'CrossSection1D':
            return CrossSection1D

        return super().find_class(module, name)

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
        self.intensity: uncertainties.UFloat = intensity
        self.from_mode: DecayMode = from_mode
        self.intensity_thu_mode: uncertainties.UFloat = intensity_thu_mode
        self.absolute_rate: uncertainties.UFloat = nuclide.decay_constant*self.intensity

    def __repr__(self):
        return "Gamma line at {0:.1f} KeV; true_intensity = {1:.2e}; decay: {2} ".format(self.erg, self.intensity, self.from_mode)


class CrossSection1D:
    def __init__(self, ergs, xss, fig_label=None, incident_particle='particle'):
        self.__ergs__ = np.array(ergs)
        self.__xss__ = np.array(xss)
        self.__fig_label__ = fig_label
        self.__incident_particle__ = incident_particle

    @cached_property
    def xss(self):
        return np.interp(self.ergs, self.__ergs__, self.__xss__)

    @cached_property
    def ergs(self):
        return np.arange(self.__ergs__[0], self.__ergs__[-1], XS_BIN_WIDTH_INTERPOLATION)


    def plot(self, ax=None, fig_title=None, units="b"):
        unit_convert = {"b": 1, "mb": 1000, "ub": 1E6, "nb": 1E9}
        try:
            unit_factor = unit_convert[units]
        except KeyError:
            assert False, "Invalid unit '{0}'. Valid options are: {1}".format(units, unit_convert.keys())
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            if fig_title is not None:
                ax.set_title(fig_title)
            else:
                if self.__fig_label__ is not None:
                    ax.set_title(self.__fig_label__)

        ax.plot(self.__ergs__, self.__xss__ * unit_factor)
        ax.set_ylabel("Cross-section [{}]".format(units))
        ax.set_xlabel("Incident {} energy [MeV]".format(self.__incident_particle__))

        return ax

    @property
    def bin_widths(self):
        return XS_BIN_WIDTH_INTERPOLATION

        # out = [high - low for low, high in zip(self.__ergs__[:-1], self.__ergs__[1:])]
        # not sure how to deduce the bin width of final bin. Hopefully bin widths are usually all the same
        # out += [np.mean(out)]
        # out = np.array(out)
        # return out

    def mean_xs(self, erg_low=None, erg_high=None, weight_callable=None):
        if erg_low is None and erg_high is None:
            xss = self.xss
            ergs = self.ergs
        else:
            if erg_high is None:
                erg_high = self.__ergs__[-1]
            if erg_low is None:
                erg_low = self.__ergs__[0]
            cut = [i for i in range(len(self.ergs)) if erg_low <= self.ergs[i] <= erg_high]
            xss = self.xss[cut]
            ergs = self.ergs[cut]
        if weight_callable is not None:
            assert hasattr(weight_callable, "__call__"), "<weighting_function> must be a callable function that takes" \
                                                         " incident energy as it's sole argument"
            weights = [weight_callable(e) for e in ergs]
            if sum(weights) == 0:
                warn("weights in mean_xs for '{0}' summed to zero. Returning 0")
                return 0
            return np.average(xss, weights=weights)
        else:
            return np.mean(xss)

    def __repr__(self):
        ergs = ", ".join(["{:.2e}".format(e) for e in self.__ergs__])
        xss = ", ".join(["{:.2e}".format(e) for e in self.__xss__])
        return "ergs: {0}\n" \
               "  xs: {1}".format(ergs, xss)


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

def get_z_a_m_from_name(name):
    _m = NUCLIDE_NAME_MATCH.match(name)

    assert _m, "\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
               "e.g. Cl38_m1, n1, Ar40".format(name)

    try:
        Z = ATOMIC_NUMBER[_m.groups()[0]]
    except KeyError:
        if name == "Nn1":
            Z = 0
        else:
            warn("invalid name: {0}".format(name))
            Z = None
    A = _m.groups()[1]
    isometric_state = _m.groups()[2]
    if isometric_state is None:
        isometric_state = 0
    isometric_state = isometric_state
    return Z, A, isometric_state


def get_name_from_z_a_m(z, a, m):
    z, a, m = map(int, [z,a ,m])
    if z == 0:
        symbol = "Nn"
    else:
        symbol = ATOMIC_SYMBOL[z]
    symbol += str(a)
    if m != 0:
        symbol += "_m{0}".format(m)
    return symbol


def __nuclide_cut__(a_z_hl_cut: str, is_stable_only, nuclide: Nuclide):
    makes_cut = True

    assert isinstance(is_stable_only, bool)
    if is_stable_only is True:
        if not nuclide.is_stable:
            return False

    if len(a_z_hl_cut) > 0:
        a_z_hl_cut = a_z_hl_cut.lower()
        if 'hl' in a_z_hl_cut and nuclide.half_life is None:
            makes_cut = False
        else:
            try:
                makes_cut = eval(a_z_hl_cut, {"hl": nuclide.half_life, 'a': nuclide.A, 'z': nuclide.Z})
                assert isinstance(makes_cut, bool), "Invalid cut: {0}".format(a_z_hl_cut)
            except NameError as e:
                invalid_name = str(e).split("'")[1]

                raise Exception("\nInvalid name '{}' used in cut. Valid names are: 'z', 'a', and 'hl',"
                                " which stand for atomic-number, mass-number, and half-life, respectively."
                                .format(invalid_name)) from e

    return makes_cut

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

        self.Z, self.A, self.isometric_state = get_z_a_m_from_name(name)

        self.half_life = kwargs.get("half_life", None)
        self.spin = kwargs.get("spin", None)
        self.mean_energies = kwargs.get("mean_energies", None)  # look into this
        self.is_stable: bool = kwargs.get("is_stable", None)  # maybe default to something else

        self.__decay_daughters_str__: List[str] = kwargs.get("__decay_daughters__", [])  # self.decay_daughters -> List[Nuclide] in corresponding order as self.decay_modes
        self.decay_gamma_lines: List[GammaLine] = kwargs.get("decay_gamma_lines", [])
        self.decay_modes: List[DecayMode] = kwargs.get("decay_modes", [])
        self.__decay_parents_str__: List[str] = kwargs.get("__decay_parents__", [])  # self.decay_parents -> List[Nuclide]

        self.__decay_mode_for_print__ = None

    @property
    def isotopic_abundance(self):
        _m = re.match('([A-Za-z]{1,2}[0-9]+)(?:m_[0-9]+)?', self.name)
        if _m:
            s = _m.groups()[0]
            try:
                return NATURAL_ABUNDANCE[s]
            except KeyError:
                pass
        return 0

    @property
    def atomic_symbol(self):
        _m = re.match('([A-Za-z]{1,2}).+', self.name)
        if _m:
            return _m.groups()[0]
        else:
            return None

    @classmethod
    def from_Z_A_M(cls, z, a, m=0):
        try:
            symbol = ATOMIC_SYMBOL[z]
        except KeyError:
            assert False, "Invalid atomic number: {0}. Cant find atonic symbol".format(z)
        name = "{0}{1}".format(symbol, a)
        if m != 0:
            name += "_m" + str(m)
        return cls.from_symbol(name)

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
            if not pickle_file.exists():
                if symbol in additional_nuclide_data:
                    instance = Nuclide(symbol, **additional_nuclide_data[symbol])
                else:
                    warn("Cannot find data for Nuclide {0}. Data for this nuclide empty".format(symbol))
                    instance = Nuclide(symbol, half_life=ufloat(np.nan, np.nan))
            else:
                with open(pickle_file, "rb") as pickle_file:
                    instance = CustomUnpickler(pickle_file).load()
        else:
            instance = NUCLIDE_INSTANCES[symbol]
        assert isinstance(instance, Nuclide)
        return instance

    def __repr__(self):
        out = "<Nuclide: {}; t_1/2 = {}>".format(self.name, self.half_life)
        if self.__decay_mode_for_print__ is not None:
            out += self.__decay_mode_for_print__.__repr__()
        return out

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

    def get_incident_proton_parents(self, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedParent]:
        pickle_path = PROTON_PICKLE_DIR / (self.name + ".pickle")
        if not pickle_path.exists():
            warn("No proton-induced data for any parents of {}".format(self.name))
            return {}

        with open(pickle_path, "rb") as f:
            daughter_reaction = CustomUnpickler(f).load()

        assert isinstance(daughter_reaction, __Reaction__)
        out = {}
        parent_nuclides = [Nuclide.from_symbol(name) for name in daughter_reaction.parent_nuclide_names]
        daughter_nuclide = self
        for parent_nuclide in parent_nuclides:
            parent_pickle_path = PROTON_PICKLE_DIR/(parent_nuclide.name + ".pickle")
            with open(parent_pickle_path, "rb") as f:
                parent_reaction = CustomUnpickler(f).load()
                assert isinstance(parent_reaction, __Reaction__)
            parent = InducedParent(daughter_nuclide, parent_nuclide, inducing_particle="proton")
            if __nuclide_cut__(a_z_hl_cut=a_z_hl_cut, is_stable_only=is_stable_only, nuclide=parent):
                parent.xs = parent_reaction.product_nuclide_names_xss[daughter_nuclide.name]
                out[parent.name] = parent

        return out

    def get_incident_proton_daughters(self, a_z_hl_cut='', is_stable_only=False):
        pickle_path = PROTON_PICKLE_DIR/(self.name + ".pickle")
        if not pickle_path.exists():
            warn("No proton-induced data for {}".format(self.name))
            return None

        with open(pickle_path, "rb") as f:
            reaction = CustomUnpickler(f).load()

        assert isinstance(reaction, __Reaction__)
        out: Dict[str, InducedParent] = {}
        for daughter_name, xs in reaction.product_nuclide_names_xss.items():
            daughter_nuclide = Nuclide.from_symbol(daughter_name)
            if __nuclide_cut__(a_z_hl_cut, is_stable_only, daughter_nuclide):
                daughter = InducedDaughter(daughter_nuclide, self, "proton")
                daughter.xs = xs
                out[daughter_name] = daughter

            # todo: Is using a dict of name and Nuclides necessary? Maybe just return a list instead.
        return out

    def __set_data_from_open_mc__(self, open_mc_decay):
        self.half_life = open_mc_decay.half_life
        self.mean_energies = open_mc_decay.average_energies
        self.spin = open_mc_decay.nuclide["spin"]
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

    @ classmethod
    def get_all_nuclides(cls, a_z_hl_cut: str = '', is_stable_only=False) -> List[Nuclide]:
        assert isinstance(a_z_hl_cut, str), 'All cuts must be a string instance.'
        nuclides = []
        for f_path in DECAY_PICKLE_DIR.iterdir():
            f_name = f_path.name.replace(".pickle", '')
            _m = NUCLIDE_NAME_MATCH.match(f_name)
            if _m:
                nuclide = cls.from_symbol(f_name)
                if __nuclide_cut__(a_z_hl_cut, is_stable_only, nuclide):
                    nuclides.append(nuclide)
        return nuclides


def pickle_decay_data(directory):
    directory = Path(directory)
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


class InducedDaughter(Nuclide):
    def __init__(self, daughter_nuclide, parent_nuclide, inducing_particle):
        assert isinstance(daughter_nuclide, Nuclide)
        assert isinstance(parent_nuclide, Nuclide)
        kwargs = {k: v for k, v in daughter_nuclide.__dict__.items() if k != "name"}
        super().__init__(daughter_nuclide.name, **kwargs)
        self.xs: CrossSection1D = None
        self.parent:Nuclide = parent_nuclide
        self.inducing_particle = inducing_particle

    def __repr__(self):
        par_symbol = self.inducing_particle[0]
        return '{0}({1},X) --> {2}'.format(self.parent, par_symbol, super().__repr__())


class InducedParent(Nuclide):
    def __init__(self, daughter_nuclide, parent_nuclide, inducing_particle):
        assert isinstance(daughter_nuclide, Nuclide)
        assert isinstance(parent_nuclide, Nuclide)
        kwargs = {k: v for k, v in parent_nuclide.__dict__.items() if k != "name"}
        super().__init__(parent_nuclide.name, **kwargs)
        self.xs: CrossSection1D = None
        self.daughter:Nuclide = daughter_nuclide
        self.inducing_particle = inducing_particle

    def __repr__(self):
        par_symbol = self.inducing_particle[0]
        return '{0}({1},X) --> {2}'.format(super().__repr__(), par_symbol, self.daughter)


# modularize the patch work of reading PADF and ENDF-B-VIII.0_protons data.
class proton_endf_file:
    def __init__(self, padf_directory, endf_b_directory):
        self.nuclide_name_and_file_path = {}

        for path in Path(padf_directory).iterdir():
            f_name = path.name
            _m = re.match("([0-9]{4,6})(?:M([0-9]))?", f_name)
            if _m:
                zaid = int(_m.groups()[0])
                if _m.groups()[1] is not None:
                    isometric_state = int(_m.groups()[1])
                else:
                    isometric_state = 0
                z = zaid // 1000
                a = zaid % 1000
                nuclide_name = get_name_from_z_a_m(z, a, isometric_state)
                self.nuclide_name_and_file_path[nuclide_name] = path

        for path in Path(endf_b_directory).iterdir():
            f_name = path.name
            _m = re.match("p-([0-9]+)_([A-Za-z]{1,2})_([0-9]+)\.endf", f_name)
            if _m:
                z = _m.groups()[0]
                a = _m.groups()[2]
                nuclide_name = get_name_from_z_a_m(z, a, 0)
                self.nuclide_name_and_file_path[nuclide_name] = path


class __Reaction__:
    def __init__(self, name):
        self.name = name
        self.product_nuclide_names_xss: Dict[str, CrossSection1D] = {}
        self.parent_nuclide_names: List[str] = []


def pickle_proton_data():
    assert PROTON_PICKLE_DIR.exists()
    i = 0
    all_reactions = {}
    files = proton_endf_file(padf_directory=proton_padf_data_dir, endf_b_directory=proton_enfd_b_data_dir)

    for nuclide_name, f_path in files.nuclide_name_and_file_path.items():
        if nuclide_name in all_reactions:
            reaction = all_reactions[nuclide_name]
        else:
            reaction = __Reaction__(nuclide_name)
            all_reactions[nuclide_name] = reaction

        e = Evaluation(f_path)
        for heavy_product in Reaction.from_endf(e, 5).products:
            heavy_product_name = heavy_product.particle
            if heavy_product_name == "photon":
                continue
            if heavy_product_name == "neutron":
                heavy_product_name = "Nn1"
            xs_fig_label = "{0}(p,X){1}".format(nuclide_name, heavy_product_name)
            xs = CrossSection1D(heavy_product.yield_.x / 1E6, heavy_product.yield_.y, xs_fig_label, 'proton')
            reaction.product_nuclide_names_xss[heavy_product_name] = xs
            if heavy_product_name in all_reactions:
                daughter_reaction = all_reactions[heavy_product_name]
            else:
                daughter_reaction = __Reaction__(heavy_product_name)
                all_reactions[heavy_product_name] = daughter_reaction
            daughter_reaction.parent_nuclide_names.append(nuclide_name)
        i += 1

    for nuclide_name, reaction in all_reactions.items():
        pickle_file_name = PROTON_PICKLE_DIR/(nuclide_name + ".pickle")
        with open(pickle_file_name, "bw") as f:
            pickle.dump(reaction, f)


#  Download PADF proton induced reaction files from https://www-nds.iaea.org/padf/
#  Download decay data from https://www.nndc.bnl.gov/endf/b8.0/download.html
proton_padf_data_dir = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/PADF_2007/Files"
proton_enfd_b_data_dir = "/Users/jeffreyburggraf/Desktop/nukeData/ENDF-B-VIII.0_protons"
decay_data_dir = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay"
dir_old = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay/"
dir_new = "/Users/jeffreyburggraf/Desktop/nukeData/ENDF-B-VIII.0_decay/"


if __name__ == "__main__":
    assert Path(decay_data_dir).exists, "Cannot find decay data files. " \
                                         "Download decay files from https://www.nndc.bnl.gov/endf/b7.1/download.html" \
                                         " and set the <decay_data_dir> " \
                                         "variable to the location of the unzipped directory"
     # from openmc.data.decay import FissionProductYields

    #  How to get SF yields:
    #  Download from https://www.cenbg.in2p3.fr/GEFY-GEF-based-fission-fragment,780
    # y = FissionProductYields("/Users/jeffreyburggraf/Downloads/gefy81_s/GEFY_92_238_s.dat")

    #  Uncomment code below to pickle Nuclide data
    pickle_decay_data(decay_data_dir)


    assert Path(proton_padf_data_dir).exists, "Cannot find proton data files. " \
                                         "Download proton files from https://www-nds.iaea.org/padf/ and set the " \
                                         "<proton_dir> variable to the location of the unzipped directory"
    #  Uncomment code below to pickle incident proton data
    pickle_proton_data()

    pass






