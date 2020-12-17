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
from uncertainties import ufloat, UFloat
from uncertainties import unumpy as unp
from uncertainties.umath import isinf, isnan
import uncertainties
import marshal
from functools import cached_property
from openmc.data.data import NATURAL_ABUNDANCE, atomic_mass, atomic_weight, AVOGADRO
from scipy.stats import norm
import zipfile
pwd = Path(__file__).parent


__all__ = ['Nuclide']

avogadros_number = AVOGADRO
__u_to_kg__ = 1.6605390666E-27  # atomic mass units to kg
__speed_of_light__ = 299792458   # c in m/s


#  Note to myself: Pickled nuclear data is on personal SSD
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
#   * Use pyne to read latest ENDSFs and fill in some missing half lives via a pickle file.
#     Then, read results into <additional_nuclide_data> variable.
#   * Uncertainty in xs values for CrossSection1D.plot? xs values are ufloats
#   * Find a way to include data files in the pip package
#   * Xe134 shows a half-life of 0, when in fact it is very large. The ENDF file has thee correct half-life.
#     Whats going on here?

# p = '/Users/jeffreyburggraf/PycharmProjects/miscellaneous/data.zip'
#
# with ZipFile(p, 'r') as zip_ref:
#     zip_ref.ext


NUCLIDE_INSTANCES = {}  # Dict of all Nuclide class objects created. Used for performance enhancements and for pickling
PROTON_INDUCED_FISSION_XS1D = {}  # all available proton induced fission xs. lodaed only when needed.
PHOTON_INDUCED_FISSION_XS1D = {}  # all available proton induced fission xs. lodaed only when needed.
# All neutron yield data loaded - to avoid loading large files twice.
NEUTRON_YIELD_DATA = {'independent': {}, 'cumulative': {}}

DECAY_PICKLE_DIR = pwd/'data'/'nuclides'  # rel. dir. of pickled nuke data
PROTON_PICKLE_DIR = pwd / "data" / "incident_proton"  # rel. dir. of pickled proton activation data
PHOTON_PICKLE_DIR = pwd / "data" / "incident_photon"  # rel. dir. of pickled photon activation data
SF_YIELD_PICKLE_DIR = pwd/'data'/'SF_yields'
NEUTRON_F_YIELD_PICKLE_DIR_GEF = pwd / 'data' / 'neutron_fiss_yields'/'gef'
NEUTRON_F_YIELD_PICKLE_DIR_ENDF = pwd / 'data' / 'neutron_fiss_yields'/'endf'

NUCLIDE_NAME_MATCH = re.compile("([A-Za-z]{1,2})([0-9]{1,3})(?:_m([0-9]+))?")  # Nuclide name in GND naming convention

# global variable for the bin-width of xs interpolation
XS_BIN_WIDTH_INTERPOLATION = 0.1

# Some additional nuclide info that aren't in ENDSFs
additional_nuclide_data = {"In101_m1": {"half_life": ufloat(10, 5)},
                           "Lu159_m1": {"half_life": ufloat(10, 5)},
                           "Rh114_m1": {"half_life": ufloat(1.85, 0.05), "__decay_daughters_str__": "Pd114"},
                           "Pr132_m1": {"half_life": ufloat(20, 5), "__decay_daughters_str__": "Ce132"}}

def human_readable_half_life(hl, include_errors):
    def get_error_print(e, sig_figs=None):
        if sig_figs is None:
            if 1 <= e <= 100:
                sig_figs = 0
            elif 0.1 <= e < 1:
                sig_figs = 1
            else:
                sig_figs = 2

        if sig_figs == 0:
            return "+/-{}%".format(int(e))
        elif sig_figs == 1:
            return "+/-{:.1f}%".format(e)
        else:
            fmt = "+/-{{:.{}e}}%".format(sig_figs)
            return fmt.format(e)

    hl_in_sec = hl

    if hl_in_sec.n == np.inf or hl_in_sec.n == np.nan:
        return str(hl_in_sec.n)

    if hl_in_sec < 1:
        percent_error = 100 * hl_in_sec.std_dev / hl_in_sec.n
        out = "{:.2e} seconds ".format(hl_in_sec.n)
        if include_errors:
            out += " ({}) ".format(get_error_print(percent_error))
        return out

    elif hl_in_sec < 60:
        percent_error = 100 * hl_in_sec.std_dev / hl_in_sec.n
        out = "{:.1f} seconds ".format(hl_in_sec.n)
        if include_errors:
            out += " ({}) ".format(get_error_print(percent_error))
        return out

    seconds_in_a_minute = 60
    seconds_in_a_hour = 60 * seconds_in_a_minute
    seconds_in_a_day = seconds_in_a_hour * 24
    seconds_in_a_month = seconds_in_a_day * 30
    seconds_in_a_year = 12 * seconds_in_a_month

    n_seconds = hl_in_sec % seconds_in_a_minute
    n_minutes = (hl_in_sec % seconds_in_a_hour) / seconds_in_a_minute
    n_hours = (hl_in_sec % seconds_in_a_day) / seconds_in_a_hour
    n_days = (hl_in_sec % seconds_in_a_month) / seconds_in_a_day
    n_months = (hl_in_sec % seconds_in_a_year) / seconds_in_a_month
    n_years = (hl_in_sec / seconds_in_a_year)

    out = None

    for value, unit in zip([n_seconds, n_minutes, n_hours, n_days, n_months, n_years],
                           ['seconds', 'minutes', 'hours', 'days', 'months', 'years']):
        error, value = value.std_dev, value.n
        if int(value) != 0:
            if np.isclose(error, value) and unit == 'years':
                percent_error = 'lower bound'
            else:
                percent_error = 100 * error / value
                if percent_error < 0.001:
                    percent_error = get_error_print(percent_error, 4)
                elif percent_error < 0.01:
                    percent_error = get_error_print(percent_error, 3)
                elif percent_error < 0.1:
                    # percent_error = '+/-{:.2e}%'.format(percent_error)
                    percent_error = get_error_print(percent_error, 2)
                elif percent_error < 1:
                    # percent_error = "+/-{:.1f}%".format(error)
                    percent_error = get_error_print(percent_error, 1)
                else:
                    percent_error = get_error_print(percent_error, 0)

            if value >= 100:
                value_str = "{:.3e}".format(value)
            else:
                value_str = "{:.1f}".format(value)


            out = "{} {}".format(value_str, unit)
            if include_errors:
                out += " ({})".format(percent_error)

    if out is None:
        assert False, 'Issue in "human_friendly_half_life'

    return out


# Needed because classes in this __init__ file will not be in scope of __main__ as required for unpickling
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'GammaLine':
            return GammaLine
        elif name == 'DecayMode':
            return DecayMode
        elif name == '_Reaction':
            return ActivationReactionContainer
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
        return "Gamma line at {0:.1f} KeV; true_intensity = {1:.2e}; decay: {2} ".format(self.erg, self.intensity,
                                                                                         self.from_mode)


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

    @classmethod
    def from_endf(cls, endf_path, product_name, mt=5):
        e = Evaluation(endf_path)
        r = Reaction.from_endf(e, mt)
        for prod in r.products:
            if prod.particle == product_name:
                break
        else:
            assert False, 'could not find product in endf file {0}. Available products: {1}'\
                .format(Path(endf_path).name, [prod.particle for prod in r.products])

        erg = prod.yield_.x/1E6
        xs = prod.yield_.y
        return CrossSection1D(erg, xs)

    def interp(self, new_energies) -> np.ndarray:
        return np.interp(new_energies, self.ergs, self.xss)

    def plot(self, ax=None, fig_title=None, units="b", erg_min=None, erg_max=None):
        unit_convert = {"b": 1, "mb": 1000, "ub": 1E6, "nb": 1E9}
        try:
            unit_factor = unit_convert[units]
        except KeyError:
            assert False, "Invalid unit '{0}'. Valid options are: {1}".format(units, unit_convert.keys())
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if ax.get_title() == '':
            if fig_title is not None:
                ax.set_title(fig_title)
            else:
                if self.__fig_label__ is not None:
                    ax.set_title(self.__fig_label__)
        if erg_max is None:
            erg_max = self.__ergs__[-1]
        if erg_min is None:
            erg_min = self.__ergs__[0]
        selector = np.where((self.__ergs__ <= erg_max) & (self.__ergs__ >= erg_min))
        ax.plot(self.__ergs__[selector], (self.__xss__[selector]) * unit_factor, label=self.__fig_label__)
        y_label = "Cross-section [{}]".format(units)
        x_label = "Incident {} energy [MeV]".format(self.__incident_particle__)
        if ax is plt:
            ax.ylabel(y_label)
            ax.xlabel(x_label)
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        return ax

    @property
    def bin_widths(self):
        return XS_BIN_WIDTH_INTERPOLATION

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
        out = "{0}:\nergs: {1}\nxs: {2}".format(self.__fig_label__, ergs, xss)
        return out


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
    A = int(_m.groups()[1])
    isometric_state = _m.groups()[2]
    if isometric_state is None:
        isometric_state = 0
    isometric_state = isometric_state
    return {'Z': Z, "A": A, "M": isometric_state}


def __nuclide_cut__(a_z_hl_cut: str, a, z, hl, is_stable_only):
    makes_cut = True

    assert isinstance(is_stable_only, bool)
    if is_stable_only and not np.isinf(hl):
        return False

    if len(a_z_hl_cut) > 0:
        a_z_hl_cut = a_z_hl_cut.lower()
        if 'hl' in a_z_hl_cut and hl is None:
            makes_cut = False
        else:
            try:
                makes_cut = eval(a_z_hl_cut, {"hl":hl, 'a': a, 'z': z})
                assert isinstance(makes_cut, bool), "Invalid cut: {0}".format(a_z_hl_cut)
            except NameError as e:
                invalid_name = str(e).split("'")[1]

                raise Exception("\nInvalid name '{}' used in cut. Valid names are: 'z', 'a', and 'hl',"
                                " which stand for atomic-number, mass-number, and half-life, respectively."
                                .format(invalid_name)) from e

    return makes_cut


proton_mass = 938.272
neutron_mass = 939.565


def get_proton_to_neutron_equiv_fission_erg(n: Nuclide, proton_erg):
    'eproton - rm[0, 1] + rm[1, 1] + rm[Z, A] - rm[1 + Z, -1 + A] + \
    rm[1 + Z, A] - rm[1 + Z, 1 + A]'
    z, a = n.Z, n.A
    if hasattr(proton_erg, '__iter__'):
        proton_erg = np.array(proton_erg)
    return proton_erg - neutron_mass + proton_mass \
           + Nuclide.get_mass_in_mev_per_c2(z=z, a=a) \
           - Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a - 1) \
           + Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a) \
           - Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a + 1)


def get_photon_to_neutron_equiv_fission_erg(n: Nuclide, photon_erg):
    z, a = n.Z, n.A
    if hasattr(photon_erg, '__iter__'):
        photon_erg = np.array(photon_erg)
    return photon_erg - neutron_mass - Nuclide.get_mass_in_mev_per_c2(z=z, a=a-1) + \
           Nuclide.get_mass_in_mev_per_c2(z=z, a=a)


def __set_neutron_fiss_yield_ergs__():
    if 'ergs' not in NEUTRON_YIELD_DATA:
        with open(NEUTRON_F_YIELD_PICKLE_DIR_GEF / 'yield_ergs.pickle', 'rb') as f:
            NEUTRON_YIELD_DATA['ergs'] = pickle.load(f)
    return NEUTRON_YIELD_DATA['ergs']


class Nuclide:
    """A nuclide that can be used to access cross sections, decay children/parent nuclides, decay modes, and much more.

        Parameters
        ----------
        ev_or_filename : str of openmc.data.endf.Evaluation
            ENDF fission product yield evaluation to read from. If given as a
            string, it is assumed to be the filename for the ENDF file.

        Attributes
        ----------
        half life : UFloat
        todo

        Notes
        -----
        """
    def __init__(self, name, **kwargs):
        assert isinstance(name, str)
        assert '__internal__' in kwargs, '\nTo generate a Nuclide instance, use the following syntax:\n\t' \
                                         'Nuclide.from_symbol(<symbol>)'

        self.is_valid: bool = True  # Was there any available data for this nuclide?
        orig_name = name
        if '-' in name:
            name = name.replace('-', '')
            name = name.replace('Nat', '0')
            if name.endswith('m'):
                name = name[:-1] + '_m1'

            msg = 'OpenMC nuclides follow the GND naming convention. Nuclide ' \
                  '"{}" is being renamed as "{}".'.format(orig_name, name)
            warn(msg)
        self.name: str = name

        self.__Z_A_iso_state__ = get_z_a_m_from_name(name)

        self.half_life: UFloat = kwargs.get("half_life", None)
        self.spin: int = kwargs.get("spin", None)
        self.mean_energies = kwargs.get("mean_energies", None)  # look into this
        self.is_stable: bool = kwargs.get("is_stable", None)  # maybe default to something else

        self.__decay_daughters_str__: List[str] = kwargs.get("__decay_daughters__", [])  # self.decay_daughters -> List[Nuclide] in corresponding order as self.decay_modes
        self.decay_gamma_lines: List[GammaLine] = kwargs.get("decay_gamma_lines", [])
        self.decay_modes: List[DecayMode] = kwargs.get("decay_modes", [])
        self.__decay_parents_str__: List[str] = kwargs.get("__decay_parents__", [])  # self.decay_parents -> List[Nuclide]

        self.__decay_mode_for_print__ = None

    def plot_decay_gamma_spectrum(self, label_first_n_lines: int = 5, min_intensity:float=0.05, ax=None,
                                  log_scale=False, label=None):
        assert isinstance(label_first_n_lines, int)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = ax.gcf(), ax.gca()

        if log_scale:
            ax.set_yscale('log')
        x = []
        x_err = []
        y = []
        y_err = []
        ax.set_xlabel('Gamma energy')
        ax.set_ylabel('Gammas per decay')
        for g in self.decay_gamma_lines:
            if g.intensity.n >= min_intensity:
                x.append(g.erg.n)
                x_err.append(g.erg.std_dev)

                y.append(g.intensity.n)
                y_err.append(g.intensity.std_dev)

        label_first_n_lines += 1
        label_first_n_lines = min(len(x), label_first_n_lines)
        used_xys = set()
        ax.set_ylim(0, ax.get_ylim()[1]*1.5)

        y_max = ax.get_ylim()[1]
        _x_ = np.array(x[:label_first_n_lines])
        _y_ = np.array(y[:label_first_n_lines])
        sorter = np.argsort(_x_)
        for index, xy in enumerate(zip(_x_[sorter], _y_[sorter])):
            xy = np.array(xy)
            erg = xy[0]
            print('Label: {:.2f}KeV'.format(erg))

            text_xy = (xy[0], y_max*0.7)
            ax.annotate('{:.2f}KeV'.format(erg), xy=xy, xytext=text_xy,
                        arrowprops=dict(width=0.1, headwidth=4, facecolor='black', shrink=0.03), rotation=90)

        ax.set_title('Gamma spectrum of {}, half-life = {}'.format(self.name, self.human_friendly_half_life()))
        ax.errorbar(x, y, yerr=y_err, xerr=x_err, label=label, marker='p', ls='none')
        return fig, ax

    @property
    def Z(self) -> int:
        """Proton number"""
        if not hasattr(self, '__Z_A_iso_state__'):
            self.__Z_A_iso_state__ = get_z_a_m_from_name(self.name)
        return self.__Z_A_iso_state__['Z']

    @property
    def isometric_state(self) -> int:
        """Meta stable excited state, starting with "0" as the ground state."""
        if not hasattr(self, '__Z_A_iso_state__'):
            self.__Z_A_iso_state__ = get_z_a_m_from_name(self.name)
        return self.__Z_A_iso_state__['M']

    @property
    def A(self) -> int:
        if not hasattr(self, '__Z_A_iso_state__'):
            self.__Z_A_iso_state__ = get_z_a_m_from_name(self.name)
        return self.__Z_A_iso_state__['A']

    def human_friendly_half_life(self, include_errors: bool=True) -> str:
        return human_readable_half_life(self.half_life, include_errors)

    @property
    def proton_induced_fiss_xs(self) -> CrossSection1D:
        global PROTON_INDUCED_FISSION_XS1D
        simple_nuclide_name = self.atomic_symbol + str(self.A)
        if simple_nuclide_name not in PROTON_INDUCED_FISSION_XS1D:
            try:
                with open(PROTON_PICKLE_DIR/'fission'/'{}.pickle'.format(simple_nuclide_name), 'rb') as f:
                    PROTON_INDUCED_FISSION_XS1D[simple_nuclide_name] = CustomUnpickler(f).load()
            except FileNotFoundError:
                assert False, 'No proton induced fission data for {0}. Download it and integrate it if it is ' \
                              'available. The conversion to pickle is done in `endf_to_pickle.py`. See "{1}" for' \
                              ' instructions.'.format(self, pwd/'endf_files'/'FissionXS'/'readme')

        return PROTON_INDUCED_FISSION_XS1D[simple_nuclide_name]

    @property
    def photon_induced_fiss_xs(self) -> CrossSection1D:
        simple_nuclide_name = self.atomic_symbol + str(self.A)
        if simple_nuclide_name not in PHOTON_INDUCED_FISSION_XS1D:
            try:
                with open(PHOTON_PICKLE_DIR / 'fission' / '{}.pickle'.format(simple_nuclide_name), 'rb') as f:
                    PHOTON_INDUCED_FISSION_XS1D[simple_nuclide_name] = CustomUnpickler(f).load()
            except FileNotFoundError:
                assert False, 'No photon induced fission data for {0}.'

        return PHOTON_INDUCED_FISSION_XS1D[simple_nuclide_name]

    def rest_energy(self, units='MeV'):  # in J or MeV
        units = units.lower()
        ev = 1.0/1.602176634E-19
        unit_dict = {'j': 1, 'mev': ev*1E-6, 'ev': ev}
        assert units in unit_dict.keys(), 'Invalid units, "{}".\nUse one of the following: {}'\
            .format(units, unit_dict.keys())
        j = self.atomic_mass*__u_to_kg__*__speed_of_light__**2
        return j*unit_dict[units]

    @staticmethod
    def get_mass_in_mev_per_c2(n: (Nuclide, str) = None, z=None, a=None):
        c = 931.494102
        if n is not None:
            if isinstance(n, Nuclide):
                return n.mass_in_mev_per_c2()
            else:
                assert isinstance(n, str)
                if '_' in n:
                    n = n[:n.index('_')]
                return atomic_mass(n)*c
        else:
            assert z is not None and a is not None
            return atomic_mass(ATOMIC_SYMBOL[z] + str(a))*c

    @property
    def mass_in_mev_per_c2(self) -> float:
        return self.rest_energy('MeV')

    @property
    def atomic_mass(self) -> float:
        try:
            return atomic_mass(self.name)
        except KeyError:
            warn('Atomic mass for {} not found'.format(self))
            return None

    @property
    def grams_per_mole(self) -> float:
        try:
            return atomic_weight(self.name)
        except KeyError:
            warn('Atomic weight for {} not found'.format(self))
            return None

    @property
    def isotopic_abundance(self) -> float:
        _m = re.match('([A-Za-z]{1,2}[0-9]+)(?:m_[0-9]+)?', self.name)
        if _m:
            s = _m.groups()[0]
            try:
                return NATURAL_ABUNDANCE[s]
            except KeyError:
                pass
        return 0

    @property
    def atomic_symbol(self) -> str:
        """Atomic symbol according to the proton number of this isotope"""
        _m = re.match('([A-Za-z]{1,2}).+', self.name)
        if _m:
            return _m.groups()[0]
        else:
            return None

    def add_proton(self) -> Nuclide:
        return self.from_Z_A_M(self.Z+1, self.A+1, self.isometric_state)

    def remove_proton(self) -> Nuclide:
        return self.from_Z_A_M(self.Z-1, self.A-1, self.isometric_state)

    def remove_neutron(self) -> Nuclide:
        return self.from_Z_A_M(self.Z, self.A-1, self.isometric_state)

    def add_neutron(self) -> Nuclide:
        return self.from_Z_A_M(self.Z, self.A+1, self.isometric_state)

    def is_heavy_FF(self) -> bool:
        return 125 < self.A < 158

    def is_light_FF(self) -> bool:
        return 78 < self.A < 112

    @classmethod
    def from_Z_A_M(cls, z: int, a: int, isometric_state: int = 0) -> Nuclide:
        try:
            z = int(z)
        except ValueError:
            assert False, 'Invalid value passed to `z` argument: "{}"'.format(z)

        try:
            a = int(a)
        except ValueError:
            assert False, 'Invalid value passed to `a` argument: "{}"'.format(a)

        try:
            isometric_state = int(isometric_state)
        except ValueError:
            if isinstance(isometric_state, str):
                if len(isometric_state) == 0:
                    isometric_state = 0
                else:
                    assert False, '\n`isometric_state` argument must be able to be converted into an integer,' \
                                  ' not "{}"'.format(isometric_state)
        try:
            symbol = ATOMIC_SYMBOL[z]
        except KeyError:
            assert False, "Invalid atomic number: {0}. Cant find atomic symbol".format(z)
        name = "{0}{1}".format(symbol, a)
        if isometric_state != 0:
            name += "_m" + str(isometric_state)
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
                    instance = Nuclide(symbol, __internal__=True, **additional_nuclide_data[symbol])
                    instance.is_valid = True
                else:
                    warn("Cannot find data for Nuclide `{0}`. Data for this nuclide is set to defaults: None, nan, ect."
                         .format(symbol))
                    instance = Nuclide(symbol,  __internal__=True, half_life=ufloat(np.nan, np.nan))
                    instance.is_valid = False
            else:
                with open(pickle_file, "rb") as pickle_file:
                    instance = CustomUnpickler(pickle_file).load()
                    instance.is_valid = True
        else:
            instance = NUCLIDE_INSTANCES[symbol]
            instance.is_valid = True
        # assert isinstance(instance, type(cls)), type(instance)
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

    def __get_sf_yield__(self, yielded_nuclide, independent_bool):
        sub_dir = ('independent' if independent_bool else 'cumulative')
        f_path = SF_YIELD_PICKLE_DIR/sub_dir/'{}.pickle'.format(self.name)
        assert f_path.exists(), 'No SF fission yield data for {}'.format(self.name)
        with open(f_path, 'rb') as f:
            yield_ = pickle.load(f)
        if yielded_nuclide == 'all':
            return yield_
        else:
            assert NUCLIDE_NAME_MATCH.match(yielded_nuclide), 'Invalid nuclide name for yield, "{}"'\
                .format(yielded_nuclide)
            try:
                return yield_[yielded_nuclide]
            except KeyError:
                warn('{} yield of {} from the SF of {} not found in data. Returning zero.'
                     .format(sub_dir, yielded_nuclide, self.name))
                return 0

    def __get_neutron_fiss_yield_gef__(self, product_nuclide: str, independent_bool, eval_ergs):
        if isinstance(product_nuclide, Nuclide):
            product_nuclide = product_nuclide.name
        assert isinstance(product_nuclide, str), product_nuclide
        product_nuclide = product_nuclide.replace('-', '')
        sub_dir = ('independent' if independent_bool else 'cumulative')
        fission_nuclide_path = NEUTRON_F_YIELD_PICKLE_DIR_GEF / sub_dir / '{}.marshal'.format(self.name)
        print(fission_nuclide_path)
        __set_neutron_fiss_yield_ergs__()
        data_ergs = NEUTRON_YIELD_DATA['ergs']
        if eval_ergs is None:
            new_ergs = data_ergs
        else:
            new_ergs = eval_ergs

        if self.name not in NEUTRON_YIELD_DATA[sub_dir]:
            assert fission_nuclide_path.exists(), 'No neutron fission yield data for {}'.format(self.name)
            with open(fission_nuclide_path, 'rb') as f:
                yield_data = marshal.load(f)
                NEUTRON_YIELD_DATA[sub_dir][self.name] = yield_data
        else:
            yield_data = NEUTRON_YIELD_DATA[sub_dir][self.name]

        def interp(yield_data_):
            nominal_values = yield_data_['yield']
            std_devs = yield_data_['yield_err']
            result = unp.uarray(np.interp(new_ergs, data_ergs, nominal_values),
                                np.interp(new_ergs, data_ergs, std_devs))
            return result

        if product_nuclide == 'all':
            out = {k: interp(v) for k, v in yield_data.items()}
            if eval_ergs is None:
                return new_ergs, out
            else:
                return out

        elif product_nuclide not in yield_data:
            warn('Gef fission yield data for fission fragment {} from fissioning nucleus {} not present. Assuming zero.'
                 .format(product_nuclide, self.name))
            _ = np.zeros_like(new_ergs)
            out = unp.uarray(_, _)
            if eval_ergs is None:
                return new_ergs, out
            else:
                return out
        else:
            y = yield_data[product_nuclide]
            if eval_ergs is None:
                return new_ergs, interp(y)
            else:
                return interp(y)

    def independent_neutron_fission_yield_gef(self, product_nuclide='all', ergs=None):
        return self.__get_neutron_fiss_yield_gef__(product_nuclide, True, ergs)

    def cumulative_neutron_fission_yield_gef(self, product_nuclide='all', ergs=None):
        return self.__get_neutron_fiss_yield_gef__(product_nuclide, False, ergs)

    def independent_gamma_fission_yield_gef(self, product_nuclide='all', ergs=None):
        non_flag = ergs is None
        if ergs is None:
            ergs = __set_neutron_fiss_yield_ergs__()
        ergs = get_photon_to_neutron_equiv_fission_erg(self, ergs)
        out = self.__get_neutron_fiss_yield_gef__(product_nuclide, True, ergs)
        if non_flag:
            return ergs, out
        else:
            return out

    def cumulative_gamma_fission_yield_gef(self, product_nuclide='all', ergs=None):
        non_flag = ergs is None
        if ergs is None:
            ergs = __set_neutron_fiss_yield_ergs__()
        ergs = get_photon_to_neutron_equiv_fission_erg(self, ergs)
        out = self.__get_neutron_fiss_yield_gef__(product_nuclide, False, ergs)
        if non_flag:
            return ergs, out
        else:
            return out

    def independent_proton_fission_yield_gef(self, product_nuclide='all', ergs=None):
        non_flag = ergs is None
        if ergs is None:
            ergs = __set_neutron_fiss_yield_ergs__()
        ergs = get_proton_to_neutron_equiv_fission_erg(self, ergs)
        out = self.__get_neutron_fiss_yield_gef__(product_nuclide, True, ergs)
        if non_flag:
            return ergs, out
        else:
            return out

    def cumulative_proton_fission_yield_gef(self, product_nuclide='all', ergs=None):
        non_flag = ergs is None
        if ergs is None:
            ergs = __set_neutron_fiss_yield_ergs__()
        ergs = get_proton_to_neutron_equiv_fission_erg(self, ergs)
        out = self.__get_neutron_fiss_yield_gef__(product_nuclide, False, ergs)
        if non_flag:
            return ergs, out
        else:
            return out

    def __get_neutron_fiss_yield_endf__(self, yield_type, product_nuclide='all', ):
        assert yield_type in ['cumulative', 'independent']
        f_path = NEUTRON_F_YIELD_PICKLE_DIR_ENDF/yield_type/(self.name + '.marshal')
        assert f_path.exists(), 'No ENDF neutron fission yield data for {}'.format(self.name)
        with open(f_path, 'rb') as f:
            yield_data = marshal.load(f)
        ergs = np.array(yield_data['ergs'])
        del yield_data['ergs']
        if product_nuclide == 'all':
            return ergs, yield_data
        else:
            assert isinstance(product_nuclide, (str, Nuclide))
            if isinstance(product_nuclide, str):
                assert NUCLIDE_NAME_MATCH.match(product_nuclide), 'Invalide nuclide name, {}'.format(product_nuclide)
            elif isinstance(product_nuclide, Nuclide):
                product_nuclide = product_nuclide.name
            else:
                assert False, '`product_nuclide` must be a string or a Nuclide instance.'

            if product_nuclide not in yield_data:
                warn('ENDF fission yield data for fission fragment {} from fissioning nucleus {} not present. '
                     'Assuming zero.'.format(product_nuclide, self.name))
                _ = np.zeros(len(ergs))
                return ergs, unp.uarray(_, _)

            return ergs, unp.uarray(yield_data[product_nuclide]['yield'], yield_data[product_nuclide]['yield_err'])

    def cumulative_neutron_fission_yield_endf(self, product_nuclide='all'):
        return self.__get_neutron_fiss_yield_endf__('cumulative', product_nuclide)

    def independent_neutron_fission_yield_endf(self, product_nuclide='all'):
        return self.__get_neutron_fiss_yield_endf__('independent', product_nuclide)

    def independent_sf_fission_yield(self, product_nuclide='all'):
        return self.__get_sf_yield__(product_nuclide, True)

    def cumulative_sf_fission_yield(self, product_nuclide='all'):
        return self.__get_sf_yield__(product_nuclide, False)

    def get_incident_proton_parents(self, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedParent]:
        pickle_path = PROTON_PICKLE_DIR / (self.name + ".pickle")
        return self.__get_parents__(pickle_path, 'proton', a_z_hl_cut, is_stable_only)

    def get_incident_proton_daughters(self, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedDaughter]:
        pickle_path = PROTON_PICKLE_DIR/(self.name + ".pickle")
        return self.__get_daughters__(pickle_path, 'proton', a_z_hl_cut, is_stable_only)

    def get_incident_photon_daughters(self, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedDaughter]:
        pickle_path = PHOTON_PICKLE_DIR/(self.name + ".pickle")
        return self.__get_daughters__(pickle_path, 'photon', a_z_hl_cut, is_stable_only)

    def get_incident_photon_parents(self, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedParent]:
        pickle_path = PHOTON_PICKLE_DIR/(self.name + ".pickle")
        return self.__get_parents__(pickle_path, 'photon', a_z_hl_cut, is_stable_only)

    def __get_daughters__(self, data_path, incident_particle, a_z_hl_cut='', is_stable_only=False):
        if not data_path.exists():
            warn("No {}-induced data for {}".format(incident_particle, self.name))
            return None

        with open(data_path, "rb") as f:
            reaction = CustomUnpickler(f).load()

        assert isinstance(reaction, ActivationReactionContainer)
        out: Dict[str, InducedDaughter] = {}
        for daughter_name, xs in reaction.product_nuclide_names_xss.items():
            daughter_nuclide = Nuclide.from_symbol(daughter_name)
            a, z, hl = daughter_nuclide.A, daughter_nuclide.Z, daughter_nuclide.half_life
            if __nuclide_cut__(a_z_hl_cut, a, z, hl, is_stable_only):
                daughter = InducedDaughter(daughter_nuclide, self, incident_particle)
                daughter.xs = xs
                out[daughter_name] = daughter

        return out

    def __get_parents__(self, data_path, incident_particle, a_z_hl_cut='', is_stable_only=False):
        if not data_path.exists():
            warn("No {}-induced data for any parents of {}".format(incident_particle, self.name))
            return {}

        with open(data_path, "rb") as f:
            daughter_reaction = CustomUnpickler(f).load()

        assert isinstance(daughter_reaction, ActivationReactionContainer)
        out = {}
        parent_nuclides = [Nuclide.from_symbol(name) for name in daughter_reaction.parent_nuclide_names]
        daughter_nuclide = self
        for parent_nuclide in parent_nuclides:
            parent_pickle_path = data_path.parent / (parent_nuclide.name + ".pickle")
            with open(parent_pickle_path, "rb") as f:
                parent_reaction = CustomUnpickler(f).load()
                assert isinstance(parent_reaction, ActivationReactionContainer)
            parent = InducedParent(daughter_nuclide, parent_nuclide, inducing_particle=incident_particle)
            a, z, hl = parent.A, parent.Z, parent.half_life
            if __nuclide_cut__(a_z_hl_cut, a, z, hl, is_stable_only):
                parent.xs = parent_reaction.product_nuclide_names_xss[daughter_nuclide.name]
                out[parent.name] = parent

        return out

    def is_effectively_stable(self, threshold_in_years=100):
        return self.half_life.n >= (365*24*60**2)*threshold_in_years

    def __set_data_from_open_mc__(self, open_mc_decay):
        self.half_life = open_mc_decay.half_life
        self.mean_energies = open_mc_decay.average_energies
        self.spin = open_mc_decay.nuclide["spin"]

        if isinf(self.half_life.n) or open_mc_decay.nuclide["stable"]:
            self.is_stable = True
            self.half_life = ufloat(np.inf, 0)
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
        with open(DECAY_PICKLE_DIR/'quick_nuclide_lookup.pickle', 'rb') as f:
            nuclides_dict = pickle.load(f)
        for (a, z, hl), nuclide_name in nuclides_dict.items():
            if __nuclide_cut__(a_z_hl_cut, a, z, hl, is_stable_only):
                nuclides.append(Nuclide.from_symbol(nuclide_name))

        return nuclides


class InducedDaughter(Nuclide):
    def __init__(self, daughter_nuclide, parent_nuclide, inducing_particle):
        assert isinstance(daughter_nuclide, Nuclide)
        assert isinstance(parent_nuclide, Nuclide)
        kwargs = {k: v for k, v in daughter_nuclide.__dict__.items() if k != "name"}
        super().__init__(daughter_nuclide.name, __internal__=True, **kwargs)
        self.xs: CrossSection1D = None
        self.parent: Nuclide = parent_nuclide
        self.inducing_particle = inducing_particle

    def __repr__(self):
        par_symbol = self.inducing_particle[0]
        return '{0}({1},X) --> {2}'.format(self.parent, par_symbol, super().__repr__())


class InducedParent(Nuclide):
    def __init__(self, daughter_nuclide, parent_nuclide, inducing_particle):
        assert isinstance(daughter_nuclide, Nuclide)
        assert isinstance(parent_nuclide, Nuclide)
        kwargs = {k: v for k, v in parent_nuclide.__dict__.items() if k != "name"}
        super().__init__(parent_nuclide.name, __internal__=True, **kwargs)
        self.xs: CrossSection1D = None
        self.daughter: Nuclide = daughter_nuclide
        self.inducing_particle = inducing_particle

    def __repr__(self):
        par_symbol = self.inducing_particle[0]
        return '{0}({1},X) --> {2}'.format(super().__repr__(), par_symbol, self.daughter)


# used for storing particle-induced data in pickle file. Imported in endf_to_pickle.py, and used in this module
# for unpickling
class ActivationReactionContainer:
    def __init__(self, name):
        self.name = name
        #  Dict with daughter nuclide names as keys as xs objects as values
        self.product_nuclide_names_xss: Dict[str, CrossSection1D] = {}
        self.parent_nuclide_names: List[str] = []

    @staticmethod
    def set(init_data_dict, nuclide_name, endf_file_path, incident_particle):
        print('Reading data from {} for {}'.format(nuclide_name, incident_particle + 's'))

        if nuclide_name in init_data_dict:
            reaction = init_data_dict[nuclide_name]
        else:
            reaction = ActivationReactionContainer(nuclide_name)
            init_data_dict[nuclide_name] = reaction

        e = Evaluation(endf_file_path)
        for activation_product in Reaction.from_endf(e, 5).products:
            activation_product_name = activation_product.particle
            if activation_product_name == "photon":
                continue
            if activation_product_name == "neutron":
                activation_product_name = "Nn1"
            try:
                par_id = {'proton': 'p', 'neutron': 'n', 'photon': 'G'}[incident_particle]
            except KeyError:
                assert False, 'Invalid incident particle: "{}"'.format(incident_particle)
            xs_fig_label = "{0}({1},X){2}".format(nuclide_name, par_id, activation_product_name)
            try:
                xs = CrossSection1D(activation_product.yield_.x / 1E6, activation_product.yield_.y, xs_fig_label,
                                    incident_particle)
            except AttributeError as e:
                continue
            reaction.product_nuclide_names_xss[activation_product_name] = xs
            if activation_product_name in init_data_dict:
                daughter_reaction = init_data_dict[activation_product_name]
            else:
                daughter_reaction = ActivationReactionContainer(activation_product_name)
                init_data_dict[activation_product_name] = daughter_reaction
            daughter_reaction.parent_nuclide_names.append(nuclide_name)

    def __len__(self):
        return len(self.parent_nuclide_names) + len(self.product_nuclide_names_xss)

    def __repr__(self):
        return 'self: {0}, parents: {1}, daughters: {2}'.format(self.name, self.parent_nuclide_names,
                                                                self.product_nuclide_names_xss.keys())


decay_data_dir = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay"
dir_old = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay/"
dir_new = "/Users/jeffreyburggraf/Desktop/nukeData/ENDF-B-VIII.0_decay/"


if __name__ == "__main__":
    s = Nuclide.from_symbol('La144')
    s.plot_decay_gamma_spectrum()
    plt.show()


