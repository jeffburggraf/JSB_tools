from __future__ import annotations
import pickle
import numpy as np
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
from openmc.data import Reaction, Decay
from matplotlib import pyplot as plt
import re
from pathlib import Path
from warnings import warn
from uncertainties import ufloat, UFloat
from uncertainties import unumpy as unp
import uncertainties
import marshal
from functools import cached_property
from openmc.data.data import NATURAL_ABUNDANCE, atomic_mass, atomic_weight, AVOGADRO
from typing import Union, List, Dict, Collection, Tuple, TypedDict
from numbers import Number
pwd = Path(__file__).parent


__all__ = ['Nuclide', 'avogadros_number']


#  Units
avogadros_number = AVOGADRO
__u_to_kg__ = 1.6605390666E-27  # atomic mass units to kg
__speed_of_light__ = 299792458   # c in m/s

#  Note to myself: Pickled nuclear data is on personal SSD. Update this regularly!
#  Todo:
#   * make cross section pulls be implemented in a nuke_data.cross_secion file. Let the endf to pickle also be
#     implemented there
#   * Make a Nuclide.fromreaction('parent_nuclide_name', inducing particle, daughter_nuclide_name )
#   * Implement FissionProductYields object, similar to CrossSection1D
#   * Add a custom exception, not AssertionError, when an unknown symbol is passed to Nuclide.
#   * Add doc strings.
#   * Add documentation, and exception messages, to explain where the data can be downloaded and how to regenerate
#     the pickle files.
#   * Get rid of <additional_nuclide_data> functionality. too complex.
#   * Why no uncertainty in xs values for CrossSection1D.plot? (some) xs values are ufloats. (no ufloat for PADF?)
#   * Find a way to include data files in the pip package, maybe create a separate repository.


yield_data_type = Dict[float, Dict[str, TypedDict('YieldDict', {'yield': float, 'yield_err': float})]]


NUCLIDE_INSTANCES = {}  # Dict of all Nuclide class objects created. Used for performance enhancements and for pickling
PROTON_INDUCED_FISSION_XS1D = {}  # all available proton induced fission xs. lodaed only when needed.
PHOTON_INDUCED_FISSION_XS1D = {}  # all available proton induced fission xs. lodaed only when needed.

DECAY_PICKLE_DIR = pwd/'data'/'nuclides'  # rel. dir. of pickled nuke data
PROTON_PICKLE_DIR = pwd / "data" / "incident_proton"  # rel. dir. of pickled proton activation data
PHOTON_PICKLE_DIR = pwd / "data" / "incident_photon"  # rel. dir. of pickled photon activation data

FISS_YIELDS_PATH = pwd/'data'/'fiss_yields'
SF_YIELD_PICKLE_DIR = pwd/'data'/'SF_yields'
NEUTRON_F_YIELD_PICKLE_DIR = pwd / 'data' / 'neutron_fiss_yields'

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
    def __init__(self, nuclide: Nuclide, erg: UFloat, intensity: UFloat, intensity_thu_mode: UFloat,
                 from_mode: DecayMode):
        """
        A container for a single gamma emission line.

        Attributes:
            erg:
                Energy of gamma in KeV
            intensity:
                Mean number of gammas with energy self.erg emitted per parent nucleus decay (through any decay channel)

            intensity_thu_mode:
                Mean number of gammas with energy self.erg emitted per parent nucleus decay (through self.from_mode
                decay channel)

            from_mode:
                DecayMode instance. Contains decay channel, branching ratio, among other information
    """
        self.erg = erg
        self.intensity = intensity
        self.from_mode = from_mode
        self.intensity_thu_mode = intensity_thu_mode
        self.absolute_rate = nuclide.decay_rate * self.intensity

    def __repr__(self):
        return "Gamma line at {0:.1f} KeV; true_intensity = {1:.2e}; decay: {2} ".format(self.erg, self.intensity,
                                                                                         self.from_mode)


def __get_fiss_yield_path__(nuclide_symbol: str, inducing_particle4path: str, data_source4path: str,
                            independent_bool: bool) -> Path:
    """
    Get path fission yield data.
    Raises:
        FileNotFoundError if data file can't be found.
    Args:
        nuclide_symbol: e.g. "U238"
        inducing_particle4path: e.g.  "neutron", "SF" (or None), "proton", "gamma"
        data_source4path:  e.g. "gef", "endf"
        independent_bool: e.g. "independent" for independent yield, "cumulative" for cumulative yield.

    Returns: Path to data file

    """
    assert isinstance(independent_bool, bool)

    path = Path(FISS_YIELDS_PATH)
    assert path.exists(), 'Fission yield directory not found: {}'.format(path)

    if inducing_particle4path is None:
        warn('None passed to inducing particle. SF assumed.')
        inducing_particle4path = 'SF'
    path /= inducing_particle4path
    if not path.exists():
        raise FileNotFoundError('\nNo path found for inducing particle "{}". Options are: {}'
                                .format(inducing_particle4path, [p.name for p in (path.parent.iterdir())
                                                                 if p.name != '.DS_Store']))

    assert isinstance(data_source4path, str)
    path /= data_source4path
    if not path.exists():
        raise FileNotFoundError('\nInvalid `data_source` for {}. Options are: {}'.format(inducing_particle4path,
                                                                                         list(path.iterdir())))
    independent4path = ('independent' if independent_bool else 'cumulative')
    path /= independent4path
    if not path.exists():
        raise FileNotFoundError("Fission yield path doesnt exist: {}".format(path))

    path /= (nuclide_symbol + ".marshal")
    if not path.exists():
        print(path)
        raise FileNotFoundError("Not {} yield data {} for {}"
                                .format(inducing_particle4path, data_source4path, nuclide_symbol))
    return path


AVAILABLE_INDUCING_PARTICLES = ['neutron', 'proton', 'sf', 'gamma']


class FissionYields:
    def __search_path__(self, _inducing_particle, data_sources, independent_bool, nuclide_symbol):
        """
        Searches for the data path. If data_source is None, Loop through the respective values in the `data_sources`
        dict above and return the first one that can be found. If none found, return None.
        A data_source of None is picked, then attempt to convert neutron yields to desired yield.
        Returns:

        """
        try:
            for _d in data_sources[_inducing_particle]:
                try:
                    if _d is None:
                        self.converted_from_neutron_fission = True
                        if _inducing_particle == 'gamma':
                            nuclide_symbol = Nuclide.from_symbol(nuclide_symbol).remove_neutron().name
                        elif _inducing_particle == 'proton':
                            nuclide_symbol = Nuclide.from_symbol(nuclide_symbol).remove_proton().name
                        return self.__search_path__('neutron', data_sources, independent_bool, nuclide_symbol)
                    out = __get_fiss_yield_path__(nuclide_symbol, _inducing_particle, _d, independent_bool)
                    return out
                except FileNotFoundError:
                    pass
        except KeyError:
            pass

    def __init__(self, nuclide: Nuclide,
                 inducing_particle: Union[str, type(None)],
                 eval_ergs: Union[Collection, type(None), float] = None,
                 data_source=None,
                 independent_bool: bool = True):
        """
        Retrieve fission yield data, if available. In some cases (e.g. gamma fiss), the code will convert neutron yield
        to the desired yield by adjusting the nucleaous
        Args:
            nuclide: Fissioning Nuclide.
            inducing_particle: Use 'sf' or None for spontaneous fission
            eval_ergs: Iterable, Number, or None. Energies to evaluate (and interpolate ) yields at.
             If None, use the energies from data source.
            data_source: Several data sources are available depending on the inducing particle.
                         neutron: ['endf', 'gef']
                         proton': ['ukfy', <<convert from neutron>>]
                         gamma: [<<convert from neutron>>]
                         'sf':['gef']
                         'alpha': []
                         electron: []
                         If None, pick best.
            independent_bool: Independent or cumulative.
        """
        assert isinstance(inducing_particle, str)
        inducing_particle = inducing_particle.lower()
        inducing_particle = inducing_particle.replace('photon', 'gamma')
        assert inducing_particle in AVAILABLE_INDUCING_PARTICLES,\
            'No data for inducing particle "{}". Your options are: {}'.format(inducing_particle,
                                                                              AVAILABLE_INDUCING_PARTICLES)
        nuclide_symbol = nuclide.name

        self.converted_from_neutron_fission = False

        # data_sources: Mapping from fission type to data source/directory name.
        #  None signifies to try to convert energy to equivalent for neutron induced fission, and use
        #  n-induced fission yields.
        data_sources = {'neutron': ['endf', 'gef'], 'proton': ['ukfy', None], 'gamma': [None], 'alpha': [],
                        'electron': [], 'sf': ['gef']}

        if data_source is None:
            fiss_yield_path = self.__search_path__(inducing_particle, data_sources, independent_bool, nuclide_symbol)
        else:
            fiss_yield_path = __get_fiss_yield_path__(nuclide_symbol, inducing_particle, data_source, independent_bool)

        if fiss_yield_path is None:
            raise FileNotFoundError(f"Cannot find {inducing_particle}-induced fission data for {nuclide_symbol}"
                                    f"{' from ' + data_source + ' library' if data_source else ''}")

        with open(fiss_yield_path, 'rb') as f:
            fiss_data = marshal.load(f)

        data_ergs = np.array(list(fiss_data.keys()))

        if self.converted_from_neutron_fission:
            if eval_ergs is None:
                if inducing_particle == 'gamma':
                    self.energies = GammaFissionErgConverter.neutron2gamma(nuclide, data_ergs)
                elif inducing_particle == 'proton':
                    self.energies = ProtonFissionErgConverter.neutron2proton(nuclide, data_ergs)
                else:
                    assert False
            else:
                if isinstance(eval_ergs, Number):
                    eval_ergs = [eval_ergs]
                else:
                    _msg = '`eval_ergs` must be an iterable of Numbers, None, or a Number'
                    assert hasattr(eval_ergs, '__iter__'), _msg
                    assert all(isinstance(e, Number) for e in eval_ergs), _msg
                eval_ergs = np.array(eval_ergs)
                if inducing_particle == 'gamma':
                    self.energies = GammaFissionErgConverter.gamma2neutron(nuclide, data_ergs)
                elif inducing_particle == 'proton':
                    self.energies = ProtonFissionErgConverter.proton2neutron(nuclide, data_ergs)
                else:
                    assert False, 'Invalid inducing particle, "{}"'.format(inducing_particle)

        self.yields = {}
        for erg, yield_dict in fiss_data.items():
            for n_name, yield_dict in yield_dict.items():
                if n_name not in self.yields:
                    self.yields[n_name] = ([], [])
                self.yields[n_name][0].append(yield_dict['yield'])
                self.yields[n_name][1].append(yield_dict['yield_err'])

        self.yields = {k: unp.uarray(np.interp(eval_ergs, data_ergs, v[0]), np.interp(eval_ergs, data_ergs, v[1]))
                       for k, v in sorted(self.yields.items(), key=lambda x: -sum(x[1][0]))}

    def __repr__(self):
        return str(self.yields)


class CrossSection1D:
    def __init__(self, ergs: List[float], xss: List[Union[UFloat, float]],
                 fig_label: str = None, incident_particle: str = 'particle'):
        """
        A container for energy dependent 1-D cross-section
        Args:
            ergs: energies for which cross-section is evaluated.
            xss: Cross sections corresponding to energies.
            fig_label: Figure label.
            incident_particle:
        """
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

    def cut_off(self, threshold=0.001):
        i = np.searchsorted(self.xss, threshold)
        return self.ergs[i]


class DecayMode:
    """
    Container for DecayMode. Effectively an openmc.DecayMode wrapper.
    """
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


def get_z_a_m_from_name(name: str) -> Dict[str, int]:
    """
    From a nuclide name, return the atomic number (Z), mass number (A) and isomeric state (ie excited state, or M)
    Args:
        name: nuclide name

    Returns:

    """
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


def __nuclide_cut__(a_z_hl_cut: str, a: int, z: int, hl: UFloat, is_stable_only) -> bool:
    """
    Does a nuclide with a given z, a, and hl (half life) make fit the criteria given by `a_z_hl_cut`?

    Args:
        a_z_hl_cut: The criteria to be evaluated as python code, where z=atomic number, a=mass_number,
            and hl=half life in seconds
        a: mass number
        z: atomic number
        hl: half life in seconds
        is_stable_only: does the half life have to be effectively infinity?

    Returns:

    """
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


class ProtonFissionErgConverter:
    @staticmethod
    def proton2neutron(n: Nuclide, proton_erg, __invert__=False):
        """
        Calculates the neutron energy that gives the proper pre-fission state for the compound nucleus
        (from proton fission on nuclide `n`). Used when proton yields are not available but neutron are.

        Args:
            n:
            proton_erg:
            __invert__: inverse funtion
        Returns:

        """
        z, a = n.Z, n.A
        if hasattr(proton_erg, '__iter__'):
            proton_erg = np.array(proton_erg)
        _term = - neutron_mass + proton_mass \
                + Nuclide.get_mass_in_mev_per_c2(z=z, a=a) \
                - Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a - 1) \
                + Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a) \
                - Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a + 1)
        if  not __invert__:
            out = proton_erg + _term
        else:
            neutron_erg = proton_erg
            out = neutron_erg - _term

        out = np.array(list(filter(lambda x: x > 0, out)), dtype=float)
        return out

    @staticmethod
    def neutron2proton(n: Nuclide, neutron_erg):
        return ProtonFissionErgConverter.proton2neutron(n,neutron_erg, True)


class GammaFissionErgConverter:
    @staticmethod
    def gamma2neutron(n: Nuclide, photon_erg, __invert__=False):
        """
            Calculates the neutron energy that gives the proper pre-fission state for the compound nucleus
            (from photo fission on nuclide `n`).  Used when photon yields are not available but neutron are.
            Values below zero are filtered.
            Args:
                n: Non-compound nucleus
                photon_erg: MeV
                __invert__: Instead solve for photon erg from neutron erg

            Returns: neutron energy

            """
        z, a = n.Z, n.A
        if hasattr(photon_erg, '__iter__'):
            photon_erg = np.array(photon_erg)
        if not __invert__:
            out = photon_erg - neutron_mass - Nuclide.get_mass_in_mev_per_c2(z=z, a=a - 1) + \
                  Nuclide.get_mass_in_mev_per_c2(z=z, a=a)
        else:
            neutron_erg = photon_erg
            out = neutron_erg + neutron_mass + Nuclide.get_mass_in_mev_per_c2(z=z, a=a - 1) - \
                  Nuclide.get_mass_in_mev_per_c2(z=z, a=a)
        out = np.array(list(filter(lambda x: x > 0, out)), dtype=float)
        return out

    @staticmethod
    def neutron2gamma(nuclide: Nuclide, neutron_erg):
        return GammaFissionErgConverter.gamma2neutron(nuclide, neutron_erg, True)


class Nuclide:
    """
    A nuclide object that can be used to access cross sections, decay children/parent nuclides, decay modes,
    and much more.

    Available data:
        Nuclide half-life
        Gamma emmissions (energy, intensity, uncertainties)
        Decay channels and the resulting child nuclides (branching ratios)
        Proton activation cross-sections (PADF)
        Neutron activation cross-sections (ENDF)
        Neutron, photon, proton, and SF fission yeilds (use caution with photon and proton data)
        Neutron, photon, and proton fission cross-sections


    Examples:
        Nuclides are not creating using __init__, as this is used for internally for pickling. Use class methods
        instead, e.g.:
            >>> Nuclide.from_symbol('Xe139')     # Xe-139
            <Nuclide: Xe139; t_1/2 = 39.68+/-0.14>
            >>> Nuclide.from_symbol('In120_m1')  # first excited state of In-120
            <Nuclide: In120_m1; t_1/2 = 46.2+/-0.8>
            >>> Nuclide.from_symbol('In120_m2')  # second excited state of In-120
            <Nuclide: In120_m2; t_1/2 = 47.3+/-0.5>

        or:
            >>> Nuclide.from_Z_A_M(54, 139)
            <Nuclide: Xe139; t_1/2 = 39.68+/-0.14>
            >>> Nuclide.from_Z_A_M(49, 120, 1) # first excited state of In-120
            <Nuclide: In120_m1; t_1/2 = 46.2+/-0.8>
            >>> Nuclide.from_Z_A_M(49, 120, 2) # second excited state of In-120
            <Nuclide: In120_m2; t_1/2 = 47.3+/-0.5>

        Fission yields:
            >>>Nuclide.from_symbol('U238').independent_sf_fission_yield().yields['Zn77']
            1.1e-05+/-1.1e-05

        (todo)


    Parameters
        ev_or_filename : str of openmc.data.endf.Evaluation
            ENDF fission product yield evaluation to read from. If given as a
            string, it is assumed to be the filename for the ENDF file.

    Attributes
        half life : UFloat
        spin:
        mean_energies: Mean gamma decay energy
        is_stable:
        decay_gamma_lines: List of GammaLine object sorted from highest to least intensity.
        decay_modes: Modes of decay. As of now, only gamma decays can be investigated with this tool (todo)

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

    def plot_decay_gamma_spectrum(self, label_first_n_lines: int = 5, min_intensity: float = 0.05, ax=None,
                                  log_scale: bool = False, label: Union[str, None] = None):
        """
        Plots the lines of a nuclides gamma spectrum.
        Args:
            label_first_n_lines: Starting from the most intense, label the next n lines.
            min_intensity: intensity threshold to be plotted
            ax: matplotlib ax
            log_scale:
            label: legend label for plot

        Returns: The axis object.

        """
        if len(self.decay_gamma_lines) == 0:
            warn('\nNo gamma lines for {}. No plot.'.format(self.name))
            return ax

        assert isinstance(label_first_n_lines, int)
        if ax is None:
            _, ax = plt.subplots()
        else:
            if ax is plt:
                ax = ax.gca()

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
        if len(y) == 0:
            warn('\nNo gamma lines for {} above intensity threshold of `{}`. No plot.'.format(self.name, min_intensity))
            return ax

        label_first_n_lines += 1
        label_first_n_lines = min(len(x), label_first_n_lines)

        y_max = max(y)
        _x_ = np.array(x[:label_first_n_lines])
        _y_ = np.array(y[:label_first_n_lines])
        sorter = np.argsort(_x_)
        for index, xy in enumerate(zip(_x_[sorter], _y_[sorter])):
            xy = np.array(xy)
            erg = xy[0]

            text_xy = (xy[0], y_max*1.1)
            ax.annotate('{:.2f}KeV'.format(erg), xy=xy, xytext=text_xy,
                        arrowprops=dict(width=0.1, headwidth=4, facecolor='black', shrink=0.03), rotation=90)

        ax.set_title('Gamma spectrum of {}, half-life = {}'.format(self.name, self.human_friendly_half_life(False)))
        ax.errorbar(x, y, yerr=y_err, xerr=x_err, label=label, marker='p', ls='none')
        if len(x) == 1:
            ax.set_xlim(x[0] - 10, x[0] + 10)
        ax.set_ylim(0, max(y+y_err)*1.5)

        return ax

    @property
    def Z(self) -> int:
        """Returns: Proton number"""
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
        """
        Returns: Mass number

        """
        if not hasattr(self, '__Z_A_iso_state__'):
            self.__Z_A_iso_state__ = get_z_a_m_from_name(self.name)
        return self.__Z_A_iso_state__['A']

    def human_friendly_half_life(self, include_errors: bool=True) -> str:
        """
        Gives the half life in units of seconds, hours, days, months, etc.
        Args:
            include_errors:  Whether to include uncertainties

        Returns:

        """
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
    def gamma_induced_fiss_xs(self) -> CrossSection1D:
        """
        Get the photon induced fission cross section for this nuclide.
        Raise error if no data available.
        Returns:

        """
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
                return n.mass_in_mev_per_c2
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
    def atomic_symbol(self) -> Union[str, None]:
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
    def from_symbol(cls, symbol: str):
        assert isinstance(symbol, str), '`symbol` argument must be a string.'
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
        return instance

    def __repr__(self):
        out = "<Nuclide: {}; t_1/2 = {}>".format(self.name, self.half_life)
        if self.__decay_mode_for_print__ is not None:
            out += self.__decay_mode_for_print__.__repr__()
        return out

    @property
    def decay_rate(self):
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

    def __get_sf_yield__(self, independent_bool) -> Dict[str, UFloat]:
        sub_dir = ('independent' if independent_bool else 'cumulative')
        f_path = SF_YIELD_PICKLE_DIR/sub_dir/'{}.marshal'.format(self.name)
        assert f_path.exists(), 'No SF fission yield data for {}'.format(self.name)
        with open(f_path, 'rb') as f:
            yield_data: yield_data_type = marshal.load(f)
        result = {}
        for n_name, data in yield_data[0.].items():
            result[n_name]: UFloat = ufloat(data['yield'], data['yield_err'])
        return result

    def independent_gamma_fission_yield(self, ergs=None, data_source=None) -> FissionYields:
        y = FissionYields(nuclide=self, inducing_particle='gamma', eval_ergs=ergs, data_source=data_source,
                          independent_bool=True)
        return y

    def cumulative_gamma_fission_yield(self, ergs=None, data_source=None) -> FissionYields:
        y = FissionYields(nuclide=self, inducing_particle='gamma', eval_ergs=ergs, data_source=data_source,
                          independent_bool=False)
        return y

    def independent_proton_fission_yield(self, ergs=None, data_source=None) -> FissionYields:
        y = FissionYields(nuclide=self, inducing_particle='proton', eval_ergs=ergs, data_source=data_source,
                          independent_bool=True)
        return y

    def cumulative_proton_fission_yield(self, ergs=None, data_source=None) -> FissionYields:
        y = FissionYields(nuclide=self, inducing_particle='proton', eval_ergs=ergs, data_source=data_source,
                          independent_bool=False)
        return y

    def independent_neutron_fission_yield(self, ergs=None, data_source=None) -> FissionYields:
        y = FissionYields(nuclide=self, inducing_particle='neutron', eval_ergs=ergs, data_source=data_source,
                          independent_bool=True)
        return y

    def cumulative_neutron_fission_yield(self, ergs=None, data_source=None) -> FissionYields:
        y = FissionYields(nuclide=self, inducing_particle='neutron', eval_ergs=ergs, data_source=data_source,
                          independent_bool=False)
        return y

    def independent_sf_fission_yield(self, data_source=None) -> FissionYields:
        y = FissionYields(nuclide=self, inducing_particle=None, data_source=data_source,
                          independent_bool=True)
        return y

    def cumulative_sf_fission_yield(self, data_source=None) -> FissionYields:
        y = FissionYields(nuclide=self, inducing_particle=None,  data_source=data_source,
                          independent_bool=False)
        return y

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
        """
        Get all product nuclides (and cross-sections, ect.) from a  reaction specified by the path to the nuclide's
        pickle file for the given reaction.
        Args:
            data_path:
            incident_particle: eg 'proton', 'photon', 'neutron'
            a_z_hl_cut:
            is_stable_only:

        Returns:

        """
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

    def is_effectively_stable(self, threshold_in_years: int=100) -> bool:
        """
        Is this nuclide effectively stable?
        Args:
            threshold_in_years: If the half life is greater than this, return True

        Returns:

        """
        return self.half_life.n >= (365*24*60**2)*threshold_in_years

    @ classmethod
    def get_all_nuclides(cls, a_z_hl_cut: str = '', is_stable_only=False) -> List[Nuclide]:
        """
        Returns a list of all nuclide instances subject to a criteria specified by `a_z_hl_cut`.
        Args:
            a_z_hl_cut: Criteria (python code) to be evaluated, where
                z=atomic number
                a=mass number,
                hl=half life in seconds.
                Defaults to all known nuclides.
            is_stable_only:  Only include stable nuclides.


        Returns: List of all nuclide instances meeting criteria

        """
        assert isinstance(a_z_hl_cut, str), 'All cuts must be a string instance.'
        nuclides = []
        with open(DECAY_PICKLE_DIR/'quick_nuclide_lookup.pickle', 'rb') as f:
            nuclides_dict = pickle.load(f)
        for (a, z, hl), nuclide_name in nuclides_dict.items():
            if __nuclide_cut__(a_z_hl_cut, a, z, hl, is_stable_only):
                nuclides.append(Nuclide.from_symbol(nuclide_name))

        return nuclides


class InducedDaughter(Nuclide):
    """
    A subclass of Nuclide representing a nucleus that was the result of an induced reaction.
    This adds some cross section attributes
    Attributes:
        xs: CrossSection1D instance
        parent: Parent nuclide
        inducing_particle:
    """
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
    """
        A subclass of Nuclide representing a nucleus that is the parent in an induced reaction.
        This adds some cross section attributes
        Attributes:
            xs: CrossSection1D instance
            daughter: Ddaughter nuclide
            inducing_particle:
        """
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


class ActivationReactionContainer:
    """
    Used for storing particle-induced data in pickle file. Imported in endf_to_pickle.py, and used in this module
    for unpickling
    """
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
    # s = Nuclide.from_symbol('U238')
    # print(__get_fiss_yield_path__('U238',"photon", 'GEF', False))
    FissionYields(Nuclide.from_symbol('Ac223'), inducing_particle='proton', data_source=None)


