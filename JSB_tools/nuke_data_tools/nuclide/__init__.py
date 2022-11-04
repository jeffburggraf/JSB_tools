from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from JSB_tools import mpl_hist
from pathlib import Path
import re
from logging import warning as warn
from typing import Dict, List, Tuple, Union, Callable
from uncertainties import UFloat, ufloat
from JSB_tools.nuke_data_tools.nuclide.data_directories import DECAY_PICKLE_DIR
import JSB_tools.nuke_data_tools.nuclide.cross_section as cross_section
import pickle
from datetime import datetime, timedelta
from JSB_tools.nuke_data_tools.nudel import LevelScheme
try:
    from openmc.data.endf import Evaluation
    from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
    from openmc.data import Reaction, Decay, Product
    from openmc.data.data import atomic_mass, atomic_weight, AVOGADRO, _ATOMIC_MASS
except ModuleNotFoundError:
    warn("OpenMC not installed! Some functionality is limited. ")

pwd = Path(__file__).parent

with open(pwd/'elements.pickle', 'rb') as f:
    _element_data = pickle.load(f)

constants = {'neutron_mass': 939.56542052,
             'proton_mass': 938.272088,
             'u_to_kg':  1.6605390666E-27,  # atomic mass units to kg
             'J_to_eV': 1.0/1.602176634E-19,  # Joules to eV
             'c':  299792458}


NATURAL_ABUNDANCE: dict = None  # Has form {SS1: {A1: abun1, A2: abum2, ...}, SS2: {...} }
ALL_ISOTOPES: dict = None  #


def all_isotopes():
    """
    Has form {SSAAA1: {A1, A2, A3}, SSAAA2: {...}, ...} represents all isotopes for a given Z which have data.
    Returns:

    """
    global  ALL_ISOTOPES
    if ALL_ISOTOPES is None:
        with open(pwd/"all_isotopes.pickle", 'rb') as f:
            ALL_ISOTOPES = pickle.load(f)
    return ALL_ISOTOPES


def get_abundance():
    """
    Return dict like
        {SS1: {A1: abun1, A2: abum2, ...},
        SS2: {...} }

    e.g.:
        {Be {9: 1.0},
         B {11: 0.8018, 10: 0.1982},
         C {12: 0.988922, 13: 0.011078},
         N {14: 0.996337, 15: 0.003663}}
    Returns:

    """
    global NATURAL_ABUNDANCE
    if NATURAL_ABUNDANCE is None:
        with open(pwd/'abundances.pickle', 'rb') as f:
            NATURAL_ABUNDANCE = pickle.load(f)
    return NATURAL_ABUNDANCE


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'GammaLine':
            return GammaLine
        elif name == 'DecayMode':
            return DecayMode
        elif name == 'Nuclide':
            return Nuclide
        elif name == '_DiscreteSpectrum':
            return _DiscreteSpectrum
        # else:
        #     assert False

        return super().find_class(module, name)


class Element:
    def __init__(self, Z):
        data = _element_data[Z]
        self.atomic_number = data['AtomicNumber']
        self.element_name = data['Element']
        self._atomic_mass_in_u = data['AtomicMass']
        self.element_period = data['Period']
        self.element_group = data['Group']
        self.element_phase = data['Phase']
        self.metalQ = data['Metal'] == 'yes'
        self.nonmetalQ = data['Nonmetal'] == 'yes'
        self.metalloidQ = data['Metalloid'] == 'yes'
        self.element_type = data['Type']
        self.atomic_radius = data['AtomicRadius']
        self.electronegativity = data['Electronegativity']
        self.firstIonization = data['FirstIonization']
        self.density = data['Density']  # in g/cm3
        self.melting_point = data['MeltingPoint']
        self.boiling_point = data['BoilingPoint']
        self.specific_heat = data['SpecificHeat']
        self.n_valence_electrons = data['NumberofValence']

    def get_atomic_mass(self, unit='g'):
        u_to_kg = constants['u_to_kg']
        if unit == 'u':
            s = 1
        elif unit == 'g':
            s = u_to_kg * 1E3
        elif unit == 'kg':
            s = u_to_kg
        else:
            raise ValueError(f"Invalid unit, {unit}")

        return self._atomic_mass_in_u * s


def get_symbol_etc(symbol):
    """
    Get (correct) symbol fro argument and return nucleus Z, A, etc.
    Args:
        symbol:

    Returns: symbol, Z, A, M

    """
    if '-' in symbol:
        symbol = symbol.replace('-', '')
        if symbol.endswith('m'):
            symbol = symbol[:-1] + '_m1'

    if symbol[0] == 'n' and symbol.lower() in ['n', 'neutron']:
        symbol = 'N1'

    _m = Nuclide.NUCLIDE_NAME_MATCH.match(symbol)

    if not _m:
        raise ValueError(
            "\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
            "e.g. Cl38_m1, n1, Ar40".format(symbol))

    if _m.groups()[2] == '0':  # ground state specification, "_m0", is redundant.
        symbol = _m.groups()[0] + _m.groups()[1]

    try:
        Z = ATOMIC_NUMBER[_m.groups()[0]]
    except KeyError:
        if symbol == "Nn1":
            Z = 0
        else:
            Z = None

    A = int(_m.groups()[1])
    isometric_state = _m.groups()[2]

    if isometric_state is None:
        isometric_state = 0
    isometric_state = isometric_state

    return symbol, Z, A, isometric_state


class Nuclide(Element):
    """
    A nuclide object that can be used to access cross sections, decay children/parent nuclides, decay modes,
    and much more.

    Available data:
        Nuclide half-life
        Gamma emissions (energy, intensity, uncertainties)
        Decay channels and the resulting child nuclides (+ branching ratios)
        Proton activation cross-sections (PADF)
        Neutron activation cross-sections (ENDF)
        Neutron, photon, proton, and SF fissionXS yeilds (use caution with photon and proton data)
        Neutron, photon, and proton fissionXS cross-sections

        """
    NUCLIDE_NAME_MATCH = re.compile(
        "^(?P<s>[A-z]{1,3})(?P<A>[0-9]{1,3})(?:_?m(?P<iso>[0-9]+))?$")  # Nuclide name in GND naming convention

    all_instances = {}  # keeping track of all Nuclides to save time on loading
    default_values = {'excitation_energy': 0,
                      'half_life': None,
                      'spin': 0,
                      'is_stable': None,
                      'decay_radiation_types': list,
                      '__decay_daughters_str__': list,
                      '__decay_gamma_lines': list,
                      '__decay_betaplus_lines': list,
                      'decay_modes': dict,
                      '__decay_parents_str__': list,
                      '__decay_mode_for_print__': None,
                      'self.mean_energies': None
                      }

    @staticmethod
    def __get_default_atrib(name):
        val = Nuclide.default_values[name]
        if isinstance(val, Callable):
            val = val()

        return val

    def __new__(cls, symbol, *args, **kwargs):
        """

        Args:
            symbol:
            *args:
            **kwargs: if _default in kwargs, run __init__, setting all values to default.
        """

        symbol,  z, a, m = get_symbol_etc(symbol)

        pickle_file = DECAY_PICKLE_DIR / (symbol + '.pickle')

        if symbol not in Nuclide.all_instances:
            if '_default' in kwargs or not pickle_file.exists():
                self = super().__new__(cls)
                self.__Z_A_iso_state__ = z, a, m
                self.__init__(symbol)

            else:
                with open(pickle_file, "rb") as pickle_file:
                    self = CustomUnpickler(pickle_file).load()
                Nuclide.all_instances[symbol] = self

        else:
            self = Nuclide.all_instances[symbol]

        return self

    def __init__(self, symbol, **kwargs):
        """


        Notes:
            __init__ is not called unless the nuclide is to be set to default values.
            The heavy lifting is done in __new__

        Args:
            symbol:
            **kwargs: Used by __new__
        """
        self.name = symbol

        super().__init__(self.Z)

        self.excitation_energy: float = 0
        self.half_life: Union[None, UFloat] = ufloat(np.nan, np.nan)
        self.spin: int = None

        self.is_stable: Union[None, bool] = None
        self.decay_radiation_types: list = []

        # self.__decay_daughters_str__: List[str] = self.__get_default_atrib('__decay_daughters_str__')
        self.__decay_daughters_str__: List[str] = []

        self.__decay_gamma_lines: List[GammaLine] = []
        # self.__decay_gamma_lines: List[GammaLine] = self.__get_default_atrib('__decay_gamma_lines')
        # self.__decay_betaplus_lines: List[BetaPlusLine] = self.__get_default_atrib('__decay_betaplus_lines')
        self.__decay_betaplus_lines: List[BetaPlusLine] = []

        # self.decay_modes: Dict[Tuple[str], List[DecayMode]] = self.__get_default_atrib('decay_modes')
        self.decay_modes: Dict[Tuple[str], List[DecayMode]] = {}

        self.__decay_parents_str__: List[str] = []
        # self.__decay_parents_str__: List[str] = self.__get_default_atrib('__decay_parents_str__')

        self.__decay_mode_for_print__ = None
        # self.__decay_mode_for_print__ = self.__get_default_atrib('__decay_mode_for_print__')
        self.is_valid = False

        # self.decay_radiation_types

    @staticmethod
    def get_z_a_m_from_name(name: str) -> Dict[str, int]:
        """
        From a nuclide name, return the atomic number (Z), mass number (A) and isomeric state (ie excited state, or M)
        Args:
            name: nuclide name

        Returns:

        """
        _m = Nuclide.NUCLIDE_NAME_MATCH.match(name)

        assert _m, "\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
                   "e.g. Cl38_m1, n1, Ar40".format(name)

        assert False


    def adopted_levels(self):
        """
        Levels
        Returns:

        """
        return LevelScheme(f"{self.atomic_symbol}{self.A}")

    @property
    def decay_gamma_lines(self) -> List[GammaLine]:
        if not self.__decay_gamma_lines:
            spec = _DiscreteSpectrum.__unpickle__(self, 'gamma')
            if spec.__is_empty__:
                return []
            self.__decay_gamma_lines = list(sorted([GammaLine(self, ds) for ds in spec.__discrete_entries__],
                                                   key=lambda x: -x.intensity))
            return self.__decay_gamma_lines
        else:
            return self.__decay_gamma_lines

    def get_gamma_nearest(self, energy: Union[float, List[float]]) -> GammaLine:
        if isinstance(energy, UFloat):
            energy = energy.n
        if len(self.decay_gamma_lines) == 0:
            raise IndexError(f"No gamma lines from {self.name}")
        ergs = np.array([g.erg.n for g in self.decay_gamma_lines])
        if isinstance(energy, (float, int)):
            out = self.decay_gamma_lines[np.argmin(np.abs(ergs-energy))]
        else:
            if not hasattr(energy, '__iter__'):
                raise ValueError(f'Invalid value for arg `energy`: {energy}')
            out = [self.decay_gamma_lines[np.argmin(np.abs(ergs - erg))] for erg in energy]
        return out

    @property
    def decay_betaplus_lines(self) -> List[BetaPlusLine]:
        if not self.__decay_betaplus_lines:
            spec = _DiscreteSpectrum.__unpickle__(self, 'ec/beta+')
            if spec.__is_empty__:
                return []
            self.__decay_betaplus_lines = list(sorted([BetaPlusLine(self, ds) for ds in spec.__discrete_entries__],
                                               key=lambda x: -x.intensity))
            return self.__decay_betaplus_lines
        else:
            return self.__decay_betaplus_lines

    @property
    def positron_intensity(self):
        out = sum(b.positron_intensity for b in self.decay_betaplus_lines)
        if not isinstance(out, UFloat):
            out = ufloat(out, 0)
        return out

    def decay_chain_gamma_lines(self, __branching__ratio__=1) -> List[GammaLine]:
        raise NotImplementedError("")

    @property
    def decay_branching_ratios(self) -> Dict[Tuple[str], UFloat]:
        return {k: sum(m.branching_ratio for m in v) for k, v in self.decay_modes.items()}

    def phits_kfcode(self) -> int:
        """
        Particle kf=code for the PHITS Montecarlo package.
        Returns:

        """
        try:
            {(1, 1): 2212, (0, 1): 2112}[(self.Z, self.A)]
        except KeyError:
            return int(self.Z*1E6 + self.A)

    def get_n_decays(self, ref_activity: float, activity_ref_date: datetime, tot_acquisition_time: float,
                     acquisition_ti: datetime = datetime.now(), activity_unit='uCi') -> UFloat:
        """
        Calculate the number of decays of o a radioactive source during specified time.
        Args:
            ref_activity: Activity in micro-curies.
            activity_ref_date: Date of source calibration
            tot_acquisition_time: float, number of seconds the acquisition was performed.
            acquisition_ti: Optional. If provided, This is the date/time of the start of acquisition. If not provided,
                today is used.
            activity_unit: Ci, uCi, Bq, kBq

        Returns: Absolute of number of decays during specified time.

        """
        assert isinstance(tot_acquisition_time, (float, int)), tot_acquisition_time

        tot_acquisition_time = timedelta(seconds=tot_acquisition_time).total_seconds()

        corr_seconds = (acquisition_ti - activity_ref_date).total_seconds()  # seconds since ref date.
        try:
            unit_factor = {'ci': 3.7E10, 'uci': 3.7E10*1E-6, 'bq': 1, "kbq": 1E3}[activity_unit.lower()]
        except KeyError:
            assert False, f'Bad activity unit, "{activity_unit}". Valid units: Ci, uCi, Bq, uBq'

        ref_num_nuclides = ref_activity*unit_factor/self.decay_rate  # # of nuclides when ref calibration was performed

        corrected_num_nuclides = ref_num_nuclides*0.5**(corr_seconds/self.half_life)
        n_decays = corrected_num_nuclides*(1-0.5**(tot_acquisition_time/self.half_life))
        return n_decays

    def plot_decay_gamma_spectrum(self, label_first_n_lines: int = 5, min_intensity: float = 0.02, ax=None,
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

        ax.set_title('Gamma spectrum of {}, half-life = {}, for Ig (%) >= {:.1f}'.format(self.name,
                                                                                         self.pretty_half_life(False),
                                                                                         min_intensity * 100))
        ax.errorbar(x, y, yerr=y_err, xerr=x_err, label=label, marker='p', ls='none')
        if len(x) == 1:
            ax.set_xlim(x[0] - 10, x[0] + 10)
        ax.set_ylim(0, max(y+y_err)*1.5)

        return ax

    @staticmethod
    def get_s_z_a_m_from_string(nuclide_name) -> Tuple[str, int, int, int]:
        """
        Get atomic_symbol, Z, A, and isomeric state (m) from nuclide name using GND convention.
        e.g.
            s_z_a_m_from_string('U235_m1') -> ("U", 92, 135, 1)
        Args:
            nuclide_name: nuclide name

        Returns:
            (atomic_symbol, Z, A, isomeric_state)
        """
        m = Nuclide.NUCLIDE_NAME_MATCH.match(nuclide_name)
        if not m:
            raise ValueError(f'Invalid nuclide name, "{nuclide_name}"')
        iso = int(m.group('iso')) if m.group('iso') is not None else 0

        return m.group('s'), ATOMIC_NUMBER[m.group('s')], int(m.group('A')), iso

    @property
    def Z(self) -> int:
        """Returns: Proton number"""
        return self.__Z_A_iso_state__[0]

    @property
    def N(self) -> int:
        return self.A-self.Z

    @property
    def isometric_state(self) -> int:
        """Meta stable excited state, starting with 0 as the ground state."""
        return self.__Z_A_iso_state__[2]

    @property
    def A(self) -> int:
        """
        Returns: Mass number

        """
        return self.__Z_A_iso_state__[1]

    def pretty_half_life(self, include_errors: bool = True, abrev_units=True) -> str:
        """
        Gives the half life in units of seconds, hours, days, months, etc.
        Args:
            include_errors:  Whether to include uncertainties
            abrev_units: seconds -> s, minutes -> m, etc.

        Returns:

        """
        return self.half_life
        # return pretty_half_life(self.half_life, include_errors, abrev_units)

    @property
    def proton_induced_fiss_xs(self) -> CrossSection1D:
        global PROTON_INDUCED_FISSION_XS1D
        simple_nuclide_name = self.atomic_symbol + str(self.A)
        if simple_nuclide_name not in PROTON_INDUCED_FISSION_XS1D:
            try:
                with open(PROTON_PICKLE_DIR/'fissionXS'/'{}.pickle'.format(simple_nuclide_name), 'rb') as f:
                    PROTON_INDUCED_FISSION_XS1D[simple_nuclide_name] = CustomUnpickler(f).load()
            except FileNotFoundError:
                assert False, 'No proton induced fissionXS data for {0}. Download it and integrate it if it is ' \
                              'available. The conversion to pickle is done in `endf_to_pickle.py`. See "{1}" for' \
                              ' instructions.'.format(self, pwd/'endf_files'/'FissionXS'/'readme.md')

        return PROTON_INDUCED_FISSION_XS1D[simple_nuclide_name]

    @property
    def gamma_induced_fiss_xs(self) -> CrossSection1D:
        """
        Get the photon induced fissionXS cross section for this nuclide.
        Raise error if no data available.
        Returns:

        """
        simple_nuclide_name = self.atomic_symbol + str(self.A)
        if simple_nuclide_name not in PHOTON_INDUCED_FISSION_XS1D:
            try:
                with open(GAMMA_PICKLE_DIR / 'fissionXS' / '{}.pickle'.format(simple_nuclide_name), 'rb') as f:
                    PHOTON_INDUCED_FISSION_XS1D[simple_nuclide_name] = CustomUnpickler(f).load()
            except FileNotFoundError:
                assert False, 'No photon induced fissionXS data for {0}.'

        return PHOTON_INDUCED_FISSION_XS1D[simple_nuclide_name]

    @property
    def neutron_induced_fiss_xs(self) -> CrossSection1D:
        """
        Get the neutron induced fissionXS cross section for this nuclide.
        Raise error if no data available.
        Returns:

        """
        simple_nuclide_name = self.atomic_symbol + str(self.A)
        if simple_nuclide_name not in PHOTON_INDUCED_FISSION_XS1D:
            try:
                with open(NEUTRON_PICKLE_DIR / 'fissionXS' / '{}.pickle'.format(simple_nuclide_name), 'rb') as f:
                    NEUTRON_INDUCED_FISSION_XS1D[simple_nuclide_name] = CustomUnpickler(f).load()
            except FileNotFoundError:
                assert False, 'No photon induced fissionXS data for {0}.'

        return NEUTRON_INDUCED_FISSION_XS1D[simple_nuclide_name]

    def rest_energy(self, units='MeV'):  # in J or MeV
        units = units.lower()
        ev = constants['J_to_eV']
        unit_dict = {'j': 1, 'mev': ev*1E-6, 'ev': ev}
        assert units in unit_dict.keys(), 'Invalid units, "{}".\nUse one of the following: {}'\
            .format(units, unit_dict.keys())
        c = constants['c']
        j = self.get_atomic_mass('kg') * c**2
        return j*unit_dict[units]

    @property
    def mass_in_mev_per_c2(self) -> float:
        return self.rest_energy('MeV')

    def neutron_separation_energy(self, n_neutrons=1):
        """
        Min. energy required to remove n_neutrons from nucleus in MeV
        Args:
            n_neutrons: Number of neutrons to obe removed.

        Returns:

        """
        z, a, m = Nuclide.get_z_a_m_from_name(self.name).values()
        a -= n_neutrons
        neutron_mass = constants['neutron_mass']
        return (Nuclide.from_Z_A_M(z, a, m).mass_in_mev_per_c2 + n_neutrons*neutron_mass) - self.mass_in_mev_per_c2

    def proton_separation_energy(self, n_protons=1):
        """
        Min. energy required to remove n_protons from the nucleus (in MeV)
        Args:
            n_protons: Number of protons to obe removed.

        Returns:

        """
        z, a, m = Nuclide.get_z_a_m_from_name(self.name).values()
        z -= n_protons
        a -= n_protons
        proton_mass = constants['proton_mass']
        return (Nuclide.from_Z_A_M(z, a, m).mass_in_mev_per_c2 + n_protons * proton_mass) - self.mass_in_mev_per_c2

    def alpha_separation_energy(self):
        """
        Min. energy required to remove He-4 from the nucleus (in MeV)

        Returns:

        """
        z, a, m = Nuclide.get_z_a_m_from_name(self.name).values()
        z -= 2
        a -= 2

        return (Nuclide.from_Z_A_M(z, a, m).mass_in_mev_per_c2 + Nuclide.from_symbol('He-4').mass_in_mev_per_c2) \
                - self.mass_in_mev_per_c2

    # @property
    # def atomic_mass(self) -> float:
    #     try:
    #         return atomic_mass(self.name)
    #     except KeyError:
    #         warn('Atomic mass for {} not found'.format(self))
    #         return None

    @property
    def grams_per_mole(self) -> float:
        try:
            return atomic_weight(self.name)
        except KeyError:
            warn('Atomic weight for {} not found'.format(self))
            return None

    @staticmethod
    def isotopic_abundance(nuclide_name) -> float:
        m = re.match("(([A-Z][a-z]*)([0-9]+).*?)", nuclide_name)
        if not m:
            raise ValueError(f"Invalid nuclide name, {nuclide_name}")
        _, symbol, A = m.groups()

        try:
            return get_abundance()[symbol][int(A)]
        except KeyError:
            return 0

    @staticmethod
    def rel_isotopic_abundance(nuclide_name) -> float:
        tot = 0
        for name in Nuclide.get_all_isotopes(nuclide_name, True):
            try:
                tot += NATURAL_ABUNDANCE[name]
            except KeyError:
                continue
        try:
            return Nuclide.isotopic_abundance(nuclide_name)/tot
        except ZeroDivisionError:
            return 0

    @staticmethod
    def get_all_isotopes(atomic_symbol: str, non_zero_abundance=False) -> List[str]:
        """
        Returns list of strings of all isotopes with atomic number according to `atomic_symbol` argument.
        Args:
            atomic_symbol:
            non_zero_abundance: If True, only return nuclides that occur naturally. Otherwise, return any for which
                                atomic data exists.
        Returns:

        """
        m = re.match('^([A-z]{0,3})(?:[0-9]+[_m]*([0-9]+)?)', atomic_symbol)
        assert m, f"Invalid argument, '{atomic_symbol}'"
        s = m.groups()[0]
        s = f"{s[0].upper()}{s[1:]}"
        outs = []
        for a in all_isotopes()[s]:
            other_s = f"{atomic_symbol}{a}"
            if non_zero_abundance:
                if get_abundance()[s][a] > 0:
                    outs.append(other_s)
            else:
                outs.append(other_s)

        return outs

    def all_isotopes(self, non_zero_abundance=False):
        return Nuclide.get_all_isotopes(self.atomic_symbol, non_zero_abundance)

    @staticmethod
    def natural_abundance(symbol) -> float:
        try:
            return NATURAL_ABUNDANCE[symbol]
        except KeyError:
            return 0.

    @staticmethod
    def max_abundance_nucleus(atomic_symbol) -> Tuple[float, str]:
        abundance = None
        out = None

        for s in Nuclide.get_all_isotopes(atomic_symbol):
            try:
                a = NATURAL_ABUNDANCE[s]
            except KeyError:
                continue

            if abundance is None or a > abundance:
                abundance = a
                out = s

        return abundance, out

    @property
    def atomic_symbol(self) -> Union[str, None]:
        """Atomic symbol according to the proton number of this isotope"""
        _m = re.match('([A-Za-z]{1,2}).+', self.name)
        if _m:
            return _m.groups()[0]
        else:
            return None

    @property
    def latex_name(self):
        if self.isometric_state != 0:
            try:
                m = 'mnop'[self.isometric_state - 1]
            except IndexError:
                m = f'_l{self.isometric_state}'
        else:
            m = ''
        return f"$^{{{self.A}{m}}}${self.atomic_symbol}"

    @property
    def mcnp_zaid(self):
        return f'{self.Z}{self.A:0>3}'

    def add_proton(self, n=1) -> Nuclide:
        return self.from_Z_A_M(self.Z+n, self.A+n, self.isometric_state)

    def remove_proton(self, n=1) -> Nuclide:
        return self.from_Z_A_M(self.Z-n, self.A-n, self.isometric_state)

    def remove_neutron(self, n=1) -> Nuclide:
        return self.from_Z_A_M(self.Z, self.A-n, self.isometric_state)

    def add_neutron(self, n=1) -> Nuclide:
        return self.from_Z_A_M(self.Z, self.A+n, self.isometric_state)

    def remove_alpha(self):
        return self.from_Z_A_M(self.Z-2, self.A-4, self.isometric_state)

    def add_alpha(self):
        return self.from_Z_A_M(self.Z+2, self.A+4, self.isometric_state)

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

    # @classmethod
    # def from_symbol(cls, symbol: str, discard_meta_state=False):
    #     """
    #
    #     Args:
    #         symbol: e.g. 'Xe139', 'Ta180_m1"
    #         discard_meta_state: If True, discard meta stable state.
    #
    #     Returns:
    #
    #     """
    #     if discard_meta_state:
    #         symbol = symbol.split('_')[0]
    #     assert isinstance(symbol, str), f'`symbol` argument must be a string, not {type(symbol)}'
    #
    #     if symbol in Nuclide.all_instances:  # check first thing for speed.
    #         return Nuclide.all_instances[symbol]
    #
    #     if '-' in symbol:
    #         symbol = symbol.replace('-', '')
    #         if symbol.endswith('m'):
    #             symbol = symbol[:-1] + '_m1'
    #
    #     if symbol.lower() in ['n', 'neutron']:
    #         symbol = 'N1'
    #     elif symbol.lower() == 'alpha':
    #         symbol = 'He4'
    #     elif symbol.lower() == 'proton':
    #         symbol = 'H1'
    #
    #     _m = Nuclide.NUCLIDE_NAME_MATCH.match(symbol)
    #     if not _m:
    #         raise ValueError("\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
    #                "e.g. Cl38_m1, n1, Ar40".format(symbol))
    #     # assert _m,
    #
    #     symbol = _m.group()
    #
    #     if _m.groups()[2] == '0':  # ground state specification, "_m0", is redundant.
    #         symbol = _m.groups()[0] + _m.groups()[1]
    #         _m = Nuclide.NUCLIDE_NAME_MATCH.match(symbol)
    #
    #     pickle_file = DECAY_PICKLE_DIR/(symbol + '.pickle')
    #
    #     if symbol not in Nuclide.all_instances:
    #         if not pickle_file.exists():
    #                 instance = Nuclide(symbol,  __internal__=True, half_life=ufloat(np.nan, np.nan))
    #                 instance.is_valid = False
    #         else:
    #             with open(pickle_file, "rb") as pickle_file:
    #                 instance = CustomUnpickler(pickle_file).load()
    #                 instance.is_valid = True
    #             Nuclide.all_instances[symbol] = instance
    #
    #     else:
    #         instance = NUCLIDE_INSTANCES[symbol]
    #         instance.is_valid = True
    #
    #     if instance.name == 'n1':
    #         instance.name = 'N1'
    #
    #     return instance

    def __repr__(self):
        try:
            hl = self.pretty_half_life()
        except ValueError:
            hl = self.half_life

        if self.isometric_state != 0:
            if self.excitation_energy == 0:
                ext_print = " (? keV)"
            else:
                ext_print = f" ({self.excitation_energy*1E-3:.6g} keV)"
        else:
            ext_print = ""

        out = f"<Nuclide: {self.name}{ext_print}; t_1/2 = {hl}>"
        if self.__decay_mode_for_print__ is not None:
            out += f" (from decay {self.__decay_mode_for_print__.__repr__()})"
        return out

    @property
    def decay_rate(self):
        return np.log(2)/self.half_life

    def get_decay_parents(self, return_branching_ratios=False) -> Union[List[Nuclide], List[Tuple[Nuclide, UFloat]]]:
        """
        Return list (or more, see `return_branching_ratios`) of nuclides which decay to self.
        Args:
            return_branching_ratios:
                If False, return list of parents.
                If True, return 2-tuple of parents and corresponding decay branching ratio, i.e.
                    as follows:
                            [(parent1, branching_ratio1), (parent2, branching_ratio2)]

        Returns:

        """
        if return_branching_ratios:
            out_dict = {}
            for n in self.__decay_parents_str__:
                par = self.from_symbol(n)
                out_dict[n] = ufloat(0, 0)
                for modes in par.decay_modes.values():
                    for mode in modes:
                        if mode.daughter_name == self.name:
                            out_dict[n] += mode.branching_ratio
            return [(Nuclide.from_symbol(k), v) for k, v in out_dict.items()]
        else:
            return list([self.from_symbol(name) for name in self.__decay_parents_str__])

    @property
    def decay_daughters(self):
        out = list([self.from_symbol(name) for name in self.__decay_daughters_str__])
        for nuclide in out:
            for decay_modes in self.decay_modes.values():
                for decay_mode in decay_modes:
                    if decay_mode.daughter_name == nuclide.name:
                        nuclide.__decay_mode_for_print__ = decay_mode

        return out

    def neutron_capture_xs(self, data_source=None) -> CrossSection1D:
        reses = self.get_incident_particle_daughters('neutron', data_source=data_source)
        s = Nuclide.from_Z_A_M(self.Z, self.A + 1).name
        try:
            return reses[s].xs
        except KeyError:
            raise FileNotFoundError(f"No neutron capture cross-section for {self.name}")

    def get_incident_particle_daughters(self, particle, data_source=None, a_z_hl_cut='', is_stable_only=False) \
            -> Dict[str, InducedDaughter]:
        f = getattr(self, f"get_incident_{particle}_daughters")
        return f(data_source=data_source, a_z_hl_cut=a_z_hl_cut, is_stable_only=is_stable_only)

    def get_incident_proton_parents(self, data_source=None, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedParent]:
        """
        See __get_daughters__
        :param data_source:
        :param a_z_hl_cut:
        :param is_stable_only:
        :return:
        """
        return self.__get_parents__('proton', a_z_hl_cut, is_stable_only, data_source)

    def get_incident_proton_daughters(self, data_source=None, a_z_hl_m_cut='', is_stable_only=False) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('proton', a_z_hl_m_cut, is_stable_only, data_source)

    def get_incident_gamma_daughters(self, data_source=None, a_z_hl_m_cut='', is_stable_only=False) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('gamma', a_z_hl_m_cut, is_stable_only, data_source)

    def get_incident_gamma_parents(self, data_source=None, a_z_hl_m_cut='', is_stable_only=False) -> Dict[str, InducedParent]:
        return self.__get_parents__('gamma', a_z_hl_m_cut, is_stable_only, data_source)

    def get_incident_neutron_daughters(self, data_source=None, a_z_hl_m_cut='', is_stable_only=False) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('neutron', a_z_hl_m_cut, is_stable_only, data_source)

    def get_incident_neutron_parents(self, data_source=None, a_z_hl_m_cut='', is_stable_only=False) -> Dict[str, InducedParent]:
        return self.__get_parents__('neutron', a_z_hl_m_cut, is_stable_only, data_source)

    def __get_daughters__(self, projectile, a_z_hl_m_cut='', is_stable_only=False,
                          data_source: Union[str, None] = None):
        """
        Get all product nuclides (and cross-sections, ect.) from a  reaction specified by the path to the nuclide's
        pickle file for the given reaction.
        Args:
            projectile: eg 'proton', 'photon', 'neutron'
            a_z_hl_m_cut: Filter reaction products. The string will be evaluated as python code.
            is_stable_only:
            data_source: None uses ONLY the default library.
                         'all' uses all of them, where higher priority libraries take precedence.
                         'endf' uses ENDF, 'talys' uses TALYS, etc.

        Returns:

        """
        reaction = ActivationReactionContainer.load(self.name, projectile, data_source)
        assert isinstance(reaction, ActivationReactionContainer), type(reaction)
        out: Dict[str, InducedDaughter] = {}

        for daughter_name, xs in reaction.product_nuclide_names_xss.items():
            daughter_nuclide = Nuclide.from_symbol(daughter_name)
            a, z, hl, m = daughter_nuclide.A, daughter_nuclide.Z, daughter_nuclide.half_life, \
                          daughter_nuclide.isometric_state
            if __nuclide_cut__(a_z_hl_m_cut, a, z, hl, is_stable_only, m):
                daughter = InducedDaughter(daughter_nuclide, self, projectile)
                daughter.xs = xs
                out[daughter_name] = daughter

        return out

    def __get_parents__(self, projectile, a_z_hl_cut='', is_stable_only=False,
                        data_source: Union[str, None] = None):
        daughter_reaction = ActivationReactionContainer.load(self.name, projectile, data_source)
        assert isinstance(daughter_reaction, ActivationReactionContainer)
        out: Dict[str, InducedParent] = {}

        parent_nuclides = [Nuclide.from_symbol(name) for name in daughter_reaction.parent_nuclide_names]

        for parent_nuclide in parent_nuclides:
            a, z, hl, m = parent_nuclide.A, parent_nuclide.Z, parent_nuclide.half_life, parent_nuclide.isometric_state
            if __nuclide_cut__(a_z_hl_cut, a, z, hl, is_stable_only, m):
                parent = InducedParent(self, parent_nuclide, inducing_particle=projectile)
                parent.xs = ActivationReactionContainer.fetch_xs(parent_nuclide.name, self.name, projectile)
                out[parent.name] = parent

        return out

    def inelastic_xs(self, projectile, data_source=None):
        warnings.warn("See U235 neutron inelastic vs fission xs. Doesn't make sense bc F is > inel. Something is wrong. ")
        return self.__get_misc_xs__('inelastic_xs', projectile, data_source)

    def elastic_xs(self, projectile, data_source=None):
        return self.__get_misc_xs__('elastic_xs', projectile, data_source)

    def total_xs(self, projectile, data_source=None):
        return self.__get_misc_xs__('total_xs', projectile, data_source)

    def __get_misc_xs__(self, attrib, projectile, data_source=None) -> CrossSection1D:
        """
        Loads ActivationReactionContainer and grabs cross-section attribute according to `attrib`.
        Args:
            attrib: inelastic_xs, total_xs, etc.
            projectile: proton, neutron, etc.
            data_source: endf, padf, etc. None will return first found.

        Returns:

        """
        assert projectile in ActivationReactionContainer.libraries, f'No data for projectile, "{projectile}"'
        out = None

        if data_source is None:
            data_sources = ActivationReactionContainer.libraries[projectile]
        else:
            data_sources = [data_source]

        for data_source in data_sources:
            try:
                reaction = ActivationReactionContainer.load(self.name, projectile, data_source=data_source)
            except FileNotFoundError:
                continue

            out = getattr(reaction, attrib)
            break

        return out

    def is_effectively_stable(self, threshold_in_years: int = 100) -> bool:
        """
        Is this nuclide effectively stable?
        Args:
            threshold_in_years: If the half life is greater than this, return True

        Returns:

        """
        return self.half_life.n >= (365*24*60**2)*threshold_in_years

    @staticmethod
    def get_all_nuclides(z_a_m_hl_cut: str = '', is_stable_only=False) -> List[Nuclide]:
        """
        Returns a list of all nuclide instances subject to a criteria specified by `a_z_hl_cut`.
        Args:
            z_a_m_hl_cut: Criteria (python code) to be evaluated, where the following variables may appear in the
             expression:
                z=atomic number
                a=mass number,
                m=isomeric level/excited state (0 for ground state)
                hl=half life in seconds.
                Defaults to all known nuclides.
            is_stable_only:  Only include stable nuclides.


        Returns: List of all nuclide instances meeting criteria

        """
        assert isinstance(z_a_m_hl_cut, str), 'All cuts must be a string instance.'
        nuclides = []
        with open(DECAY_PICKLE_DIR/'quick_nuclide_lookup.pickle', 'rb') as f:
            nuclides_dict = pickle.load(f)
        for (a, z, hl, m), nuclide_name in nuclides_dict.items():
            if __nuclide_cut__(z_a_m_hl_cut, a, z, hl, is_stable_only, m=m):
                nuclides.append(Nuclide.from_symbol(nuclide_name))

        yield from nuclides


class _DiscreteSpectrum:
    """
    Container for pickled spectra data
    """
    all_instances: Dict[Tuple[str, str], _DiscreteSpectrum] = {}

    def __init__(self, nuclide: Nuclide, spectra_data):
        self.__nuclide_name = nuclide.name
        self.radiation_type: str = spectra_data['type']
        self.mean_erg_per_decay = spectra_data['energy_average']
        self.__is_empty__ = False
        discrete_normalization = spectra_data['discrete_normalization']

        try:
            self.__discrete_entries__ = spectra_data["discrete"]
            for emission_data in self.__discrete_entries__:
                emission_data['intensity'] = discrete_normalization*emission_data['intensity']
                emission_data['energy'] = emission_data['energy']*1E-3
                emission_data['from_mode'] = tuple(emission_data['from_mode'])
        except KeyError as e:
            self.__discrete_entries__ = []

        try:
            self.__continuous_entries__ = []
        except KeyError:
            pass

    def __repr__(self):
        return f'{self.__nuclide_name} _DiscreteSpectrum; {self.radiation_type}...'

    @classmethod
    def __blank_entry(cls, nuclide: Nuclide, radiation_type) -> _DiscreteSpectrum:
        out = cls(nuclide, {'energy_average': ufloat(0, 0),
                             'type': radiation_type,
                             'discrete_normalization': 1})
        out.__is_empty__ = True
        return out

    @classmethod
    def __unpickle__(cls, nuclide: Nuclide, radiation_type: str) -> _DiscreteSpectrum:
        if (radiation_type, nuclide.name) not in _DiscreteSpectrum.all_instances:
            try:
                with open(cls.__pickle_path(radiation_type, nuclide.name), 'rb') as f:
                    out = CustomUnpickler(f).load()
            except FileNotFoundError:
                out = _DiscreteSpectrum.__blank_entry(nuclide, radiation_type)
            _DiscreteSpectrum.all_instances[(radiation_type, nuclide.name)] = out
        else:
            out = _DiscreteSpectrum.all_instances[(radiation_type, nuclide.name)]
        return out

    @staticmethod
    def __pickle_path(radiation_type: str, nuclide_name):
        radiation_type = radiation_type.replace('/', '_')
        # if radiation_type == 'ec/beta+':
        #     radiation_type = 'ec_beta_plus'
        return pwd/'data'/'nuclides'/f'{radiation_type}_spectra'/f'{nuclide_name}.pickle'

    def __pickle__(self):
        path = self.__pickle_path(self.radiation_type, self.__nuclide_name)
        if not path.parent.exists():
            path.parent.mkdir()
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class DecayMode:
    """
    Container for DecayMode. Effectively an openmc.DecayMode wrapper.
    Attributes:
        modes: Strings of successive decay modes (usually length one, but not always.)
        daughter_name: Name of daughter nucleus.
        parent_name: Name of parent nucleus.
        branching_ratio: Branching intensity of this mode.
        partial_decay_constant: Rate of decays through this mode.
        partial_half_life: Half life for decays through this mode.
    """
    def __init__(self, openmc_decay_mode, parent_half_life):
        self.q_value = openmc_decay_mode.energy*1E-6
        self.modes = tuple(openmc_decay_mode.modes)
        self.daughter_name = openmc_decay_mode.daughter
        self.parent_name = openmc_decay_mode.parent
        self.branching_ratio = openmc_decay_mode.branching_ratio
        parent_decay_rate = np.log(2)/parent_half_life
        self.partial_decay_constant = parent_decay_rate*self.branching_ratio
        if self.partial_decay_constant != 0:
            self.partial_half_life = np.log(2)/self.partial_decay_constant
        else:
            self.partial_half_life = np.inf

    def is_mode(self, mode_str):
        return mode_str in self.modes

    def __repr__(self):
        out = "{0} -> {1} via {2} with BR of {3}".format(self.parent_name, self.daughter_name, self.modes,
                                                   self.branching_ratio)
        if not DEBUG:
            return out
        else:
            return f"{out}, Q-value: {self.q_value}"


class DecayModeHandlerMixin:
    def __init__(self, nuclide, emission_data):
        #  In some cases, a given decay mode can populate ground and excited states of the same child nucleus,
        #  leading to multiple DecayMode objects for a given decay path (same parent and daughter).
        #  Making obj.from_mode return a list in general is undesirable.
        #  This issue is managed in self.decay_mode property.
        self.parent_nuclide_name = nuclide.name
        try:
            self._from_modes: List[DecayMode] = nuclide.decay_modes[emission_data['from_mode']]
        except KeyError:
            self._from_modes = []

    @property
    def parent_nuclide(self) -> Nuclide:
        return Nuclide.from_symbol(self.parent_nuclide_name)

    @property
    def from_mode(self):
        if len(self._from_modes) == 1:
            return self._from_modes[0]
        elif len(self._from_modes) > 1:
            raise AssertionError("This gamma line does not have a single specified mode. Use from_modes to get list of "
                                 "decay modes")
        elif len(self._from_modes) == 0:
            warn(f"Attempt to access decay mode from spectra of nuclide ??. The decay mode"
                 f"in this case was not in the nuclear library. Returning None.")
            return None

    @property
    def from_modes(self):
        return self._from_modes


class GammaLine(DecayModeHandlerMixin):
    """
    Attributes:
        erg:
            Energy of gamma in KeV
        intensity:
            Mean number of gammas with energy self.erg emitted per parent nucleus decay (through any decay channel)
    """
    def __new__(cls, *args, **kwargs):
        obj = super(GammaLine, cls).__new__(cls)
        if kwargs:  # alternative __init__
            obj._from_modes = kwargs['_from_modes']
            obj.intensity = kwargs['intensity']
            obj.erg = kwargs['erg']
            obj.absolute_rate = kwargs['absolute_rate']
        return obj

    def __copy__(self):
        return GammaLine.__new__(GammaLine, _from_modes=self._from_modes, intensity=self.intensity, erg=self.erg,
                                 absolute_rate=self.absolute_rate)

    def __init__(self, nuclide: Nuclide, emission_data):
        """
        A container for a single gamma emission line.

        Args:
            nuclide:
            emission_data: Emission dictionary from _DiscreteSpectrum
    """
        super(GammaLine, self).__init__(nuclide, emission_data)
        self.erg = emission_data['energy']
        self.intensity = emission_data['intensity']
        self.absolute_rate = nuclide.decay_rate * self.intensity

    def get_n_gammas(self, ref_activity: float, activity_ref_date: datetime, tot_acquisition_time: float,
                     acquisition_ti: datetime = datetime.now(), activity_unit='uCi') -> UFloat:
        """
        Get the number of times this gamma line is produced by a radioactive source.
        Args:
            ref_activity: Activity in micro-curies when source was calibrated.
            activity_ref_date: Date of the source calibration
            tot_acquisition_time: float, number of seconds the acquisition was performed
            acquisition_ti: Optional. If provided, This is the date/time of the start of acquisition. If not provided,
                today is used.
            activity_unit: kBq, Bq, uCi, or Ci

        Returns: Absolute of number of photons emitted for this line.

        """
        assert isinstance(tot_acquisition_time, (float, int)), tot_acquisition_time
        n = Nuclide.from_symbol(self.parent_nuclide_name)
        n_decays = n.get_n_decays(ref_activity=ref_activity, activity_ref_date=activity_ref_date,
                                  tot_acquisition_time=tot_acquisition_time, acquisition_ti=acquisition_ti,
                                  activity_unit=activity_unit)
        return n_decays*self.intensity

    def __repr__(self, compact=False):
        if len(self.from_modes) > 1:
            mode = self.from_modes
        else:
            mode = self.from_mode

        return "Gamma line at {0:.2f} KeV; intensity = {1:.2e}; decay: {2} "\
            .format(self.erg, self.intensity, mode)


class BetaPlusLine(DecayModeHandlerMixin):
    """
    Attributes:
        erg: This appears to be the energy of available to an EC transition, not that of the beta particle.
        positron_intensity:
    """
    def __init__(self, nuclide: Nuclide, emission_data):
        """
        A container for a single gamma emission line.

        Args:
            nuclide:
            emission_data: Emission dictionary from __DiscreteSpectrum
    """
        super(BetaPlusLine, self).__init__(nuclide, emission_data)

        self.erg = emission_data['energy']*1E-6
        self.intensity = emission_data['intensity']
        self.absolute_rate = nuclide.decay_rate * self.intensity
        self.positron_intensity = emission_data['positron_intensity']

    def __repr__(self):
        return f'Beta+/EC: e+ intensity: {self.positron_intensity}'


if __name__ == '__main__':
    Nuclide().decay_modes
    n = Nuclide.from_symbol("U235")
    print(n.all_isotopes())
    # from openmc.data import NATURAL_ABUNDANCE
    # isotopes = {}
    # for path in Path('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/nuclide/data/nuclides').iterdir():
    #     if m := re.match("(([A-Z][a-z]*)([0-9]+).*?)\.pickle", path.name):
    #         n_name, symbol, A = m.groups()
    #         A = int(A)
    #         try:
    #             isotopes[symbol].add(A)
    #         except KeyError:
    #             isotopes[symbol] = set()
    # def a(s):
    #     try:
    #         return ATOMIC_NUMBER[s]
    #     except KeyError:
    #         return 0
    #
    # isotopes = {k: v for k, v in sorted(isotopes.items(), key=lambda k_v: a(k_v[0]))}
    #
    # with open("all_isotopes.pickle", 'wb')as f:
    #     pickle.dump(isotopes, f)
    #
    # for k, v in isotopes.items():
    #     print(k, v)

