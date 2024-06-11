from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import re
from logging import warning as warn
from typing import Dict, List, Tuple, Union, Literal
from JSB_tools.nuke_data_tools.nuclide.data_directories import DECAY_PICKLE_DIR
from JSB_tools.nuke_data_tools.nuclide.cross_section import (CrossSection1D, ActivationCrossSection,
                                                             ActivationReactionContainer, activation_libraries)
import pickle
from datetime import datetime, timedelta
from JSB_tools.nuke_data_tools.nudel import LevelScheme
import functools
import uncertainties.unumpy as unp
from uncertainties import UFloat, ufloat
from uncertainties.core import Variable, AffineScalarFunc
# from scipy.integrate import odeint
from typing import Callable
from pendulum import Date, Interval
import pendulum
try:
    from openmc.data.endf import Evaluation
    from openmc.data import Reaction, Decay, Product
except ModuleNotFoundError:
    warn("OpenMC not installed! Some functionality is limited. ")
from JSB_tools.nuke_data_tools.nuclide.atomic_data import ATOMIC_SYMBOL, ATOMIC_NUMBER, AVOGADRO, atomic_mass, \
    atomic_weight
# import JSB_tools.nuke_data_tools.nuclide.fission_yields

pwd = Path(__file__).parent

with open(pwd / 'elements.pickle', 'rb') as f:
    _element_data = pickle.load(f)

constants = {'neutron_mass': 939.56542052,
             'proton_mass': 938.272088,
             'u_to_kg': 1.6605390666E-27,  # atomic mass units to kg
             'J_to_eV': 1.0 / 1.602176634E-19,  # Joules to eV
             'c': 299792458,
             'A': 6.0221421E23}

NATURAL_ABUNDANCE: dict = None  # Has form {SS1: {A1: abun1, A2: abum2, ...}, SS2: {...} }
ALL_ISOTOPES: dict = None  #


def all_isotopes():
    """
    Has form {SSAAA1: {A1, A2, A3}, SSAAA2: {...}, ...} represents all isotopes for a given Z which have data.
    Returns:

    """
    global ALL_ISOTOPES
    if ALL_ISOTOPES is None:
        with open(pwd / "all_isotopes.pickle", 'rb') as f:
            ALL_ISOTOPES = pickle.load(f)
    return ALL_ISOTOPES


def get_abundance_dict():
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
        with open(pwd / 'abundances.pickle', 'rb') as f:
            NATURAL_ABUNDANCE = pickle.load(f)
    return NATURAL_ABUNDANCE


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return {'GammaLine': GammaLine,
                    'DecayMode': DecayMode,
                    'Nuclide': Nuclide,
                    '_DiscreteSpectrum': _DiscreteSpectrum}[name]
        except KeyError:
            return super().find_class(module, name)


class Element:
    def __init__(self, Z):
        try:
            data = _element_data[Z]
            self.element_name = data['Element']
            self.element_symbol = data['Symbol']
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
        except KeyError:
            for k in ['element_name ', 'element_period ', 'element_group ', 'element_phase ', 'metalQ ', 'nonmetalQ ',
                      'metalloidQ ', 'element_type ', 'atomic_radius ', 'electronegativity ', 'firstIonization ',
                      'density ', 'melting_point ', 'boiling_point ', 'specific_heat ', 'n_valence_electrons ']:
                setattr(self, k, None)
                self.atom_density = 0

    @property
    def is_noble_gas(self):
        return self.element_group == 18


def get_symbol_etc(symbol):
    """
    Get (correct) symbol from argument and return nucleus Z, A, etc.
    m_or_e is 'e' if name refers to nuclear level, None for gs and 'm' for an isomeric state.
    Args:
        symbol:

    Returns: symbol, Z, A, M, m_or_e

    """
    if '-' in symbol:
        symbol = symbol.replace('-', '')
        if symbol.endswith('m'):
            symbol = symbol[:-1] + '_m1'

    if symbol[0] == 'n' and symbol.lower() in ['n', 'neutron']:
        symbol = 'Nn1'

    _m = Nuclide.NUCLIDE_NAME_MATCH.match(symbol)  # "^(?P<s>[A-z]{1,3})(?P<A>[0-9]{1,3})(?:_?(?P<m_e>[me])(?P<iso>[0-9]+))?$")  # Nuclide name in GND naming convention

    if not _m:
        raise ValueError(
            "\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
            "e.g. Cl38_m1, n1, Ar40".format(symbol))

    if _m.group('iso') == '0':  # ground state specification, "_m0", is redundant.
        symbol = _m.group('s') + _m.group('A')

    m_or_e = _m.group('m_e')

    try:
        Z = ATOMIC_NUMBER[_m.group('s')]
    except KeyError:
        if symbol == "Nn1":
            Z = 0
        else:
            Z = None

    A = int(_m.group('A'))
    isometric_state = _m.group('iso')

    if isometric_state is None:
        isometric_state = 0
    else:
        isometric_state = int(isometric_state)

    isometric_state = isometric_state

    return symbol, Z, A, isometric_state, m_or_e


class Nuclide(Element):
    """
    A nuclide object that can be used to access cross-sections, decay children/parent nuclides, decay modes,
    and much more.

    Available data:
        Nuclide half-life
        Gamma emissions (energy, intensity, uncertainties)
        Decay channels and the resulting child nuclides (+ branching ratios)
        Proton activation cross-sections (PADF)
        Neutron activation cross-sections (ENDF)
        Neutron, photon, proton, and SF fission xs yeilds (use caution with photon and proton data)
        Neutron, photon, and proton fission cross-sections

        """
    NUCLIDE_NAME_MATCH = re.compile(
        "^(?P<s>[A-z]{1,3})(?P<A>[0-9]{1,3})(?:_?(?P<m_e>[me])(?P<iso>[0-9]+))?$")  # Nuclide name in GND naming convention

    all_instances = {}  # keeping track of all Nuclides to save time on loading

    @classmethod
    def _from_e(cls, symbol, z, a, e: int) -> Nuclide:
        # todo: Decay modes. and return isomer if exists.

        self = Nuclide(symbol, _default=True)
        self.__decay_gamma_lines = []
        self.__decay_betaplus_lines = []

        self.__decay_daughters_str__ = [symbol]

        level = LevelScheme(symbol).levels[e]

        self.excitation_energy = level.energy
        self.nuclear_level = level

        self.__Z_A_iso_state__ = z, a, None

        self.half_life = level.half_life
        self.name = f"{symbol}_e{e}"

        return self

    def copy(self) -> Nuclide:
        return Nuclide(self.name, _copy=True)

    def __new__(cls, symbol=None, *args, **kwargs):
        """

        Args:
            symbol:
            *args: unused
            **kwargs: if _default in kwargs, set all values to default. if _copy in kwargs, don't pull intance from RAM
        """
        if symbol is None:  # Handles call to __new__ which occurs during unpickling.
            return object.__new__(cls)
        else:
            if not isinstance(symbol, str):
                raise TypeError(f"Invalid type, '{type(symbol)}' "
                                f"passed to Nuclide. Argument `symbol` must have str type")

        symbol, z, a, m, m_or_e = get_symbol_etc(symbol)

        if m_or_e == 'e':
            symbol = symbol.replace(f'_e{m}', '')  # remove "_ei" from symbol
            return Nuclide._from_e(symbol, z, a, m)

        if symbol not in Nuclide.all_instances or '_copy' in kwargs:
            pickle_file = DECAY_PICKLE_DIR / (symbol + '.pickle')

            if '_default' in kwargs or not pickle_file.exists():
                self = super().__new__(cls)
                self.__Z_A_iso_state__ = z, a, m
                self.is_valid = False  # triggers default routine in __init__
                return self
            else:
                with open(pickle_file, "rb") as pickle_file:
                    self = CustomUnpickler(pickle_file).load()
                self.is_valid = True

                Nuclide.all_instances[symbol] = self
                return self

        else:
            self = Nuclide.all_instances[symbol]
            self.is_valid = True

            # self.half_life.tag = f'half_life_{self.name}'

            return self

    def __init__(self, symbol, **kwargs):
        """

        Attributes:
            name: e.g. 'Xe139',  or 'U235_m1' for long-lived isomers, or 'U238_e2' for U238 in the second excited state

            excitation_energy: excited state energy in keV (always 0 for ground-state,
                None for nuclei in short-lived excited states.)

            nuclear_level: 0 for gs, 1 for first excited state, etc.

            isometric_state: A property that refers to the isomeric state. A value of None means nuclear excited level
                is directly specified by self.nuclear_level. isometric_state is not always the same as nuclear_level,
                since the isomeric_state number increments for each nuclear level with a long half-life.

            half_life: half life in seconds.

            spin: Spin state in units of hbar

            is_stable:  Equal to True is nuclei is stable, False otherwise.

            decay_modes: Dictionary with values DecayMode instances and keys tuple of strings (e.g. ('beta-',) ),
                usually of len=1 except for case of delayed-decay ratiotion, such as beta-delayrd neutron emission,
                in which case would be ('beta-', 'n')

        Notes:
            In __new__, attempt to load nuclide data from pickle file.
            If not file present or "__default__" in kwargs, set attribs to defaults.
            The reason for doing this in __init__ is so that autocomplete and type hinting work as usual.

        Args:
            symbol:
            **kwargs: Used by __new__
        """
        self.name = symbol
        super().__init__(self.Z)  # set element information. This is

        self.__decay_gamma_lines: List[GammaLine] = None
        self.__decay_betaplus_lines: List[BetaPlusLine] = None

        self.__decay_mode_for_print__ = None

        if self.is_valid:  # means instances were set in __new__
            return

        self.excitation_energy: float = None

        self.nuclear_level: float = None

        self.fissionable: bool = None

        self.half_life: Union[None, Variable] = None
        self.spin: int = None

        self.is_stable: Union[None, bool] = None

        self.__decay_daughters_str__: List[str] = []

        self.decay_modes: Dict[Tuple[str], List[DecayMode]] = {}

        self.__decay_parents_str__: List[str] = []

    @staticmethod
    def get_z_a_m_from_name(name: str):
        """
        From a nuclide name, return the atomic number (Z), mass number (A) and isomeric state (ie excited state, or M)
        Args:
            name: nuclide name, e.g. 'Xe139'

        Returns:

        """
        try:
            _m = Nuclide.NUCLIDE_NAME_MATCH.match(name)
            if not _m:
                raise ValueError
            z = ATOMIC_NUMBER[_m.group('s')]
            a = int(_m.group('A'))
            m = _m.group('iso')

        except (ValueError, KeyError) as e:
            raise ValueError(
                "\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
                "e.g. Cl38_m1, Nn1, Ar40".format(name)) from e

        if m is None:
            m = 0

        return z, a, m

    def adopted_levels(self) -> LevelScheme:
        """
        Levels
        Returns:

        """
        return LevelScheme(f"{self.atomic_symbol}{self.A}")

    def get_gammas(self, decay_rate_ratio_thresh=10, n_gen=1, **kwargs):
        """
        Include daughter decays, assuming equilibrium is achieved.
        Args:
            decay_rate_ratio_thresh:
                If daughter decays less than this times as fast, then don't include. Set to None to include all
                If None, no restriction on gammas

            n_gen:

            **kwargs: Internal use only. Do not use.

        Returns:

        """
        out: List[GammaLine] = kwargs.get('out', self.decay_gamma_lines)
        parent_decay_rate = kwargs.get('parent_decay_rate', self.decay_rate)
        cum_branching_ratio = kwargs.get('cum_branching_ratio', 1)
        cum_modes = kwargs.get('cum_modes', [])

        if n_gen == 0:
            return []

        for decay_type, decay_modes in self.decay_modes.items():
            for mode in decay_modes:
                daughter = mode.daughter_nuclide

                if decay_rate_ratio_thresh is None:
                    proceed_flag = True
                else:
                    proceed_flag = (daughter.decay_rate > parent_decay_rate * decay_rate_ratio_thresh)

                if not proceed_flag:
                    continue

                cum_modes.append(mode)

                new_branching_ratio = cum_branching_ratio * mode.branching_ratio

                for g in daughter.decay_gamma_lines:
                    intensity = new_branching_ratio * g.intensity
                    new_g = GammaLine.__new__(GammaLine, intensity=intensity, erg=g.erg, absolute_rate=parent_decay_rate * intensity, _from_modes=cum_modes,
                                              parent_nuclide_name=cum_modes[0].parent_name)

                    out.append(new_g)

                daughter.get_gammas(decay_rate_ratio_thresh=decay_rate_ratio_thresh, n_generations=n_gen - 1,
                                    out=out, parent_decay_rate=parent_decay_rate, cum_branching_ratio=new_branching_ratio, cum_modes=cum_modes)

        return out

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
            out = self.decay_gamma_lines[np.argmin(np.abs(ergs - energy))]
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
            return int(self.Z * 1E6 + self.A)

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

        if self.is_stable:
            return ufloat(0, 0)

        tot_acquisition_time = timedelta(seconds=tot_acquisition_time).total_seconds()

        corr_seconds = (acquisition_ti - activity_ref_date).total_seconds()  # seconds since ref date.
        try:
            unit_factor = {'ci': 3.7E10, 'uci': 3.7E10 * 1E-6, 'bq': 1, "kbq": 1E3}[activity_unit.lower()]
        except KeyError:
            assert False, f'Bad activity unit, "{activity_unit}". Valid units: Ci, uCi, Bq, uBq'

        ref_num_nuclides = ref_activity * unit_factor / self.decay_rate  # # of nuclides when ref calibration was performed

        corrected_num_nuclides = ref_num_nuclides * 0.5 ** (corr_seconds / self.half_life)
        n_decays = corrected_num_nuclides * (1 - 0.5 ** (tot_acquisition_time / self.half_life))
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

            text_xy = (xy[0], y_max * 1.1)
            ax.annotate('{:.2f}KeV'.format(erg), xy=xy, xytext=text_xy,
                        arrowprops=dict(width=0.1, headwidth=4, facecolor='black', shrink=0.03), rotation=90)

        ax.set_title('Gamma spectrum of {}, half-life = {}, for Ig (%) >= {:.1f}'.format(self.name,
                                                                                         self.pretty_half_life(False),
                                                                                         min_intensity * 100))
        ax.errorbar(x, y, yerr=y_err, xerr=x_err, label=label, marker='p', ls='none')
        if len(x) == 1:
            ax.set_xlim(x[0] - 10, x[0] + 10)
        ax.set_ylim(0, max(y + y_err) * 1.5)

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
        return self.A - self.Z

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

    def pretty_half_life(self, include_errors: bool = False, precision=3) -> str:
        """
        Gives the half life in units of seconds, hours, days, months, etc.
        Args:
            precision:
            include_errors:  Whether to include uncertainties

        Returns:

        """
        if self.half_life is None:
            return None
        elif np.isinf(self.half_life.n):
            return 'inf'
        elif self.half_life == 0:
            return ufloat(0, 0)

        dts = [('ns', 1E-9), ('us', 1E-6), ('ms', 1E-3), ('s', 1), ('min', 60), ('hr', 60 ** 2), ('day', 24 * 60 ** 2),
               ('month', 30 * 24 * 60 ** 2), ('year', 365 * 24 * 60 ** 2)]

        n_units = 0
        unit = 0

        for unit, dt in dts[::-1]:
            n_units = self.half_life.n / dt
            if n_units > 1:
                break

        if include_errors:
            rel_err = self.half_life.std_dev / self.half_life.n
            n_units = ufloat(n_units, rel_err * n_units)
            nominal_n_units = n_units.n
        else:
            nominal_n_units = n_units

        if np.abs(np.log10(nominal_n_units)) > 3:
            n_units = f"{n_units:.{precision}e}"
        else:
            n_units = f"{n_units:.{precision + 1}g}"

        return f"{n_units} {unit}"

    def get_fission_xs(self, projectile: str, library=None):
        if library is None:
            try:
                library = ActivationReactionContainer.libraries[projectile][0]
            except KeyError:
                raise KeyError(f"No nuclear data for library, '{library}'")

        dir_var = f'{projectile.upper()}_PICKLE_DIR'

        try:
            pickle_dir = eval(dir_var)
        except NameError:
            raise NameError(f"No global variable defined to point to directory for {projectile}'s. Define {dir_var}")

        try:
            with open(pickle_dir / library / 'fission_xs' / f'{self.name}.pickle', 'rb') as f:
                return CustomUnpickler(f).load()
        except FileNotFoundError:
            raise FileNotFoundError(f'No {projectile} induced fission xs data for {self.name} from library {library}.')

    @functools.cached_property
    def proton_induced_fiss_xs(self) -> CrossSection1D:
        try:
            return self.get_fission_xs('proton', 'endf')
        except FileNotFoundError:
            p = Path(__file__).parent / 'endf_files' / 'proton_fission_xs' / 'readme'
            with open(p) as f:
                msg = '\n'.join(f.readlines())
            raise FileNotFoundError(
                f'No proton induced fission xs data for {self.name}. Download it and integrate it if it is available. '
                f'Saving to pickle is done in `endf_to_pickle.py`.\n{msg}')

    @functools.cached_property
    def gamma_induced_fiss_xs(self) -> CrossSection1D:
        """
        Get the photon induced fission cross-section for this nuclide.
        Raise error if no data available.
        Returns:

        """
        return self.get_fission_xs('gamma')

    @property
    def neutron_induced_fiss_xs(self) -> CrossSection1D:
        """
        Get the neutron induced fission cross-section for this nuclide.
        Raise error if no data available.
        Returns:

        """
        return self.get_fission_xs('neutron')

    def neutron_energy_spectrum(self, ergs, binsQ=False):
        assert self.name == 'Cf252'
        from JSB_tools.nuke_data_tools.nuclide.misc_data.watt_neutron_spectra import Cf252_watt_spectrum
        return Cf252_watt_spectrum(ergs, binsQ)

    def rest_energy(self, units='MeV'):  # in J or MeV
        units = units.lower()
        ev = constants['J_to_eV']
        unit_dict = {'j': 1, 'mev': ev * 1E-6, 'ev': ev}
        assert units in unit_dict.keys(), 'Invalid units, "{}".\nUse one of the following: {}' \
            .format(units, unit_dict.keys())
        c = constants['c']
        j = self.atomic_mass('kg') * c ** 2
        return j * unit_dict[units]

    def neutron_separation_energy(self, n_neutrons=1):
        """
        Min. energy required to remove n_neutrons from nucleus in MeV
        Args:
            n_neutrons: Number of neutrons to obe removed.

        Returns:

        """
        warn("Verify!")
        z, a, m = Nuclide.get_z_a_m_from_name(self.name)
        a -= n_neutrons
        neutron_mass = constants['neutron_mass']
        return (Nuclide.from_Z_A_M(z, a, m).rest_energy('MeV') + n_neutrons * neutron_mass) - self.rest_energy('MeV')

    def proton_separation_energy(self, n_protons=1):
        """
        Min. energy required to remove n_protons from the nucleus (in MeV)
        Args:
            n_protons: Number of protons to obe removed.

        Returns:

        """
        warn("Verify!")

        z, a, m = Nuclide.get_z_a_m_from_name(self.name)
        z -= n_protons
        a -= n_protons
        proton_mass = constants['proton_mass']
        return (Nuclide.from_Z_A_M(z, a, m).rest_energy('MeV') + n_protons * proton_mass) - self.rest_energy('MeV')

    def alpha_separation_energy(self):
        """
        Min. energy required to remove He-4 from the nucleus (in MeV)

        Returns:

        """
        warn("Verify!")

        z, a, m = Nuclide.get_z_a_m_from_name(self.name)
        z -= 2
        a -= 2

        return (Nuclide.from_Z_A_M(z, a, m).rest_energy('MeV') + Nuclide('He-4').rest_energy('MeV')) \
            - self.rest_energy('MeV')

    @property
    def grams_per_mole(self) -> Union[float, None]:
        """
        Grams per mole of natural isotopic substance

        Returns:

        """
        return self.atomic_mass('u')

    @staticmethod
    def isotopic_abundance(nuclide_name) -> float:
        """
        Return isotopic abundance.
        Args:
            nuclide_name:

        Returns:

        """
        m = re.match("(([A-Z][a-z]*)([0-9]+).*?)", nuclide_name)
        if not m:
            raise ValueError(f"Invalid nuclide name, {nuclide_name}")
        _, symbol, A = m.groups()

        try:
            return get_abundance_dict()[symbol][int(A)]
        except KeyError:
            return 0

    @staticmethod
    def standard_atomic_weight(element_symbol):
        """
        Return atomic weight of an element in atomic mass units (g/mol)

        Args:
            element_symbol:

        Returns:

        """
        return atomic_weight(element_symbol)

    @functools.cached_property
    def atom_density(self):
        """
        Return Atoms/cm3 for material at STP conditions.
        Returns:

        """
        return self.density / self.atomic_mass(unit='g')

    def atomic_mass(self, unit='g'):
        """
        Args:
            unit: can be 'u', 'g', or 'kg'

        Returns:

        """
        u_to_kg = constants['u_to_kg']

        if self.excitation_energy != 0:
            eV_to_J = 1.0 / constants['J_to_eV']
            excitation_mass = eV_to_J * 1E3 * self.excitation_energy / constants['c'] ** 2  # mass in kg
            excitation_mass /= u_to_kg  # kg -> u
        else:
            excitation_mass = 0

        if unit == 'u':
            s = 1
        elif unit == 'g':
            s = u_to_kg * 1E3
        elif unit == 'kg':
            s = u_to_kg
        else:
            raise ValueError(f"Invalid unit, {unit}")

        return (atomic_mass(self.name) + excitation_mass) * s

    @staticmethod
    def get_all_isotopes(atomic_symbol: str, non_zero_abundance=True) -> List[str]:
        """
        Returns list of strings of all isotopes (with existing data) with atomic number according to `atomic_symbol`
        argument.

        Args:
            atomic_symbol:
            non_zero_abundance: If True, only return nuclides that occur naturally. Otherwise, return any for which
                                atomic data exists.
        Returns:

        """
        # m = re.match('^([A-z]{0,3})(?:[0-9]+[_m]*([0-9]+)?)', atomic_symbol)
        m = re.match(r'^([A-z]{0,3})$', atomic_symbol)
        assert m, f"Invalid argument, '{atomic_symbol}'"

        s = m.groups()[0]
        s = f"{s[0].upper()}{s[1:].lower()}"
        outs = []

        abun = get_abundance_dict()
        for a in all_isotopes()[s]:
            other_s = f"{s}{a}"

            if non_zero_abundance:
                if a in abun[s] and abun[s][a] > 0:
                    outs.append(other_s)
            else:
                outs.append(other_s)

        return outs

    @staticmethod
    def isotopic_breakdown(atomic_symbol) -> Dict[int, float]:
        """

        Args:
            atomic_symbol:

        Returns: Dictionary, e.g.
                    {A1: frac1, A2: frac2, ...}
                 where,
                    A is mass number

        """
        m = re.match(r'^([A-z]{0,3})$', atomic_symbol)
        assert m, f"Invalid argument, '{atomic_symbol}'"

        s = m.groups()[0]
        s = f"{s[0].upper()}{s[1:].lower()}"

        return get_abundance_dict()[s]

    def all_isotopes(self, non_zero_abundance=False):
        return Nuclide.get_all_isotopes(self.atomic_symbol, non_zero_abundance)

    @property
    def natural_abundance(self):
        return Nuclide.get_natural_abundance(self.name)

    @staticmethod
    def get_natural_abundance(symbol: Union[Nuclide, str]) -> float:
        if isinstance(symbol, Nuclide):
            symbol = symbol.name
        try:
            return Nuclide.isotopic_abundance(symbol)
        except KeyError:
            return 0.

    @staticmethod
    def max_abundance_isotope(atomic_symbol) -> Tuple[float, str]:
        abundance = None
        out = None

        for s in Nuclide.get_all_isotopes(atomic_symbol):
            try:
                a = Nuclide.isotopic_abundance(s)
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

    def get_latex_name(self, unboldmath_cmd=False):
        """
        Return latex str for nuclide name. E.g. $^{136m}$I
        Args:
            unboldmath_cmd: Makes exponent not in bold. Needed to override JSB_tools.mpl_style, which by default adds
                a \boldmath declaration to the preamble.

        Returns:

        """
        if unboldmath_cmd:
            if self.isometric_state != 0:
                try:
                    m = 'mnop'[self.isometric_state - 1]
                except IndexError:
                    m = f'_l{self.isometric_state}'
            else:
                m = ''
            return fr"\unboldmath{{$^{{{self.A}{m}}}$}}{self.atomic_symbol}"
        else:
            return self.latex_name

    @property
    def mcnp_zaid(self):
        return f'{self.Z}{self.A:0>3}'

    def add_proton(self, n=1) -> Nuclide:
        return self.from_Z_A_M(self.Z + n, self.A + n, self.isometric_state)

    def remove_proton(self, n=1) -> Nuclide:
        return self.from_Z_A_M(self.Z - n, self.A - n, self.isometric_state)

    def remove_neutron(self, n=1) -> Nuclide:
        return self.from_Z_A_M(self.Z, self.A - n, self.isometric_state)

    def add_neutron(self, n=1) -> Nuclide:
        return self.from_Z_A_M(self.Z, self.A + n, self.isometric_state)

    def remove_alpha(self):
        return self.from_Z_A_M(self.Z - 2, self.A - 4, self.isometric_state)

    def add_alpha(self):
        return self.from_Z_A_M(self.Z + 2, self.A + 4, self.isometric_state)

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
        return cls(name)

    def __repr__(self):
        try:
            hl = self.pretty_half_life()
        except ValueError:
            hl = self.half_life

        if self.isometric_state != 0:
            if self.excitation_energy == 0:
                ext_print = " (? keV)"
            else:
                ext_print = f" ({self.excitation_energy * 1E-3:.3g} keV)"
        else:
            ext_print = ""

        out = f"<Nuclide: {self.name}{ext_print}; t_1/2 = {hl}>"
        if self.__decay_mode_for_print__ is not None:
            out += f" (from decay {self.__decay_mode_for_print__.__repr__()})"
        return out

    @property
    def decay_rate(self):
        return np.log(2) / self.half_life

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
                par = Nuclide(n)
                out_dict[n] = ufloat(0, 0)
                for modes in par.decay_modes.values():
                    for mode in modes:
                        if mode.daughter_name == self.name:
                            out_dict[n] += mode.branching_ratio
            return [(Nuclide(k), v) for k, v in out_dict.items()]
        else:
            return list([Nuclide(name) for name in self.__decay_parents_str__])

    @property
    def decay_daughters(self) -> List[Nuclide]:
        out = list([Nuclide(name) for name in self.__decay_daughters_str__])
        for nuclide in out:
            for decay_modes in self.decay_modes.values():
                for decay_mode in decay_modes:
                    if decay_mode.daughter_name == nuclide.name:
                        nuclide.__decay_mode_for_print__ = decay_mode

        return out

    def thermal_neutron_capture_xs(self, T=270, data_source=None):
        erg = T * 8.6E-11
        return self.neutron_capture_xs(data_source=data_source)(np.array([erg]))[0]

    def neutron_capture_xs(self, data_source=None) -> CrossSection1D:
        reses = self.get_incident_particle_daughters('neutron', data_source=data_source)
        s = Nuclide.from_Z_A_M(self.Z, self.A + 1).name
        try:
            return reses[s].xs
        except KeyError:
            raise FileNotFoundError(f"No neutron capture cross-section for {self.name}")

    def get_incident_particle_daughters(self, particle, data_source=None, a_z_hl_m_cut='', is_stable_only=False) \
            -> Dict[str, InducedDaughter]:
        f = getattr(self, f"get_incident_{particle}_daughters")
        return f(data_source=data_source, a_z_hl_m_cut=a_z_hl_m_cut, is_stable_only=is_stable_only)

    def get_incident_proton_parents(self, data_source=None, a_z_hl_m_cut='', is_stable_only=False) -> Dict[
        str, InducedParent]:
        return self.__get_parents__('proton', a_z_hl_m_cut, is_stable_only, data_source)

    def get_incident_proton_daughters(self, data_source: Union[Literal['all'], str] = None, a_z_hl_m_cut='',
                                      is_stable_only=False, no_levels=True) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('proton', a_z_hl_m_cut, is_stable_only, data_source, no_levels)

    def get_incident_gamma_daughters(self, data_source: Union[Literal['all'], str] = None, a_z_hl_m_cut='',
                                     is_stable_only=False, no_levels=True) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('gamma', a_z_hl_m_cut, is_stable_only, data_source, no_levels)

    def get_incident_gamma_parents(self, data_source: Union[Literal['all'], str] = None, a_z_hl_m_cut='',
                                   is_stable_only=False) -> Dict[str, InducedParent]:
        return self.__get_parents__('gamma', a_z_hl_m_cut, is_stable_only, data_source, )

    def get_incident_neutron_daughters(self, data_source: Union[Literal['all'], str] = None, a_z_hl_m_cut='',
                                       is_stable_only=False, no_levels=True) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('neutron', a_z_hl_m_cut, is_stable_only, data_source, no_levels)

    def get_incident_neutron_parents(self, data_source: Union[Literal['all'], str] = None, a_z_hl_m_cut='',
                                     is_stable_only=False) -> Dict[str, InducedParent]:
        return self.__get_parents__('neutron', a_z_hl_m_cut, is_stable_only, data_source)

    def __get_daughters__(self, projectile, a_z_hl_m_cut='', is_stable_only=False,
                          data_source: Union[str, None] = None, no_levels=True):
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

            no_levels: If True, short-liveed excited nuclides, like Ta180_e2, are not included.

        Returns:

        """
        reaction = ActivationReactionContainer.load(self.name, projectile, data_source)
        assert isinstance(reaction, ActivationReactionContainer), type(reaction)
        out: Dict[str, InducedDaughter] = {}

        for daughter_name, xs in reaction.product_nuclide_names_xss.items():
            if daughter_name == 'photon':  # todo: Dont pickle these
                continue

            if no_levels:
                if '_e' in daughter_name:
                    continue

            daughter_nuclide = Nuclide(daughter_name)

            if a_z_hl_m_cut != '' or is_stable_only:
                a, z, hl, m = daughter_nuclide.A, daughter_nuclide.Z, daughter_nuclide.half_life, \
                    daughter_nuclide.isometric_state

                if not __nuclide_cut__(a_z_hl_m_cut, a, z, hl, is_stable_only, m):
                    continue

            daughter = InducedDaughter(daughter_nuclide, self, projectile)
            daughter.xs = xs
            out[daughter_name] = daughter

        return out

    def __get_parents__(self, projectile, a_z_hl_cut='', is_stable_only=False,
                        data_source: Union[str, None] = None):
        daughter_reaction = ActivationReactionContainer.load(self.name, projectile, data_source)
        assert isinstance(daughter_reaction, ActivationReactionContainer)
        out: Dict[str, InducedParent] = {}

        parent_nuclides = [Nuclide(name) for name in daughter_reaction.parent_nuclide_names if name not in
                           ['photon', 'neutron']]

        for parent_nuclide in parent_nuclides:
            a, z, hl, m = parent_nuclide.A, parent_nuclide.Z, parent_nuclide.half_life, parent_nuclide.isometric_state
            if __nuclide_cut__(a_z_hl_cut, a, z, hl, is_stable_only, m):
                parent = InducedParent(self, parent_nuclide, inducing_particle=projectile)
                parent.xs = ActivationReactionContainer.fetch_xs(parent_nuclide.name, self.name, projectile,
                                                                 data_source=data_source)
                out[parent.name] = parent

        return out

    def inelastic_xs(self, projectile, data_source=None):
        warn("See U235 neutron inelastic vs fission xs. Doesn't make sense bc F is > inel. Something is wrong. ")
        return self.__get_misc_xs__('inelastic_xs', projectile, data_source)

    def elastic_xs(self, projectile, data_source=None):
        return self.__get_misc_xs__('elastic_xs', projectile, data_source)

    def total_xs(self, projectile, data_source=None):
        return self.__get_misc_xs__('total_xs', projectile, data_source)

    def get_sf_nubar(self):
        from JSB_tools.nuke_data_tools.nuclide.misc_data.nubar import get_nubar
        return get_nubar(self.Z, self.A)

    def get_neutron_emission_rate(self):
        """Neutron emission rate from SF  (per atom)"""

        if ('sf',) not in self.decay_modes:
            return ufloat(0, 0)

        br_ratio = Nuclide('Cf252').decay_modes['sf',][0].branching_ratio
        return self.decay_rate * br_ratio * self.get_sf_nubar()

    def __get_misc_xs__(self, attrib, projectile, data_source=None) -> CrossSection1D:
        """
        Loads ActivationReactionContainer and grabs cross-section attribute according to `attrib`.
        Args:
            attrib: inelastic_xs, total_xs, etc.
            projectile: proton, neutron, etc.
            data_source: endf, padf, etc. None will return first found.

        Returns:

        """
        assert projectile in activation_libraries, f'No data for projectile, "{projectile}"'
        out = None

        if data_source is None:
            data_sources = activation_libraries[projectile]
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
        return self.half_life.n >= (365 * 24 * 60 ** 2) * threshold_in_years

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
                hl=half life in seconds. Variables for year, day and hr will also be in scope.
                Defaults to all known nuclides.
            is_stable_only:  Only include stable nuclides.


        Returns: List of all nuclide instances meeting criteria

        """
        assert isinstance(z_a_m_hl_cut, str), 'All cuts must be a string instance.'
        nuclides = []
        with open(DECAY_PICKLE_DIR / 'quick_nuclide_lookup.pickle', 'rb') as f:
            nuclides_dict = pickle.load(f)
        for (a, z, hl, m), nuclide_name in nuclides_dict.items():
            if __nuclide_cut__(z_a_m_hl_cut, a, z, hl, is_stable_only, m=m):
                nuclides.append(Nuclide(nuclide_name))

        yield from nuclides


def __nuclide_cut__(a_z_hl_cut: str, a: int, z: int, hl: UFloat, is_stable_only=False, m=0) -> bool:
    """
    Does a nuclide with a given z, a, and time_in_seconds (half life) make fit the criteria given by `a_z_hl_cut`?
    Args:
        a_z_hl_cut: The criteria to be evaluated as python code, where z=atomic number, a=mass_number,
            and time_in_seconds=half life in seconds
        a: mass number
        z: atomic number
        hl: half life in seconds. Can also use variables year, day, and hr
        is_stable_only: does the half life have to be effectively infinity?
        m: Excited level.
    Returns: bool
    """
    makes_cut = True

    assert isinstance(is_stable_only, bool)
    if is_stable_only:
        if hl is None:
            return False
        elif not np.isinf(unp.nominal_values(hl)):
            return False

    if len(a_z_hl_cut) > 0:
        a_z_hl_cut = a_z_hl_cut.lower()
        if 'hl' in a_z_hl_cut and hl is None:
            makes_cut = False
        else:
            try:
                makes_cut = eval(a_z_hl_cut, {"hl": hl, 'a': a, 'z': z, 'm': m, 'year': 31536000, 'day': 86400, 'hr': 3600})
                assert isinstance(makes_cut, bool), "Invalid cut: {0}".format(a_z_hl_cut)
            except NameError as e:
                invalid_name = str(e).split("'")[1]

                raise Exception("\nInvalid name '{}' used in cut. Valid names are: 'z', 'a', and 'hl',"
                                " which stand for atomic-number, mass-number, and half-life, respectively."
                                .format(invalid_name)) from e

    return makes_cut


class InducedDaughter(Nuclide):
    """
    A subclass of Nuclide representing a nucleus that was the result of an induced reaction.
    This adds cross-section info

    Attributes:
        xs: CrossSection1D instance
        parent: Parent nuclide
        inducing_particle:
    """

    def __new__(cls, daughter_nuclide, parent_nuclide, inducing_particle, *args, **kwargs):
        return daughter_nuclide.copy()

    def __init__(self, daughter_nuclide, parent_nuclide, inducing_particle):
        assert isinstance(daughter_nuclide, Nuclide)
        assert isinstance(parent_nuclide, Nuclide)
        # kwargs = {k: v for k, v in daughter_nuclide.__dict__.items() if k != "name"}
        # super().__init__(daughter_nuclide.name, __internal__=True, **kwargs)
        self.xs: ActivationCrossSection = None
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
            daughter: Daughter nuclide
            inducing_particle:
        """

    def __new__(cls, daughter_nuclide, parent_nuclide, inducing_particle, *args, **kwargs):
        return parent_nuclide

    def __init__(self, daughter_nuclide, parent_nuclide, inducing_particle):
        assert isinstance(daughter_nuclide, Nuclide)
        assert isinstance(parent_nuclide, Nuclide)
        self.xs: ActivationCrossSection = None
        self.daughter: Nuclide = daughter_nuclide
        self.inducing_particle = inducing_particle

    def __repr__(self):
        par_symbol = self.inducing_particle[0]
        return '{0}({1},X) --> {2}'.format(super().__repr__(), par_symbol, self.daughter)


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
                intensity = discrete_normalization * emission_data['intensity']
                energy = emission_data['energy'] * 1E-3

                if isinstance(energy, AffineScalarFunc):
                    energy = ufloat(energy.n, energy.std_dev, f'γEnergy{energy.n:.2f}keV')

                if isinstance(intensity, AffineScalarFunc):
                    intensity = ufloat(intensity.n, intensity.std_dev, tag=f"γIntensity_{energy.n:.2f}keV")

                emission_data['from_mode'] = tuple(emission_data['from_mode'])

                emission_data['intensity'] = intensity
                emission_data['energy'] = energy

        except KeyError:
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
        return pwd / 'pickled_data' / 'nuclides' / f'{radiation_type}_spectra' / f'{nuclide_name}.pickle'

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

    @classmethod
    def from_values(cls, daughter_name, parent_name, parent_half_life, branching_ratio, q_value=None, modes: Union[None, tuple[str]] = None,
                    intermediates=None):
        out = cls.__new__(cls)

        out.q_value = q_value

        if modes is None:
            out.modes = []
        else:
            out.modes = modes

        out.daughter_name = daughter_name
        out.parent_name = parent_name
        out.branching_ratio = branching_ratio
        out.partial_decay_constant = np.log(2) / parent_half_life * branching_ratio

        if len(intermediates) == 0:
            intermediates = None
        out.intermediates = intermediates
        return out

    def __init__(self, openmc_decay_mode, parent_half_life):
        self.q_value = openmc_decay_mode.energy * 1E-6
        self.modes = tuple(openmc_decay_mode.modes)
        self.daughter_name = openmc_decay_mode.daughter
        self.parent_name = openmc_decay_mode.parent
        self.branching_ratio = openmc_decay_mode.branching_ratio
        parent_decay_rate = np.log(2) / parent_half_life
        self.partial_decay_constant = parent_decay_rate * self.branching_ratio
        if self.partial_decay_constant != 0:
            self.partial_half_life = np.log(2) / self.partial_decay_constant
        else:
            self.partial_half_life = np.inf

        self.intermediates = None

    @property
    def daughter_nuclide(self):
        return Nuclide(self.daughter_name)

    def is_mode(self, mode_str):
        return mode_str in self.modes

    def __repr__(self):
        # if self.intermediates is not None:
        #     chain = [self.parent_name] + self.intermediates + [self.daughter_name]
        #
        #     ss = '->'.join(chain)
        # else:
        ss = f"{self.parent_name} -> {self.daughter_name}"

        out = f"{ss} via {self.modes} with BR of {self.branching_ratio}"
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

        self._parents = [nuclide.name]  # for the case of gamma lines from decay chain

    @property
    def parent_nuclide(self) -> Nuclide:
        return Nuclide(self.parent_nuclide_name)

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
            try:
                obj.parent_nuclide_name = kwargs['parent_nuclide_name']
            except KeyError:
                obj.parent_nuclide_name = args[0].name

        return obj

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

    def __copy__(self):
        return GammaLine.__new__(GammaLine, _from_modes=self._from_modes, intensity=self.intensity, erg=self.erg,
                                 absolute_rate=self.absolute_rate)

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
        n = Nuclide(self.parent_nuclide_name)
        n_decays = n.get_n_decays(ref_activity=ref_activity, activity_ref_date=activity_ref_date,
                                  tot_acquisition_time=tot_acquisition_time, acquisition_ti=acquisition_ti,
                                  activity_unit=activity_unit)
        return n_decays * self.intensity

    def __repr__(self, compact=False):
        if len(self.from_modes) > 1:
            mode = self.from_modes
        else:
            mode = self.from_mode

        return "Gamma line at {0:.2f} KeV; intensity = {1:.2e}; decay: {2} " \
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

        self.erg = emission_data['energy'] * 1E-6
        self.intensity = emission_data['intensity']
        self.absolute_rate = nuclide.decay_rate * self.intensity
        self.positron_intensity = emission_data['positron_intensity']

    def __repr__(self):
        return f'Beta+/EC: e+ intensity: {self.positron_intensity}'


_unit_scales = {'n': 1E-9, 'u': 1E-6, 'm': 1E-3, '': 1, 'k': 1E3, 'M': 1E6, 'G': 1E9}


def get_scale_unit(unit):
    m = re.match("([numkM]?)(.+)", unit)

    scale_str = m.groups()[0]
    unit = m.groups()[1].lower()

    try:
        scale = _unit_scales[scale_str]
    except KeyError:
        raise ValueError(f"Invalid unit scale, '{scale_str}' ")

    return scale, unit


def get_seconds(dt):
    if isinstance(dt, (float, int)):
        out = dt

    elif isinstance(dt, Date):  # assume dt from then to now
        now = pendulum.now()
        out = (pendulum.date(now.year, now.month, now.day) - dt).total_seconds()

    elif isinstance(dt, Interval):
        out = dt.total_seconds()

    else:
        raise ValueError('Invalid type, "{type(dt)}"')

    return out


class Sample:
    """
    Class for calculating quantities of samples.

    e.g., going to ug of Cf-252 to uCi, etc.

    """
    valid_units = ['nCi', 'uCi', 'mCi', 'Ci','Bq', 'kBq', 'MBq', 'ng', 'ug', 'mg', 'g']

    def __init__(self, symbols: Union[str, List[str]], init_amount,
                 units=Literal['nCi', 'uCi', 'mCi', 'Ci', 'Bq', 'kBq', 'MBq', 'ng', 'ug', 'mg', 'g'],
                 atom_fractions=None):

        if isinstance(symbols, str):
            if Nuclide.NUCLIDE_NAME_MATCH.match(symbols):
                self.nuclides: List[Nuclide] = [Nuclide(symbols)]
                self.atom_fractions = [1]

            else:
                if not symbols in ATOMIC_SYMBOL.keys():
                    raise ValueError(f'Invalid nuclide: "{symbols}"')

                self.atom_fractions = []
                self.nuclides: List[Nuclide] = []
                for A, frac in Nuclide.isotopic_breakdown(symbols):
                    self.atom_fractions.append(frac)
                    self.nuclides.append(Nuclide(f'{symbols}{A}'))

        else:
            self.atom_fractions = atom_fractions
            self.nuclides: List[Nuclide] = [Nuclide(s) for s in symbols]

        self.atom_fractions = np.array(self.atom_fractions) / sum(self.atom_fractions)

        assert units in self.valid_units

        scale, unit = get_scale_unit(units)

        if unit == 'g':
            mass = init_amount * scale
            self.n_atoms = mass/self.grams_per_mole * AVOGADRO

        else:
            if not len(self.nuclides) == 1:
                raise ValueError("Cannot specify amount for a group of different nuclides")

            if unit.lower() == 'ci':
                assert self.nuclide.decay_rate > 0
                scale *= 3.7E10
                decays_per_second = init_amount * scale
                self.n_atoms = decays_per_second / self.nuclide.decay_rate

            elif unit.lower() == 'bq':
                self.n_atoms = scale * init_amount / self.nuclide.decay_rate

    def get_decay_rate(self, unit='Bq', dt: Union[float, int, Interval, Date] = 0, return_type=float):
        """

                Args:
                    unit:
                    dt: Decay sample according to delta time.
                        If Date: Assume a date is past, and decay until now
                        if float: assume a number of seconds

                    return_type: If a float, returns a number. If a string, returns a formatted string including units.

                Returns:
                    str or float, depending on return_type

                """
        scale, unit = get_scale_unit(unit)

        norm = 1

        if unit == 'ci':
            norm /= 3.7E10

        norm /= scale

        if dt != 0:
            dt = get_seconds(dt)

            norm *= 0.5**(dt/self.nuclide.half_life)

        out = norm * self.n_atoms * self.nuclide.decay_rate

        if return_type is float:
            return out
        elif return_type is str:
            return f"{out:.3g} {unit}"
        else:
            raise ValueError("Invalid return type. ")

    def get_atoms(self,  dt: Union[float, int, Interval, Date] = 0,):
        out = self.n_atoms

        if dt != 0:
            dt = get_seconds(dt)
            out *= 0.5 ** (dt / self.nuclide.half_life)

        return out

    def get_mass(self, unit='g', dt: Union[float, int, Interval, Date] = 0, return_type=float):
        """

        Args:
            unit:
            dt: Decay sample according to delta time.
                If Date: Assume a date is past, and decay until now
                if float: assume a number of seconds

            return_type: If a float, returns a number. If a string, returns a formatted string including units.

        Returns:
            str or float, depending on return_type

        """
        scale, unit = get_scale_unit(unit)

        norm = 1

        norm /= scale

        moles = self.n_atoms / AVOGADRO

        mass = moles * self.grams_per_mole

        if dt != 0:
            dt = get_seconds(dt)
            norm *= 0.5**(dt/self.nuclide.half_life)

        out = norm * mass
        if return_type is float:
            return out
        elif return_type is str:
            return f"{out:.3g} {unit}"
        else:
            raise ValueError("Invalid return type. ")

    @property
    def mass(self):
        return self.get_mass()

    @property
    def nuclide(self):
        assert len(self.nuclides) == 1
        return self.nuclides[0]

    @property
    def grams_per_mole(self,):
        return sum(f * n.grams_per_mole for f, n in zip(self.atom_fractions, self.nuclides))



if __name__ == '__main__':
    dt = pendulum.Date(2003, 9, 24)
    s = Sample('Cf252', 0.546, units='ug')
    # s = Sample('Cf252', 5.3, units='mCi')
    print(s.get_decay_rate(unit='mCi', dt=dt))
