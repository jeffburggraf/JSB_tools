from __future__ import annotations
import pickle
import numpy as np
import warnings
from matplotlib import pyplot as plt
import re
from pathlib import Path
from warnings import warn
from uncertainties import ufloat, UFloat
from uncertainties import unumpy as unp
import marshal
from typing import Union, List, Dict, Collection, Tuple
from numbers import Number
from JSB_tools.nuke_data_tools.global_directories import DECAY_PICKLE_DIR, PROTON_PICKLE_DIR, GAMMA_PICKLE_DIR, \
    NEUTRON_PICKLE_DIR, FISS_YIELDS_PATH
from functools import cached_property
from scipy.interpolate import interp1d
from uncertainties import nominal_value
from datetime import datetime, timedelta
# from JSB_tools import TabPlot
# from JSB_tools.nuke_data_tools.talys import talys_dir, run, pickle_result
# import JSB_tools.nuke_data_tools.talys as talys


def openmc_not_installed_warning():
    warnings.warn("openmc not installed! some functionality is limited. ")


try:
    from openmc.data.endf import Evaluation
    from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
    from openmc.data import Reaction, Decay, Product
    from openmc.data.data import NATURAL_ABUNDANCE, atomic_mass, atomic_weight, AVOGADRO, _ATOMIC_MASS
except ModuleNotFoundError:
    openmc_not_installed_warning()


__all__ = ['Nuclide', 'FissionYields', 'openmc_not_installed_warning', 'GammaLine']
pwd = Path(__file__).parent

DEBUG = False
#  Units
__u_to_kg__ = 1.6605390666E-27  # atomic mass units to kg
__speed_of_light__ = 299792458   # c in m/s

#  Note to myself: Pickled nuclear data is on personal SSD. Update this regularly!
#  Todo:
#   * make cross section pulls be implemented in a nuke_data.cross_secion file. Let the endf to pickle also be
#     implemented there
#   * Make a Nuclide.fromreaction('parent_nuclide_name', inducing particle, daughter_nuclide_name )
#   * Add documentation, and exception messages, to explain where the data can be downloaded and how to regenerate
#     the pickle files.
#   * Get rid of <additional_nuclide_data> functionality. too complex and stupid


NUCLIDE_INSTANCES = {}  # Dict of all Nuclide class objects created. Used for performance enhancements and for pickling
PROTON_INDUCED_FISSION_XS1D = {}  # all available proton induced fissionXS xs. lodaed only when needed.
PHOTON_INDUCED_FISSION_XS1D = {}  # all available gamma induced fissionXS xs. lodaed only when needed.
NEUTRON_INDUCED_FISSION_XS1D = {}  # all available neutron induced fissionXS xs. lodaed only when needed.


# global variable for the bin-width of xs interpolation
XS_BIN_WIDTH_INTERPOLATION = 0.1

# Some additional nuclide info that aren't in ENDSFs
additional_nuclide_data = {"In101_m1": {"half_life": ufloat(10, 5)},
                           "Lu159_m1": {"half_life": ufloat(10, 5)},
                           "Rh114_m1": {"half_life": ufloat(1.85, 0.05), "__decay_daughters_str__": "Pd114"},
                           "Pr132_m1": {"half_life": ufloat(20, 5), "__decay_daughters_str__": "Ce132"}}

_all_nuclides_ = None


def nuclide_list():
    """
    Return all nuclides that have had their mass quantified.
    Returns:

    """
    global _all_nuclides_

    if _all_nuclides_ is None:
        atomic_mass('H1')
        _all_nuclides_ = [f"{x[0].upper()}{x[1:]}" for x in _ATOMIC_MASS.keys()]

        [_all_nuclides_.remove(x) for x in ['C0', 'Zn0', 'Pt0', 'Os0', 'Tl0']]


    return _all_nuclides_


def human_readable_half_life(hl, include_errors, abrev_units=True):
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

    def process_return(_s: str):
        if abrev_units:
            for u, up in [('seconds', 's'), ('minutes', 'm'), ('hours', 'h'), ('days', 'd'), ('years', 'a')]:
                _s = _s.replace(u, up)
        return _s.rstrip().lstrip()

    hl_in_sec = hl

    if hl_in_sec is None:
        raise ValueError

    if isinstance(hl_in_sec, float):
        hl_in_sec = ufloat(hl_in_sec, 0)

    if np.isinf(hl_in_sec.n) or np.isnan(hl_in_sec.n):
        return str(hl_in_sec.n).rstrip()

    if hl_in_sec < 1:
        percent_error = 100 * hl_in_sec.std_dev / hl_in_sec.n
        out = "{:.2e} seconds ".format(hl_in_sec.n)
        if include_errors:
            out += " ({}) ".format(get_error_print(percent_error))
        return process_return(out)

    elif hl_in_sec < 60:
        percent_error = 100 * hl_in_sec.std_dev / hl_in_sec.n
        out = "{:.1f} seconds ".format(hl_in_sec.n)
        if include_errors:
            out += " ({}) ".format(get_error_print(percent_error))
        return process_return(out)

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
                    percent_error = get_error_print(percent_error, 2)
                elif percent_error < 1:
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

    return process_return(out)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'GammaLine':
            return GammaLine
        elif name == 'DecayMode':
            return DecayMode
        elif name == 'ActivationReactionContainer':
            return ActivationReactionContainer
        elif name == 'CrossSection1D':
            return CrossSection1D

        return super().find_class(module, name)


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
                    out = pickle.load(f)
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


class DecayModeHandlerMixin:
    def __init__(self, nuclide, emission_data):
        #  In some cases, a given decay mode can populate ground and excited states of the same child nucleus,
        #  leading to multiple DecayMode objects for a given decay path (same parent and daughter).
        #  Making obj.from_mode return a list in general is undersirable.
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

    # @property
    # def parent_nuclide_name(self):
    #     try:
    #         return self.from_mode.parent_name
    #     except AssertionError:
    #
    #         if not all([m.parent_name == self._from_modes[0].parent_name for m in self._from_modes]):
    #             warn("WTF")
    #         return self._from_modes[0].parent_name


class GammaLine(DecayModeHandlerMixin):
    """
    Attributes:
        erg:
            Energy of gamma in KeV
        intensity:
            Mean number of gammas with energy self.erg emitted per parent nucleus decay (through any decay channel)

        from_mode:
            DecayMode instance. Contains decay channel, branching ratio, among other information
    """
    def __new__(cls, *args, **kwargs):
        obj = super(GammaLine, cls).__new__(cls)
        if kwargs:
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
            emission_data: Emission dictionary from __DiscreteSpectrum
    """
        super(GammaLine, self).__init__(nuclide, emission_data)
        self.erg = emission_data['energy']
        self.intensity = emission_data['intensity']
        self.absolute_rate = nuclide.decay_rate * self.intensity

    def get_n_gammas(self, ref_activity: float, activity_ref_date: datetime, tot_acquisition_time: float,
                     acquisition_ti: datetime = datetime.now(), activity_unit='uCi') -> UFloat:
        """
        Get the number of time this gamma line is produced by a radioactive source.
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
        if DEBUG:
            print(f"BetaPlusLine emission data of {nuclide}: ")
            for k, v in emission_data.items():
                print('\t', k, v)
            print(self.from_modes)

        self.erg = emission_data['energy']*1E-6
        self.intensity = emission_data['intensity']
        self.absolute_rate = nuclide.decay_rate * self.intensity
        self.positron_intensity = emission_data['positron_intensity']

    def __repr__(self):
        return f'Beta+/EC: e+ intensity: {self.positron_intensity}'


def rest_mass(z=None, a=None, n=None):
    """
    return the rest mass of nucleus with atomic numbers Z and n
    Args:
        z: proton number
        a: mass number (z + a)
        n: neutron number

    Returns:

    """
    if z == a == 0:
        return 0
    if z is None:
        assert not (a is n is None), 'Incomplete nucleon number specification'
        z = a-n
    elif a is None:
        assert not (z is n is None), 'Incomplete nucleon number specification'
        a = z + n
    elif n is None:
        assert (z is not a is not None), 'Incomplete nucleon number specification'
        n = a-z
    assert z + n == a, 'Invalid nucleon number specification (Z + N != A)'

    c = 931.494102  # 1 AMU = 931.494102 MeV/c^2
    try:
        if z == 0 and n == 1:
            return neutron_mass
        else:
            symbol = ATOMIC_SYMBOL[z] + str(a)
            return atomic_mass(symbol)*c
    except KeyError:
        raise KeyError(f'No atomic mass data for Z={z} A={a}')


proton_mass = 938.272  # MeV/c^2
neutron_mass = 939.565
alpha_mass = 3727.37


class RawFissionYieldData:
    """
    Container for marshal'd fission yield data.
    Internal use only.
    """
    instances: Dict[Path, RawFissionYieldData] = {}

    def __new__(cls, *args, **kwargs):
        """For cashed class instances."""
        _dir = RawFissionYieldData.__get_dir(*args, **kwargs)
        if _dir in RawFissionYieldData.instances:
            return RawFissionYieldData.instances[_dir]
        instance = super(RawFissionYieldData, cls).__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance

    @staticmethod
    def __get_dir(inducing_particle, library, independent_bool: bool, nuclide_name):
        _dir = FISS_YIELDS_PATH / inducing_particle
        if not _dir.exists():
            assert False, f'Invalid inducing particle, "{inducing_particle}", for induced fission'
        _dir /= library
        if not _dir.exists():
            assert False, f'Invalid library, "{library}", for {inducing_particle}-induced fission'
        _dir = _dir / ('independent' if independent_bool else 'cumulative') / (nuclide_name + '.marshal')
        if not _dir.exists():
            raise FileNotFoundError(f"No fission yield file for {inducing_particle}-induced fission on {nuclide_name} "
                                    f"from library, '{library}'")
        return _dir

    def __init__(self, inducing_particle, library, independent_bool: bool, nuclide_name):
        self.file_path = self.__get_dir(inducing_particle, library, independent_bool, nuclide_name)
        # define this here so that the whole file isn't loaded in the event that only energies are needed.
        self.file = open(self.file_path, 'rb')
        self.ergs = marshal.load(self.file)
        RawFissionYieldData.instances[self.file_path] = self

    @cached_property
    def data(self) -> Dict:
        out = marshal.load(self.file)
        del self.file
        return out


class DecayedFissionYields:
    def __init__(self, target: str, inducing_par: Union[None, str], energies: Union[Collection[float], None] = None,
                 library: Union[str, None] = None, independent_bool: bool = True):
        """
        Calculates fission yields at a time after factoring in decays.

        Args:
            target: Fission target nucleus
            inducing_par: None for SF. Or, e.g., 'proton', 'neutron', 'gamma'
            energies: If None, use energies from the data file.
            library: Fission yield library. See FissionYields.FISSION_YIELD_SUBDIRS for available yield libraries.
            independent_bool:
        """
        self.indep_fission_yields = FissionYields(target=target, inducing_par=inducing_par, energies=energies,
                                                  library=library, independent_bool=independent_bool)

    def decay(self, times, yield_thresh=1E-4) -> Dict[str, np.ndarray]:
        """
        Returns dict of yields at each time t.
        Args:
            times: An array or float.
            yield_thresh: If yield is below this, don't include it.

        Returns: Dict[str, ]

        """
        out = {}

        for n_name, yield_ in self.indep_fission_yields.yields.items():
            if hasattr(yield_, '__iter__'):
                assert len(yield_) == 1, 'Weight yields before applying the decay function! '
                yield_ = yield_[0]

            if yield_ < yield_thresh:
                if n_name not in out:
                    out[n_name] = ufloat(0, 0) if not hasattr(times, '__iter__') else unp.uarray(np.zeros_like(times), np.zeros_like(times))
                continue

            decay_func = decay_nuclide(n_name)
            for n_name_decay, decay_yield in decay_func(times).items():
                if max(decay_yield) < yield_thresh:  # todo: wont work with non-iter `times`?
                    if n_name not in out:
                        out[n_name] = ufloat(0, 0) if not hasattr(times, '__iter__') else unp.uarray(np.zeros_like(times), np.zeros_like(times))
                    continue
                try:
                    out[n_name_decay] += yield_*decay_yield
                except KeyError:
                    out[n_name_decay] = yield_ * decay_yield
        if hasattr(times, '__iter__'):
            out = {k: v for k, v in sorted(out.items(), key=lambda x: -x[1][-1])}
        else:
            out = {k: v for k, v in sorted(out.items(), key=lambda x: -x[1])}

        return out


class FissionYields:
    """
    Retrieve fission yield data, if available. In some cases, the code will convert neutron yield
        to the desired yield by adjusting the nucleus and energies accordingly

    Attributes:
        energies: Energies at which the yield is determined
        library: Library where data was taken from, e.g. ukfy, or gef
        weights: Yields can be weighted by a quantity, e.g. flux, cross-sections

    """
    FISSION_YIELD_SUBDIRS = {'neutron': ['endf', 'gef'], 'proton': ['ukfy', None], 'gamma': ['ukfy', None],
                             'sf': ['gef'], 'alpha': [None], 'electron': [], }

    @property
    def file_path(self):
        return self.__data.file_path

    @staticmethod
    def particle_energy_convert(nuclide_name: str, ergs, from_par, to_par) -> Tuple[np.ndarray, str]:
        """
        Converts the particle energies of a given `from_par`-induced fission on `n`, to that of `to_par`-induced fission
            on`n_prime`. The problem solved is the following: what do the energies of `to_par` have to have to be in
            order to produce the same pre-fission nucleus as `from_par` with regard to
            nucleon numbers and compound nucleus excitation energy.
        Args:
            nuclide_name: Nuclide instance
            ergs: The energies to be converted
            from_par: Particle that has kinetic energy specified by `ergs`.
            to_par: The particle for which the new energies will be calculated.

        Returns: Tuple[(calculated energies, pre fission nucleus for to_par)]

        """
        n = Nuclide.from_symbol(nuclide_name)
        if not hasattr(ergs, '__iter__'):
            assert isinstance(ergs, Number)
            ergs = [ergs]
        else:
            assert isinstance(ergs[0], Number)
        from_par_KEs = np.array(ergs)
        err_msg = 'Particle, {} not implemented or is invalid. To add more particles, add values to the ' \
                  'FissionYields.__par_masses__ dictionary.'

        def get_par_z_a(par: str):
            """Return Z and A of particle if applicable, else (0, 0). Implement more particles here, e.g. electron
            Args:
                par: Particle symbol/name.
            """
            # assert par in FissionYields.__par_masses__.keys(), err_msg.format(par)
            if par == 'gamma':
                return 0, 0
            else:
                _n = Nuclide.from_symbol(par)
                assert _n.is_valid, f"Invalid particle, {par}"
                return _n.Z, _n.A

        from_par_z, from_par_a = get_par_z_a(from_par)
        from_target_z, from_target_a = n.Z, n.A
        from_par_target_nuclide_mass = n.rest_energy()
        from_par_rest_mass = rest_mass(from_par_z, from_par_a)

        compound_z, compound_a = (from_par_z + from_target_z), (from_par_a + from_target_a)

        to_par_z, to_par_a = get_par_z_a(to_par)
        to_target_z, to_target_a = compound_z - to_par_z, compound_a - to_par_a

        to_par_nuclide = Nuclide.from_Z_A_M(to_target_z, to_target_a)
        to_par_target_nuclide_mass = to_par_nuclide.rest_energy()
        to_par_rest_mass = rest_mass(to_par_z, to_par_a)

        to_par_KEs = from_par_rest_mass + from_par_target_nuclide_mass + from_par_KEs - to_par_rest_mass \
                     - to_par_target_nuclide_mass
        return to_par_KEs, to_par_nuclide.name

    @staticmethod
    def __score(xnew, xp):
        if len(xp) == 2:
            if not all([i in xp for i in xnew]):
                return 0
        l = len(list(filter(lambda x: xp[0] <= x <= xp[-1], xnew)))
        return l / len(xnew)

    def __find_best_library__(self):
        """
        Find fission yield library that has the most data points within the desired energy range.

        """
        scores = []
        libraries = []
        datums = []
        for lib in FissionYields.FISSION_YIELD_SUBDIRS[self.inducing_par]:
            if lib is None:
                continue
            try:
                data = RawFissionYieldData(self.inducing_par, lib, self.independent_bool, self.target)
            except FileNotFoundError:
                continue
            datums.append(data)
            data_ergs = data.ergs
            libraries.append(lib)
            scores.append(self.__score(self.energies if self.energies is not None else data_ergs, data_ergs))
        if len(scores):
            i = np.argmax(scores)
            self.library = libraries[i]
            return datums[i]

    @classmethod
    def from_neutron(cls, target, inducing_par, ergs, independent_bool=True):
        assert ergs is not None, 'Must supply energies when converting neutron-fission data to another projectile'
        new_ergs, new_target = FissionYields.particle_energy_convert(target, ergs, inducing_par, 'neutron')

        yield_ = cls(new_target, 'neutron', new_ergs, None, independent_bool)
        return yield_

    @staticmethod
    def interp(xnew, x, y):
        """
        Interp fission yield for a given nuclide. Todo: consider other extrapolation methods.
        Args:
            xnew: Desired energy points.
            x: Energy at which yield data exists.
            y: The yield Data

        Returns: interpolated yields.

        """
        interp_func = interp1d(x, y, fill_value='extrapolate', bounds_error=False)
        result = interp_func(xnew)
        negative_select = np.where(xnew < 0)
        result[negative_select] = 0
        result = np.where(result >= 0, result, 0)
        return result

    def __init__(self, target: str, inducing_par: Union[None, str], energies: Union[Collection[float], None] = None,
                 library: Union[str, None] = None, independent_bool: bool = True):
        """
        Load fission yield data into class instance. If `library` is None, find the best option for `energies`.

        Args:
            target: Fission target nucleus
            inducing_par: None for SF. Or, e.g., 'proton', 'neutron', 'gamma'
            energies: If None, use energies from the data file (in MeV).
            library: Fission yield library. See FissionYields.FISSION_YIELD_SUBDIRS for available yield libraries.
            independent_bool:
        """
        if isinstance(energies, (float, int)):
            energies = [energies]
        if hasattr(energies, '__iter__') and len(energies) == 0:
            warn('length of `energies` is zero! Falling back on default library energy points.')
            energies = None
        assert isinstance(target, str), f'Bar `target` argument type, {type(target)}'

        self.target = target
        yield_dirs = FissionYields.FISSION_YIELD_SUBDIRS
        if inducing_par is None:
            inducing_par = 'sf'
        else:
            assert isinstance(inducing_par, str)
            inducing_par = inducing_par.lower()
        if inducing_par == 'sf':
            assert energies is None, 'Cannot specify energies for SF'
            energies = [0]
        self.inducing_par = inducing_par
        self.independent_bool = independent_bool
        assert inducing_par in yield_dirs, f"No fission yield data for projectile, '{self.inducing_par}'.\n" \
                                           f"Your options are:\n{list(yield_dirs.keys())}"
        self.energies = energies
        if not hasattr(self.energies, '__iter__'):
            if self.energies is not None:
                assert isinstance(self.energies, Number), "Invalid `energies` argument."
                self.energies = [self.energies]
        self.library = library
        if self.library is not None:
            assert self.library in FissionYields.FISSION_YIELD_SUBDIRS[self.inducing_par], \
                f"Library '{self.library}' for {self.inducing_par}-induced fission doesn't exist!"
            self.__data: RawFissionYieldData = RawFissionYieldData(self.inducing_par, self.library, self.independent_bool,
                                                                   self.target)
        else:
            self.__data: RawFissionYieldData = self.__find_best_library__()  # sets self.library if data is found
            if self.library is None:  # data wasn't found
                raise FileNotFoundError(
                    f"No fission yield file for {self.inducing_par}-induced fission on {target}.")
        self.data_ergs = self.__data.ergs
        if self.energies is None:
            self.energies = self.data_ergs

        self.energies = np.array(self.energies)

        if self.__score(self.energies, self.data_ergs) != 1:
            warn(f"\nExtrapolation being used in the {inducing_par}-fission yields of {target} for energies greater "
                 f"than {self.data_ergs[-1]} and/or less than {self.data_ergs[0]}\n Plot yields to make \nsure results"
                 f"are reasonable")
        __yields_unsorted__ = {}  # data placeholder so as not to calculate again during sorting.
        _sorter = []  # used for sorting
        _keys_in_order = []
        for n, yield_ in self.__data.data.items():
            if len(self.__data.ergs) > 1:
                yield_nominal = self.interp(self.energies, self.__data.ergs, yield_[0])
                yield_error = self.interp(self.energies, self.__data.ergs, yield_[1])
                __yields_unsorted__[n] = unp.uarray(yield_nominal, yield_error)

            else:
                yield_nominal = [yield_[0][0]]
                __yields_unsorted__[n] = ufloat(yield_[0][0], yield_[1][0])

            tot_yield = sum(yield_nominal)
            i = np.searchsorted(_sorter, tot_yield)
            _sorter.insert(i, tot_yield)
            _keys_in_order.insert(i, n)

        self.yields = {k: __yields_unsorted__[k] for k in _keys_in_order[::-1]}
        self.__unweighted_yields = None
        self.__weighted_by_xs = False
        self.weights = np.ones_like(self.energies)

    def threshold(self, frac_of_max=0.02) -> List[str]:
        """
        Remove all yields that are less than frac_of_max*y_m, where y_m is the nucleus with maximum yield.
        Modifies self.
        Args:
            frac_of_max:

        Returns: List of nuclides that were excluded

        """
        if frac_of_max == 0.0:
            return []

        if self.inducing_par != 'sf':
            cut_off = sum(self.yields[list(self.yields.keys())[0]]) * frac_of_max

            def get_rel_yield():
                return sum(unp.nominal_values(self.yields[k]))

        else:
            cut_off = max(self.yields.values()).n * frac_of_max

            def get_rel_yield():
                return self.yields[k].n

        out = []
        for k in list(self.yields.keys()):
            if get_rel_yield() < cut_off:
                del self.yields[k]
                out.append(k)

        return out

    def get_yield(self, nuclide_name):
        """
        Simply return yield of nuclide.
        """

        try:
            assert isinstance(nuclide_name, str)
            return self.yields[nuclide_name]
        except KeyError:
            return ufloat(0, 0)

    @property
    def _is_weighted(self):
        return not (self.__unweighted_yields is None)

    @cached_property
    def mass_number_vs_yield(self) -> Dict:
        a_dict = {}
        for n, y in self.yields.items():
            mass_num = int(re.match("[A-Za-z]*([0-9]+)", n).groups()[0])
            try:
                a_dict[mass_num] += y
            except KeyError:
                a_dict[mass_num] = np.array(y)

        mass_nums = np.array(sorted(a_dict.keys()))
        a_dict = {k: a_dict[k] for k in mass_nums}
        return a_dict

    def plot_A(self, weights=None, at_energy=None, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        # else:
        leg_label = self.target
        title = f'Fragment mass distribution of {self.inducing_par}-induced fission of {self.target}'
        if self._is_weighted or weights is not None:
            title += ' (weighted)'
        if weights is None:
            if at_energy is not None:
                weights = np.zeros(len(self.energies))
                i = np.abs(at_energy - self.energies).argmin()
                weights[i] = 1
                title += f'\n {self.inducing_par} energy = {at_energy} MeV'
            else:
                weights = np.ones_like(self.energies)

        x = []
        y = []
        y_err = []
        for a, yield_ in self.mass_number_vs_yield.items():
            x.append(a)
            if self.inducing_par == 'sf':
                _y = np.sum(yield_)
                # print(type(_y), _y)
            else:
                _y = np.average(yield_, weights=weights)
            y.append(_y.n)
            y_err.append(_y.std_dev)

        if leg_label is None:
            ax.set_title(title)

        ax.set_xlabel('A')
        ax.set_ylabel('Yield per fission')
        ax.errorbar(x, y, y_err, label=leg_label)
        if leg_label is not None:
            ax.legend()
        return ax

    def plot(self, nuclide: Union[List[str], str, None] = None, first_n_nuclides=12, plot_data=False, c=1):

        assert self.inducing_par != "sf"
        if nuclide is not None:
            if not isinstance(nuclide, str):
                assert hasattr(nuclide, '__iter__')
                plot_keys = nuclide
            else:
                assert isinstance(nuclide, str)
                y = c*self.yields[nuclide]
                y_data = self.__data.data[nuclide]
                plt.errorbar(self.energies, unp.nominal_values(y), unp.std_devs(y), label=f'{nuclide}: interp')
                if plot_data and not self._is_weighted and c == 1:
                    plt.errorbar(self.__data.ergs, y_data[0], y_data[1], label=f'{nuclide}: data')
                plt.legend()
                return

        else:
            assert isinstance(first_n_nuclides, int)
            plot_keys = list(self.yields.keys())[:first_n_nuclides]
        for index, k in enumerate(plot_keys):
            axs_i = index % 4
            if axs_i == 0:
                fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
                axs = axs.flatten()
            ax = axs[axs_i]
            y_interp = unp.nominal_values(self.yields[k])
            y_err_interp = unp.std_devs(self.yields[k])
            y_data = self.__data.data[k][0]
            y_err_data = self.__data.data[k][1]
            ax.fill_between(self.energies, y_interp-y_err_interp, y_interp + y_err_interp,
                            color='red', alpha=0.4)
            interp_label = f'{k}: interp' if plot_data else k
            ax.plot(self.energies, y_interp, label=interp_label, ls='--', c='red',
                    marker='p' if not len(self.energies) else None)

            if plot_data and not self._is_weighted and c == 1:
                ax.errorbar(self.__data.ergs, y_data, y_err_data, label=f'{k}: data', ls='None', c='black',
                            marker='o')
            if axs_i > 1:
                ax.set_xlabel('Projectile Energy [MeV]')
            if axs_i in [0, 2]:
                ax.set_ylabel('Yield per fission')

            ax.legend()

    def undo_weighting(self):
        """Undo any weighting"""
        if self.__unweighted_yields is not None:
            self.yields = self.__unweighted_yields

            try:
                del self.mass_number_vs_yield
            except AttributeError:
                warn("Bug? ")

        else:
            warn("No weights to undo!")
        self.weights = np.ones_like(self.energies)
        self.__unweighted_yields = None
        self.__weighted_by_xs = False

    def weight_by_fission_xs(self):
        """
        Weight yield by cross section as a function of energy.

        Returns: cross-section data used to weight

        """
        xs: CrossSection1D
        n = Nuclide.from_symbol(self.target)
        if self.inducing_par == 'gamma':
            xs = n.gamma_induced_fiss_xs
        elif self.inducing_par == 'proton':
            xs = n.proton_induced_fiss_xs
        elif self.inducing_par == 'neutron':
            xs = n.neutron_induced_xs
        else:
            assert False, f"Cannot weight by fission cross-section for inducing particle '{self.inducing_par}'"
        xs_values = xs.interp(self.energies)
        self.weight_by_erg(xs_values)
        self.__weighted_by_xs = True
        return xs_values

    def weight_by_erg(self, weights):
        """
        Weight all yields (modifies self) by a value for each energy. e.g., a projectile energy distribution.
        Replace each yield with weights*yields.

        Args:
            weights: Array with same length of self.energies

        Returns:

        """
        if self.__unweighted_yields is None:
            self.__unweighted_yields = self.yields.copy()

        try:
            del self.mass_number_vs_yield  # undo any past weighting.
            # If weighting hasn't been performed, this will raise AttributeError
        except AttributeError:
            pass
        assert self.inducing_par != 'sf', "Cannot weight by energy for SF"
        assert len(weights) == len(self.energies), f"Weights must be the same length of self.energies, \n" \
                                                   f"len(self.energies) :{len(self.energies)} " \
                                                   f"!= len(weights): {len(weights)}"
        self.yields = {k: v*weights for k, v in self.yields.items()}
        self.yields = {k: v for k, v in sorted(self.yields.items(), key=lambda k_v: -np.sum(unp.nominal_values(k_v[1])))}

        # __sorter = []
        # __keys = []
        # for k, v in self.yields.items():
        #     new_values = v*weights
        #     n = np.sum(unp.nominal_values(new_values))
        #     i = np.searchsorted(__sorter, n)
        #     __sorter.insert(i, n)
        #     __keys.insert(i, k)
        #     self.yields[k] = new_values
        #
        # self.yields = {k: self.yields[k] for k in __keys[::-1]}
        self.weights = self.weights*weights

        return None

    def plot_weights(self, ax=None):
        if not self._is_weighted:
            warn("No weights. No plot")
            return
        if ax is None:
            plt.figure()
            ax = plt.gca()

        assert self.energies is not None, "No energies to weight. Is this SF?"
        ax.plot(self.energies, self.weights)
        ax.set_xlabel("Energy [MeV]")
        ax.set_ylabel("Weight")

    def __getitem__(self, item):
        try:
            return self.yields[item]
        except KeyError:
            return ufloat(0, 0)


class CrossSection1D:
    def __init__(self, ergs: List[float], xss: List[Union[UFloat, float]],
                 fig_label: str = None, incident_particle: str = 'particle', data_source='', mt_value=None,
                 **misc_data):
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
        self.data_source = data_source
        self.misc_data = misc_data
        self.mt_value = mt_value

    def cut(self, erg_min=None, erg_max=None):
        if erg_min is None:
            erg_min = self.ergs[0]

        if erg_max is None:
            erg_max = self.ergs[-1]

        sel = np.where((self.ergs <= erg_max) & (self.ergs >= erg_min))

        self.__ergs__ = self.__ergs__[sel]

        self.__xss__ = self.__xss__[sel]
        try:
            del self.xss
        except AttributeError:
            pass
        try:
            del self.ergs
        except AttributeError:
            pass

        return self

    @property
    def nominal_xs(self):
        return unp.nominal_values(self.xss)

    @cached_property
    def xss(self):
        return np.interp(self.ergs, self.__ergs__, self.__xss__)

    @cached_property
    def ergs(self):
        return self.__ergs__
        # return np.arange(self.__ergs__[0], self.__ergs__[-1], XS_BIN_WIDTH_INTERPOLATION)

    def threshold_erg(self, thresh_barn=50E-3):
        if max(self.xss) < thresh_barn:
            return np.inf
        return self.ergs[np.argmax(self.xss > thresh_barn)]

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
        out = np.interp(new_energies, self.ergs, self.xss)

        if not hasattr(new_energies, '__len__'):
            return out

        return out

    def plot(self, ax=None, fig_title=None, units="b", erg_min=None, erg_max=None, **mpl_kwargs):
        unit_convert = {"b": 1, "mb": 1000, "ub": 1E6, "nb": 1E9}
        try:
            unit_factor = unit_convert[units]
        except KeyError:
            assert False, "Invalid unit '{0}'. Valid options are: {1}".format(units, unit_convert.keys())
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_title(fig_title)
        elif ax is plt:
            ax = plt.gca()
            ax.set_title('')

        if erg_max is None:
            erg_max = self.__ergs__[-1]
        if erg_min is None:
            erg_min = self.__ergs__[0]
        selector = np.where((self.__ergs__ <= erg_max) & (self.__ergs__ >= erg_min))

        label = mpl_kwargs.pop('label', None)
        if label is None:
            try:
                src = self.data_source.lower() if isinstance(self.data_source, str) else ""
            except AttributeError:
                src = 'No src data'
            label = f"{self.__fig_label__}" + (f"({src})" if src else "")

        ax.plot(self.__ergs__[selector], (self.__xss__[selector]) * unit_factor, label=label, **mpl_kwargs)

        y_label = "Cross-section [{}]".format(units)
        x_label = "Incident {} energy [MeV]".format(self.__incident_particle__)

        if ax is plt:
            ax.ylabel(y_label)
            ax.xlabel(x_label)
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        if label != '':
            ax.legend()
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

        # self.m1_population_branching_ratio = 0
        # self.m2_population_branching_ratio = 0

    def is_mode(self, mode_str):
        return mode_str in self.modes

    def __repr__(self):
        out = "{0} -> {1} via {2} with BR of {3}".format(self.parent_name, self.daughter_name, self.modes,
                                                   self.branching_ratio)
        if not DEBUG:
            return out
        else:
            return f"{out}, Q-value: {self.q_value}"


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


def __nuclide_cut__(a_z_hl_cut: str, a: int, z: int, hl: UFloat, is_stable_only=False, m=0) -> bool:
    """
    Does a nuclide with a given z, a, and time_in_seconds (half life) make fit the criteria given by `a_z_hl_cut`?

    Args:
        a_z_hl_cut: The criteria to be evaluated as python code, where z=atomic number, a=mass_number,
            and time_in_seconds=half life in seconds
        a: mass number
        z: atomic number
        hl: half life in seconds
        is_stable_only: does the half life have to be effectively infinity?
        m: Excited level.

    Returns: bool

    """
    makes_cut = True

    assert isinstance(is_stable_only, bool)
    if is_stable_only:
        if hl is None:
            return False
        elif not np.isinf(nominal_value(hl)):
            return False

    if len(a_z_hl_cut) > 0:
        a_z_hl_cut = a_z_hl_cut.lower()
        if 'hl' in a_z_hl_cut and hl is None:
            makes_cut = False
        else:
            try:
                makes_cut = eval(a_z_hl_cut, {"hl": hl, 'a': a, 'z': z, 'm': m})
                assert isinstance(makes_cut, bool), "Invalid cut: {0}".format(a_z_hl_cut)
            except NameError as e:
                invalid_name = str(e).split("'")[1]

                raise Exception("\nInvalid name '{}' used in cut. Valid names are: 'z', 'a', and 'hl',"
                                " which stand for atomic-number, mass-number, and half-life, respectively."
                                .format(invalid_name)) from e

    return makes_cut


def decay_default_func(nuclide_name):
    """
    For the trivial case of stable nuclides.
    Also used for nuclides with no data

    Args:
        nuclide_name:

    Returns:

    """
    def func(ts, scale=1, decay_rate=False, *args, **kwargs):
        if decay_rate:
            scale = 0

        if hasattr(ts, '__iter__'):
            out = {nuclide_name: scale*np.ones_like(ts)}
        else:
            out = {nuclide_name: scale}

        return out

    return func


def decay_nuclide(nuclide_name: str, init_quantity=1., init_rate=None, driving_term=0.,
                  fiss_prod=False, fiss_prod_thresh=0, max_half_life=None):
    """
    Generate and return a function that solves the following problem:
        Starting with some amount (see init_quantity) of a given unstable nuclide, what amount of the parent and its
         progeny nuclides remain after time t?

    Solves the diff. eq y' == M.y, where M is a matrix derived from decay rates/branching ratios. Either y or y' can be
    returned (see

    Use driving_term arg for the case of nuclei being generated at a constant rate (in Hz), e.g. via a beam.
    A negative driving_term can be used, however, not that if the number of parent nuclei goes negative
    the solution is unphysical.

    Spontaneous fission products are not (yet) included in decay progeny.

    Solution is exact in the sense that the only errors are from machine precision.

    The coupled system of linear diff. egs. is solved "exactly" by solving the corresponding eigenvalue problem
        (or inhomogeneous system of lin. diff. eq. in the case of a nonzero driving_term).

    Args:
        nuclide_name: Name of parent nuclide.

        init_quantity: scalar representing the amount of parent nuclei at t=0 - i.e. the initial condition.

        init_rate: Similar to init_quantity, except specify a initial rate (decays/second).
            Only one of the two init args can be used

        driving_term: Set to non-zero number to have the parent nucleus be produced at a constant rate (in Hz).

        fiss_prod: If True, decay fission products as well.

        fiss_prod_thresh: Threshold of a given fission product yield (rel to highest yielded product).
            Between 0 and 1.0. 0 means accept all.

        max_half_life: A given decay chain will halt after reaching a nucleus with half life (in sec.) above this value.

    Returns:
        A function that takes a time (or array of times), and returns nuclide fractions at time(s) t.
        Return values of this function are of form: Dict[nuclide_name: Str, fractions: Union[np.ndarray, float]]
        See 'return_func' docstring below for more details.

    """

    nuclide = Nuclide.from_symbol(nuclide_name)
    if (not nuclide.is_valid) or nuclide.is_stable:
        return decay_default_func(nuclide_name)

    if fiss_prod:
        assert ('sf',) in nuclide.decay_modes, f"Cannot include fission product on non-SF-ing nuclide, {nuclide}"
        fission_yields = FissionYields(nuclide.name, None, independent_bool=True)
        fission_yields.threshold(fiss_prod_thresh)
    else:
        fission_yields = None

    assert isinstance(init_quantity, (float, int, type(None))), "`init_quantity` must be a float or int"
    assert isinstance(init_rate, (float, int, type(None))), "`init_rate` must be a float or int"
    assert not init_quantity is init_rate is None, "Only one of the two init args can be used"
    if init_rate is not None:
        init_quantity = init_rate/nuclide.decay_rate.n

    column_labels = [nuclide_name]  # Nuclide names corresponding to lambda_matrix.
    lambda_matrix = [[-nuclide.decay_rate.n]]  # Seek solutions to F'[t] == lambda_matrix.F[t]

    completed = set()

    def loop(parent_nuclide: Nuclide, decay_modes):
        if not len(decay_modes):  # or parent_nuclide.name in _comp:  # stable nuclide. Terminate recursion.
            return

        if parent_nuclide.name in completed:  # this decay chain has already been visited. No need to repeat.
            return

        # Loop through all decay channels
        for mode_name_tuple, modes in decay_modes.items():
            if not len(modes):
                continue

            if mode_name_tuple == ('sf',):
                if fiss_prod:
                    fiss_branching = modes[0].branching_ratio

                    modes = []  # new modes that mimics structure of typical decay but for all fission products.

                    for fp_name, y in fission_yields.yields.items():
                        _mode = type('', (), {})()
                        _mode.parent_name = parent_nuclide.name
                        _mode.daughter_name = fp_name
                        _mode.branching_ratio = y*fiss_branching.n
                        modes.append(_mode)
                else:
                    continue

            # A given decay channels (e.g. beta- -> gs or 1st excited state, also fission) can have multiple
            # child nuclides, so loop through them all.
            for mode in modes:

                parent_index = column_labels.index(mode.parent_name)

                child_nuclide = Nuclide.from_symbol(mode.daughter_name)

                child_lambda = child_nuclide.decay_rate.n

                try:
                    # index of row/column for child nuclide in lambda matrix.
                    child_index = column_labels.index(mode.daughter_name)
                    child_row = lambda_matrix[child_index]

                except ValueError:  # First time encountering this nuclide. Add new row/column to lambda-matrix.
                    column_labels.append(mode.daughter_name)
                    child_index = len(column_labels) - 1

                    for _list in lambda_matrix:
                        _list.append(0)  # add another column to maintain an nxn matrix

                    child_row = [0]*len(lambda_matrix[-1])  # create source(/sink) vector for current daughter nucleus

                    child_row[child_index] = -child_lambda  # Set entry for decay of child (diagonal term).

                    lambda_matrix.append(child_row)  # finally add new row to matrix.

                # Do not use += below. The parent feeding rate is a constant no matter how many times the same
                # parent/daughter combo is encountered.
                child_row[parent_index] = mode.branching_ratio.n*parent_nuclide.decay_rate.n  # parent feeding term

                if (max_half_life is not None) and child_nuclide.half_life.n > max_half_life:
                    continue  # don't worry about daughter bc nucleus decays too slow according to `max_half_life`
                else:
                    loop(child_nuclide, child_nuclide.decay_modes)  # recursively loop through all daughters

        completed.add(parent_nuclide.name)  # Add parent to list of completed decay chains to avoid repeats

    loop(nuclide, nuclide.decay_modes)  # initialize recursion.

    lambda_matrix = np.array(lambda_matrix)

    eig_vals, eig_vecs = np.linalg.eig(lambda_matrix)

    eig_vecs = eig_vecs.T

    b = [init_quantity] + [0.] * (len(eig_vals) - 1)

    if driving_term != 0:
        # coefficients of the particular solution (which will be added to homo. sol.)
        particular_coeffs = np.linalg.solve(-lambda_matrix, [driving_term] + [0.] * (len(eig_vals) - 1))
    else:
        particular_coeffs = np.zeros_like(eig_vals)  # No driving term. Will have no effect in this case.

    coeffs = np.linalg.solve(eig_vecs.T, b - particular_coeffs)  # solve for initial conditions

    def return_func(ts, scale=1, decay_rate=False, threshold=None, plot=False) -> Dict[str, np.ndarray]:
        """
        Evaluates the diff. eq. solution for all daughter nuclides at the provided times (`ts`).
        Can determine, as a function of time, the quantity of all decay daughters or the decay rate.
        This is controlled by the `decay_rate` argument.
        Args:
            ts: Times at which to evaluate the specified quantity

            scale: Scalar to be applied to the yield of all daughters.

            decay_rate: If True, return decays per second instead of fraction remaining. I.e. return y[i]*lambda_{i}

            threshold: Fraction of the total integrated yield below which solution arent included. None includes all.

            plot: Plot for debugging.

        Returns: dict of the form, e.g.:
            {'U235': [1., 0.35, 0.125],
             'Pb207': [0, 0.64, 0.87],
              ...}

        """
        if threshold is not None:
            raise NotImplementedError("Todo")

        if hasattr(ts, '__iter__'):
            iter_flag = True
        else:
            iter_flag = False
            ts = [ts]

        yields = [np.sum([c * vec * np.e ** (val * t) for c, vec, val in
                          zip(coeffs, eig_vecs, eig_vals)], axis=0) for t in ts]

        if driving_term != 0:
            yields += particular_coeffs

        yields = np.array(yields).T

        if not decay_rate:
            out = {name: scale*yield_ for name, yield_ in zip(column_labels, yields)}
        else:
            out = {name: scale*yield_*lambda_ for name, yield_, lambda_ in
                    zip(column_labels, yields, np.abs(np.diagonal(lambda_matrix)))}

        if not iter_flag:
            for k, v in out.items():
                out[k] = v[0]

        if plot:
            # if not (plot is True)
            assert iter_flag, 'Cannot plot for only one time'
            plt.figure()
            for k, v in out.items():
                plt.plot(ts, v, label=k)
            if decay_rate:
                plt.ylabel("Decays/s")
            else:
                plt.ylabel("Rel. abundance")
            plt.xlabel('Time [s]')
            plt.legend()

        return out

    return return_func


class Nuclide:
    """
    A nuclide object that can be used to access cross sections, decay children/parent nuclides, decay modes,
    and much more.

    Available data:
        Nuclide half-life
        Gamma emmissions (energy, intensity, uncertainties)
        Decay channels and the resulting child nuclides (+ branching ratios)
        Proton activation cross-sections (PADF)
        Neutron activation cross-sections (ENDF)
        Neutron, photon, proton, and SF fissionXS yeilds (use caution with photon and proton data)
        Neutron, photon, and proton fissionXS cross-sections


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
            ENDF fissionXS product yield evaluation to read from. If given as a
            string, it is assumed to be the filename for the ENDF file.

    Attributes
        half life : UFloat
        spin:
        mean_energies: Mean gamma decay energy
        is_stable:
        decay_gamma_lines: List of GammaLine object sorted from highest to least intensity.
        decay_modes: Dict mapping from modes of decay to DecayMode objects. Keys are Tuple[str], e.g. ('beta-','alpha')

        Notes
        -----
        """
    NUCLIDE_NAME_MATCH = re.compile(
        "(?P<s>[A-Za-z]{1,2})(?P<A>[0-9]{1,3})(?:_[me](?P<iso>[0-9]+))?")  # Nuclide name in GND naming convention

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

        self.excitation_energy = kwargs.get("excitation_energy", 0)
        self.half_life: UFloat = kwargs.get("half_life", None)
        self.spin: int = kwargs.get("spin", None)
        self.mean_energies = kwargs.get("mean_energies", None)  # look into this
        self.is_stable: bool = kwargs.get("is_stable", None)  # maybe default to something else
        self.decay_radiation_types = []

        self.__decay_daughters_str__: List[str] = kwargs.get("__decay_daughters__", [])  # self.decay_daughters -> List[Nuclide] in corresponding order as self.decay_modes

        self.__decay_gamma_lines: List[GammaLine] = []
        self.__decay_betaplus_lines: List[BetaPlusLine] = []

        self.decay_modes: Dict[Tuple[str], List[DecayMode]] = kwargs.get("decay_modes", {})
        self.__decay_parents_str__: List[str] = kwargs.get("__decay_parents__", [])  # self.decay_parents -> List[Nuclide]

        self.__decay_mode_for_print__ = None

    def adopted_levels(self):
        from nudel import LevelScheme
        return LevelScheme(f"{self.atomic_symbol}{self.A}")

    def potential_coincidence_summing(self):
        outs = []
        gamma_ergs = np.array([g.erg for g in self.decay_gamma_lines])
        for index, g1 in enumerate(self.decay_gamma_lines):
            for g2 in self.decay_gamma_lines[index+1:]:
                _sum = g1.erg + g2.erg
                err = _sum.std_dev
                arg_closest = np.argmin(abs(_sum-gamma_ergs))
                closest_erg = self.decay_gamma_lines[arg_closest].erg
                intensity = self.decay_gamma_lines[arg_closest].intensity
                if abs(closest_erg-_sum) < err:
                    outs.append(f'{closest_erg, intensity} (diff={closest_erg-_sum} KeV) is potential coinc. sum of {g1.erg, g1.intensity} and {g2.erg, g2.intensity}')
        return outs

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
                                                                           self.human_friendly_half_life(False),
                                                                           min_intensity*100))
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
        if not hasattr(self, '__Z_A_iso_state__'):
            self.__Z_A_iso_state__ = get_z_a_m_from_name(self.name)
        return self.__Z_A_iso_state__['Z']

    @property
    def N(self) -> int:
        return self.A-self.Z

    @property
    def isometric_state(self) -> int:
        """Meta stable excited state, starting with 0 as the ground state."""
        if not hasattr(self, '__Z_A_iso_state__'):
            self.__Z_A_iso_state__ = get_z_a_m_from_name(self.name)
        return int(self.__Z_A_iso_state__['M'])

    @property
    def A(self) -> int:
        """
        Returns: Mass number

        """
        if not hasattr(self, '__Z_A_iso_state__'):
            self.__Z_A_iso_state__ = get_z_a_m_from_name(self.name)
        return self.__Z_A_iso_state__['A']

    def human_friendly_half_life(self, include_errors: bool = True, abrev_units=True) -> str:
        """
        Gives the half life in units of seconds, hours, days, months, etc.
        Args:
            include_errors:  Whether to include uncertainties

        Returns:

        """
        return human_readable_half_life(self.half_life, include_errors, abrev_units)

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
        ev = 1.0/1.602176634E-19
        unit_dict = {'j': 1, 'mev': ev*1E-6, 'ev': ev}
        assert units in unit_dict.keys(), 'Invalid units, "{}".\nUse one of the following: {}'\
            .format(units, unit_dict.keys())
        j = self.atomic_mass*__u_to_kg__*__speed_of_light__**2
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
        z, a, m = get_z_a_m_from_name(self.name).values()
        a -= n_neutrons

        return (Nuclide.from_Z_A_M(z, a, m).mass_in_mev_per_c2 + n_neutrons*neutron_mass) - self.mass_in_mev_per_c2

    def proton_separation_energy(self, n_protons=1):
        """
        Min. energy required to remove n_protons from the nucleus (in MeV)
        Args:
            n_protons: Number of protons to obe removed.

        Returns:

        """
        z, a, m = get_z_a_m_from_name(self.name).values()
        z -= n_protons
        a -= n_protons
        return (Nuclide.from_Z_A_M(z, a, m).mass_in_mev_per_c2 + n_protons * proton_mass) - self.mass_in_mev_per_c2

    def alpha_separation_energy(self):
        """
        Min. energy required to remove He-4 from the nucleus (in MeV)

        Returns:

        """
        z, a, m = get_z_a_m_from_name(self.name).values()
        z -= 2
        a -= 2

        return (Nuclide.from_Z_A_M(z, a, m).mass_in_mev_per_c2 + Nuclide.from_symbol('He-4').mass_in_mev_per_c2) \
                - self.mass_in_mev_per_c2

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

    @staticmethod
    def isotopic_abundance(nuclide_name) -> float:
        _m = re.match('([A-Za-z]{1,2}[0-9]+)(?:m_[0-9]+)?', nuclide_name)
        if _m:
            s = _m.groups()[0]
            try:
                return NATURAL_ABUNDANCE[s]
            except KeyError:
                pass
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
                atomic weight data has been tabulated..
        Returns:

        """
        m = re.match('([A-Z][a-z]{0,2})([0-9]*_m[0-9])?', atomic_symbol)
        assert m, f"Invalid argument, '{atomic_symbol}'"
        atomic_symbol = m.groups()[0]

        l = len(atomic_symbol)
        outs = []
        if non_zero_abundance:
            options = NATURAL_ABUNDANCE.keys()
        else:
            options = nuclide_list()

        for k in options:
            if k[:l] == atomic_symbol:
                outs.append(k)
        return outs
        # outs = []
        # for f in DECAY_PICKLE_DIR.iterdir():
        #     if m := re.match(r"(.+)\.pickle", f.name):
        #         symbol = m.groups()[0]
        #         m = Nuclide.NUCLIDE_NAME_MATCH.match(symbol)
        #         if not m:
        #             continue
        #         s = m.group('s')
        #
        #         if s == atomic_symbol:
        #             A = m.group('A')
        #             if stable_only:
        #                 n = Nuclide.from_symbol(symbol)
        #                 if n.half_life is None:
        #                     continue
        #                 if not n.half_life > (1E2*365*24*60**2):
        #                     continue
        #             outs.append(f"{s}{A}")
        # return outs

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

    @classmethod
    def from_symbol(cls, symbol: str, discard_meta_state=False):
        """

        Args:
            symbol: e.g. 'Xe139', 'Ta180_m1"
            discard_meta_state: If True, discard meta stable state.

        Returns:

        """
        if discard_meta_state:
            symbol = symbol.split('_')[0]
        assert isinstance(symbol, str), '`symbol` argument must be a string.'

        if symbol in NUCLIDE_INSTANCES:  # check first thing for speed.
            return NUCLIDE_INSTANCES[symbol]

        if '-' in symbol:
            symbol = symbol.replace('-', '')
            if symbol.endswith('m'):
                symbol = symbol[:-1] + '_m1'

        if symbol.lower() in ['n', 'neutron']:
            symbol = 'N1'
        elif symbol.lower() == 'alpha':
            symbol = 'He4'
        elif symbol.lower() == 'proton':
            symbol = 'H1'

        _m = Nuclide.NUCLIDE_NAME_MATCH.match(symbol)

        assert _m, "\nInvalid Nuclide name '{0}'. Argument <name> must follow the GND naming convention, Z(z)a(_mi)\n" \
                   "e.g. Cl38_m1, n1, Ar40".format(symbol)

        if _m.groups()[2] == '0':  # ground state specification, "_m0", is redundant.
            symbol = _m.groups()[0] + _m.groups()[1]
            _m = Nuclide.NUCLIDE_NAME_MATCH.match(symbol)

        pickle_file = DECAY_PICKLE_DIR/(symbol + '.pickle')

        if symbol not in NUCLIDE_INSTANCES:
            if not pickle_file.exists():
                if symbol in additional_nuclide_data:
                    instance = Nuclide(symbol, __internal__=True, **additional_nuclide_data[symbol])
                    instance.is_valid = True

                else:
                    # warn("Cannot find data for Nuclide `{0}`. Data for this nuclide is set to defaults: None, nan, ect."
                    #      .format(symbol))
                    instance = Nuclide(symbol,  __internal__=True, half_life=ufloat(np.nan, np.nan))
                    instance.is_valid = False

            else:
                with open(pickle_file, "rb") as pickle_file:
                    instance = CustomUnpickler(pickle_file).load()
                    instance.is_valid = True
                NUCLIDE_INSTANCES[symbol] = instance

        else:
            instance = NUCLIDE_INSTANCES[symbol]
            instance.is_valid = True

        if instance.name == 'n1':
            instance.name = 'N1'

        return instance

    def __repr__(self):
        try:
            hl = self.human_friendly_half_life()
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

    def get_incident_proton_daughters(self, data_source=None, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('proton', a_z_hl_cut, is_stable_only, data_source)

    def get_incident_gamma_daughters(self, data_source=None, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('gamma', a_z_hl_cut, is_stable_only, data_source)

    def get_incident_gamma_parents(self, data_source=None, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedParent]:
        return self.__get_parents__('gamma', a_z_hl_cut, is_stable_only, data_source)

    def get_incident_neutron_daughters(self, data_source=None, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedDaughter]:
        return self.__get_daughters__('neutron', a_z_hl_cut, is_stable_only, data_source)

    def get_incident_neutron_parents(self, data_source=None, a_z_hl_cut='', is_stable_only=False) -> Dict[str, InducedParent]:
        return self.__get_parents__('neutron', a_z_hl_cut, is_stable_only, data_source)

    def __get_daughters__(self, projectile, a_z_hl_cut='', is_stable_only=False,
                          data_source: Union[str, None] = None):
        """
        Get all product nuclides (and cross-sections, ect.) from a  reaction specified by the path to the nuclide's
        pickle file for the given reaction.
        Args:
            projectile: eg 'proton', 'photon', 'neutron'
            a_z_hl_cut: string to be evaluated as python. Variables are
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
            if __nuclide_cut__(a_z_hl_cut, a, z, hl, is_stable_only, m):
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

    def is_effectively_stable(self, threshold_in_years: int=100) -> bool:
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

# def get_activation_products(e: Evaluation, projectile, debug=False) -> dict:
#     """
#     Get all activation products from ENDF Evaluation.
#     Notes:
#         MT 4: If r.product is same as target, this is production of ground state (i.e. same Z, A)
#                 If ZZAAA_e_1 is the product, then first level
#         MT 51: Different from ZZAAA_e_1 in MT=4 in that this is direct inelastic scattering (e.g. like setting isomer to ~0 in TALYS)
#
#     Args:
#         e:
#         projectile:
#         debug:
#
#     Returns:dict of form {'ergs': [...], 'xss': [...], 'mt'=int}
#
#     """
#     target = e.gnd_name
#     s, z, a, m = Nuclide.get_s_z_a_m_from_string(target)
#     # z = ATOMIC_NUMBER[s]
#
#     def to_string(z_, a_, m_=None):
#         return f"{ATOMIC_SYMBOL[z_]}{a_}" + ("" if (m_ is None or m_ == 0) else f"_m{m_}")
#
#     outs = {}
#
#     for mf, mt, _, _ in e.reaction_list:
#         if mf == 3:
#             if debug:
#                 print(f"MT = {mt}")
#
#             r = Reaction.from_endf(e, mt)
#
#             ergs = r.xs['0K'].x * 1E-6
#             xs = r.xs['0K'].y
#
#             for prod in r.products:
#                 if Nuclide.NUCLIDE_NAME_MATCH.match(prod.particle):
#                     if prod.particle == target:  # Not sure how to interpret this, so ignore.
#                         continue
#                     if '_e' in prod.particle:  # Handle this in a special case
#                         continue
#                     yield_y = prod.yield_.y
#                     yield_x = prod.yield_.x * 1E-6
#
#                     yield_y = np.interp(ergs, yield_x, yield_y)
#                     prod_xs = xs * yield_y
#
#                     outs[prod.particle] = {'ergs': ergs, "xss": prod_xs, "mt": mt}
#
#                     if debug:
#                         print(f"\t{prod} [included]")
#                 else:
#                     if debug:
#                         print(f"\t{prod} [ignored]")
#
#     if projectile == 'neutron':
#         r = Reaction.from_endf(e, 102)
#         try:
#             outs[to_string(z, a + 1, m)] = find_xs(r, 'photon')
#         except FileNotFoundError:
#             pass
#
#         r = Reaction.from_endf(e, 4)
#         for prod in r.products:
#             if m := re.match(".+_e([0-9]+)", prod.particle):
#                 iso = int(m.groups()[0])
#                 outs[to_string(z, a, iso)] = find_xs(r, prod.particle)
#
#     if debug:
#         tab = None
#         i = 0
#
#         for prod, xs_data in outs.items():
#             if i % 10 == 0:
#                 tab = TabPlot()
#             ax = tab.new_ax(f"{prod} (MT {xs_data['mt']})")
#             ax.plot(xs_data["ergs"], xs_data["xss"])
#
#             i += 1
#
#     return outs


class NuclearLibraries:
    # list of nuclear library sources for each incident particle. The order in which a given data source is
    # first used (in endf_to_pickle.py) determines the order in which they appear in `libraries`,
    # which in turn determines priority.
    xs_libraries = {'proton': ['endf', 'tendl', 'padf'],
                    'gamma': ['endf', 'tendl'],
                    'neutron': ['endf', 'tendl']}


class ActivationReactionContainer:
    """
    Used for storing particle-induced data in pickle file. Imported in endf_to_pickle.py, and used in this module
    for unpickling. There is only one of these created per target/projective pair.

    Attributes:
        self.name: The name of the target nuclide.

        self.product_nuclide_names_xss: A dict mapping between target name and CrossSection1D objects for each of the
            targets activation reaction products.

        self.parent_nuclide_names: A list of names of all parent nuclides that can produce this residue nuclide
            (e.g. self) via activation. This allows traveling both directions in an activation chain.
    """
    # all_instances set in code below
    all_instances: Dict[str, Dict[str, Dict[str, ActivationReactionContainer]]] = {}
    directories: Dict[str, Path] = \
        {'proton': PROTON_PICKLE_DIR,
         'gamma': GAMMA_PICKLE_DIR,
         'neutron': NEUTRON_PICKLE_DIR}

    libraries = NuclearLibraries.xs_libraries

    for __proj, __list_of_libraries in libraries.items():
        all_instances[__proj] = {k: {} for k in __list_of_libraries}

    @staticmethod
    def list_targets(projectile, data_source):
        """
        List all the nuclei available for data_source (e.g. "ENDF", "TENDL")

        Returns:

        """
        path_dir = ActivationReactionContainer.directories[projectile]/data_source
        outs = []
        zs = []
        for path in path_dir.iterdir():
            name = path.name[:-len(path.suffix)]
            m = re.match("([A-Z][a-z]{0,3})[0-9]+", name)
            if m is None:
                continue
            try:
                z = ATOMIC_NUMBER[m.groups()[0]]
            except KeyError:
                continue
            zs.append(z)
            outs.append(path.name[:-len(path.suffix)])
        outs = [outs[i] for i in np.argsort(zs)]
        return outs

    def __init__(self, nuclide_name: str, projectile: str, data_source: str):
        """

        Args:
            nuclide_name:
            projectile:
            data_source:
        """
        data_source = data_source.lower()
        assert data_source in self.libraries[projectile], f'Data source "{data_source}" not included in ' \
                                                          'ActivationReactionContainer.libraries. ' \
                                                          'Add new source if needed.'
        self.projectile = projectile
        self.nuclide_name = nuclide_name
        self.product_nuclide_names_xss: Dict[str, CrossSection1D] = {}
        self.parent_nuclide_names: List[str] = []

        self.elastic_xs: Union[CrossSection1D, None] = None
        self.inelastic_xs: Union[CrossSection1D, None] = None
        self.total_xs: Union[CrossSection1D, None] = None

        self._evaluation: Union[Evaluation, None] = None  # Only set when instance is created by from_endf() factory.

        self._available_mts: Union[None, set] = None  # MT values in ENDF file. Not all are pickled, however (yet). Todo?

        ActivationReactionContainer.all_instances[projectile][data_source][nuclide_name] = self

        self.data_source = data_source

    @classmethod
    def fetch_xs(cls, parent_name, residue_name, projectile, data_source=None) -> CrossSection1D:
        """
        Find pickled CrossSection1D instance for specified reaction.
        Args:
            parent_name: Tame of target nuclide, e.g. C11
            residue_name: Name of residue nuclide, e.g. Be11
            projectile: "proton", "neutron", "gamma".
            data_source: If None, find best using library priority. Else, look for specific library.

        Returns:

        """
        assert projectile in cls.libraries, f'Invalid projectile, {projectile}.'
        assert data_source in cls.libraries[projectile], f"No library named {data_source} for projectile {projectile}."

        for data_source in (cls.libraries[projectile] if data_source is None else [data_source]):
            try:
                r = cls.from_pickle(parent_name, projectile, data_source)
                if residue_name in r.product_nuclide_names_xss:
                    return r.product_nuclide_names_xss[residue_name]
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                continue
        msg = "" if data_source is None else f" in library {data_source}"
        raise FileNotFoundError(f"Cannot find any xs data for {parent_name}({projectile}, X){residue_name}{msg}.")

    @classmethod
    def load(cls, nuclide_name, projectile, data_source) -> ActivationReactionContainer:
        """
        Like from_pickle, but combines multiple data_sources.
        Args:
            nuclide_name:
            projectile:"proton", "neutron", "gamma"
            data_source:
             "all" to combine all libraries, with more reliable data taking precedence,
             None to use only the default library (usually ENDF),
             or a specific library name to use just that library.

        Returns:

        """
        if isinstance(data_source, str):
            data_source = data_source.lower()

        reactions = []

        reaction = None

        if data_source is None:
            return cls.from_pickle(nuclide_name, projectile, data_source=None)

        elif data_source in cls.libraries[projectile]:
            return cls.from_pickle(nuclide_name, projectile, data_source=data_source)

        elif data_source == 'all':
            for library in cls.libraries[projectile]:
                try:
                    r = cls.from_pickle(nuclide_name, projectile, library)
                except FileNotFoundError:
                    continue
                if reaction is None:  # reaction is the obj that will be returned (after possible additions below)
                    reaction = r
                else:
                    reactions.append(r)
        else:
            if projectile not in cls.libraries:
                raise ValueError(f'Invalid/no data for projectile, "{projectile}"')
            raise FileNotFoundError(f'`data_source, "{data_source}" not found for projectile "{projectile}". '
                                    f'Available data libraries are:\n\t{cls.libraries[projectile]}\n, '
                                    f'or, you can specify "all"')

        # If we reach this far, then that means 'all' was specified as data source
        if reaction is None:
            raise FileNotFoundError(f'No data found for {projectile} on {nuclide_name}')

        for _other_reaction in reactions:
            for name, xs in _other_reaction.product_nuclide_names_xss.items():
                if name not in reaction.product_nuclide_names_xss:
                    reaction.product_nuclide_names_xss[name] = xs
            reaction.parent_nuclide_names.extend([pname for pname in _other_reaction.parent_nuclide_names
                                                  if pname not in reaction.parent_nuclide_names])

        return reaction

    @classmethod
    def from_pickle(cls, nuclide_name, projectile, data_source) -> ActivationReactionContainer:

        assert projectile in cls.libraries, f'No activation data for incident particle "{projectile}"'

        if data_source is None:  # default library is always the first element
            data_source = cls.libraries[projectile][0]

        if data_source not in cls.libraries[projectile]:
            raise FileNotFoundError(f'No data source "{data_source}" for projectile "{projectile}"')

        all_instances = cls.all_instances[projectile][data_source]  # shouldn't have KeyError here

        try:  # check RAM for existing instance
            existing_instance = all_instances[nuclide_name]
        except KeyError:  # not in RAM
            try:  # load existing pickle file
                pickle_path = cls.get_pickle_path(nuclide_name, projectile, data_source)
                with open(str(pickle_path), "rb") as f:
                    existing_instance = CustomUnpickler(f).load()
            except FileNotFoundError:  # no existing pickle file. Raise error
                raise FileNotFoundError(f'No {projectile} activation data for {nuclide_name} and data source '
                                        f'"{data_source}"')

            all_instances[nuclide_name] = existing_instance

        return existing_instance

    @staticmethod
    def __get_product_name__(n1, n2, projectile) -> str:
        """For a reaction like C13(p, X) -> B10, find X (e.g. a in this case for alpha). """
        _1 = Nuclide.from_symbol(n1)
        _2 = Nuclide.from_symbol(n2)
        z1, n1 = _1.Z, _1.N
        z2, n2 = _2.Z, _2.N
        z = z1-z2
        n = n1-n2
        if projectile == 'proton':
            z += 1
        if projectile == 'neutron':
            n += 1
        if z == n == 2:
            return r'$\alpha$'
        elif z == 1 == n:
            return 'd'
        elif z == 1 and n == 2:
            return 't'
        elif z > 0 and n == 0:
            return f'{z}p'
        elif z == 0 and n > 0:
            return f'{n}n'
        elif z == 2 and n == 1:
            return 'He3'

        return 'X'  # X for something

    def set_misc_xs(self):
        """
        Write non_elastic, elastic, and total if available.
        Returns:
        """

        assert isinstance(self._evaluation, Evaluation)

        def get_xy(mt):
            r = Reaction.from_endf(self._evaluation, mt)
            xs = r.xs["0K"]  # cross-sections at zero Kelvin
            return xs.x * 1E-6, xs.y

        non_elastic = 3

        fig_label = f"{self.nuclide_name}({self._get_projectile_short}, inelastic)"

        x, y = None, None

        if self.projectile == 'neutron':
            xtot, ytot = get_xy(1)
            xel, yel = get_xy(2)

            x = xel
            if not (xel[0] > xtot[0] and xel[-1] < xtot[-1]):
                x = xtot
                yel = np.interp(x, xel, yel)
            else:
                ytot = np.interp(x, xtot, ytot)
            y = ytot - yel
        else:
            if non_elastic in self._available_mts:
                x, y = get_xy(non_elastic)

        if x is not None:
            self.inelastic_xs = CrossSection1D(x, y, fig_label, self.projectile, self.data_source)

        if 2 in self._available_mts:
            x, y = get_xy(2)
            fig_label = f"{self.nuclide_name}({self._get_projectile_short}, elastic)"
            self.elastic_xs = CrossSection1D(x, y, fig_label, self.projectile, self.data_source)

        if 1 in self._available_mts:
            x, y = get_xy(1)
            fig_label = f"{self.nuclide_name}({self._get_projectile_short}, total)"
            self.total_xs = CrossSection1D(x, y, fig_label, self.projectile, self.data_source)

    @property
    def _get_projectile_short(self):
        try:
            return {'proton': 'p', 'neutron': 'n', 'gamma': 'g', 'electron': 'e', 'alpha': 'a'}[self.projectile]
        except KeyError:
            assert False, 'Invalid incident particle: "{}"'.format(self.projectile)

    @staticmethod
    def _find_yield(r, product):
        out = ActivationReactionContainer._get_xs(r)

        for prod in r.products:
            if prod.particle == product:
                break
        else:
            raise FileNotFoundError(f"No product {product} for MT {r.mt}")

        yield_y = prod.yield_.y
        yield_x = prod.yield_.x * 1E-6

        yield_y = np.interp(out['ergs'], yield_x, yield_y)
        out['xss'] *= yield_y

        return out

    @staticmethod
    def _get_xs(r: Reaction):
        """
        Get cross-section from Reaction instance
        Args:
            r:

        Returns:

        """
        try:
            ergs = r.xs['0K'].x * 1E-6
            xs = r.xs['0K'].y
        except KeyError:
            raise FileNotFoundError

        return {'ergs': ergs, "xss": xs, "mt": r.mt}

    @staticmethod
    def _get_activation_products(e: Evaluation, projectile, debug=False) -> dict:
        """
        Get all activation products from ENDF Evaluation.
        Notes:
            MT 4: If r.product is same as target, this is production of ground state (i.e. same Z, A)
                    If ZZAAA_e_1 is the product, then first level
            MT 51: Different from ZZAAA_e_1 in MT=4 in that this is direct inelastic scattering (e.g. like setting isomer to ~0 in TALYS)

        Args:
            e:
            projectile:
            debug:

        Returns:dict of form {'ergs': [...], 'xss': [...], 'mt'=int}

        """
        if debug:
            from JSB_tools import TabPlot

        target = e.gnd_name
        s, z, a, m = Nuclide.get_s_z_a_m_from_string(target)
        z = ATOMIC_NUMBER[s]

        def to_string(z_, a_, m_=None):
            return f"{ATOMIC_SYMBOL[z_]}{a_}" + ("" if (m_ is None or m_ == 0) else f"_m{m_}")

        outs = {}

        for mf, mt, _, _ in e.reaction_list:
            if mf == 3:
                if debug:
                    print(f"MT = {mt}")
                try:
                    r = Reaction.from_endf(e, mt)
                except IndexError:
                    warn(f"openmc bug: Creating Reaction for {projectile} on {e.target['zsymam']}")
                    continue

                ergs = r.xs['0K'].x * 1E-6
                xs = r.xs['0K'].y

                for prod in r.products:
                    if Nuclide.NUCLIDE_NAME_MATCH.match(prod.particle):
                        if prod.particle == target:  # Not sure how to interpret this, so ignore.
                            continue
                        if '_e' in prod.particle:  # Handle this in a special case
                            continue
                        yield_y = prod.yield_.y
                        yield_x = prod.yield_.x * 1E-6

                        yield_y = np.interp(ergs, yield_x, yield_y)
                        prod_xs = xs * yield_y

                        outs[prod.particle] = {'ergs': ergs, "xss": prod_xs, "mt": mt}

                        if debug:
                            print(f"\t{prod} [included]")
                    else:
                        if debug:
                            print(f"\t{prod} [ignored]")

        if projectile == 'neutron':  # radiative capture
            r = Reaction.from_endf(e, 102)
            try:
                outs[to_string(z, a + 1, m)] = ActivationReactionContainer._get_xs(r)
            except FileNotFoundError:
                pass

            r = Reaction.from_endf(e, 4)
            for prod in r.products:
                if m := re.match(".+_e([0-9]+)", prod.particle):
                    iso = int(m.groups()[0])
                    outs[to_string(z, a, iso)] = ActivationReactionContainer._find_yield(r, prod.particle)

        if debug:
            tab = None
            i = 0

            for prod, xs_data in outs.items():
                if i % 10 == 0:
                    tab = TabPlot()
                ax = tab.new_ax(f"{prod} (MT {xs_data['mt']})")
                ax.plot(xs_data["ergs"], xs_data["xss"])

                i += 1

        return outs

    class EvaluationException(Exception):
        pass

    @classmethod
    def from_endf(cls, endf_path, projectile, data_source: str, debug=False):
        """
        Build the instance from ENDF file using openmc. Instance is saved to ActivationReactionContainer.all_instances
        Args:
            endf_path: Path to relevant target nuclide endf file

            projectile: incident particle

            data_source: Source of data, e.g. 'ENDF', or 'TENDL'. Default 'None'. Determines the name of directory of
                pickled data.

            debug: Plots cross-sections and prints some useful information.

        Returns: None

        """
        endf_path = Path(endf_path)
        assert endf_path.exists(), endf_path

        try:
            e = Evaluation(endf_path)
        except (ValueError, KeyError):
            raise ActivationReactionContainer.EvaluationException(f"Could not create Evaluation for {endf_path}")

        nuclide_name = e.gnd_name

        print(f'Read data from {data_source} for {projectile}s on {nuclide_name}')

        try:
            all_instances = ActivationReactionContainer.all_instances[projectile][data_source]
        except KeyError:
            raise KeyError("Need to add particle/data_source to NuclearLibraries stucture. "
                           "(search for NuclearLibraries)")

        try:
            self = all_instances[nuclide_name]
        except KeyError:
            self = ActivationReactionContainer(nuclide_name, projectile, data_source)

        self._evaluation = e
        self._available_mts = set([x[1] for x in self._evaluation.reaction_list])

        self.set_misc_xs()

        par_id = self._get_projectile_short

        for activation_product_name, reaction_dict in ActivationReactionContainer._get_activation_products(e, projectile, debug).items():
            if activation_product_name == nuclide_name:
                continue

            _product_label = cls.__get_product_name__(nuclide_name, activation_product_name, projectile)

            xs_fig_label = f"{self.nuclide_name}({par_id},{_product_label}){activation_product_name}"

            xs = CrossSection1D(reaction_dict['ergs'], reaction_dict['xss'], xs_fig_label,
                                self.projectile,  self.data_source, reaction_dict['mt'])

            self.product_nuclide_names_xss[activation_product_name] = xs

            try:
                daughter_reaction = all_instances[activation_product_name]
            except KeyError:  # initialize fresh instance
                daughter_reaction = ActivationReactionContainer(activation_product_name, self.projectile,
                                                                self.data_source)

            daughter_reaction.parent_nuclide_names.append(self.nuclide_name)

        return self

    @staticmethod
    def get_pickle_path(nuclide_name, projectile, data_source):
        path = ActivationReactionContainer.directories[projectile] / data_source
        path = path / nuclide_name
        path = path.with_suffix('.pickle')
        return path

    @property
    def pickle_path(self):
        return self.get_pickle_path(self.nuclide_name, self.projectile, self.data_source)

    def __pickle__(self):
        print(f'Creating and writing {self.pickle_path.relative_to(self.pickle_path.parents[3])}')
        path = self.pickle_path

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def pickle_all(projectile, data_source=None):
        for _data_source, _dict in ActivationReactionContainer.all_instances[projectile].items():
            if data_source is None or data_source == _data_source:
                for nuclide_name, reaction in _dict.items():
                    reaction.__pickle__()

    @staticmethod
    def __bug_test__(openmc_reaction: Reaction, openmc_product: Product, nuclide_name, incident_particle):
        """When activation_product.yield_.y == [1, 1], it indicates what seems to be a bug for ( or at least for)
        (G, 1n) reactions in fissionable nuclides. In this case (again, at least) the correct yield can be found
         by accessing the xs attribute on the openmc.Reaction instance itself."""
        activation_product_name = openmc_product.particle
        warn_other = False
        try:
            if len(openmc_product.yield_.y) == 2 and all(openmc_product.yield_.y == np.array([1, 1])):
                one_less_n_name = Nuclide.from_symbol(nuclide_name).remove_neutron().name
                warn_other = True
                if activation_product_name == one_less_n_name:
                    if incident_particle == 'gamma':
                        try:
                            yield_ = openmc_reaction.xs['0K']
                            openmc_product.yield_.y = yield_.y
                            openmc_product.yield_.x = yield_.x
                            warn(f'Bad (G, 1n) yield (ie [1., 1.]) for {nuclide_name}. Correcting'
                                 f' (see __bug_test__ in __init__.py)')
                            return True
                        except KeyError:
                            pass

        except AttributeError:
            pass
        if warn_other:
            warn(f'Bad yield (ie [1., 1.])  for {nuclide_name}({incident_particle}, X){activation_product_name}')

    def __len__(self):
        return len(self.parent_nuclide_names) + len(self.product_nuclide_names_xss)

    def __repr__(self):
        return 'self: {0}, parents: {1}, daughters: {2}'.format(self.nuclide_name, self.parent_nuclide_names,
                                                                self.product_nuclide_names_xss.keys())


if __name__ == "__main__":
    print(Nuclide.NUCLIDE_NAME_MATCH.match("U238").groups('A'))
    # n = Nuclide.from_symbol("C13")
    # print(Nuclide.from_symbol('U235').decay_daughters)
    # times = np.linspace(0, Nuclide.from_symbol('U235').half_life.n*3, 30)
    # # f = decay_nuclide('U235')
    # # print(f(times))
    # f0 = decay_nuclide('U235', init_quantity=10)
    # f1 = decay_nuclide('U235', init_quantity=10)
    # y0 = f0(times)
    # y1 = f1(times)
    # fig, (ax0, ax1) = plt.subplots(1, 2)
    # for k in y0:
    #     if sum(y0[k]) < 1E-7:
    #         continue
    #     ax0.plot(times, unp.nominal_values(y0[k]), label=k)
    #     ax1.plot(times, unp.nominal_values(y1[k]), label=k)
    # ax1.legend()
    # ax0.legend()
    # #
    # plt.show()
    # print()

    # talys_calculation('C13', 'g')
    # f = decay_nuclide('Xe139', True)
