from __future__ import annotations
from typing import Tuple, Dict, List, Union, Collection
import numpy as np
from numbers import Number
from JSB_tools import TabPlot
from JSB_tools.nuke_data_tools.nuclide import DecayNuclide
from JSB_tools.nuke_data_tools.nuclide.cross_section import CrossSection1D
from scipy.interpolate import interp1d
from warnings import warn
from uncertainties import ufloat
from uncertainties import unumpy as unp
import re
import matplotlib.pyplot as plt
from pathlib import Path
from JSB_tools.nuke_data_tools.nuclide.data_directories import FISS_YIELDS_PATH
import marshal
from functools import cached_property


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
        n = Nuclide(nuclide_name)
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
                _n = Nuclide(par)
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
    def mass_number_vs_yield(self) -> Dict[int, np.ndarray]:
        """
        Sorted dict of the form, e.g.
            {139: [0.02, 0.023, ...],
             136: [...],
              ... and so on ...}
        where in this example mass 139 has the highest total yields.
        Returns:

        """
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

    def plot_mass_yield(self, energies=None):
        """

        Args:
            energies:

        Returns:

        """
        if energies is None:
            di = max(1, int(len(self.energies)/10))
            indices = [i for i in range(0, len(self.energies), di)]
        else:
            indices = np.searchsorted(self.energies, energies)

        tab = TabPlot()

        xy = {}

        for A, yields in self.mass_number_vs_yield.items():
            for i in indices:
                try:
                    xy[i][A] = yields[i]
                except KeyError:
                    xy[i] = {A: yields[i]}

        for i, vals in xy.items():
            ax = tab.new_ax(f"E={self.energies[i]:.1f}")
            x = list(vals.keys())
            y = list(vals.values())
            yerr = unp.std_devs(y)
            y = unp.nominal_values(y)

            yerr = np.where(yerr/y < 0.6, yerr, 0.6*y)

            ax.errorbar(x, y, yerr, ls='None', marker='.')

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
                fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex='all')
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
                pass

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
        n = Nuclide(self.target)
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

    def weight_by_erg(self, weights, integrate=False, nominal_values_sum=False):
        """
        Weight all yields (modifies self) by a value for each energy. e.g., a projectile energy distribution.
        Replace each yield with weights*yields.

        Args:
            weights: Array with same length of self.energies
            integrate: If True, replace list of yields with single number (the weighted sum).
            nominal_values_sum: If True, the simple_sum procedure becomes sum(unp.nominal_values)

        Returns:

        """
        if nominal_values_sum:
            assert integrate, "nominal_values_sum is only to be used if `integrate`=True"

        if self.__unweighted_yields is None and (not integrate):
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

        if integrate:
            if nominal_values_sum:
                self.yields = {k: [sum(unp.nominal_values(v)*weights)] for k, v in self.yields.items()}
            else:
                self.yields = {k: [sum(v*weights)] for k, v in self.yields.items()}
            self.energies = np.array([np.average(self.energies, weights=weights)])
            self.__unweighted_yields = None
        else:
            self.yields = {k: v*weights for k, v in self.yields.items()}
        self.yields = {k: v for k, v in sorted(self.yields.items(), key=lambda k_v: -np.sum(unp.nominal_values(k_v[1])))}

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

            decay_func = DecayNuclide(n_name)
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

