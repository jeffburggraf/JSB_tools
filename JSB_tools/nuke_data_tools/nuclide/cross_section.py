from __future__ import annotations
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, List, Union, Tuple, Set
from JSB_tools.nuke_data_tools.nuclide.data_directories import pickle_dir, all_particles, activation_directories
import re
import JSB_tools.nuke_data_tools.nuclide as nuclide_module
from JSB_tools.nuke_data_tools.nudel import LevelScheme
import pickle
from logging import warning as warn
from JSB_tools.nuke_data_tools.nuclide.gen_mt_dict import mt_data
try:
    from openmc.data import ATOMIC_NUMBER, ATOMIC_SYMBOL, Tabulated1D, Evaluation, Reaction, Product, \
        ResonancesWithBackground, IncidentNeutron
except ModuleNotFoundError:
    openmc = None
    from JSB_tools import no_openmc_warn
    no_openmc_warn()

TOTAL_MTS = [2, 4, 5, 11, 16, 17, 18, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 41, 42, 44, 45, 102,
             103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'CrossSection1D':
            return CrossSection1D

        elif name == 'ActivationCrossSection':
            return ActivationCrossSection

        elif name == 'ActivationReactionContainer':
            return ActivationReactionContainer

        return super().find_class(module, name)


class CrossSection1D:
    # todo: Make attribs for parent and daughter nucleus (do this during pickling)
    class Identity(Tabulated1D):
        def __init__(self):
            super().__init__([0, 1], [1, 1])

        def __call__(self, x):
            return np.ones_like(x)

    identity = Identity()

    def get_reaction_name(self, mt=None):
        if mt is None:
            mt = self.mt_value
        try:
            return f'({self.__incident_particle__[0]}, {mt_data[mt]["name"]})'
        except KeyError:
            return f'({self.__incident_particle__[0]}, X)'

    def __init__(self, xs: Tabulated1D, yield_: Tabulated1D = None,
                 fig_label: str = None, incident_particle: str = 'particle', data_source='', mt_value=None,
                 endf_path=None, **misc_data):
        """
        A container for energy dependent 1-D cross-section
        Args:
            # x: energies for which cross-section is evaluated.
            # xss: Cross-sections corresponding to energies.
            fig_label: Figure label.
            incident_particle:

        """
        self.__xs__ = xs
        self.__yield__ = yield_
        self.__fig_label__ = fig_label
        self.__incident_particle__ = incident_particle
        self.data_source = data_source
        self.misc_data = misc_data
        self.mt_value = mt_value
        self.endf_path: Path = endf_path

    @property
    def emin(self):
        if self.__yield__ is None:
            if isinstance(self.__xs__, ResonancesWithBackground):
                return self.__xs__.background.x[0]
            return self.__xs__.x[0]
        else:
            return 1E-6 * max(self.__xs__.x[0], self.__yield__.x[0])

    @property
    def emax(self):
        if self.__yield__ is None:
            if isinstance(self.__xs__, ResonancesWithBackground):
                return self.__xs__.background.x[-1]
            return self.__xs__.x[-1]
        else:
            return 1E-6 * min(self.__xs__.x[-1], self.__yield__.x[-1])

    @property
    def ergs(self):
        return np.logspace(np.log10(self.emin), np.log10(self.emax), 250)

    def get_plot_ergs(self, npoints):
        return np.logspace(np.log10(self.emin), np.log10(self.emax), npoints)

    def __call__(self, ergs):
        """

        Args:
            ergs: Energy in MeV

        Returns:

        """
        out = self.__xs__(1E6 * ergs)
        if self.__yield__ is not None:
            out *= self.__yield__(1E6 * ergs)
        return out

    def plot(self, ergs=None, ax=None, npoints=3500, fig_title=None, units="b", erg_min=None, erg_max=None, return_handle=False,
             **mpl_kwargs):
        if ergs is None or isinstance(ergs, int):
            self.__call__(self.emin)  # induce TENDL fetch if resonance bug is present

            ergs = self.get_plot_ergs(npoints=npoints)

        if erg_max is not None:
            i1 = np.searchsorted(ergs, erg_max)
            ergs = ergs[:i1]

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

        label = mpl_kwargs.pop('label', None)
        y = self(ergs) * unit_factor

        if label is None:
            try:
                src = self.data_source.lower() if isinstance(self.data_source, str) else ""
            except AttributeError:
                src = 'No src data'
            label = f"{self.__fig_label__}" + (f"({src})" if src else "")

        handle = ax.plot(ergs, y, label=label, **mpl_kwargs)[0]

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
        if not return_handle:
            return ax
        else:
            return ax, handle

    def mean_xs(self, erg_low=None, erg_high=None, weight_callable=None):
        if erg_low is None and erg_high is None:
            ergs = self.ergs
            xss = self(ergs)
        else:
            if erg_high is None:
                erg_high = self.emax
            if erg_low is None:
                erg_low = self.emin
            cut = [i for i in range(len(self.ergs)) if erg_low <= self.ergs[i] <= erg_high]
            ergs = self.ergs[cut]
            xss = self(ergs)

        if weight_callable is not None:
            assert hasattr(weight_callable, "__call__"), "<weighting_function> must be a callable function that takes" \
                                                         " incident energy as it's sole argument"
            weights = [weight_callable(e) for e in ergs]
            if sum(weights) == 0:
                warn("weights in mean_xs for '{0}' summed to zero. Returning 0", )
                return 0
            return np.average(xss, weights=weights)
        else:
            return np.mean(xss)

    def __repr__(self):
        return f"{self.__fig_label__} ({self.data_source})"


class ActivationCrossSection(CrossSection1D):
    """
    Represents a case where several cross-sections contribute to the total cross-section for the production of a given
    product from a given parent.

    Behaves much like CrossSection1D instance.

    """
    def __init__(self, xss: Dict[int, Tabulated1D], yields: Dict[int, Union[None, List[Tabulated1D]]], fig_label: str = None,
                 incident_particle: str = 'particle', data_source='', mt_values=None, endf_path=None,
                 **misc_data):
        super().__init__(None, None, fig_label=fig_label, incident_particle=incident_particle,
                         data_source=data_source, mt_value=None, endf_path=endf_path, **misc_data)

        if yields is None:
            yields = {k: [CrossSection1D.identity] for k in xss.keys()}

        self.__xss__: Dict[int, Tabulated1D] = {}
        self.__yields__: Dict[int, List[Tabulated1D]] = {}

        for mt in xss:
            if mt not in yields:
                yields[mt] = None

            self._add_xs(mt, xss[mt], yields[mt])

        self.parent = self.product = None  # Set later (unfortunately, should fix by pickling this info)

    def set_TENDL_resonance(self):

        tendlr = ActivationReactionContainer.from_pickle(self.parent, 'neutron', 'tendl')

        for k, v in self.__xss__.items():
            try:
                new_xs = tendlr.product_nuclide_names_xss[self.product].__xss__[k]
                if hasattr(self.__xss__[k], 'resonances'):
                    self.__xss__[k].resonances = new_xs.resonances  # Replace bugged END wiht TENDKL
                # self.__xss__[k] = new_xs
            except KeyError:
                continue

        self.data_source = f'{self.data_source}/tendl'

    def __iadd__(self, other: ActivationCrossSection, exclude_MTs=None):
        for mt, xs in other.__xss__.items():
            if exclude_MTs is not None and mt in exclude_MTs:
                continue
            self._add_xs(mt, xs, other.__yields__[mt])

        return self

    def get_reaction_name(self, mt=None):
        if len(self.__xss__) == 1:
            mt = list(self.__xss__.keys())[0]
        else:
            mt = -1

        return super().get_reaction_name(mt=mt)

    def threshold(self, frac=0) -> float:
        """
        Return the lowest energy where the cross-section * yield increases beyond frac of max.

        Args:
            frac: Fraction of max val to be considered above threshold. 0 means first non-zero value.

        Returns:

        """
        # def find_thresh(tab, yields_, X=None):
        X = np.linspace(1, 30, 200)
        Y = self(X)
        val = frac * max(Y)
        i = np.argmax(Y>val)
        if i < len(X):
            return X[i]
        else:
            return X[-1]

    def __call__(self, ergs, mt_value=None, ith_channel=None):
        """
        Evaluate the cross-section at ergs. Default is to sum all channels.
        Can specify the ith channel of a given mt value.
        Args:
            ergs:
            mt_value: If None, sum all mts.
            ith_channel: if None, sum all channels for mt = mt_value

        Returns:

        """
        if mt_value is None:
            ith_channel = None

        if isinstance(ergs, int):
            ergs = float(ergs)

        out = np.zeros_like(ergs)
        ergs = 1E6 * ergs

        for _mt in self.mt_values:
            if mt_value is not None:
                if _mt != mt_value:
                    continue

            _call = self.__xss__[_mt]

            try:
                xs_y = _call(ergs)
            except AttributeError:  # bug
                # print()
                # raise
                if self.data_source != 'tendl':
                    self.set_TENDL_resonance()
                    return self.__call__(ergs=ergs*1E-6, mt_value=mt_value, ith_channel=ith_channel)

                if isinstance(_call, ResonancesWithBackground):
                    _call = self.__xss__[_mt] = _call.background
                    # self.__xss__[_mt] = self.__xss__[_mt]
                    xs_y = self.__xss__[_mt](ergs)
                else:
                    raise

            if self.__yields__[_mt] is not None:
                yield_y = np.zeros_like(xs_y)

                for i, yield_ in enumerate(self.__yields__[_mt]):
                    if ith_channel is not None:
                        if i != ith_channel:
                            continue

                    if yield_ is not None:
                        yield_y += yield_(ergs)
            else:
                yield_y = np.ones_like(xs_y)

            out += xs_y * yield_y

        return out

    @property
    def mts(self):
        return list(self.__xss__.keys())

    def _add_xs(self, mt: int, xs: Union[ResonancesWithBackground, Tabulated1D],
                yield_: Union[None, Tabulated1D, List[Tabulated1D]]):
        if yield_ is None:
            yield_ = CrossSection1D.identity

        try:
            if isinstance(yield_, list):
                self.__yields__[mt].extend(yield_)
            else:
                self.__yields__[mt].append(yield_)
        except KeyError:
            if isinstance(yield_, list):
                self.__yields__[mt] = [k for k in yield_]
            else:
                self.__yields__[mt] = [yield_]

        if isinstance(xs, ResonancesWithBackground):
            assert xs.mt == mt
            xs.x = xs.background.x

        self.__xss__[mt] = xs

    @property
    def mt_values(self):
        return set(self.__xss__.keys())

    def debug_plot(self):
        """
        Plots the xs and yields for each MT

        Returns:

        """
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
        self.plot(ax=ax1, plot_mts=True)

        x = self.ergs * 1E6

        for k, yields in self.__yields__.items():
            for i, y in enumerate(yields):
                ax2.scatter(x * 1E-6, y(x), label=f"MT={k} ({i})", s=2)

        ax2.set_ylabel('yields')
        ax2.set_xlabel(ax1.get_xlabel())
        ax1.set_xlabel('')

        ax2.legend()

    def plot(self, ergs=None, ax=None, npoints=3000, plot_mts=False, fig_title=None, units="b", erg_min=None, erg_max=None,
             return_handle=False, color=None,
             **mpl_kwargs):
        """

        Args:
            ergs:
            ax:
            plot_mts:
            fig_title:
            units:
            erg_min:
            erg_max:
            return_handle:
            **mpl_kwargs:

        Returns:

        """
        color_ = 'black' if plot_mts else mpl_kwargs.pop('c', color)
        ax, handle = super(ActivationCrossSection, self).plot(ergs=ergs, ax=ax, npoints=npoints, fig_title=fig_title, units=units,
                                                              erg_min=erg_min,
                                                              erg_max=erg_max, return_handle=True, lw=2,
                                                              c=color_, **mpl_kwargs)

        return ax

    @property
    def emin(self):
        out = None
        for xs in self.__xss__.values():
            if isinstance(xs, ResonancesWithBackground):
                min_ = min(xs.background.x)
            else:
                min_ = min(xs.x)
            if out is None or min_ < out:
                out = min_

        return 1E-6 * out

    @property
    def emax(self):
        out = None
        for xs in self.__xss__.values():
            if isinstance(xs, ResonancesWithBackground):
                max_ = max(xs.background.x)
            else:
                max_ = max(xs.x)

            if out is None or max_ > out:
                out = max_

        return 1E-6 * out


# dist of available libraries for each particle
activation_libraries = {k: list(v.keys()) for k, v in activation_directories.items()}


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
    parents_ = {}  # Dict of parents grouped by data source. Loaded during cls.from_pickle.

    # all_instances set in code below. Format is: {'neutron': {'endf': {'Ge74': self, ...} ...}, 'proton': {...}}
    all_instances: Dict[str, Dict[str, Dict[str, ActivationReactionContainer]]] = {}

    directories = {par: pickle_dir / 'activation' / par for par in all_particles}

    for __proj, __list_of_libraries in activation_libraries.items():
        all_instances[__proj] = {k: {} for k in __list_of_libraries}

    @staticmethod
    def list_targets(projectile, data_source):
        """
        List all the nuclei available for data_source (e.g. "ENDF", "TENDL")

        Returns:

        """
        path_dir = ActivationReactionContainer.directories[projectile] / data_source
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
        assert data_source in activation_libraries[projectile], f'Data source "{data_source}" not included in ' \
                                                                  'ActivationReactionContainer.libraries. ' \
                                                                  'Add new source if needed.'
        self.projectile = projectile
        self.nuclide_name = nuclide_name
        self.product_nuclide_names_xss: Dict[str, ActivationCrossSection] = {}
        self.parent_nuclide_names: Set[str] = set()

        self.elastic_xs: Union[CrossSection1D, None] = None
        self.nonelastic_xs: Union[CrossSection1D, None] = None
        self.total_xs: Union[CrossSection1D, None] = None

        self.path = None  # Path to ENDF file.

        self._evaluation: Union[Evaluation, None] = None  # Only set when instance is created by from_endf() factory.

        self._available_mts: Union[None, set] = None  # MT values in ENDF file.

        ActivationReactionContainer.all_instances[projectile][data_source][nuclide_name] = self

        self.data_source = data_source

    def delete(self):
        """
        Remove self from cls.all_instances for garbage collection.
        Returns:

        """
        try:
            del self.all_instances[self.projectile][self.data_source][self.nuclide_name]
        except KeyError:
            warn(f"KeyError where I wouldn't expect it...  for {self.projectile}'s on nucleus {self.nuclide_name} "
                 f"from library {self.data_source}")

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
        assert projectile in activation_libraries, f'Invalid projectile, {projectile}.'
        assert data_source in activation_libraries[projectile], \
            f"No library named {data_source} for projectile {projectile}."

        for data_source in (activation_libraries[projectile] if data_source is None else [data_source]):
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
            nuclide_name: e.g. 'U238'

            projectile:"proton", "neutron", "gamma"

            data_source: None or str with the following behavior:
                 "all": Combine all libraries, with more reliable data taking precedence.
                  None: Use only the default library (usually ENDF),
                  'library': Use ONLY data from a specific library, e.g. 'tendl'

        Returns:

        """
        if isinstance(data_source, str):
            data_source = data_source.lower()

        return_reaction = None

        if data_source is None:
            return cls.from_pickle(nuclide_name, projectile, data_source=None)

        elif data_source in activation_libraries[projectile]:
            return cls.from_pickle(nuclide_name, projectile, data_source=data_source)

        elif data_source == 'all':
            for library in activation_libraries[projectile][::-1]:  # reversed so that higher priority xs data overwrites lower
                # , e.g. ENDF overwrites TENDL.

                try:
                    r: ActivationReactionContainer = cls.from_pickle(nuclide_name, projectile, library)
                except FileNotFoundError:
                    continue

                if return_reaction is None:
                    return_reaction = r
                else:
                    for prod, xs in r.product_nuclide_names_xss.items():
                        return_reaction.product_nuclide_names_xss[prod] = xs  # overwrites value if already exists.

            if return_reaction is None:
                raise FileNotFoundError(f'No data found for {projectile} on {nuclide_name}')

            return return_reaction

        else:
            if projectile not in cls.libraries:
                raise ValueError(f'Invalid/no data for projectile, "{projectile}"')
            raise FileNotFoundError(f'`data_source, "{data_source}" not found for projectile "{projectile}". '
                                    f'Available data libraries are:\n\t{cls.libraries[projectile]}\n, '
                                    f'or, you can specify "all"')

    def set_misc_attribs(self):
        for k, v in self.product_nuclide_names_xss.items():
            if not hasattr(v, 'product'):
                v.parent = self.nuclide_name
                v.product = k

    @classmethod
    def from_pickle(cls, nuclide_name, projectile, data_source) -> ActivationReactionContainer:
        """

        Args:
            nuclide_name: 'neutron', 'proton', 'gamma', etc...
            projectile:
            data_source:

        Returns:

        """

        assert projectile in activation_libraries, f'No activation data for incident particle "{projectile}"'

        if data_source is None:  # default library is always the first element
            data_source = activation_libraries[projectile][0]
        else:
            data_source = data_source.lower()

        if data_source not in activation_libraries[projectile]:
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
                                        f'"{data_source}"') from None

            all_instances[nuclide_name] = existing_instance

        if data_source not in cls.parents_:  # look for and  set activation parents
            try:
                with open(existing_instance.parents_dict_path, 'rb') as f:
                    cls.parents_[data_source] = pickle.load(f)
            except FileNotFoundError:
                cls.parents_[data_source] = {}

        try:  # update parents
            existing_instance.parent_nuclide_names.update(cls.parents_[data_source][nuclide_name])
        except KeyError:
            pass

        existing_instance.set_misc_attribs()

        return existing_instance

    @staticmethod
    def __get_product_name__(n1, n2, projectile) -> str:
        """For a reaction like C13(p, X) -> B10, find X (e.g. a in this case for alpha). """
        _1 = nuclide_module.Nuclide(n1)
        _2 = nuclide_module.Nuclide(n2)
        z1, n1 = _1.Z, _1.N
        z2, n2 = _2.Z, _2.N
        z = z1 - z2
        n = n1 - n2
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
        mts_labels = [(2, 'elastic')]   # nonelastic is not working (3, 'nonelastic')]  # (1, 'total')

        reactions = self._get_reactions(self._evaluation, self.projectile)

        for mt, label in mts_labels:
            if mt in reactions:
                fig_label = f"{self.nuclide_name}({self._get_projectile_short}, {label})"

                r = reactions[mt]
                xs = r.xs['0K']

                XS = CrossSection1D(xs, fig_label=fig_label, incident_particle=self.projectile,
                                    data_source=self.data_source, mt_value=mt, endf_path=self._evaluation.path)

                setattr(self, f'{label}_xs', XS)

        if self.total_xs is None:
            for res, xs in self.product_nuclide_names_xss.items():
                if self.total_xs is None:
                    fig_label = f"{self.nuclide_name}({self._get_projectile_short}, total)"
                    self.total_xs = ActivationCrossSection(xs.__xss__, xs.__yields__, fig_label=fig_label)
                else:
                    for mt in xs.__xss__.keys():
                        if mt in TOTAL_MTS:
                            self.total_xs._add_xs(mt, xs.__xss__[mt], xs.__yields__[mt])

    @property
    def _get_projectile_short(self):
        try:
            return {'proton': 'p', 'neutron': 'n', 'gamma': 'γ', 'electron': 'e', 'alpha': 'a'}[self.projectile]
        except KeyError:
            assert False, 'Invalid incident particle: "{}"'.format(self.projectile)

    @staticmethod
    def find_isomeric_gdr_name(name: str) -> Union[str, None]:
        """
        Starting from a nuclear isomer named in terms of excited level, e.g. Pt_193_e5, return the nucleus name
        in terms of isomeric number.

        e.g. 'Pa234_e2' -> 'Pa234_m1'  (the ~1 min half-life Pa234_m1 occurs at the second excited level of Pa234)

        Args:
            name:

        Returns: str

        """
        m = re.match(".+_e([0-9]+)", name)

        if not m:
            return None

        base_name = name.replace(f"_e{m.groups()[0]}", '')
        level_i = int(m.groups()[0])

        level = LevelScheme(base_name).levels[level_i]
        erg = level.energy.n
        hl = level.half_life.n

        for i in range(1, 5):
            n = nuclide_module.Nuclide(f"{base_name}_m{i}")
            if n.is_valid and np.isclose(erg, n.excitation_energy, rtol=0.15) and hl > 0.1:  # 0.1: arb. threshold
                return n.name

        return name  # leave name as is

    def pickle_fission_xs(self, r: Reaction):
        assert r.mt == 18

        fission_xs = list(r.xs.values())[0]

        xs_fig_label = f'{self.nuclide_name}({self.projectile[0].upper()},F)'
        xs = CrossSection1D(fission_xs, None, fig_label=xs_fig_label, incident_particle=self.projectile,
                            data_source=self.data_source, mt_value=18, endf_path=self.path)
        path = self.pickle_path.parent / 'fission_xs'
        path.mkdir(exist_ok=True, parents=True)
        with open(path / '{0}.pickle'.format(self.nuclide_name), 'wb') as f:
            pickle.dump(xs, f)

    @staticmethod
    def get_product_name(target, projectile, mt, ):
        """
        Deduce residue nucleus from target nucleus, projectile and MT value.
        Args:
            target:
            projectile:
            mt:

        Returns:

        """
        dz_in, dn_in = {'neutron': (0, 1), 'proton': (1, 0), 'gamma': (0, 0), 'alpha': (2, 2)}[projectile]
        target = nuclide_module.Nuclide(target)
        try:
            data = mt_data[mt]
            dz_out, dn_out, level = data['outz'], data['outn'], data['level']
        except KeyError:
            return None
        Z = target.Z + dz_in - dz_out
        N = target.N + dn_in - dn_out
        out = nuclide_module.Nuclide.from_Z_A_M(Z, Z + N).name
        if level > 0:
            out += f'_e{level}'
        return out

    @staticmethod
    def _get_reactions(e, projectile):
        if projectile == 'neutron':
            return IncidentNeutron.from_endf(e).reactions
        else:
            out = {}
            for mf, mt, _, _ in e.reaction_list:
                if mt in [2]:  # elastic scattering, don't use product paradigm.
                    continue

                if mf == 3:
                    try:
                        r = Reaction.from_endf(e, mt)
                    except IndexError:
                        warn(f"openmc bug: Creating Reaction for {projectile} on {e.target['zsymam']}")
                        continue
                    out[mt] = r

            return out

    def _get_activation_products(self, e: Evaluation, projectile, debug=False) -> Dict[str, ActivationCrossSection]:
        """
        Get all activation products from an ENDF Evaluation.
        Notes:
            MT 4: If r.product is same as target, this is production of ground state (i.e. same Z, A)
                  If ZZAAA_e_1 is the product, then first level
            MT 51: Different from ZZAAA_e_1 in MT=4 in that this is direct inelastic scattering (e.g. like setting isomer to ~0 in TALYS)

        Args:
            e:
            projectile:
            debug:

        Returns:dict of form {mt: {'xs': xs_tab, 'products': {'nuclide_name': yield_tab, 'nuclide_name': yield_tab}}}
            where tab means the object is an openmc.Tabulated1D instance.

        """
        if debug:
            from JSB_tools import TabPlot

        target = e.gnds_name
        # s, z, a, m = nuclide_module.Nuclide.get_s_z_a_m_from_string(target)

        outs = {}

        for mt, r in self._get_reactions(e, projectile).items():
            if mt == 18:  # fission is done separately
                self.pickle_fission_xs(r)
                continue

            xs: Tabulated1D = r.xs['0K']

            all_products = [p for p in r.products if nuclide_module.Nuclide.NUCLIDE_NAME_MATCH.match(p.particle)]

            _proj_name = self.get_product_name(target, projectile, mt)

            if _proj_name is not None:
                if _proj_name not in [p.particle for p in all_products]:
                    p = Product(_proj_name)
                    p._yield_ = CrossSection1D.identity
                    all_products.append(p)

            for prod in all_products:
                if prod.particle in ['neutron', 'photon']:
                    continue

                yield_: Tabulated1D = prod.yield_

                if hasattr(yield_, 'y') and all(yield_.y == 0):
                    continue

                if prod.particle == target:  # Not sure how to interpret this, so ignore.
                    continue

                if '_e' in prod.particle:  # Handle this in a special case
                    new_name = self.find_isomeric_gdr_name(prod.particle)
                    if new_name is None:
                        continue
                    else:
                        prod.particle = new_name

                try:
                    activation_cross_section = outs[prod.particle]

                except KeyError:
                    fig_label = f"{target}({projectile}, X){prod.particle}"
                    activation_cross_section = ActivationCrossSection({}, {}, fig_label,
                                                                      projectile, self.data_source, mt_values=[mt],
                                                                      endf_path=self.path)
                    outs[prod.particle] = activation_cross_section

                activation_cross_section._add_xs(mt, xs, yield_)

                if debug:
                    print(f"\t{prod} [included]")

        # if projectile == 'neutron':  # radiative capture
        #     r = Reaction.from_endf(e, 102)
        #     if '0K' in r.xs:
        #         m = nuclide_module.Nuclide.NUCLIDE_NAME_MATCH.match(target)
        #         prod_name = f"{m.group('s')}{int(m.group('A')) + 1}"
        #         outs[prod_name] = ActivationCrossSection({102: r.xs['0K']}, None,
        #                                                  fr"{target}({projectile}, $\gamma$){prod_name}",
        #                                                  projectile, self.data_source, mt_values=[103],
        #                                                  endf_path=self.path)
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
    def from_endf(cls, endf_path, projectile, data_source: str, parents_dict=None, debug=False):
        """
        Build the instance from ENDF file using openmc. Instance is saved to ActivationReactionContainer.all_instances
        Args:
            endf_path: Path to relevant target nuclide endf file

            projectile: incident particle

            data_source: Source of data, e.g. 'ENDF', or 'TENDL'. Default 'None'. Determines the name of directory of
                pickled data.

            parents_dict: For keeping track of activation parents.

            debug: Plots cross-sections and prints some useful information.

        Returns: None

        """
        endf_path = Path(endf_path)
        assert endf_path.exists(), endf_path

        try:
            e = Evaluation(endf_path)
        except (ValueError, KeyError):
            raise ActivationReactionContainer.EvaluationException(f"Could not create Evaluation for {endf_path}")

        nuclide_name = e.gnds_name

        self = ActivationReactionContainer(nuclide_name, projectile, data_source)

        self.path = endf_path

        self._evaluation = e
        self._evaluation.path = endf_path

        self._available_mts = set([x[1] for x in self._evaluation.reaction_list])

        for product_name, reaction_cross_section in self._get_activation_products(e, projectile, debug).items():
            self.product_nuclide_names_xss[product_name] = reaction_cross_section

            if parents_dict is not None:
                try:
                    parents_dict[product_name].add(self.nuclide_name)
                except KeyError:
                    parents_dict[product_name] = {self.nuclide_name}

        self.set_misc_xs()

        return self

    @staticmethod
    def get_pickle_path(nuclide_name, projectile, data_source):
        path = ActivationReactionContainer.directories[projectile] / data_source
        path = path / nuclide_name
        path = path.with_suffix('.pickle')
        return path

    @staticmethod
    def get_parents_dict_path(projectile, data_source):
        return ActivationReactionContainer.directories[projectile] / data_source / 'parents.pickle'

    @property
    def parents_dict_path(self):
        return self.get_parents_dict_path(projectile=self.projectile, data_source=self.data_source)

    @property
    def pickle_path(self):
        return self.get_pickle_path(self.nuclide_name, self.projectile, self.data_source)

    def __pickle__(self):
        path = self.pickle_path

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        return path

    @staticmethod
    def pickle_all(projectile, library=None, paths=None):
        if paths is not None:
            paths = [Path(p) for p in paths]

        try:
            for _data_source, _dict in ActivationReactionContainer.all_instances[projectile].items():
                if (library is not None) and (library.lower() != _data_source.lower()):
                    continue

                for nuclide_name, reaction in _dict.items():
                    if paths is not None:
                        if len(paths) == 0:
                            raise StopIteration

                        if reaction.path not in paths:
                            continue
                        else:
                            paths.remove(reaction.path)

                    reaction.__pickle__()

        except StopIteration:
            pass

    @staticmethod
    def __bug_test__(openmc_reaction: Reaction, openmc_product: Product, nuclide_name, incident_particle):
        """When activation_product.yield_.y == [1, 1], it indicates what seems to be a bug for ( or at least for)
        (G, 1n) reactions in fissionable nuclides. In this case (again, at least) the correct yield can be found
         by accessing the xs attribute on the openmc.Reaction instance itself."""
        activation_product_name = openmc_product.particle
        warn_other = False
        try:
            if len(openmc_product.yield_.y) == 2 and all(openmc_product.yield_.y == np.array([1, 1])):
                one_less_n_name = Nuclide(nuclide_name).remove_neutron().name
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

    def __repr__(self):
        return f'<ActivationReactionContainer: {self.nuclide_name}> ({self.data_source})'


if __name__ == "__main__":
    from JSB_tools.nuke_data_tools import Nuclide
    Nuclide('H1').elastic_xs('neutron').plot()
    print()
    nuclide_module.Nuclide('In113').get_incident_neutron_daughters(data_source='TENDL')['In112'].xs.plot()

    ax = nuclide_module.Nuclide('In114').neutron_capture_xs().plot()
    print(nuclide_module.Nuclide('In114').thermal_neutron_capture_xs())
    ax.axvline(0.025*1E-6)
    print()
    plt.show()
    # ActivationReactionContainer.get_product_name('Ar41', 'neutron', 102)
    # from openmc.data import Evaluation, Reaction
    # e = Evaluation('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/nuclide/endf_files/ENDF-neutrons/n-018_Ar_040.endf')
    # r = Reaction.from_endf(e, 102)
    # xs = r.xs['0K']
    # xs.x
    # print()
    # e = Evaluation(
    #     '/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/nuclide/endf_files/TENDL-protons/p-U238.tendl')
    # mts = [11, 17, 33, 42]
    # for mt in mts:
    #     print(f"MT: {mt}")
    #     for p in Reaction.from_endf(e, mt).products:
    #         print('\t', p.particle, p)
    #
    # from JSB_tools.nuke_data_tools import Nuclide
    #
    # for k, v in Nuclide("U238").get_incident_proton_daughters(data_source='tendl').items():
    #     print(k, v.xs.mt_values, v.xs.data_source)
    #     if k == 'Ac222':
    #         v.xs.plot(plot_mts=True)
    #
    # # xs = Nuclide("Pb208").get_incident_proton_daughters(data_source='all')['He3'].xs
    # # xs.plot(plot_mts=True)
    # plt.show()
