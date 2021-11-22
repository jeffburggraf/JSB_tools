from __future__ import annotations
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from JSB_tools.MCNP_helper.geometry import MCNPNumberMapping, get_comment
from typing import Dict, Union
from JSB_tools.nuke_data_tools import Nuclide
from JSB_tools.MCNP_helper.atomic_data import atomic_weight, ATOMIC_NUMBER, atomic_mass
try:
    import openmc.material
except ModuleNotFoundError:
    from JSB_tools.nuke_data_tools import openmc_not_installed_warning
    openmc_not_installed_warning()
import numpy as np
import re
from typing import List
from numbers import Number
from typing import Tuple, List
from inspect import signature
# Todo: For windows report, get rid of openmc when it isn't absolutely necessary.


chemical_regex = re.compile(r'(?P<symbol>[A-Z][a-z]{0,2}-?(?P<A>[0-9]{0,3}))(?:\((?P<n_atoms>[0-9]+)\))?')


class ChemicalFormula:
    def __init__(self, formula: str):
        """

        Args:
            formula: Chemical formula, e.g. N(2), O(3) or Ar(1) or Ar
        """
        self.atomic_weights, self.atom_numbers = [], []
        for m in chemical_regex.finditer(formula):
            a = m.group('A')
            symbol = m.group('symbol')
            n_atoms = m.group('n_atoms')

            if a:
                self.atomic_weights.append(atomic_mass(symbol))
            else:
                self.atomic_weights.append(atomic_weight(symbol))

            if n_atoms:
                n_atoms = int(n_atoms)
            else:
                n_atoms = 1
            self.atom_numbers.append(n_atoms)
        self.atom_numbers = np.array(self.atom_numbers)
        self.atomic_weights = np.array(self.atomic_weights)
        self.total_grams_peer_mole = np.sum(self.atom_numbers*self.atomic_weights)


class _IdealGas:
    R = 8.3144626181

    def __init__(self, list_of_chemicals: List[str]):
        """

        Args:
            list_of_chemicals: e.g., ['O2', 'He'] for O2 and helium. ["O", "H2"] for free O and He.
        """
        list_of_chemical_formulas = [ChemicalFormula(s) for s in list_of_chemicals]
        self.total_grams_per_mole_list = np.array([a.total_grams_peer_mole for a in list_of_chemical_formulas])

    @staticmethod
    def __temp_and_pressure__(temp, pressure, temp_units, press_units):
        """Convert/Parse arguments"""
        temp_units = temp_units.lower()
        press_units = press_units.lower()
        if temp_units == 'k':
            temp = temp
        elif temp_units == 'c':
            temp = temp + 273.15
        elif temp_units == 'f':
            temp = (temp-32)*5/9+273.15
        else:
            assert False, 'Invalid units for temperature: {}'.format(temp_units)

        if press_units == 'atm':
            pressure = pressure*101325
        elif 'bar' in press_units:
            pressure = 1E5*pressure
        elif press_units == 'p':
            pressure = pressure
        else:
            assert False, 'Invalid units for pressure: {}'.format(press_units)
        return temp, pressure

    def get_density_from_mass_ratios(self, mass_ratios: List[Number], temperature=293.15, temp_units: str = 'K',
                                     pressure: float = 1, pressure_units: str = 'bars', n_sig_digits=4):
        temperature, pressure = self.__temp_and_pressure__(temperature, pressure, temp_units, pressure_units)
        mass_ratios = np.array(mass_ratios)
        assert len(mass_ratios) == len(self.total_grams_per_mole_list)
        norm = sum(mass_ratios)
        p_over_r_t = pressure/(_IdealGas.R * temperature)
        _x = np.sum((mass_ratios/norm)/self.total_grams_per_mole_list)
        out = 1E-6*p_over_r_t/_x
        fmt = '{' + ':.{}E'.format(n_sig_digits) + '}'

        out = float(fmt.format(out))
        return out

    def get_density_from_atom_fractions(self, atom_fractions: List[Number], temperature=293.15, temp_units: str = 'K',
                                        pressure: float = 1, pressure_units: str = 'bars', n_sig_digits=4):
        temperature, pressure = self.__temp_and_pressure__(temperature, pressure, temp_units, pressure_units)
        atom_fractions = np.array(atom_fractions)
        assert len(atom_fractions) == len(self.total_grams_per_mole_list)
        mean_g_per_mole = np.average(self.total_grams_per_mole_list, weights=atom_fractions)
        p_over_r_t = pressure/(_IdealGas.R * temperature)
        out = 1E-6*p_over_r_t*mean_g_per_mole  # 1E-6 converts from g/m3 to g/cm3
        fmt = '{' + ':.{}E'.format(n_sig_digits) + '}'
        out = float(fmt.format(out))
        return out

    def get_atom_fractions_from_mass_ratios(self, mass_ratios: List[Number]):
        mass_ratios = np.array(mass_ratios)
        assert len(mass_ratios) == len(self.total_grams_per_mole_list)
        norm = np.sum(mass_ratios/self.total_grams_per_mole_list)
        return mass_ratios/self.total_grams_per_mole_list/norm


def get_most_abundant_isotope(symbol):
    openmc_mat = openmc.material.Material()
    openmc_mat.add_element(symbol, 0.01)
    out = openmc_mat.nuclides[np.argmax([n[1] for n in openmc_mat.nuclides])][0]
    return out


class Material:
    all_materials = MCNPNumberMapping('Material', 1000, 1000)

    @property
    def _get_elements_and_fractions(self) -> Tuple[List[str], List[float]]:
        """
        sorted list of atomic symbols comprising material, e.g. ['Ar', 'He'], as well as the fractions.
        When isotopes appear, add up the fractions for the given element.
        (used for PHITS SRIM incorporation)
        Returns: (elements, fractions)
        """
        elements_dict = {}
        for zaid, frac in zip(self._zaids, self._zaid_proportions):
            z = zaid//1000
            a = zaid%1000
            s = Nuclide.from_Z_A_M(z, a).atomic_symbol
            try:
                elements_dict[s] += frac
            except KeyError:
                elements_dict[s] = frac
        fractions = np.array(list(elements_dict.values()))
        elements = np.array(list(elements_dict.keys()))
        arg_sort = np.argsort(elements)
        fractions = fractions[arg_sort] / sum(fractions)
        elements = elements[arg_sort]
        return elements, fractions

    @staticmethod
    def clear():
        Material.all_materials = MCNPNumberMapping('Material', 1000, 1000)

    @staticmethod
    def get_all_material_cards():
        cards = []
        for mat in Material.all_materials.values():
            mat: Material
            cards.append(mat.mat_card)
        return '\n'.join(cards)

    # @staticmethod
    # def all_materials() -> List[Material]:
    #     out = []
    #     for mat in Material.__all_materials.values():
    #         mat: Material
    #         out.append(mat)
    #     return out

    def __init__(self, density: float, mat_number: int = None, mat_name: str = None, mat_kwargs: Dict[str, str] = None,
                 is_mcnp=True):
        self.mat_number = mat_number
        self.__name__ = mat_name
        Material.all_materials[self.mat_number] = self
        self.density = density
        self._zaids = []
        self._zaid_proportions = []

        if mat_kwargs is None:
            self.mat_kwargs = {}
        else:
            self.mat_kwargs = mat_kwargs

        self.is_weight_fraction = None
        self.is_gas = False
        self.is_mcnp = is_mcnp

    def set_srim_dedx(self, dedx_path=Path.expanduser(Path("~"))/'phits'/'data'/'dedx', scaling=None):
        """
        Sets the DeDx file from an SRIM output. See JSB_tools/SRIM. Only works for PHITS.
        Args:
            dedx_path: Path where PHITS looks for user supplied stopping powers
            scaling: A function or a constant. If a constant, scale sopping powers by this value.
                If function, scale the stopping powers by f(e)
        Returns:

        """
        assert dedx_path.exists(), dedx_path

        from JSB_tools.SRIM import existing_outputs, SRIMTable

        elements, fractions = self._get_elements_and_fractions

        class _SrimRun:
            def __init__(self, atoms, fractions, density, projectile, is_gas):
                self.atoms = atoms
                self.fractions = np.array(fractions)
                self.density = float(density)
                self.projectile = projectile
                self.is_gas = is_gas
                self.args = atoms, fractions, self.density, projectile, is_gas

            def __repr__(self):
                out = f'elements: {self.atoms}, %s: {100*np.array(list(map(int, self.fractions)))}, ' \
                      f'density: {self.density}, gas = {self.is_gas}'
                return out

        options = [_SrimRun(*args) for args in existing_outputs()]

        # remove SRIM entries without the same element composition
        options = list(filter(lambda x: tuple(x.atoms) == tuple(elements), options))

        # remove SRIM entries without the similar fractional element composition
        options = list(filter(lambda x: all(np.isclose(x.fractions, fractions)), options))

        # filter for gaseous state or not
        options = list(filter(lambda x: x.is_gas == self.is_gas, options))

        _srim_outputs = {}
        for option in options:  # group SRIM entries by same target (but different projectile)
            try:
                _srim_outputs[option.projectile].append(option)
            except KeyError:
                _srim_outputs[option.projectile] = [option]
        srim_outputs = {}
        lines = ['unit = 1']
        if not len(_srim_outputs.items()):
            assert False, f"No SRIM data available for the following material: \n{self}." \
                          f"\nSee JSB_tools.SRIM to add this data."

        if scaling is None:
            scaling = lambda erg: 1
        else:
            if isinstance(scaling, Number):
                c = scaling  # avoid infinite recursion
                scaling = lambda erg: c
            else:
                assert hasattr(scaling, '__call__'), "Invalid type for `scaling` parameter. Must be Number of callable."
                assert len(signature(scaling).parameters) == 1, "`scaling` function should take only one argument"

        # within each group of SRIM entries with same projectile, chose one with best density.
        # Warn if density is not exact.
        # Then, write lines for PHITS dedx file.
        for proj, options in _srim_outputs.items():
            if len(options):
                arg_min = np.argmin([abs(o.density-self.density) for o in options])
                best_option: _SrimRun = options[arg_min]
                srim_outputs[proj] = best_option
                if not np.isclose(best_option.density, self.density, rtol=0.02):
                    warnings.warn(f"\n ------------------ DENSITY WARNING ----------------"
                                  f"\nSRIM run for projectile {best_option.projectile} in material,\n"
                                  f"elements:  {best_option.atoms}\n"
                                  f"fractions: {best_option.fractions},'\n"
                                  f"has a density of {best_option.density}\n"
                                  f"instead of       {self.density}  (the intended density)")
                srim_table = SRIMTable(*best_option.args)

                lines.append(f"kf = {Nuclide.from_symbol(best_option.projectile).phits_kfcode()}")
                for erg, dedx in zip(srim_table.ergs, srim_table.total_dedx):
                    lines.append(f"{erg:.3E} {scaling(erg)*dedx:.4E}")
        dedxfile = (dedx_path/str(self.mat_number)).with_suffix('.txt')
        with open(dedxfile, 'w') as f:
            f.write('\n'.join(lines))

        self.mat_kwargs['dedxfile'] = f'{dedxfile.name} $ - dedx from SRIM'

    @classmethod
    def gas(cls, list_of_chemicals: List[str],
            mass_ratios: List[Number] = None, atom_fractions: List[Number] = None, use_natural_abundences=False,
            temperature=293,
            temp_units: str = 'K',
            pressure: float = 1,
            pressure_units: str = 'bars',
            n_sig_digits=4,
            mat_number: int = None,
            mat_name: str = None,
            mat_kwargs: Dict[str, str] = None,
            is_mcnp=True
            ):
        # Todo: Use N(2) or N_2 to specify number of N atoms. Thus, allowing isotope specification normally.

        g = _IdealGas(list_of_chemicals)
        if len(list_of_chemicals) == 1:
            is_weight_fraction = True
            fractions = [1]
            density = g.get_density_from_atom_fractions([1], temperature, temp_units, pressure, pressure_units,
                                                        n_sig_digits)
        else:
            if mass_ratios is not None:
                assert hasattr(mass_ratios, '__iter__')
                assert atom_fractions is None, 'Can use `atom_fractions` or `mass_ratios`, but not both.'
                density = g.get_density_from_mass_ratios(mass_ratios, temperature, temp_units, pressure, pressure_units,
                                                         n_sig_digits)

                is_weight_fraction = True
                fractions = mass_ratios
            else:
                assert atom_fractions is not None, 'Must specify either `atom_fractions` or `mass_ratios`.'
                density = g.get_density_from_atom_fractions(atom_fractions,  temperature, temp_units, pressure,
                                                            pressure_units, n_sig_digits)
                is_weight_fraction = False
                fractions = atom_fractions

        if mat_kwargs is None:
            mat_kwargs = {'GAS': "1"}
        else:
            assert isinstance(mat_kwargs, dict)
            mat_kwargs['GAS'] = "1"

        out = Material(density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs, is_mcnp=is_mcnp)

        for s, fraction in zip(list_of_chemicals, fractions):
            if m := chemical_regex.match(s):
                a = m.group('A')
                symbol = m.group('symbol')
                if not a:
                    a = None
                if a is None:
                    nuclide_name = get_most_abundant_isotope(symbol)
                else:
                    nuclide_name = symbol + a
                n = Nuclide.from_symbol(nuclide_name)
                assert n.is_valid, 'Invalid atomic symbol, "{}"'.format(nuclide_name)
            else:
                assert False, 'Invalid atomic symbol, "{}"'.format(s)
            out.add_zaid(n, fraction, is_weight_fraction)
        out.is_gas = True
        return out

    def add_element_natural(self, element_symbol, fraction: float = 1.) -> None:
        """
        Add all of an element's isotopes according to their natural abundances.

        Args:
            element_symbol: e.g. "Ar", "U", "W"
            fraction: negative for weight fraction, positive for atom fraction.

        Returns: None
        """
        open_mc_mat = openmc.material.Material()
        open_mc_mat.add_element(element_symbol, 1)
        z = ATOMIC_NUMBER[element_symbol]
        for symbol, percent, _, in open_mc_mat.nuclides:
            m = re.match('[A-Z][a-z]{0,2}([0-9]+)', symbol)
            a = int(m.groups()[0])
            zaid = 1000 * z + a
            self.add_zaid(zaid, percent*fraction, False)

    def add_zaid(self, zaid_or_nuclide: Union[Nuclide, int], fraction, is_weight_fraction=False,
                 elemental_zaid=False):
        if self.is_weight_fraction is None:
            self.is_weight_fraction = is_weight_fraction
        if isinstance(zaid_or_nuclide, int):
            pass
        elif isinstance(zaid_or_nuclide, Nuclide):
            zaid_or_nuclide = 1000*zaid_or_nuclide.Z + zaid_or_nuclide.A
        else:
            assert False, 'Incorrect type, "{}", passed in `zaid_or_nuclide` argument.\n' \
                          'Must be zaid (int) or Nuclide'.format(type(zaid_or_nuclide))
        if elemental_zaid:
            zaid_or_nuclide = 1000 * (zaid_or_nuclide//1000)
        self._zaids.append(zaid_or_nuclide)
        self._zaid_proportions.append(fraction)

    @property
    def mat_card(self, mat_kwargs=None) -> str:
        if mat_kwargs is not None:
            self.mat_kwargs.update(mat_kwargs)

        assert len(self._zaids) > 0, 'No materials added! Use Material.add_zaid'
        comment = get_comment('density = {}'.format(self.density), self.name )
        outs = ['M{}  {}'.format(self.mat_number, comment)]
        for n, zaid in zip(self._zaid_proportions, self._zaids):
            outs.append('     {} {}'.format(zaid, '-{}'.format(n) if self.is_weight_fraction else n))

        outs.extend(["     {} = {}".format(k, v) for k, v in self.mat_kwargs.items()])

        return '\n'.join(outs)

    def __repr__(self):
        return self.mat_card

    @property
    def name(self):
        return self.__name__

    def delete_mat(self):
        raise NotImplementedError("Don't do this! Modify the features of an existing material instead. ")
        # if self.mat_number in Material.__all_materials:
        #     del Material.__all_materials[self.mat_number]


class PHITSOuterVoid:
    pass


class DepletedUranium(Material):
    def __init__(self, density=19.1, mat_number: int = None, mat_name: str = "DepletedUranium",
                 mat_kwargs: Dict[str, str] = None):
        super(DepletedUranium, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        self.add_element_natural('U')


class Tungsten(Material):
    def __init__(self, density=19.28, mat_number: int = None, mat_name: str = "Tungsten",
                 mat_kwargs: Dict[str, str] = None):
        super(Tungsten, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        self.add_element_natural('W')


class Titanium(Material):
    def __init__(self, density=4.54, mat_number: int = None, mat_name: str = "Titanium",
                 mat_kwargs: Dict[str, str] = None):
        super(Titanium, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        self.add_element_natural('Ti')


class Nickel(Material):
    def __init__(self, density=8.902, mat_number: int = None, mat_name: str = "Nickle",
                 mat_kwargs: Dict[str, str] = None):
        super(Nickel, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        self.add_element_natural('Ni', 1)


class StainlessSteel(Material):
    def __init__(self, density=None, mat_number: int = None, mat_name: str = "Stainless steel",
                 mat_kwargs: Dict[str, str] = None, elemental_zaid=False, number=304):
        """

        Args:
            density:
            mat_number:
            mat_name:
            mat_kwargs:
            elemental_zaid: If True, use, e.g., 6000 instead of 6012.
            number: Steel type, e.g. 304 stainless steel (most common)
        """
        # zaids_fracs  {steel_num: (density, [zaid1, frac1, ...]])}
        zaids_fracs = {304: (8.0,
                             [(6000, 0.00183),
                              (14000, 0.009781),
                              (15031, 0.000408),
                              (16000, 0.000257),
                              (24000, 0.200762),
                              (25055, 0.010001),
                              (26000, 0.690375),
                              (28000, 0.086587)]),
                       302: (7.86,
                             [(6000, 0.006356),
                              (14000, 0.0018057),
                              (15031, 0.000739),
                              (16000, 0.000476),
                              (24000, 0.188773),
                              (25055, 0.018462),
                              (26000, 0.683520),
                              (28000, 0.083616)]),
                       202: (7.86,
                             [(6000, 0.003405),
                              (7014, 0.004866),
                              (14000, 0.009708),
                              (15031, 0.000528),
                              (16000, 0.000255),
                              (24000, 0.188773),
                              (25055, 0.086851),
                              (26000, 0.659160),
                              (28000, 0.046454)])
                       }
        if number not in zaids_fracs:
            raise KeyError(f"No specification for steel {number}")
        if density is None:
            density = zaids_fracs[number][0]
        super(StainlessSteel, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)

        for zaid, frac in zaids_fracs[number][-1]:
            self.add_zaid(zaid, frac, elemental_zaid=elemental_zaid)
        # self.add_element_natural('Fe', 0.659)
        # self.add_element_natural('Cr', 0.18)
        # self.add_element_natural('Mn', 0.08)
        # self.add_element_natural('Ni', 0.04)


class Aluminum(Material):
    def __init__(self, density=2.63, mat_number: int = None, mat_name: str = "Aluminum",
                 mat_kwargs: Dict[str, str] = None):
        super(Aluminum, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        self.add_element_natural('Al')


class Lead(Material):
    def __init__(self, density=11.43, mat_number: int = None, mat_name: str = "Lead",
                 elemental=True,
                 mat_kwargs: Dict[str, str] = None):
        """
        Lead material. All examples use only zaid=82000. I don;t know why this is, but that's why `elemental` is True
        by default.
        Args:
            density:
            mat_number:
            mat_name:
            elemental:
            mat_kwargs:
        """
        super(Lead, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)

        if elemental:
            zaids = [82000]
            fractions = [1]
        else:
            zaids = [82204, 82206, 82207, 82208]
            fractions = np.array([1.4, 24.1, 22.1, 52.4])/100

        [self.add_zaid(z, f) for z, f in zip(zaids, fractions)]


class Mylar(Material):
    def __init__(self, density=1.38, mat_number: int = None, mat_name: str = "Mylar",
                 mat_kwargs: Dict[str, str] = None, elemental_zaids=False):
        super(Mylar, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        fractions = [0.363632, 0.454552, 0.181816]

        zaids = [1001, 6000, 8016]

        if elemental_zaids:
            zaids = [1000*(z//1000) for z in zaids]

        [self.add_zaid(z, atom_fraction) for z, atom_fraction in zip(zaids, fractions)]


class Air(Material):
    def __init__(self, temperature=293, temp_units: str = 'K', pressure: float = 1, pressure_units: str = 'bars',
                 n_sig_digits=4, density=None,  mat_number: int = None, mat_name: str = "air",
                 mat_kwargs: Dict[str, str] = None):
        elements = ['C', 'N(2)', 'O(2)', 'Ar']
        zaids = [6000, 7014, 8016, 18000]
        fractions = [1.5E-4, 0.78, 0.21, 0.0047]
        if density is not None:
            assert temperature is pressure is None, "`density` was specified. `pressure` and `temperature` must be None"

        else:
            g = _IdealGas(elements)
            density = g.get_density_from_atom_fractions(atom_fractions=fractions,
                                 temperature=temperature, temp_units=temp_units, pressure=pressure,
                                 pressure_units=pressure_units,  n_sig_digits=n_sig_digits)
        super(Air, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)

        for zaid, f in zip(zaids, fractions):
            self.add_zaid(zaid, f)


class ThoriumTetrafluoride(Material):
    def __init__(self, density=6.3, mat_number: int = None, mat_name: str = "ThF4",
                 mat_kwargs: Dict[str, str] = None, elemental_zaids=False):
        super(ThoriumTetrafluoride, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        fractions = [4, 1]
        zaids = [9019, 90232]

        if elemental_zaids:
            zaids = [1000*(z//1000) for z in zaids]
        [self.add_zaid(z, atom_fraction) for z, atom_fraction in zip(zaids, fractions)]


class Graphite(Material):
    def __init__(self, density=1.7, mat_number: int = None, mat_name: str = "Graphite",
                 mat_kwargs: Dict[str, str] = None):
        super(Graphite, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        self.add_zaid(6000, 1)


if __name__ == "__main__":
    print(StainlessSteel())
    # m = Material.gas(['He', 'Ar'], atom_fractions=[1,1], pressure=1.35)
    # m.set_srim_dedx()
    # print(m)
    # print(DepletedUranium()._get_elements_and_fractions)
    # print(atomic_weight, ATOMIC_NUMBER, atomic_mass)