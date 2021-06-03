from JSB_tools.MCNP_helper.geometry import MCNPNumberMapping, get_comment
from typing import Dict, Union
from JSB_tools.nuke_data_tools import Nuclide
from openmc.data import atomic_weight, ATOMIC_NUMBER, atomic_mass
import numpy as np
import re
from typing import List
from numbers import Number
import openmc.material
# Todo: Revamp the IdealGas interface.


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
                self.atomic_weights.append(atomic_mass(symbol + a))
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
    __all_materials = MCNPNumberMapping('Material', 1000, 1000)

    @staticmethod
    def clear():
        Material.__all_materials = MCNPNumberMapping('Material', 1000, 1000)

    @staticmethod
    def get_all_material_cards():
        cards = []
        for mat in Material.__all_materials.values():
            cards.append(mat.mat_card)
        return '\n'.join(cards)

    def __init__(self, density: float, mat_number: int = None, mat_name: str = None, mat_kwargs: Dict[str, str] = None):
        self.mat_number = mat_number
        self.__name__ = mat_name
        Material.__all_materials[self.mat_number] = self
        self.density = density
        self._zaids = []
        self._zaid_proportions = []

        if mat_kwargs is None:
            self.mat_kwargs = {}
        else:
            self.mat_kwargs = mat_kwargs

        self.is_weight_fraction = None

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

        out = Material(density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)

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

        return out

    def add_element_natural(self, element_symbol, fraction: float = 1.) -> None:
        """
        Add all of an element's isotopes according to their natural abundances.

        Args:
            element_symbol: e.g. "Ar", "U", "W"
            fraction:

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

    def add_zaid(self, zaid_or_nuclide: Union[Nuclide, int], fraction, is_weight_fraction=False):
        if self.is_weight_fraction is None:
            self.is_weight_fraction = is_weight_fraction
        if isinstance(zaid_or_nuclide, int):
            pass
        elif isinstance(zaid_or_nuclide, Nuclide):
            zaid_or_nuclide = 1000*zaid_or_nuclide.Z + zaid_or_nuclide.A
        else:
            assert False, 'Incorrect type, "{}", passed in `zaid_or_nuclide` argument.\n' \
                          'Must be zaid (int) or Nuclide'.format(type(zaid_or_nuclide))

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
        if self.mat_number in Material.__all_materials:
            del Material.__all_materials[self.mat_number]


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


class StainlessSteel(Material):
    def __init__(self, density=7.86, mat_number: int = None, mat_name: str = "Stainless steel",
                 mat_kwargs: Dict[str, str] = None):
        super(StainlessSteel, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        self.add_element_natural('Fe', 0.659)
        self.add_element_natural('Cr', 0.18)
        self.add_element_natural('Mn', 0.08)
        self.add_element_natural('Ni', 0.04)


class Aluminum(Material):
    def __init__(self, density=2.63, mat_number: int = None, mat_name: str = "Aluminum",
                 mat_kwargs: Dict[str, str] = None):
        super(Aluminum, self).__init__(density=density, mat_number=mat_number, mat_name=mat_name, mat_kwargs=mat_kwargs)
        self.add_element_natural('Al')


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


