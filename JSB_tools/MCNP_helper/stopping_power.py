from __future__ import annotations
import re
import numpy as np
from warnings import warn
from pathlib import Path
from typing import List, Dict, Set, Iterable, Sized, Union, Optional
from numbers import Number
from JSB_tools.MCNP_helper.atomic_data import ATOMIC_NUMBER, ATOMIC_SYMBOL
from JSB_tools.MCNP_helper.materials import Material
import platform
import subprocess
from functools import cached_property
import matplotlib.pyplot as plt
import JSB_tools.MCNP_helper.outp_reader as outp_reader

cwd = Path(__file__).parent


def get_material(material_element_symbols: Union[List[str], str],
                 density: float,
                 material_atom_percents: Optional[Union[List[float], float]],
                 material_mass_percents: Optional[Union[List[float], float]],
                 temperature=None):
    assert isinstance(density, Number), '`grams_per_cm3` argument must be a number'

    density = abs(density)

    def make_iter(a):
        if a is None:
            return a
        elif not hasattr(a, '__iter__'):
            return [a]
        else:
            if isinstance(a, str):
                return [a]
            else:
                return a

    material_element_symbols = make_iter(material_element_symbols)
    material_atom_percents = make_iter(material_atom_percents)
    material_mass_percents = make_iter(material_mass_percents)

    assert all([isinstance(s, str) for s in material_element_symbols]), 'Element symbols must be string'
    if material_atom_percents is not None:
        assert all([isinstance(f, Number) for f in material_atom_percents]), \
            '`element_atom_percents` must be float(s)'
    if material_mass_percents is not None:
        assert all([isinstance(f, Number) for f in material_mass_percents]), \
            '`element_mass_percents` must be float(s)'

    if len(material_element_symbols) == 1:
        if hasattr(material_atom_percents, '__iter__'):
            if len(material_atom_percents) != 1:
                warn('Ignoring `element_atom_percents` argument. Not need for a single element')
        if hasattr(material_mass_percents, '__iter__'):
            if len(material_mass_percents) != 1:
                warn('Ignoring `element_mass_percents` argument. Not need for a single element')
        material_mass_percents = None
        material_atom_percents = [1]

    if material_atom_percents is None:
        assert material_mass_percents is not None, 'Must either provide `element_mass_percents` or ' \
                                                   '`element_atom_percents` arguments'
        atom_fractions = material_mass_percents
    else:
        atom_fractions = material_atom_percents
    assert len(material_element_symbols) == len(atom_fractions), '`element_symbols` arg must be equal in length to atom ' \
                                                                 'fractions'
    m = Material(density, 1000, )

    for element, fraction in zip(material_element_symbols, atom_fractions):
        m.add_zaid(element, fraction)

    return m


def read_stopping_power(outp: outp_reader.OutP,
                          particle: str, material: Material):
    """

    Args:
        outp:
        particle:

        # material_id: Used for material density and material ID
        #
        # cell_num: If no `material_id` is given, used to find material density and material ID

    Returns:

    """
    particle = particle.lower()
    out = MCNPStoppingPowerData()

    if re.match('[0-9]+', particle):
        particle_type = 'heavy_ion'

    elif particle in ['electron', 'proton', 'positron']:
        particle_type = particle

    else:
        particle_type = None
        warn('Reading stopping powers from {} type of particle might not behave normally. Verify.'.format(particle))

    out.material = material

    if particle == 'electron':
        electron_flag = True  # just for speed
        c = re.compile(f'1range +table .+material +{out.material.mat_number} .+ print table 85')

    else:
        electron_flag = False
        c = re.compile(f'1.*{particle}.+{out.material.mat_number}.+print table 85')

    looking_for_beginning = False

    for index, line in enumerate(outp.__outp_lines__):
        if c.match(line) and not looking_for_beginning:
            if electron_flag:
                assert 'electron' in outp.__outp_lines__[index + 2]
                looking_for_beginning = True
            else:
                index += 8  # beginning of dEdx data begins 8 lines after the start of print table 85
                break
        elif looking_for_beginning:
            if re.match(' +[0-9]+ +([0-9.E+-]+ *){11}', line):
                break
    else:
        assert False, f"Could not find dEdx table for '{particle}' and material '{material_id}'"

    length = int(outp.__outp_lines__[index].split()[0])  # This is the number of data entries in the table.
    ergs = []
    dedxs = []
    ranges = []

    # add to these upon further insights if needed
    total_dedx_index = {'ion': 6, 'proton': 6, 'electron': 4}.get(particle_type, -6)
    range_index = {'ion': -3, 'proton': -3, 'electron': 5}.get(particle_type, -3)

    for index in range(index, index + length):
        values = list(map(float, (outp.__outp_lines__[index].split())))
        ergs.append(values[1])
        dedxs.append(values[total_dedx_index])
        ranges.append(values[range_index])

    out.ranges = np.array(ranges)
    out.energies = np.array(ergs)
    out.dedxs = np.array(dedxs)
    out.par = particle
    return out


class MCNPStoppingPowerData:
    def __init__(self):
        self.__energies__: np.ndarray = None  # MeV
        self.ranges: np.ndarray = None  # cm
        self.dedxs: np.ndarray = None   # MeV/(g/cm2)
        self.par = None
        self.material: Material = None
        # self.cell_density = None
        self.erg_bin_widths = None
        self.__dx_de__ = 0

    @property
    def cell_density(self):
        return self.material.density

    def __call__(self, erg, mul_rho=True, units='MeV'):
        """
        Return stopping power as a function of energy

        Args:
            erg:
            mul_rho: If True, multply by density to get MeV/cm

        Returns:

        """
        if units == 'eV':
            s = 1E6
        elif units == 'keV':
            s = 1E3
        elif units == 'MeV':
            s = 1
        else:
            raise NotImplementedError(f"No scaling for units, '{units}'")

        out = s * np.interp(erg, self.energies, self.dedxs)

        if mul_rho:
            out *= self.cell_density

        return out

    @property
    def ranges_cm(self):
        assert self.cell_density is not None
        return self.ranges/self.cell_density

    @cached_property
    def __dx_des__(self):
        return 1.0/self.dedxs

    @property
    def energies(self):
        return self.__energies__

    @energies.setter
    def energies(self, value):
        value = np.array(value)
        self.__energies__ = value
        self.erg_bin_widths = np.array([b2 - b1 for b1, b2 in zip(value[:-1], value[1:])])

    def eval_de_dx_at_erg(self, erg, density=None):
        if density is None:
            density = self.cell_density
        return np.interp(erg, self.energies, self.dedxs*density)

    def eval_dx_de_at_erg(self, erg, density=None):
        if density is None:
            density = self.cell_density
        return np.interp(erg, self.energies, self.__dx_des__/density)

    def get_range_at_erg(self, erg, density=None):
        if density is None:
            assert self.cell_density is not None, 'Must either provide a cell number or manually supply a density.\n' \
                                                  'Set density to 1 to get range ion units of cm2/g'
            density = self.cell_density
        return np.interp(erg, self.energies, self.ranges/density)

    def plot_dedx(self, ax=None, label=None, title=None, material_name_4_title=None, density=None, per_g_cm3=False):
        """

        Args:
            ax: matplotlib axes object.
            label: Legend label.
            title: Specify title completely manually.
            material_name_4_title: Will include this in figure title.
            density: Provide a density. Overrides using sekf.density.
            per_g_cm3: If True, do not multiply by density. Units will be (MeV)/cm per g/cm3 (i.e. MeV.cm2/g)

        Returns:

        """
        if ax is None:
            fig, ax = plt.subplots()

        y = self.dedxs
        if not per_g_cm3:
            if density is not None:
                y = self.dedxs*density
            else:
                if self.cell_density is not None:
                    density = self.cell_density
                    y = self.dedxs * density

        ax.plot(self.energies, y, label=label)

        if label is not None:
            ax.legend()

        ax.set_xlabel("Energy [MeV]")

        if density is None or per_g_cm3:
            ax.set_ylabel("dEdx [MeV cm2/g]")
        else:
            ax.set_ylabel("dEdx [MeV/cm]")

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if title is None:
            title = ""
            if material_name_4_title is not None:
                title += "{0} in material {1}".format(self.par, material_name_4_title)
            else:
                title += "{0} in material {1}".format(self.par, self.material.name)

            if density is not None:
                title += " density: {0:.4E} g/cm3".format(density)

            ax.set_title(title)
        return ax

    def plot_energy_loss(self, init_erg=None, ax=None, units=None, max_distance=None, density=None,
                         N_points=160, **mpl_kwargs):
        """
        Starting at x=0 with energy of init_erg, plot the energy as a function of distance traveled.

        Args:
            init_erg: Initial particle energy.
            ax:
            N_points: Number of points on x-axis
            units:
            max_distance: Cuts of plot at distance = max_distance
            density: Uses self.cell_density by default
        Returns:

        """
        if density is None:
            density = self.cell_density

        if ax is None:
            _, ax = plt.subplots()

        if init_erg is None:
            init_erg = self.energies[-1]

        full_range = self.get_range_at_erg(init_erg)

        if max_distance is None:
            max_x = full_range
        else:
            max_x = max_distance
        xs = np.linspace(0, max_x, N_points)

        unit_conversion, units = self._dist_unit([xs[-1]], units)

        ys = np.interp(full_range - xs, self.ranges/density, self.energies)
        xs *= unit_conversion
        ax.plot(xs, ys, **mpl_kwargs)

        ax.set_xlabel(f'Distance traveled [{units}]')
        ax.set_ylabel(f'Energy [MeV]')

        return xs, ys

    @staticmethod
    def _dist_unit(ys, units=None):
        unit_names = ["nm", "um", "mm", "cm", "m", "km"]
        orders = np.array([-7, -4, -1, 0, 2, 5])
        if units is None:
            test_value = np.max(ys)
            i = np.searchsorted(orders, np.log10(test_value), side='right') - 1
            i = max([0, i])
            units = unit_names[i]
        else:
            assert units in unit_names, f"Invalid units, {units}"
            i = unit_names.index(units)

        unit_conversion = 10. ** -orders[i]
        return unit_conversion, units

    def plot_range(self, ax=None, energies=None , label=None, title=None, material_name_4_title=None, particle_name_4_title=None,
                   density=None, units=None):

        mat_name = self.material.name if self.material.name is not None else 'material'

        if ax is None:
            fig, ax = plt.subplots()

        y = self.ranges
        if density is not None:
            y = self.ranges/density
            if self.cell_density is not None:
                warn('Overriding cell density deduced from `cell_num_4_density` argument.')
        else:
            if self.cell_density is not None:
                density = self.cell_density
                y = self.ranges/density

        ax.set_xlabel("Energy [MeV]")

        if density is None:
            unit_conversion, units = 1, 'cm'
            ax.set_ylabel("Range [g/cm2]")
        else:
            unit_conversion, units = self._dist_unit(y, units)
            ax.set_ylabel("Range [{}]".format(units))

        if energies is None:
            x = self.energies
        else:
            x = energies
            y = np.interp(energies, self.energies, y)

        ax.plot(x, y*unit_conversion, label=label)

        if label is not None:
            ax.legend()

        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

        if title is None:
            title = ""
            if particle_name_4_title is not None:
                par_name = particle_name_4_title
            else:
                try:
                    zaid = int(self.par)
                    z = zaid //1000
                    a = zaid%1000
                    par_name = '{}-{}'.format(ATOMIC_SYMBOL[z], a)

                except (ValueError, KeyError):
                    par_name = self.par

            title += f"Range of {par_name} in {mat_name}"

            if density is not None:
                title += "; density: {0:.4E} g/cm3".format(density)
            title = "\n" + title

            ax.set_title(title, y=1.05)
        return ax

    @classmethod
    def gen_stopping_power(cls, particle: str,
                           material_element_symbols: Union[List[str], str] = None,
                           density: float = None,
                           material_atom_percents: Optional[Union[List[float], float]] = None,
                           material_mass_percents: Optional[Union[List[float], float]] = None,
                           gas: bool = False,
                           material: Material = None,
                           emax=200,
                           temperature: Optional[float] = None,
                           mcnp_command='mcnp6',
                           verbose: bool = False) -> MCNPStoppingPowerData:
        """

        Args:
            particle: Either a zaid in the form of zzaaa, i.e. str(1000*z+a), or a nuclide name, i.e. 'Xe139'.
                Can also be 'proton' or 'electron'.

            material_element_symbols: List of elements that make up the medium using similar convention as in`particle`
                argument. Can also use "U" instead of "U238" to automatically isotopic composition of natural U

            density: Density of medium in g/cm3.

            material_atom_percents: relative atoms percents for each element in `material_element_symbols`

            material_mass_percents: relative mass ratios for each element in `material_element_symbols`

            gas: True is medium is gas.

            material: If used, then above material params will not be used.

            emax: Max energy to calculate stopping powers.

            temperature:  Does nothing for now.

            mcnp_command: The command that executes MCNP in youtr terminal.

            verbose: For debugging.

        Returns:
        """
        assert platform.system() != 'Windows', '`get_stopping_power` is not compatible with windows systems!'

        mode_dict = {'electron': 'e', 'proton': 'h', 'positron': 'f'}
        mode = None
        invalid_par_des_msg = '\nParticle designator must be one of the following:\n\t{}\n,' \
                              '*or* specify an ion via use particle=zzaaa, where,\n\t"zz" is atomic number\n\t"aaa" ' \
                              'is mass number.\t*or* use nuclide name, i.e. U238' \
            .format(list(mode_dict.keys()) + list(mode_dict.values()))
        element_match = re.compile('([a-zA-Z]+)([0-9]+)')

        def get_zaid(symbol):
            _m = element_match.match(symbol)
            assert _m, 'Invalid element specification: "{}"'.format(symbol)
            e_symbol = _m.groups()[0]
            a = int(_m.groups()[1])
            assert e_symbol in ATOMIC_NUMBER, 'Invalid element symbol: "{}.\nOptions:\n{}"' \
                .format(s, list(ATOMIC_NUMBER.keys()))
            z = ATOMIC_NUMBER[e_symbol]
            return z * 1000 + a

        try:  # check for zaid specification as int
            particle = int(particle)
        except ValueError:
            pass
        if isinstance(particle, str):
            particle = particle.replace('-', '')

        if element_match.match(str(particle)): # check for zaid specification as str
            particle = get_zaid(particle)

        if isinstance(particle, str):
            if particle in mode_dict:
                mode = mode_dict[particle]
                sdef_par = mode
                outp_par = particle
            elif particle in mode_dict.values():
                mode = sdef_par = particle
                outp_par = {v: k for k, v in mode_dict}[particle]
            else:
                assert False, invalid_par_des_msg

        elif isinstance(particle, int):
            mode = '#'
            outp_par = particle
            sdef_par = particle
        else:
            assert False, invalid_par_des_msg

        if material is None:
            material = get_material(material_element_symbols=material_element_symbols, density=density,
                                    material_atom_percents=material_atom_percents, material_mass_percents=material_mass_percents,
                                    temperature=temperature)

        mat_card = material.mat_card

        cwd = Path(__file__).parent
        sim_dir = cwd / 'misc'

        if gas:
            mat_card += '\n     GAS=1'
        with open(sim_dir / 'mcnp_basic_inp') as f:
            inp_lines = f.readlines()

        with open(sim_dir/'__temp__.inp', 'w') as f:
            for line in inp_lines:
                new_line = line.format(**locals())
                f.write(new_line)
                if verbose:
                    print(new_line)

        cmd = f'cd {sim_dir};'
        cmd += f'{mcnp_command} i=__temp__.inp '
        outp_path = (sim_dir/'outp')
        temp_paths = [sim_dir/'runtpe.h5', sim_dir/'runtpe', sim_dir/'meshtal']

        for path in [outp_path] + temp_paths:
            path.unlink(missing_ok=True)

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out = p.stdout.read()

        outp = outp_reader.OutP(outp_path)

        return read_stopping_power(outp, str(outp_par), material)


if __name__ == '__main__':
    from JSB_tools.MCNP_helper.materials import EJ600
    from JSB_tools import mpl_style
    mpl_style(fig_size=(7,7))

    mat = EJ600()
    # Material.get_nuclide_atom_densities()
    # mat.get_nuclide_atom_densities()
    # assert isinstance(mat, Material)
    stope = MCNPStoppingPowerData.gen_stopping_power('electron', material=mat)

    ax = stope.plot_range(energies=np.linspace(0, 2.5, 100), label='Electron', units='mm')

    stop_p = MCNPStoppingPowerData.gen_stopping_power('proton', material=mat)

    stop_p.plot_range(energies=np.linspace(0, 2.5, 100), ax=ax, label='Proton', units='mm')

    stope.plot_dedx(density=1)

    ax.set_title(r'\textbf{Range of protons/electrons in EJ600}')
    ax.set_xlabel('Particle energy [MeV]')

    plt.gcf().subplots_adjust(left=0.135)

    plt.show()
