import re
import numpy as np
import os
from warnings import warn
from uncertainties import ufloat
import uncertainties.unumpy as unp
from matplotlib import pyplot as plt
from pathlib import Path
from typing import List, Dict
from numbers import Number
from openmc import Material
from openmc.data import ATOMIC_NUMBER, ATOMIC_SYMBOL
import platform
import subprocess
from functools import cached_property
#  Todo:
#   Add function to access num particles entering cells


class F4Tally:
    def __init__(self, tally_number_or_name, outp):
        assert isinstance(tally_number_or_name, (str, int))
        if isinstance(tally_number_or_name, str):
            self.tally_name = tally_number_or_name.lower()
        else:
            self.tally_name = None
        if isinstance(tally_number_or_name, int):
            tally_number = tally_number_or_name
            self.tally_number = str(tally_number)
        else:
            self.tally_number = None

        assert isinstance(outp, OutP)

        self.tally_modifiers = set()
        match_1 = re.compile(r'^([fcemt]+)([0-9]*4): *. +([0-9]+) *\$.*name: *([^\s\\]+)')
        for card in outp.input_deck:
            if self.tally_number is not None:
                _m = re.match("^([fcemt]+){}.+".format(self.tally_number), card)
                if _m:
                    if _m.group(1) != "f":
                        self.tally_modifiers.add(_m.group(1))
            else:
                _m = match_1.match(card)
                if _m and _m.groups()[-1] == self.tally_name:
                    self.tally_number = _m.groups()[1]
        if self.tally_number is None:
            if self.tally_name is not None:
                assert False, '\nCould not find tally with name "{0}"Ex\nExample of using a name tag to access tally:'\
                              '\nF84:p 13 $ name:<tally_name_here>\nThe key usage syntax is in the comment. The text '\
                              'after the string "name:" is the name tag (white spaces are stripped, chars are ' \
                              'case-insensitive)'.format(self.tally_name)

        assert self.tally_number[-1] == "4", "Incorrect tally type!"

        found_n_tallies = 0
        tally_begin_index = None
        for index, line in enumerate(outp.__outp_lines__):
            _m = re.match(r"1tally +{}".format(self.tally_number), line)
            if _m:
                found_n_tallies += 1
                tally_begin_index = index
        if found_n_tallies == 0:
            assert False, "Cannot find tally {} in {}".format(self.tally_number, outp.__f_path__)
        elif found_n_tallies > 1:
            warn('\nSeveral dumps of tally {0} found. Using last entry.'.format(self.tally_number))
        index = tally_begin_index
        # initialize
        if len((self.tally_modifiers - {'e', 'fm'})) == 0:
            index += 2
            _m = re.match(r" +particle\(s\): +([a-z]+)", outp.__outp_lines__[index])
            if _m:
                self.particle = _m.group(1)
            else:
                warn("Could not find particle for tally {}".format(self.tally_number))
            index += 2

            cell_number = None

            if "volumes" in outp.__outp_lines__[index]:
                index += 1
                _m = re.match(" +cell: +([0-9]+)", outp.__outp_lines__[index])
                if _m:
                    cell_number = int(_m.group(1))

                index += 1

            self.cell: Cell = None
            if cell_number is not None and cell_number in outp.cells:
                self.cell = outp.cells[cell_number]
            else:
                warn("Could not find cell for tally {0}".format(self.tally_number))

            index += 1
        else:
            assert False, "Tally modifiers {} not yet supported!".format(self.tally_modifiers)

        if (self.tally_modifiers - {'fm'}) == set():
            if 'fm' in self.tally_modifiers:
                index += 1
            self.underflow = None
            self.energies = np.array([])
            self.__fluxes__ = np.array([])
            flux_out_put = outp.__outp_lines__[index+2].split()
            _flux = float(flux_out_put[0])
            _flux_error = _flux*float(flux_out_put[1])
            self.__flux__ = ufloat(_flux, _flux_error)

        elif (self.tally_modifiers - {'fm'}) == {"e"}:  # modified with e
            # loop until 'energy' appears on line, indicating beginning of energy bin data
            while index < len(outp.__outp_lines__):
                line = outp.__outp_lines__[index]
                if "energy" == line.rstrip().lstrip():
                    index += 1
                    break
                index += 1
            else:
                assert False, "Could not find energy data for tally {}".format(self.tally_number)

            self.__energy_bins__ = []
            fluxes = []
            flux_errors = []

            while index < len(outp.__outp_lines__):
                line = outp.__outp_lines__[index]
                _m = re.match(r" +[0-9\.E+-]+", line)

                if _m:

                    try:
                        erg_bin, flux, rel_error = tuple(map(float, line.split()))
                    except ValueError as e:
                        assert False, 'Error parsing tally {0}. Outp line:\n{1}\n{2}'.format(self.tally_number,
                                                                                             line, e)

                    fluxes.append(flux)
                    flux_errors.append(flux*rel_error)
                    self.__energy_bins__.append(erg_bin)
                    index += 1
                else:
                    break
            self.underflow = ufloat(fluxes[0], flux_errors[0])
            self.__fluxes__ = unp.uarray(fluxes, flux_errors)[1:]
            self.energies = np.array([0.5*(b_low+b_high) for b_low, b_high in
                                      zip(self.__energy_bins__[:-1], self.__energy_bins__[1:])])

            self.__energy_bins__ = np.array(self.__energy_bins__)
            _flux = float(outp.__outp_lines__[index].split()[1])
            _flux_error = _flux * float(outp.__outp_lines__[index].split()[2])
            self.__flux__ = ufloat(_flux, _flux_error)  # total flux

        else:
            assert False, "Tally modifiers {} not supported yet! (from tally {})"\
                .format(self.tally_modifiers, self.tally_number)

    @ property
    def flux(self):
        if 'fm' in self.tally_modifiers:
            assert False, 'Tally {} has the "FM" modifier. Tally is a reaction rate, not a flux.'\
                .format(self.tally_number)
        else:
            return self.__flux__

    @property
    def rate(self):
        if 'fm' in self.tally_modifiers:
            return self.__flux__
        else:
            assert False, 'Tally {} does not have "FM" modifier. Tally is flux, not reaction rate.'\
                .format(self.tally_number)

    @property
    def rates(self):
        if 'fm' in self.tally_modifiers:
            return self.__fluxes__
        else:
            assert False, 'Tally {} does not have "FM" modifier. Tally is flux, not reaction rate.'\
                .format(self.tally_number)

    @property
    def fluxes(self):
        if 'fm' in self.tally_modifiers:
            assert False, 'Tally {} has the "FM" modifier. Tally is a reaction rate, not a flux.'\
                .format(self.tally_number)
        else:
            return self.__fluxes__

    @property
    def erg_bin_widths(self):
        out = [e2-e1 for e1, e2 in zip(self.__energy_bins__[:-1], self.__energy_bins__[1:])]
        return np.array(out)

    def interp_energy(self, new_ergs):
        out = unp.uarray(np.interp(new_ergs, self.energies, unp.nominal_values(self.__fluxes__)),
                         np.interp(new_ergs, self.energies, unp.std_devs(self.__fluxes__)))
        return out

    def __repr__(self):
        return 'F4 Tally ({2}) in cell {0}, With {1} modifier.'.format(self.cell.cell_num, self.tally_modifiers,
                                                                      self.tally_number)


class Cell:
    def __init__(self, data, outp=None):
        self.cell_num = int(data[1])
        self.mat = int(data[2])
        self.atom_density = float(data[3])  # atoms/barn*cm
        self.density = float(data[4])
        self.volume = float(data[5])
        self.outp = outp

        if self.outp is not None:
            assert isinstance(outp, OutP)

    def __repr__(self):
        return "Cell {0}, mat:{1}".format(self.cell_num, self.mat)

    def get_tallys(self) -> List[F4Tally]:
        tallies = []
        for line in self.outp.input_deck:
            _m = re.match(' *f([0-9]*([0-9])): *. +([0-9]+)', line)
            if _m:
                cell_num = int(_m.groups()[2])
                tally_num = int(_m.groups()[0])
                tally_type = int(_m.groups()[1])
                if tally_type != 4:
                    warn('Tally other than 4 not implemented yet (from Cell.get_tallys)')
                else:
                    if cell_num == self.cell_num:
                        tallies.append(self.outp.get_tally(tally_num))
        return tallies




class OutP:
    def __init__(self, file_path):
        self.__f_path__ = file_path
        self.__outp_lines__ = open(file_path).readlines()

        self.input_deck = []
        self.nps = None
        for line in self.__outp_lines__:
            _m = re.match("^ {0,9}[0-9]+- {7}(.+)", line)
            if _m:
                card = _m.group(1).rstrip().lower()
                self.input_deck.append(card)
            _m_nps = re.match(" *dump no\. +[0-9].+nps = +([0-9]+)", line)
            if _m_nps:
                self.nps = int(_m_nps.group(1))

        self.cells = {}

        for index, line in enumerate(self.__outp_lines__):
            if re.match("^1cells", line):
                break
        else:
            warn("Could not find '1cell' (print table 60) in outp file")
            index = None
        if index is not None:
            index += 5
            while index < len(self.__outp_lines__):
                if self.__outp_lines__[index].split():
                    data = self.__outp_lines__[index].split()
                    cell_num = int(data[1])
                    self.cells[cell_num] = Cell(data, self)
                else:
                    break
                index += 1

    def get_tally(self, tally_number_or_name):
        return F4Tally(tally_number_or_name, self)

    def read_stopping_powers(self, particle, material_id=None, cell_num_4_density=None):
        particle = particle.lower()
        s = StoppingPowerData()
        if re.match('[0-9]+', particle):
            particle_type = 'heavy_ion'
        elif particle == 'electron' or particle == 'proton' or particle == 'positron':
            particle_type = particle
        else:
            particle_type = None
            warn('Reading stopping powers from {} type of particle might not behave normally. Verify.'.format(particle))

        if cell_num_4_density is not None:
            assert isinstance(cell_num_4_density, (int, str))
            cell_num_4_density = int(cell_num_4_density)
            assert cell_num_4_density in self.cells, "Cell {0} could not be found in output file!"\
                .format(cell_num_4_density)

            if material_id is not None:
                material_id = str(material_id)
                assert str(
                    self.cells[cell_num_4_density].mat) == material_id, 'material_id and material of cell {} are ' \
                                                                        ' inconsistent. '.format(cell_num_4_density)
            else:
                material_id = self.cells[cell_num_4_density].mat
            s.cell_density = self.cells[cell_num_4_density].density


        else:
            assert material_id is not None, "At least one of `material_id` or `cell_num` must be given."
            material_id = str(material_id)

        if particle == 'electron':
            electron_flag = True  # just for speed
            c = re.compile('1range +table .+material +{} .+ print table 85'.format(material_id))
        else:
            electron_flag = False
            c = re.compile("1.*{0}.+{1}.+print table 85".format(particle, material_id))

        looking_for_beginning = False
        for index, line in enumerate(self.__outp_lines__):
            if c.match(line) and not looking_for_beginning:
                if electron_flag:
                    assert 'electron' in self.__outp_lines__[index+2]
                    looking_for_beginning = True
                else:
                    index += 8  # beginning of dEdx data begins 8 lines after the start of print table 85
                    break
            elif looking_for_beginning:
                if re.match(' +[0-9]+ +([0-9.E+-]+ *){11}', line):
                    break
        else:
            assert False, "Could not find dEdx table for '{0}' and material '{1}'".format(particle, material_id)
        length = int(self.__outp_lines__[index].split()[0])  # This is the number of data entries in the table.
        ergs = []
        dedxs = []
        ranges = []

        # add to these upon further insights if needed
        total_dedx_index = {'ion': 6, 'proton': 6, 'electron': 4}.get(particle_type, -6)
        range_index = {'ion': -3, 'proton': -3, 'electron': 5}.get(particle_type, -3)

        for index in range(index, index + length):
            values = list(map(float, (self.__outp_lines__[index].split())))
            ergs.append(values[1])
            dedxs.append(values[total_dedx_index])
            ranges.append(values[range_index])

        s.ranges = np.array(ranges)
        s.energies = np.array(ergs)
        s.dedxs = np.array(dedxs)
        s.par = particle
        s.mat = material_id
        return s


class StoppingPowerData:
    def __init__(self):
        self.__energies__: np.ndarray = None  # MeV
        self.ranges: np.ndarray = None  # cm
        self.dedxs: np.ndarray = None   # MeV/(g/cm2)
        self.par = None
        self.mat = None
        self.cell_density = None
        self.erg_bin_widths = None
        self.__dx_de__ = 0

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

    def plot_dedx(self, ax=None, label=None, title=None, material_name_4_title=None, density=None):
        if ax is None:
            fig, ax = plt.subplots()

        y = self.dedxs
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
        if density is None:
            ax.set_ylabel("dEdx [MeV cm2/g]")
        else:
            ax.set_ylabel("dEdx [MeV/cm]")

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if title is None:
            title = ""
            if material_name_4_title is not None:
                title += "{0} in material {1}".format(self.par, material_name_4_title)
            else:
                title += "{0} in material {1}".format(self.par, self.mat)

            if density is not None:
                title += " density: {0:.4E} g/cm3".format(density)

            ax.set_title(title)
        return ax

    def plot_range(self, ax=None, label=None, title=None, material_name_4_title=None, particle_name_4_title=None,
                   density=None, use_best_units=True):

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
        conversion, units = 1, 'cm'

        if density is None:
            ax.set_ylabel("range [g/cm2]")
        else:
            if use_best_units:
                y /= 100
                mean_range_in_meters = np.max(y)
                unit_converts = [(10**-3, 'km'), (10**1, 'm'), (10**2, 'cm'), (10**3, 'mm'), (10**6, 'μm'), (10**9, 'nm')]
                conversion, units = \
                    unit_converts[int(np.argmin([abs(mean_range_in_meters*c-1) for c, unit in unit_converts]))]

            ax.set_ylabel("range [{}]".format(units))

        ax.plot(self.energies, y*conversion, label=label)

        if label is not None:
            ax.legend()

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

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

            if material_name_4_title is not None:
                title += "Range of {0} in {1}".format(par_name, material_name_4_title)
            else:
                title += "Range of {0} in material {1}".format(par_name, self.mat)

            if density is not None:
                title += "; material density: {0:.4E} g/cm3".format(density)

            ax.set_title(title)
        return ax

    @classmethod
    def get_stopping_power(cls, particle, material_element_symbols, grams_per_cm3, material_atom_percents=None,
                           material_mass_percents=None, gas=False, emax=200, temperature=None, mcnp_command='mcnp6',
                           verbose=False):
        assert platform.system() != 'Windows', '`get_stopping_power` is not compatible with windows systems!'
        assert isinstance(gas, bool), '`gas` argument must be either True or False'
        assert isinstance(grams_per_cm3, Number), '`grams_per_cm3` argument must be a number'
        grams_per_cm3 = abs(grams_per_cm3)
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
            assert all([isinstance(f, Number) for f in material_atom_percents]),\
                '`element_atom_percents` must be float(s)'
        if material_mass_percents is not None:
            assert all([isinstance(f, Number) for f in material_mass_percents]),\
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
            mass_flag = True
        else:
            atom_fractions = material_atom_percents
            mass_flag = False
        assert len(material_element_symbols) == len(atom_fractions), '`element_symbols` arg must be equal in length to atom ' \
                                                            'fractions'
        m = Material(temperature=temperature)
        percent_type = 'wo' if mass_flag else 'ao'
        for element, fraction in zip(material_element_symbols, atom_fractions):
            if re.match('[a-zA-Z]+[0-9]+', element):
                m.add_nuclide(element, fraction, percent_type)
            else:
                m.add_element(element, fraction, percent_type)
        m.set_density('g/cm3', grams_per_cm3)
        mcnp_zaids = []
        mcnp_atom_percents = []
        for key, value in m.get_nuclide_atom_densities().items():
            s = value[0]
            d = value[1]
            mcnp_zaids.append(get_zaid(s))
            mcnp_atom_percents.append(d)
        mcnp_atom_percents = np.array(mcnp_atom_percents)
        mcnp_atom_percents /= sum(mcnp_atom_percents)
        mat_card = 'M1000\n' + '\n'.join(['{s}{zaid} {atomic_fraction:.5E}'
                             .format(s=' '*5, zaid=zaid,atomic_fraction=atomic_fraction)
                              for zaid, atomic_fraction in zip(mcnp_zaids, mcnp_atom_percents)])
        cwd = Path(__file__).parent
        if gas:
            mat_card += '\n     GAS=1'
        with open(cwd/'mcnp_basic_inp') as f:
            lines = f.readlines()

        with open(cwd/'__temp__.inp', 'w') as f:
            for line in lines:
                new_line = line.format(**locals())
                f.write(new_line)
                if verbose:
                    print(new_line)

        cmd = 'cd {};'.format(cwd)
        cmd += '{} i=__temp__.inp '.format(mcnp_command)
        outp_path = (cwd/'outp')
        runtpe_path = (cwd/'runtpe')
        if outp_path.exists():
            outp_path.unlink()
        if runtpe_path.exists():
            runtpe_path.unlink()

        # print(os.system(cmd), 'ppps')
        # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out = p.stdout.read()
        # (out, err) = proc.communicate()
        if runtpe_path.exists():
            runtpe_path.unlink()
        # todo: find a way top make sure mcnp6 command can be found.
        outp = OutP(outp_path)

        # outp_path.unlink()
        return outp.read_stopping_powers(str(outp_par), 1000, 10)


if __name__ == "__main__":
    # test_outp = (Path(__file__).parent.parent / "FFandProtonSims" / "Protons" / "outp_saved")
    # o = OutP(test_outp)
    # d = o.read_stopping_powers("proton", 2000)
    # d.plot_dedx()
    # d.plot_range()
    # plt.show()
    s_h = StoppingPowerData.get_stopping_power('Cs133', ['U'], 19, material_atom_percents=[1, 2])
    s_l = StoppingPowerData.get_stopping_power('Tc99', ['U'], 19, material_atom_percents=[1, 2])

    ax = s_h.plot_range(material_name_4_title='U', label='Cs133')
    s_l.plot_range(ax, material_name_4_title='U', label='Tc99')
    ax.set_title("Range of typical light and heavy FF's in U238")
    print('Heavy fragment range in U: {} um'.format( s_h.get_range_at_erg(70)*1E4))
    print('Light fragment range in U {} um'.format( s_l.get_range_at_erg(105)*1E4))
    plt.show()


