from __future__ import annotations
import re
import numpy as np
import os
from warnings import warn
from uncertainties import ufloat, UFloat
import uncertainties.unumpy as unp
from matplotlib import pyplot as plt
from pathlib import Path
from typing import List, Dict, Set, Iterable, Sized, Union, Type, Optional
from typing import List, Dict, Set, Iterable, Sized, Union, Type, Optional
from numbers import Number
from JSB_tools.MCNP_helper.stopping_power import MCNPStoppingPowerData
import warnings
try:
    from openmc import Material
    from openmc.data import ATOMIC_NUMBER, ATOMIC_SYMBOL
except ModuleNotFoundError:
    from JSB_tools import no_openmc_warn
    no_openmc_warn()
import hashlib
import platform
import subprocess
from functools import cached_property
import pickle
from pickle import UnpicklingError
from JSB_tools import mpl_hist
from scipy.integrate import cumulative_trapezoid, trapezoid
#  Todo:
#   Add function to access num particles entering cells


class Tally:
    def __init__(self):
        self.energies: Sized
        self.flux: UFloat
        self.fluxes: Sized
        self.cells: Set[OutpCell]
        self.rates: Sized
        self.rate: UFloat
        self.tally_name: str
        self.tally_modifiers: Set[str]


class F8Tally:
    card_match = re.compile('[+*]?F(?P<number>[0-9]*8)', re.IGNORECASE)
    mev2j = 1.602E-13

    def __init__(self, tally_number, outp, units='MeV'):
        units = units.lower()
        if units != 'mev':
            erg_scale = {'kev': 1E3, 'ev': 1E6}[units]
        else:
            erg_scale = 1

        assert isinstance(outp, OutP)
        self.tally_number = str(tally_number)
        index = outp.__find_tally_indicies__[self.tally_number]

        self.counts = []  # counts
        erg_bins = None  # used as a flag
        self.underflow = 0

        self.cells = []  # either None's or cell numbers in same shape as self.counts
        read_flag = False
        cell_num = None

        while index < len(outp.__outp_lines__):
            line = outp.__outp_lines__[index]
            if m := re.match(' +cell + ([0-9]+)', line):
                cell_num = int(m.groups()[0])

                read_flag = True

                index += 1
                line = outp.__outp_lines__[index]

            elif re.match(' +energy', line):
                read_flag = True

                index += 1
                line = outp.__outp_lines__[index]

            if read_flag:
                if re.match(' +([0-9.E+-]+) +([0-9.E+-]+) *$', line):
                    val, rel_err = map(float, line.split())
                    err = val * rel_err
                    val = ufloat(val, err)
                    self.cells.append(cell_num)
                    self.counts.append(val)

                elif re.match(" +([0-9.E+-]+) +([0-9.E+-]+) +([0-9.E+-]+)", line):
                    erg, val, rel_err = map(float, line.split())
                    err = val * rel_err
                    val = ufloat(val, err)

                    scaled_erg = erg_scale * erg  # erg in correct units

                    self.cells.append(cell_num)

                    if erg_bins is None:  # first row
                        self.underflow = val
                        erg_bins = [scaled_erg]  # starting bin
                    else:
                        erg_bins.append(scaled_erg)
                        self.counts.append(val)

                    index += 1
                    line = outp.__outp_lines__[index]

                    if 'total' in line:
                        break
                    else:
                        continue

            if re.match(" *=+", line):
                break

            index += 1

        self.counts = unp.uarray([float(x.n) for x in self.counts], [x.std_dev for x in self.counts])
        self.cells = np.array(self.cells)

        if erg_bins is not None:
            self.erg_bins = np.array(erg_bins)
        else:
            self.erg_bins = None

    def select_erg_range(self, emin=None, emax=None):
        if emin is not None:
            i0 = np.searchsorted(self.erg_bins, emin, side='right') - 1
        else:
            i0 = 0

        if emax is not None:
            i1 = np.searchsorted(self.erg_bins, emax, side='right') - 1
        else:
            i1 = len(self.energies)

        self.erg_bins = self.erg_bins[i0: i1 + 1]

    @property
    def energies(self):
        if self.erg_bins is None:
            return None

        return 0.5 * (self.erg_bins[1:] + self.erg_bins[:-1])

    @property
    def bin_widths(self):
        return self.erg_bins[1:] - self.erg_bins[:-1]

    def plot(self, ax=None, nominal_values=False, unit='keV', **kwargs):
        if unit == 'keV':
            xscale = 1E3
        elif unit == 'MeV':
            xscale = 1
        else:
            raise ValueError

        if ax is None:
            fig, ax = plt.subplots()

        if nominal_values:
            counts = unp.nominal_values(self.counts)
        else:
            counts = self.counts

        mpl_hist(xscale * self.erg_bins, counts, ax=ax, **kwargs)
        return ax

    @staticmethod
    def get_bins_card(emin, emax, nbins):
        """
        Return the string for creating uniform bins from emin to emax.
        Args:
            emin:
            emax:
            nbins:

        Returns:

        """
        if emin == 0:
            emin = 1E-5

        return f'{emin} {nbins - 1}i {emax}'


class F6Tally:
    card_match = re.compile('[+*]?F(?P<number>[0-9]*6)(?:: *[a-z, ]+)? +(?P<cell>[0-9]+)', re.IGNORECASE)
    mev2j = 1.602E-13

    def __init__(self, tally_number_or_name, outp):
        assert isinstance(outp, OutP)

        if isinstance(tally_number_or_name, str):
            tally_number_or_name = tally_number_or_name.lower()
            try:
                card = outp.__named_cards__[tally_number_or_name]['card']
            except KeyError:
                raise KeyError(f'could not find F6 tally with name {tally_number_or_name}')
            m = F6Tally.card_match.match(card)
            assert m, f"Invalid F6 tally match from card: {card}"
            self.tally_number = m.groupdict()['number']
            self.cell_number = int(m.groupdict()['cell'])
        else:
            assert isinstance(tally_number_or_name, (int, str))
            tally_number_or_name = str(tally_number_or_name)
            for card in outp.input_deck:
                if m := F6Tally.card_match.match(card):
                    self.tally_number = m.groupdict()['number']
                    self.cell_number = int(m.groupdict()['cell'])
                    if self.tally_number == tally_number_or_name:
                        break
            else:
                raise KeyError(f'could not find F6 tally with number {tally_number_or_name}')

        self.cell = outp.cells[self.cell_number]

        index = outp.__find_tally_indicies__[self.tally_number]
        self.heating = None
        self.units = ''
        self.mass = None

        while index < len(outp.__outp_lines__):
            line = outp.__outp_lines__[index]
            if m := re.match(' +tally type 6.+units +(.+)', line):
                self.units = m.groups()[0].rstrip().lstrip()

            elif re.match(' +masses', line):
                index += 2
                self.mass = float(outp.__outp_lines__[index])

            elif re.match(fr' +cell +{self.cell.number}', line):
                index += 1
                value, err = tuple(map(float, outp.__outp_lines__[index].split()))

                self.heating = ufloat(value, value*err)
                break
            elif re.match(" =+ ", line):
                assert False, f'Failed to find tally {tally_number_or_name}'
            index += 1

        self.heating_joules = None
        if 'mev' in self.units:
            self.heating_joules = self.mev2j*self.heating


class OutpCell:
    def __init__(self, data, outp=None):
        self.number = int(data[1])
        self.mat = int(data[2])
        self.atom_density = float(data[3])  # atoms/barn*cm
        self.density = float(data[4])
        self.volume = float(data[5])
        self.mass = float(data[6])
        self.outp = outp
        self.name = None

        if self.outp is not None:
            assert isinstance(outp, OutP)

    def __repr__(self):
        return "Cell {0}, mat:{1}, name: {2}".format(self.number, self.mat, self.name)

    def get_f4_tally(self) -> F4Tally:
        assert len(self.f4_tallys) <= 1, f'Multiple tallys for cell {self.number}. Use `get_tallys`.'
        assert len(self.f4_tallys) > 0, f'No tally_n for cell {self.number}.'
        return self.f4_tallys[0]

    @cached_property
    def f4_tallys(self) -> List[F4Tally]:
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
                    if cell_num == self.number:
                        tallies.append(self.outp.get_f4_tally(tally_num))
        return tallies


class F4Tally(Tally):
    """
    Todo: Merge this with inputdeck.py/F4Tally.
     Also, rework things. THERE HAS TO BE A BETTER WAY TO COLLECT THE DATA FROM OUTP.

    Load tally data from Outp

    Attributes:
        fluxes: The fluxes from the tally in 1/cm2 (note that first bin is an underflow bin and is not included.)
        energies: Centers of tally energy bins
        dx_per_src: self.fluxes*cell_volume (i.e. track length traversed per source particle).


    """
    tally_name_match = re.compile(
        r' *f(?P<tally_num>[0-9]*[48]):(?P<particle>.) +(?P<cell>[0-9]+) *(?:\$.*name: *(?P<name>[\w -_]*\w) *)?')

    @staticmethod
    def __find_tally_cards__(cards):
        outs = {}
        for card in cards:
            if m := F4Tally.tally_name_match.match(card):
                num = m['tally_num']
                result = {'name': m['name'], 'cell': m['cell'], 'particle': m['particle']}
                outs[num] = result
        return outs

    #  todo: Verify summing methods.
    def __init__(self, tally_number_or_name=None, outp=None, __copy__tally__=None):
        self.energy_bins = None
        if __copy__tally__ is not None:
            assert isinstance(__copy__tally__, F4Tally)
            self.__flux__ = __copy__tally__.__flux__
            self.__fluxes__ = __copy__tally__.__fluxes__
            self.cells = __copy__tally__.cells
            self.underflow = __copy__tally__.underflow
            self.tally_name = __copy__tally__.tally_name
            self.tally_modifiers = __copy__tally__.tally_modifiers
            self.energies = __copy__tally__.energies

        else:
            assert outp is not None
            assert tally_number_or_name is not None
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
            # find tally_n number is not given
            found_tally = False
            names_found = []

            if self.tally_number is None:
                for tally_num, tally_info in F4Tally.__find_tally_cards__(outp.input_deck).items():
                    name = tally_info['name']
                    names_found.append(name)
                    if name == self.tally_name:
                        self.tally_number = tally_num
                        found_tally = True
                        break
                else:
                    assert False, \
                        '\nCould not find tally_n with name "{0}"\nExample of using a name tag to access tally_n:' \
                        '\nF84:p 13 $ name:<tally_name_here>\nThe key usage syntax is in the comment. The text ' \
                        'after the string "name:" is the name tag (names are case insensitive)\n' \
                        'Names found:\n{1}' \
                        .format(self.tally_name, names_found)

            if self.tally_name is None:
                self.tally_name = str(self.tally_number)

            tally_declaration_match = re.compile(r' *f{}:(?P<particle>.) +(?P<cell>[0-9]+) *'.format(self.tally_number))

            tally_modifier_match1 = re.compile('^([etc]){}:?'.format(self.tally_number))
            tally_modifier_match2 = re.compile('(fm|em|tm|cm){}'.format(self.tally_number))

            __cell__: OutpCell = None
            self.particle: str = None
            #  Todo: make fm, em, ect compatible. Do these multipliers change the form of the outp?
            for card in outp.input_deck:
                card = card.lower()
                if _m := tally_declaration_match.match(card):
                    cell_number = int(_m.group('cell'))
                    assert cell_number in outp.cells, 'Invalid cell number for tally_n {}. Card:\n\t{}' \
                        .format(self.tally_number, card)
                    __cell__ = outp.cells[cell_number]
                    self.particle = _m.group('particle')
                    found_tally = True
                elif _m := tally_modifier_match1.match(card):
                    self.tally_modifiers.add(_m.groups()[0])
                elif _m := tally_modifier_match2.match(card):
                    self.tally_modifiers.add(_m.groups()[0])

            self.cells: Set[OutpCell] = {__cell__}  # list of cells for when self.__add__ etc. is used

            if not found_tally:
                assert False, "Could not find tally_n {} in input deck!".format(self.tally_number)

            assert self.tally_number[-1] == "4", "Incorrect tally_n type!"
            self.tally_number = int(self.tally_number)

            found_n_tallies = 0
            f4tallies_found = set()
            tally_begin_index = None
            for index, line in enumerate(outp.__outp_lines__):
                if any_f4_tally_m := re.match('1tally +([0-9]*4)', line):
                    f4tallies_found.add(any_f4_tally_m.groups()[0])
                _m = re.match(r"1tally +{}".format(self.tally_number), line)
                if _m:
                    found_n_tallies += 1
                    tally_begin_index = index
            if found_n_tallies == 0:
                msg = "\nCannot find tally number {} in '{}'".format(self.tally_number, outp.__f_path__.relative_to(
                    outp.__f_path__.parent.parent.parent))
                if len(f4tallies_found):
                    msg += '\nF4 tallies found:\n{}'.format(f4tallies_found)
                else:
                    msg += '\nNo tallys found in output file!'
                assert False, msg
            # elif found_n_tallies > 1:
            #     warn('Several dumps of tally_n {0} found. Using last entry.'.format(self.tally_number))
            index = tally_begin_index
            # initialize
            if len((self.tally_modifiers - {'e', 'fm'})) == 0:
                index += 4
                if "volumes" in outp.__outp_lines__[index]:
                    index += 1

                index += 1
            else:
                assert False, "Tally modifiers {} not yet supported!".format(self.tally_modifiers)
            if (self.tally_modifiers - {'fm'}) == set():
                if 'fm' in self.tally_modifiers:
                    index += 1
                self.underflow = None
                self.energies = np.array([])
                self.__fluxes__ = np.array([])
                if not re.match('.+[0-9.E+-]+ [0-9.E+-]+', outp.__outp_lines__[index+2]):  # bug? Whatever.
                    index += 1
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
                    assert False, "Could not find energy data for tally_n {}".format(self.tally_number)

                self.energy_bins = []
                fluxes = []
                flux_errors = []

                while index < len(outp.__outp_lines__):
                    line = outp.__outp_lines__[index]
                    _m = re.match(r" +[0-9\.E+-]+", line)

                    if _m:

                        try:
                            erg_bin, flux, rel_error = tuple(map(float, line.split()))
                        except ValueError as e:
                            assert False, 'Error parsing tally_n {0}. Outp line:\n{1}\n{2}'.format(self.tally_number,
                                                                                                 line, e)

                        fluxes.append(flux)
                        flux_errors.append(flux*rel_error)
                        self.energy_bins.append(erg_bin)
                        index += 1
                    else:
                        break
                self.underflow = ufloat(fluxes[0], flux_errors[0])
                self.__fluxes__ = unp.uarray(fluxes, flux_errors)[1:]
                self.energies = np.array([0.5 * (b_low+b_high) for b_low, b_high in
                                          zip(self.energy_bins[:-1], self.energy_bins[1:])])

                self.energy_bins = np.array(self.energy_bins)
                _flux = float(outp.__outp_lines__[index].split()[1])
                _flux_error = _flux * float(outp.__outp_lines__[index].split()[2])
                self.__flux__ = ufloat(_flux, _flux_error)  # total flux
            else:
                assert False, "Tally modifiers {} not supported yet! (from tally_n {})"\
                    .format(self.tally_modifiers, self.tally_number)

    def __assert_no_fm__(self):
        if 'fm' in self.tally_modifiers:
            assert False, 'Tally {} has the "FM" modifier. Tally is a reaction rate, not a flux.'\
                .format(self.tally_number)

    def __assert_fm__(self):
        if 'fm' not in self.tally_modifiers:
            assert False, 'Tally {} does not have "FM" modifier. Tally is flux, not reaction rate.' \
                .format(self.tally_number)

    @property
    def per_barn(self):
        """
        Return track length times atoms per barn per cm. Resulting units are barn^-1
        """
        return self.dx_per_src*self.cell.atom_density

    @property
    def density(self):
        return self.cell_mass/self.total_volume

    @property
    def cell_mass(self):
        return self.cell.volume*self.cell.density

    @property
    def atom_density(self):
        return self.cell.atom_density

    @property
    def cell(self) -> OutpCell:
        assert len(self.cells) == 1,\
            'Multiple cells are used for this tally_n since __add__, ect was used. Use self.cells instead'
        return next(iter(self.cells))

    @property
    def dx_per_src(self):
        assert len(self.fluxes) > 0, "can't use `dx_per_src` unless tally_n has bins. Use `total_dx_per_src` instead"
        return self.total_volume*self.__fluxes__

    @property
    def total_dx_per_src(self):
        return self.total_volume * self.__flux__

    @property
    def nominal_fluxes(self):
        self.__assert_no_fm__()
        return unp.nominal_values(self.fluxes)

    @property
    def std_devs_of_fluxes(self):
        self.__assert_no_fm__()
        return unp.std_devs(self.fluxes)

    @ property
    def flux(self):
        self.__assert_no_fm__()
        return self.__flux__

    @property
    def fluxes(self):
        self.__assert_no_fm__()
        return self.__fluxes__

    @property
    def rate(self):
        self.__assert_fm__()
        return self.__flux__

    @property
    def rates(self):
        self.__assert_fm__()
        return self.__fluxes__

    @property
    def erg_bin_widths(self):
        out = [e2 - e1 for e1, e2 in zip(self.energy_bins[:-1], self.energy_bins[1:])]
        return np.array(out)

    @property
    def total_volume(self):
        return sum([c.volume for c in self.cells])

    def interp_energy(self, new_ergs):
        out = unp.uarray(np.interp(new_ergs, self.energies, unp.nominal_values(self.__fluxes__)),
                         np.interp(new_ergs, self.energies, unp.std_devs(self.__fluxes__)))
        return out

    def __repr__(self):
        return 'F4 Tally ({2}) in cell {0}, With {1} modifier.'.format(self.cell.number, self.tally_modifiers,
                                                                       self.tally_number)

    def __copy__(self):
        return F4Tally(None, None, self)

    def __add__(self, other):
        copied_tally = self.__copy__()
        copied_tally += other
        return copied_tally

    def __iadd__(self, other, subtract=False):
        if subtract:
            c = -1
        else:
            c = 1
        if isinstance(other, F4Tally):
            assert self.tally_modifiers == other.tally_modifiers,\
                'Tally {} and tally_n {} do not have same bins/modifiers'.format(self.tally_name, other.tally_name)
            assert len(self.energies) == len(other.energies), \
                'Tally {} and tally_n {} do not have same length'.format(len(self.__fluxes__), len(other.__fluxes__))
            self.tally_name += '{} {}'.format('-' if subtract else '+', other.tally_name)
            new_total_volume = other.total_volume + self.total_volume

            def volume_weighted_mean(self_x, other_x):
                return (self_x * self.total_volume + other_x * other.total_volume) / new_total_volume

            self.__fluxes__ = c*volume_weighted_mean(self.__fluxes__, other.__fluxes__)
            self.__flux__ = c*volume_weighted_mean(self.__flux__, other.__flux__)
            self.underflow += other.underflow
            self.cells = self.cells.union(other.cells)

        elif isinstance(other, Sized):
            assert len(other) == len(self.__fluxes__), 'Adding iterable of incompatible length to tally_n: {} != {}'\
                .format(len(self.__fluxes__), other.__len__())
            if isinstance(other, list):
                if isinstance(other[0], UFloat):
                    other = unp.uarray([n.n for n in other], [n.std_dev for n in other])
                else:
                    other = np.array(other)
            self.__fluxes__ += c*other
            self.__flux__ += c*np.sum(other)
            self.tally_name += ' (modified)'
        else:
            assert False, 'Invalid type passed to F4Tally.__iadd__(), `{}`'.format(type(other))

        return self

    def __isub__(self, other):
        self.__iadd__(other, True)
        return self

    def __sub__(self, other):
        copied_tally = self.__copy__()
        copied_tally -= other
        return copied_tally

    def plot(self, ax=None, track_length=True, title=None, label=None, norm=1, ylabel=None):
        assert self.energy_bins is not None, 'Tally has no energy bins. Cannot plot.'

        if ax is not None:
            if ax is plt:
                ax = ax.gca()
        else:
            _, ax = plt.subplots(1,1)

        if title is None:
            title = self.tally_name
        ax.set_title(title)
        ax.set_xlabel("Energy [Mev]")
        if 'fm' in self.tally_modifiers:
            ax.set_ylabel('reaction rate')
            c = norm
        else:
            if track_length:
                ax.set_ylabel('Track length [cm]')
                c = self.total_volume * norm
            else:
                ax.set_ylabel('Particle flux [1/cm^2]')
                c = norm

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        mpl_hist(self.energy_bins, c*self.fluxes, label=label, ax=ax)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
        return ax


def load_globals(pickle_path):
    # pickle_path = Path(self.input_file_path).with_suffix('.pickle')
    out = {}
    assert pickle_path.exists(), 'Pickle file of globals does not exist for {}'.format(pickle_path)
    with open(pickle_path, 'rb') as f:
        while True:
            try:
                var_name, var = pickle.load(f)
                out[var_name] = var
            except (EOFError, UnpicklingError):
                break
    return out


def read_input_file_lines(all_lines):
    out = {'input_deck': [], 'input_file_name': None}
    begin_flag = False
    for line in all_lines:
        if m := re.match("^ {0,9}[0-9]+- {7}(.*)", line):
            if not begin_flag:
                begin_flag = True

            card = m.group(1).rstrip().lower()
            out['input_deck'].append(card)
        else:
            if begin_flag and re.match(r"^ *\*+ *$", line):
                break

        if out['input_file_name'] is None:
            if m := re.match('.+i=([^ ]*)', line):
                out['input_file_name'] = m.groups()[0].strip()

    return out


class OutP:
    def __init__(self, file_path):
        self.__f_path__ = Path(file_path)
        with open(file_path) as f:
            self.hash = self.get_hash(file_path)
            f.seek(0)

            self.__outp_lines__ = f.readlines()
        if not re.match(' +.+Version *= *MCNP', self.__outp_lines__[0], re.IGNORECASE):
            warn('\nThe file\n"{}"\ndoes not appear to be an MCNP output file!\n'.format(file_path))

        inp_data = read_input_file_lines(self.__outp_lines__)

        self.input_deck = inp_data['input_deck']

        self.nps = self.ctime = None

        self.input_file_path = self.__f_path__.parent / inp_data['input_file_name']

        nps_match = re.compile(" +dump no. *[0-9]+.+nps ?= *([0-9]+).+ctm ?= *([0-9.]+)")  # matched dump line containing NPS and CTIME info
        for line in self.__outp_lines__[::-1]:
            if m := nps_match.match(line):
                self.nps = int(m.groups()[0])
                self.ctime = float(m.groups()[1])
                break

        if not len(self.input_deck):
            raise ValueError("No input deck found in Outp file. Possible that file is not an MCNP outp file.")

        self.inp_title = self.input_deck[0]

        self.cells: Dict[int, OutpCell] = {}

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
                    self.cells[cell_num] = OutpCell(data, self)
                else:
                    break
                index += 1
        cell_match = re.compile(r'([0-9]+).*(?:name: ?(.+))')
        for card in self.input_deck:  # get cell names
            if re.match(r'^\s*$', card):
                break
            if m := cell_match.match(card):
                cell_num = int(m.groups()[0])
                self.cells[cell_num].name = m.groups()[1]

    @staticmethod
    def get_hash(path):
        with open(path, 'rb') as f:
            out = hashlib.md5(f.read()).hexdigest()

            return out

    def pickle(self, path=None):
        if path is None:
            path = self.__f_path__.with_suffix('.pickle')

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def pickle_out_of_date(self):
        text_path = self.__f_path__
        return OutP.get_hash(text_path) != self.hash

    @classmethod
    def from_pickle(cls, path):
        path = Path(path).with_suffix('.pickle')

        with open(path, 'rb') as f:
            self = pickle.load(f)
        return self

    def get_cell_activity(self, particle):
        """
        Table 126. Print # of particles entering cells
        Args:
            particle:

        Returns:
            Dict[cell_num, data_dict]

        """
        flag = False
        out = {}
        for line in self.__outp_lines__:
            if flag:
                if re.match(' +[0-9]+ +[0-9]+ +[0-9]+.+', line):
                    vals = line.split()
                    cell_num = int(vals[1])
                    out[cell_num] = {'entering':  int(vals[2]),
                                     'population':  int(vals[3]),
                                     }
                if re.match(' +total.+', line):
                    break

            elif re.match(f'1{particle.lower()} +activity in each cell +print table 126', line):
                flag = True
        else:
            raise Exception("WTF")

        return out

    @cached_property
    def __find_tally_indicies__(self):
        """
        Locates the location in the output file where every tally results are written.

        Finds the line indices for the final time each tally's data from the problem is written to the output file.

        Returns: Dict[tally_number, card_index]
            , where card_index can be used as in Outp.__outp_lines__[card_index] to get the text of the input card.
        """
        out = {}
        matcher = re.compile('1tally +([0-9]*[1245678]) ?')
        for index, line in enumerate(self.__outp_lines__):
            if line[:6] == '1tally':
                if m := matcher.match(line):
                    out[m.groups()[0]] = index
        return out

    @cached_property
    def __named_cards__(self) -> Dict[str, dict]:
        """
        Find named cards.
        Returns: Dict[name, {'card': `card_text`, 'index': card_index}]
            , where card_index can be used as in Outp.__outp_lines__[card_index] to get the text of the input card.

        """
        matcher = re.compile('.+name: (.+)')
        out = {}
        for index, line in enumerate(self.input_deck):
            if m := matcher.match(line):
                name = m.groups()[0].rstrip().lstrip()
                out[name] = {'card': line, 'index': index}
        return out

    def get_cell_by_name(self, name: str):
        match = name.rstrip().lstrip().lower()
        for _, cell in self.cells.items():
            if match == cell.name:
                out = cell
                break
        else:
            assert False, f"No cell named {name} found! {[c.name for c in self.cells.values()]}"
        return out

    def get_f4_tally(self, tally_number_or_name):
        return F4Tally(tally_number_or_name, self)

    def get_f6_tally(self, tally_number_or_name):
        return F6Tally(tally_number_or_name, self)

    def get_f8_tally(self, tally_number, erg_units='Mev'):
        return F8Tally(tally_number, self, units=erg_units)

    def get_globals(self):
        pickle_path = Path(self.input_file_path).with_suffix('.pickle')
        return load_globals(pickle_path)



if __name__ == "__main__":
    o = OutP('/Users/burggraf1/PycharmProjects/ISU/darkMatter/mcnp/0_inp/outp')
    print(o.get_f6_tally('Al heat').heating_joules*6.24E14)
    print(o.get_f6_tally('C13 heat').heating_joules*6.24E14)

    # test_outp = (Path(__file__).parent.parent / "FFandProtonSims" / "Protons" / "outp_saved")
    # o = OutP(test_outp)
    # d = o.read_stopping_powers("proton", 2000)
    # d.plot_dedx()
    # d.plot_range()
    # plt.show()
    # s_h = StoppingPowerData.get_stopping_power('Cs133', ['U'], 19, material_atom_percents=[1, 2])
    # s_l = StoppingPowerData.get_stopping_power('Tc99', ['U'], 19, material_atom_percents=[1, 2])
    #
    # ax = s_h.plot_range(material_name_4_title='U', label='Cs133')
    # s_l.plot_range(ax, material_name_4_title='U', label='Tc99')
    # ax.set_title("Range of typical light and heavy FF's in U238")
    # print('Heavy fragment range in U: {} um'.format( s_h.get_range_at_erg(70)*1E4))
    # print('Light fragment range in U {} um'.format( s_l.get_range_at_erg(105)*1E4))
    # plt.show()


