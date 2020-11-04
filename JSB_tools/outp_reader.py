import re
import numpy as np
import os
from warnings import warn
from uncertainties import ufloat
import uncertainties.unumpy as unp
from matplotlib import pyplot as plt
from pathlib import Path


class F4Tally:
    def __init__(self, tally_number, outp):
        self.tally_number = tally_number
        assert isinstance(outp, OutP)
        tally_number = str(tally_number)

        assert tally_number[-1] == "4", "Incorrect tally!"

        tally_modifiers = set()

        for card in outp.input_deck:
            _m = re.match("^([fcemt]+){}.+".format(tally_number), card)
            if _m:
                if _m.group(1) != "f":
                    tally_modifiers.add(_m.group(1))
        for index, line in enumerate(outp.__outp_lines__):
            _m = re.match(r"1tally +{} +nps".format(tally_number), line)
            if _m:
                break
        else:
            assert False, "Cannot find tally {}".format(tally_number)

        # initialize
        if tally_modifiers == set() or tally_modifiers == {"e"}:
            index += 2
            _m = re.match(r" +particle\(s\): +([a-z]+)", outp.__outp_lines__[index])
            if _m:
                self.particle = _m.group(1)
            else:
                warn("Could not find particle for tally {}".format(tally_number))
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
                warn("Could not find cell for tally {0}".format(tally_number))

            index += 1
        else:
            assert False, "Tally modifiers {} not yet supported!".format(tally_modifiers)

        if tally_modifiers == set():
            self.underflow = None
            self.energies = np.array([])
            self.fluxes = np.array([])
            _flux = float(outp.__outp_lines__[index+2].split()[0])
            _flux_error = _flux*float(outp.__outp_lines__[index+2].split()[1])
            self.flux = ufloat(_flux, _flux_error)

        elif tally_modifiers == {"e"}:
            while index < len(outp.__outp_lines__):
                line = outp.__outp_lines__[index]
                if "energy" in line:
                    index += 1
                    break
                index += 1
            else:
                assert False, "Could not find energy data for tally {}".format(tally_number)

            self.__energy_bins__ = []
            fluxes = []
            flux_errors = []

            while index < len(outp.__outp_lines__):
                line = outp.__outp_lines__[index]
                _m = re.match(r" +[0-9\.E+-]+", line)

                if _m:
                    erg_bin, flux, rel_error = tuple(map(float, line.split()))
                    fluxes.append(flux)
                    flux_errors.append(flux*rel_error)
                    self.__energy_bins__.append(erg_bin)
                    index += 1
                else:
                    break
            self.underflow = ufloat(fluxes[0], flux_errors[0])
            self.fluxes = unp.uarray(fluxes, flux_errors)[1:]
            self.energies = np.array([0.5*(b_low+b_high) for b_low, b_high in
                                      zip(self.__energy_bins__[:-1], self.__energy_bins__[1:])])
            self.__energy_bins__ = np.array(self.__energy_bins__)
            _flux = float(outp.__outp_lines__[index].split()[1])
            _flux_error = _flux * float(outp.__outp_lines__[index].split()[2])
            self.flux = ufloat(_flux, _flux_error)

        else:
            assert False, "Tally modifiers {} not supported yet!".format(tally_modifiers)


class Cell:
    def __init__(self, data):
        self.cell_num = int(data[1])
        self.mat = int(data[2])
        self.atom_density = float(data[3])  # atoms/barn*cm
        self.density = float(data[4])
        self.volume = float(data[5])

    def __repr__(self):
        return "Cell {0}, mat:{1}".format(self.cell_num, self.mat)


class OutP:
    def __init__(self, file_path):
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
                    self.cells[cell_num] = Cell(data)
                else:
                    break
                index += 1

    def get_tally(self, tally_number):
        return F4Tally(tally_number, self)

    def read_stopping_powers(self, particle, material_id=None, cell_num=None):
        s = StoppingPowerData()

        if cell_num is not None:
            assert isinstance(cell_num, int)
            assert material_id is None,  "When using 'cell_num' don't supply a 'material_id'."
            try:
                material_id = self.cells[cell_num].mat
                s.cell_density = self.cells[cell_num].density
            except KeyError:
                assert False, "Cell {0} could not be found!".format(cell_num)
        else:
            assert material_id is not None, "Must supply 'material_id' or 'cell_num' arguments"

        c = re.compile("1.*{0}.+{1}.+print table 85".format(particle, material_id))

        for index, line in enumerate(self.__outp_lines__):
            if c.match(line):
                index += 8  # beginning of dEdx data begins 8 lines after the start of print table 85
                break
        else:
            assert False, "Could not find dEdx table for {0} and material {1}".format(particle, material_id)
        length = int(self.__outp_lines__[index].split()[0])  # This is the number of data entries kin the table.
        ergs = []
        dedxs = []
        ranges = []

        for index in range(index, index + length):
            values = list(map(float, (self.__outp_lines__[index].split())))
            ergs.append(values[1])
            dedxs.append(values[6])
            ranges.append(values[-3])

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

    @property
    def energies(self):
        return self.__energies__

    @energies.setter
    def energies(self, value):
        value = np.array(value)
        self.__energies__ = value
        self.erg_bin_widths = np.array([b2 - b1 for b1, b2 in zip(value[:-1], value[1:])])

    def eval_de_dx(self, erg):
        return np.interp(erg, self.energies, self.dedxs)

    def eval_dx_de(self, erg):
        return np.interp(erg, self.energies, 1.0/self.dedxs)

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

    def plot_range(self, ax=None, label=None, title=None, material_name_4_title=None, density=None):

        if ax is None:
            fig, ax = plt.subplots()

        y = self.ranges
        if density is not None:
            y = self.ranges/density
        else:
            if self.cell_density is not None:
                density = self.cell_density
                y = self.ranges/density

        ax.plot(self.energies, y, label=label)

        if label is not None:
            ax.legend()

        ax.set_xlabel("Energy [MeV]")

        if density is None:
            ax.set_ylabel("range [g/cm2]")
        else:
            ax.set_ylabel("range [cm]")

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if title is None:
            title = ""
            if material_name_4_title is not None:
                title += "{0} in material {1}".format(self.par, material_name_4_title)
            else:
                title += "{0} in material {1}".format(self.par, self.mat)

            if density is not None:
                title += "; density: {0:.4E} g/cm3".format(density)

            ax.set_title(title)
        return ax


if __name__ == "__main__":
    test_outp = (Path(__file__).parent.parent / "FFandProtonSims" / "Protons" / "outp_saved")
    o = OutP(test_outp)
    d = o.read_stopping_powers("proton", 2000)
    d.plot_dedx()
    d.plot_range()
    plt.show()



    # print(o.get_cell_population_info(14, "proton").n_par_entering)
    # print(o.nps)


#
# class F4Tally:
#     def __init__(self, tally_cards, tally_number, outp):
#         outp.seek(0)
#         outp = outp.read()
#         self.tally_start_line = None
#         self.tally_end_line = None
#         self.__outp_tally_only__ = outp.splitlines()
#         for index, line in enumerate(self.__outp_tally_only__):
#             if re.match("1tally +{} +nps += +".format(tally_number), line):
#                 self.tally_start_line = index
#                 continue
#             if self.tally_start_line is not None \
#                     and re.match("1tally +[0-9]+ +nps += +".format(tally_number), line):
#                 self.tally_end_line = index
#                 break
#         else:
#             self.tally_end_line = len(outp.splitlines())
#
#         assert self.tally_start_line is not None, "Couldn't find tally"
#
#         self.__outp_tally_only__ = "\n".join(self.__outp_tally_only__[self.tally_start_line: self.tally_end_line])
#
#         self.modifying_cards = set()
#         self.card = None
#         for card in tally_cards:
#             _m = re.match("^([fcemt]+){}.+".format(tally_number), card)
#             print("card:", card, _m)
#             assert _m
#             if _m.group(1) == "e":
#                 self.modifying_cards.add("e")
#             elif _m.group(1) == "f":
#                 self.card = card
#             elif _m.group(1) == "em":
#                 self.modifying_cards.add("em")
#             elif _m.group(1) == "t":
#                 self.modifying_cards.add("t")
#                 raise ValueError("T card not supported yet!")
#             elif _m.group(1) == "c":
#                 self.modifying_cards.add("C")
#                 raise ValueError("C card not supported yet!")
#             elif _m.group(1) == "cf":
#                 self.modifying_cards.add("cf")
#                 raise ValueError("Cell flagging (CF) not supported yet!")
#             else:
#                 raise ValueError("{} card not supported yet!".format(_m.group(1)))
#
#         cell_match = re.search(
#             "1tally +{} +nps = +(?:.*\n){{0,7}}.+volumes.*\n +cell: +(?P<cell>[0-9]+) +\n +(?P<cell_volume>[0-9.E+-]+)".
#             format(tally_number), self.__outp_tally_only__)
#
#         if cell_match:
#             self.cell = int(cell_match.group("cell"))
#             self.cell_volume = float(cell_match.group("cell_volume"))
#         else:
#             self.cell = None
#             self.cell_volume = None
#
#         flux_match = re.search(
#             r"1tally +{} +nps = +(?:.*\n){{0,7}}.+volumes.*\n +cell: +(?P<cell>[0-9]+)[ \n0-9.E+-]+cell +(?P=cell) [ \n]+(?P<flux>[.0-9E+-]+) +(?P<flux_error>[.0-9E+-]+)".format(tally_number), self.__outp_tally_only__)
#         if flux_match:
#             self.flux = float(flux_match.group("flux"))
#             self.flux_error = float(flux_match.group("flux_error"))
#         else:
#             self.flux = None
#             self.flux_error = None
#
#         self.energies = []
#         self.fluxes = []
#         self.flux_rel_errors = []
#         print("dsjgfkhed", self.modifying_cards)
#         if "e" in self.modifying_cards and self.modifying_cards == self.modifying_cards - {"c", "t"}:
#             data = re.findall(" +[0-9.E+-]+ +[0-9.E+-]+ +[0-9.E+-]+\n", self.__outp_tally_only__)
#
#             for string in data:
#                 erg, flux, flux_rel_error = map(float, string.split())
#                 self.energies.append(erg)
#                 self.fluxes.append(flux)
#                 self.flux_rel_errors.append(flux_rel_error)
#
#             _m = re.match(r"[\s\S]+?(?: +[0-9.E+-]+ +[0-9.E+-]+ +[0-9.E+-]+\n)+ +total +(?P<total>[0-9.E+-]+) +(?P<total_err>[0-9.E+-]+)", self.__outp_tally_only__)
#             if _m:
#                 self.flux = float(_m.group("total"))
#                 self.flux_error = float(_m.group("total_err"))
#
#         self.energies = np.array(self.energies)
#         self.fluxes = np.array(self.fluxes)
#         self.flux_rel_errors = np.array(self.flux_rel_errors)
#
#
# class F8Tally:
#     def __init__(self, tally_cards, outp):
#         raise ValueError("Tally 8 not supported")
#
#
# class Outp:
#     def __init__(self, file_path):
#         self.__outp__ = open(file_path)
#         self.__outp_lines__ = self.__outp__.readlines()
#         self.__outp__.seek(0)
#
#         self.input_deck = []
#         self.nps = None
#         for line in self.__outp_lines__:
#             _m = re.match("^ {0,9}[0-9]+- {7}(.+)", line)
#             if _m:
#                 card = _m.group(1).rstrip().lower()
#                 self.input_deck.append(card)
#             _m_nps = re.match(" *dump no\. +[0-9].+nps = +([0-9]+)", line)
#             if _m_nps:
#                 self.nps = int(_m_nps.group(1))
#
#     def get_tally(self, tally_number):
#         s = "^([fcemt]+){}.+".format(tally_number)
#         cards = []
#         for line in self.input_deck:
#             card_match = re.match(s, line)
#             if card_match:
#                 cards.append(line)
#         tally_type = int(str(tally_number)[-1])
#         if tally_type == 4:
#             return F4Tally(cards, tally_number, self.__outp__)
#         else:
#             raise ValueError("Tallies of type {} not supported!".format(tally_type))
#
#     def get_cell_properties(self, cell_num):
#         lines_after_match = 0
#         cell_num = int(cell_num)
#
#         for line in self.__outp_lines__:
#             if re.match("1cells +print table 60", line):
#                 lines_after_match += 1
#             elif 1 <= lines_after_match <= 6:
#                 _m = re.match(" +[0-9]+ +(?P<cell>[0-9]+) +(?P<mat>[0-9]+) +(?P<atom_density>[0-9.E+-]+) +"
#                               "(?P<gram_density>[0-9.E+-]+) +(?P<volume>[0-9.E+-]+)", line)
#                 if _m and int(_m.group("cell")) == cell_num:
#
#                     result = {"material": float(_m.group("mat")), "atom_density": float(_m.group("atom_density")),
#                               "gram_density": float(_m.group("gram_density")), "volume": float(_m.group("volume"))}
#                     result["n_atoms"] = result["volume"]*1E24*result["atom_density"]
#                     return result
#         raise ValueError("Cell {} not found!".format(cell_num))
#
#     def get_cell_population_info(self, cell, par):
#         lines_since_table_126 = 0
#         for line in self.__outp_lines__:
#             if lines_since_table_126 == 0:
#                 _m = re.match("1{} +activity +in +each +cell +print table 126\n".format(par), line)
#                 if _m:
#                     lines_since_table_126 = 1
#             elif lines_since_table_126 is None or 1 <= lines_since_table_126 <= 5:
#                 _m = re.match(" +[0-9] +(?P<cell>[0-9]+) +(?P<tr_entering>[0-9]+)"
#                               " +(?P<total_un_pars>[0-9]+) +[0-9]+ +[0-9.E+-]+", line)
#                 if _m:
#                     assert len(line.split()) == 10, "Outp format has changed!"
#                     lines_since_table_126 = None
#                     data = line.split()
#
#                     if int(data[1]) == int(cell):
#                         result = {"n_pars_entering": int(data[2]), "n_unique_pars_in_cell": int(data[3]),
#                                   "flux_weighted_energy": float(data[7]), "number_weighted_energy": float(data[6]),
#                                   "mean_track_weight": float(data[8])}
#
#                         class _CellPopulation:
#                             def __init__(self, n_pars_entering, n_unique_pars_in_cell, flux_weighted_energy,
#                                          number_weighted_energy, mean_track_weight):
#                                 self.n_pars_entering = n_pars_entering
#                                 self.n_unique_pars_in_cell = n_unique_pars_in_cell
#                                 self.flux_weighted_energy = flux_weighted_energy
#                                 self.number_weighted_energy = number_weighted_energy
#                                 self.mean_track_weight = mean_track_weight
#
#                         return _CellPopulation(**result)
#
#
#
#
#


