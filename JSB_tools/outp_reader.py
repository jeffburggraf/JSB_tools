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

    def read_stopping_powers(self, particle, material_id=None, cell_num_4_density=None):
        s = StoppingPowerData()
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

        c = re.compile("1.*{0}.+{1}.+print table 85".format(particle, material_id))

        for index, line in enumerate(self.__outp_lines__):
            if c.match(line):
                index += 8  # beginning of dEdx data begins 8 lines after the start of print table 85
                break
        else:
            assert False, "Could not find dEdx table for '{0}' and material '{1}'".format(particle, material_id)
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

