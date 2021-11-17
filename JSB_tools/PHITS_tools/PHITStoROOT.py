"""
Creates ROOT Trees from PHITS dump files. Requires pyROOT to be installed.

To have PHITS create a dump file, add the following section:
    [ T-Userdefined ]
      file= <name of file>
where <name of file> is the name of the simulation dump file.

Definition of TTree branches: Todo
    nps
"""
import ROOT
from pathlib import Path
import os
import re
import numpy as np
from JSB_tools import TBrowser, ProgressReport, mpl_hist
import time
from typing import Union
from JSB_tools.nuke_data_tools import Nuclide
import warnings

__all__ = ["phits_to_root"]


class PHITSTTreeHelper:
    """
    A wrapper for a PHITS ROOT tree. Everything done in this class can be done manually, but this mkaes it smoother by,
    e.g., adding autocomplete, the ability to look up a nuclide that's being tracked, etc.
    """
    nuclides = {}
    root_files = set()

    def __init__(self, path_to_root_file_or_tree: Union[ROOT.TTree, Path, str]):
        if isinstance(path_to_root_file_or_tree, (str, Path)):
            self.path = Path(path_to_root_file_or_tree)
            assert self.path.exists()
            self.root_file = ROOT.TFile(str(path_to_root_file_or_tree))
            self.tree = self.root_file.Get('tree')
        elif isinstance(path_to_root_file_or_tree, ROOT.TTree):
            self.tree = path_to_root_file_or_tree
            self.root_file = self.tree.GetCurrentFile()
            self.path = self.root_file.GetName()
        PHITSTTreeHelper.root_files.add(self.root_file)

    @staticmethod
    def t_browser():
        TBrowser()

    @property
    def nps(self):
        return self.tree.nps

    def particle(self):
        par_id = self.tree.par_id
        if par_id == 19:
            return self.nuclide.name
        elif par_id == 1:
            return "proton"
        elif par_id == 2:
            return "neutron"
        elif par_id == 12:
            return "electron"
        elif par_id == 13:
            return "positron"
        elif par_id == 14:
            return "photon"
        elif 15 <= par_id <= 18:
            n = self.nuclide
            return n.name

    @property
    def is_nucleus(self):
        """Helium or larger"""
        return 17 <= self.tree.par_id <= 19

    @property
    def nuclide(self) -> Union[None, Nuclide]:
        if self.tree.par_id != 19:
            return None
        elif 15 <= self.tree.par_id <= 18:
            par_id = self.tree.par_id
            try:
                return PHITSTTreeHelper.nuclides[par_id]
            except KeyError:
                if par_id == 15:
                    n = Nuclide.from_Z_A_M(1, 2)
                    PHITSTTreeHelper.nuclides[par_id] = n
                elif par_id == 16:
                    n = Nuclide.from_Z_A_M(1, 3)
                    PHITSTTreeHelper.nuclides[par_id] = n
                elif par_id == 17:
                    n = Nuclide.from_Z_A_M(2, 3)
                    PHITSTTreeHelper.nuclides[par_id] = n
                else:
                    n = Nuclide.from_Z_A_M(2, 4)
                    PHITSTTreeHelper.nuclides[par_id] = n
                return n

        zaid = self.tree.zaid
        z = zaid // 1000
        a = zaid % 1000
        if zaid in PHITSTTreeHelper.nuclides:
            return PHITSTTreeHelper.nuclides[zaid]
        nuclide = Nuclide.from_Z_A_M(z, a)
        PHITSTTreeHelper.nuclides[zaid] = nuclide
        return nuclide

    @property
    def is_term(self):
        return self.tree.is_term

    @property
    def is_src(self):
        return self.tree.is_src

    @property
    def dirx(self):
        return self.tree.dirx

    @property
    def diry(self):
        return self.tree.diry

    @property
    def dirz(self):
        return self.tree.dirz

    @property
    def x(self):
        return self.tree.x

    @property
    def y(self):
        return self.tree.y

    @property
    def z(self):
        return self.tree.z

    @property
    def energy(self):
        return self.tree.erg

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    @property
    def cell(self):
        return self.tree.cell

    @property
    def is_boundary_crossing(self):
        """
        True if particle is on a surface. Use self.cell_before_crossing to get the cell the particle was just in, and
        self.cell for the cell the particle is heading into.
        Returns:

        """
        if self.ncol == 10 or self.ncol == 12:
            assert self.cell_before_crossing != self.cell, "Hmmm this shouldn't happen. Thought I understood this." \
                                                           " Advise"
            return True
        else:
            return False

    @property
    def cell_before_crossing(self):
        return self.tree.cell_before_crossing

    def to_text(self):  # todo
        return f'pos: {self.pos}, cell: {self.cell}, erg: {self.energy} '

    @property
    def ncol(self):
        return self.tree.ncol

    @property
    def term(self):
        if not self.is_term:
            return None
        elif self.ncol == 7:
            return "geom_err"
        elif self.ncol == 8:
            return "wgt_cutoff"
        elif self.ncol == 9:
            return "time_cutoff"
        elif self.ncol == 11:
            return "erg_cutoff"
        elif self.ncol == 12:
            return "escape"
        else:
            assert False, (self.ncol, self.is_term)

    def __iter__(self):
        if self.tree.GetEntries() == 0:
            warnings.warn(f'No entries in tree as {self.path}')

        self.__i__ = 0
        return self

    def __next__(self):
        self.tree.GetEntry(self.__i__)

        self.__i__ += 1
        if self.__i__ >= self.tree.GetEntries():
            raise StopIteration

        return self








# TOdo write lookup tables.
"see https://root.cern.ch/doc/v610/staff_8py_source.html"
ROOT.gROOT.ProcessLine(
    "struct secondary {"
    "Int_t           parent_Z;"
    "Int_t           parent_N;"
    "Int_t           JCOLL;"
    "Int_t           KCOLL;"
    "Int_t           par_id;"  # Get this is from JCULSTS
    "Int_t           zaid;"  # Get this is from JCULSTS
    "Int_t           Z;"
    "Int_t           N;"
    "Int_t           x;"  # QCLUSTS
    "Int_t           y;"
    "Int_t           z;"
    "Int_t           erg;"
    "Int_t           time;"
    "Int_t           wgt;"
    "}"
)
"Should I include dirx/y/z in the above?"


class Container:
    root_files = []  # refs to all root files created

    @staticmethod
    def __close_root_files():
        for f in Container.root_files:
            f.Close()

    def __init__(self, sim_dump_file_path, output_file_name, output_directory, max_events, tree_name, overwrite):
        self.sim_dump_file_path = sim_dump_file_path
        assert Path(self.sim_dump_file_path).exists(), "Cannot find simulation dump file: {}".format(
            self.sim_dump_file_path)
        self.output_file_name = output_file_name
        self.output_file_dir = output_directory
        self.max_events = max_events
        self.overwrite = overwrite
        self.root_file_path: Union[Path, None] = None
        self.tree_name = "tree" if tree_name is None else tree_name
        self.root_file, self.tree = self.__make_ROOT_file_and_tree__()
        # self.r

        self.__br_arrays__ = []

        self.make_branch("ncol", int)
        self.prev_ncol = 0
        self.make_branch("nps", int)  # NOCHAS
        self.make_branch("is_src", int, reset=0)  # is this a source particle?

        self.make_branch("mat", int)   # IDMN; material number
        self.make_branch("par_id", int)  # particle ID
        self.make_branch("zaid", int)  # 1000*Z + A
        self.make_branch("rest_mass", float)  # rest mass in MeV
        self.make_branch("_cas_num", int)  # cascade number. Not sure what this is.
        self.make_branch("charge_number", int)  # charge num of the intrinsic particle. e.g. U-239 = 92

        self.make_branch("cell", int)  # Cell number
        self.make_branch("cell_before_crossing", int)  # Cell number during a surface or escape event (ncol=10 or 12)
        self.make_branch("cell_level", int)  # Cell level. Only used for complex geom.

        self.make_branch("col_num", int, reset=0) # Collision number
        self.make_branch("_col_counts", int, reset=0, length=3)  # Collision number

        self.make_branch("wgt", float)  # Particle weight
        self.make_branch("dirx", float)  # unit vector x momentum
        self.make_branch("diry", float)  # ...
        self.make_branch("dirz", float)  # ...

        self.make_branch("erg", float)  # Energy [MeV]
        self.make_branch("time", float)  # Time in ns
        self.make_branch("x", float)  # x position in cm
        self.make_branch("y", float)  # ...
        self.make_branch("z", float)  # ...

        self.make_branch("spinx", float)  # unit vector of spin direction
        self.make_branch("spiny", float)  # ...
        self.make_branch("spinz", float)  # ...

        self.make_branch("charge_state", float)  # Charge state (this varies with ionization)

        self.make_branch("de_dx", float)

        self.make_branch("is_term", int, reset=0)

    @property
    def is_term(self):
        return self.is_term_br[0]

    @is_term.setter
    def is_term(self, value):
        self.is_term_br[0] = value

    @property
    def de_dx(self):
        return self.de_dx_br[0]

    @de_dx.setter
    def de_dx(self, value):
        self.de_dx_br[0] = value

    @property
    def charge_state(self):
        return self.charge_state_br[0]

    @charge_state.setter
    def charge_state(self, value):
        self.charge_state_br[0] = value

    @property
    def spinz(self):
        return self.spinz_br[0]

    @spinz.setter
    def spinz(self, value):
        self.spinz_br[0] = value

    @property
    def spiny(self):
        return self.spiny_br[0]

    @spiny.setter
    def spiny(self, value):
        self.spiny_br[0] = value

    @property
    def spinx(self):
        return self.spinx_br[0]

    @spinx.setter
    def spinx(self, value):
        self.spinx_br[0] = value

    @property
    def z(self):
        return self.z_br[0]

    @z.setter
    def z(self, value):
        self.z_br[0] = value

    @property
    def y(self):
        return self.y_br[0]

    @y.setter
    def y(self, value):
        self.y_br[0] = value

    @property
    def x(self):
        return self.x_br[0]

    @x.setter
    def x(self, value):
        self.x_br[0] = value

    @property
    def time(self):
        return self.time_br[0]

    @time.setter
    def time(self, value):
        self.time_br[0] = value

    @property
    def erg(self):
        return self.erg_br[0]

    @erg.setter
    def erg(self, value):
        self.erg_br[0] = value

    @property
    def dirz(self):
        return self.dirz_br[0]

    @dirz.setter
    def dirz(self, value):
        self.dirz_br[0] = value

    @property
    def diry(self):
        return self.diry_br[0]

    @diry.setter
    def diry(self, value):
        self.diry_br[0] = value

    @property
    def dirx(self):
        return self.dirx_br[0]

    @dirx.setter
    def dirx(self, value):
        self.dirx_br[0] = value

    @property
    def wgt(self):
        return self.wgt_br[0]

    @wgt.setter
    def wgt(self, value):
        self.wgt_br[0] = value

    @property
    def _col_counts(self):
        return self._col_counts_br

    @_col_counts.setter
    def _col_counts(self, values):
        self._col_counts_br[:] = values

    @property
    def col_num(self):
        return self.col_num_br[0]

    @col_num.setter
    def col_num(self, value):
        self.col_num_br[0] = value

    @property
    def cell_level(self):
        return self.cell_level_br[0]

    @cell_level.setter
    def cell_level(self, value):
        self.cell_level_br[0] = value

    @property
    def cell(self):
        return self.cell_br[0]

    @cell.setter
    def cell(self, value):
        self.cell_br[0] = value

    @property
    def cell_before_crossing(self):
        return self.cell_before_crossing_br[0]

    @cell_before_crossing.setter
    def cell_before_crossing(self, value):
        """
        Usually the same as self.cell, except for when
            ncol = 10 or 12 (geometry crossing, e.g. changing cells, or escape, respectively)
        """
        self.cell_before_crossing_br[0] = value

    @property
    def charge_number(self):
        return self.charge_number_br[0]

    @charge_number.setter
    def charge_number(self, value):
        self.charge_number_br[0] = value

    @property
    def zaid(self):
        return self.zaid_br[0]

    @zaid.setter
    def zaid(self, value):
        self.zaid_br[0] = value

    @property
    def _cas_num(self):
        return self._cas_num_br[0]

    @_cas_num.setter
    def _cas_num(self, value):
        self._cas_num_br[0] = value

    @property
    def rest_mass(self):
        return self.rest_mass_br[0]

    @rest_mass.setter
    def rest_mass(self, value):
        self.rest_mass_br[0] = value

    @property
    def mat(self):
        return self.mat_br[0]

    @mat.setter
    def mat(self, value):
        self.mat_br[0] = value

    @property
    def par_id(self):
        return self.par_id_br[0]

    @par_id.setter
    def par_id(self, value):
        self.par_id_br[0] = value

    @property
    def is_src(self):
        return self.is_src_br[0]

    @is_src.setter
    def is_src(self, value):
        self.is_src_br[0] = value

    @property
    def ncol(self):
        return self.ncol_br[0]

    @ncol.setter
    def ncol(self, value):
        self.prev_ncol = self.ncol_br[0]
        self.ncol_br[0] = value

    @property
    def nps(self):
        return self.nps_br[0]

    @nps.setter
    def nps(self, value):
        self.nps_br[0] = value


    def fill(self):
        self.tree.Fill()
        self.reset_branch_values()

    def reset_branch_values(self):
        [b.fill(v) for b, v in self.__br_arrays__]

    def close(self):
        self.tree.Write()
        self.root_file.Close()

    def make_branch(self, b_name, dtype, reset=None, length=1):
        """
        :param b_name: name of branch in tree, as well as name of getter and setter.
        :param dtype: float or int
        :param reset: If not None, then every time the tree is filled, this branch will be initialized to this value.
        :param length: Length of array
        :return: None
        """
        _dtype = np.float if dtype == float else np.int
        if length == 1:
            root_dtype = "{0}/{1}".format(b_name, "D" if dtype == float else "I")
        else:
            assert isinstance(length, int)
            _ = "D" if dtype == float else "I"
            root_dtype = "{0}[{1}]/{2}".format(b_name, length, _)

        setattr(self, b_name + "_br", np.array([0]*length, dtype=_dtype))
        self.tree.Branch(b_name, getattr(self, b_name + "_br"), root_dtype)

        if reset is not None:  # some values must be reset to default value at each point in simulation
            self.__br_arrays__.append((getattr(self, b_name + "_br"), reset))
        assert hasattr(self, b_name), "define a getter and setter named '{0}' for Container".format(b_name)

    def __make_ROOT_file_and_tree__(self, __rename_index__=0):
        if self.output_file_dir is None:
            directory = Path(self.sim_dump_file_path).parent
        else:
            directory = Path(self.output_file_dir)

        if self.output_file_name is None:
            new_file_name = Path(self.sim_dump_file_path).name
        else:
            new_file_name = self.output_file_name

        if m := re.match(r"(^.+)(?:\.root|\.txt)", new_file_name):
            new_file_name = m.groups()[0]
        if __rename_index__ > 0:
            new_file_name = "{0}_{1}".format(new_file_name, __rename_index__)
        # new_file_name += ".root"
        # new_file_name =

        new_file_path = Path(directory) / new_file_name
        new_file_path = new_file_path.with_suffix('.root')

        if (not self.overwrite) and Path.exists(new_file_path):
            self.__make_ROOT_file_and_tree__(__rename_index__ + 1)

        file = ROOT.TFile(str(new_file_path), "RECREATE")
        self.root_file_path = new_file_path
        print("Creating: {0}".format(new_file_path))

        tree = ROOT.TTree("tree", self.tree_name)
        return file, tree

"""prev_params.erg = values[0]
            prev_params.time = values[1]
            prev_params.x = values[2]
            prev_params.y = values[3]
            prev_params.z = values[4]"""
# Todo: I don't think this is useful actually ???


class PrevParameters:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.erg = 0
        self.time = 0

    def set(self, values):
        self.erg = values[0]
        self.time = values[1]
        self.x = values[2]
        self.y = values[3]
        self.z = values[4]

    def de_dx(self, container):
        x = container.x
        y = container.y
        z = container.z
        erg = container.erg
        dx = np.sqrt((self.x-x)**2 + (self.y-y)**2 + (self.z-z)**2)
        de = self.erg - erg
        if dx == 0:
            return 0
        else:
            return de/dx


# _containers = []  # to preserve memory of ROOT files


def phits_to_root(input_file_path: Union[Path, str], output_file_name: Union[str, None] = None,
                  output_directory: Union[str, None] = None, max_histories=None, tree_name="tree",
                  overwrite=True, max_time=None) -> ROOT.TTree:
    """

    Args:
        input_file_path: Absolute path to PHITS "PTRAC" file.
        output_file_name: Name of root file to be created. If None, then use input_file_path.name.
        output_directory:
        max_histories:
        tree_name:
        overwrite: If false, don't overwrite existing file.
        max_time: Maximum time to run in seconds.

    Returns: Root tree.

    """

    container = Container(input_file_path, output_file_name, output_directory, max_histories, tree_name, overwrite)
    # _containers.append(container)
    file_size = Path(input_file_path).stat().st_size
    if max_histories is None:
        progress = ProgressReport(file_size)
    else:
        progress = None

    ncol_re = re.compile("NCOL=")
    nps_re = re.compile("NOCAS,")
    info_1_re = re.compile("NO,MAT,ITYP")  # cascade ID, Mat, particle, ...
    info_2_re = re.compile("IBLZ1,IBLZ2")  # Cell num, level structure, ...
    info_3_re = re.compile("NAME,NCNT\(1\)")
    info_4_re = re.compile("WT,U,V")
    info_5_re = re.compile("EC,TC,XC")  # EC stand energie Current
    info_6_re = re.compile("SPX,SPY")
    info_7_re = re.compile("NZST=")

    prev_params_re = re.compile("E,T,X")

    prev_params = PrevParameters()

    file = open(container.sim_dump_file_path)
    try:
        file.readline()
        file.seek(0, 0)
    except UnicodeDecodeError as e:
        assert False, "Cannot read simulation file: {}".format(container.sim_dump_file_path)

    n_events = 0

    def line_of_values(l, map_func=None):
        if map_func is None:
            return l.replace("D", "E").split()
        else:
            return list(map(map_func, l.replace("D", "E").split()))

    t_start = time.time()
    next_print_time = t_start + 1
    line_number = 0

    while max_histories is None or n_events < max_histories:
        if max_time is not None:
            t_now = time.time()
            dt = t_now - t_start
            if t_now >= next_print_time:
                next_print_time = t_now + 1
                print(f'{int(max_time-dt)} seconds remaining until max time reached')
            if dt > max_time:
                break
        line_number += 1

        line = file.readline()
        if line == "":
            break

        bytes_read = file.tell()

        if max_time is None:
            if progress is not None:
                progress.log(bytes_read)

            else:
                if n_events > 5000:
                    i_final = bytes_read * max_histories / n_events
                    progress = ProgressReport(i_final, i_init=bytes_read)

        if ncol_re.match(line):
            # If previous value ncol does not indicate start of calculation, then fill previous event

            line = file.readline()
            if container.ncol > 1:  # the current ncol is taken from the previous block
                container.fill()
                n_events += 1

            container.ncol = int(line)

            if container.ncol == 4:
                container.is_src = 1

            if container.ncol in [7, 8, 9, 11, 12]:
                container.is_term = 1
            #todo: handle NCOLS 13 and 14, after which particle reaction information is written.

        elif nps_re.match(line):
            line_number += 1
            line = file.readline()
            container.nps = int(line.split()[0])

        elif info_1_re.match(line):
            line_number += 1
            values = line_of_values(file.readline())
            container._cas_num = int(values[0])
            container.mat = int(values[1])
            container.par_id = int(values[2])  # ITYP
            zaid = int(values[3])
            container.charge_number = int(values[4])
            if container.par_id == 1:  # proton
                container.zaid = 1000
            elif container.par_id == 2:  # neutron
                container.zaid = 1
            elif 15 <= container.par_id <= 19:  # nucleus, can use ZAID
                container.zaid = zaid // 1000 + zaid % 1000000
            else:  # everything else. In this case use par_id for particle identification.
                container.zaid = 0

            container.rest_mass = float(values[-2])

        elif info_2_re.match(line):
            values = line_of_values(file.readline(), int)
            # Todo: Verify that I should use values[1] for cell ID at XC,YC,ZC.
            container.cell_before_crossing = values[0]
            container.cell = values[1]
            container.cell_level = values[-1]

        elif info_3_re.match(line):
            values = np.array(line_of_values(file.readline(), int))
            container.col_num = values[0]
            container._col_counts = values[1:]

        elif info_4_re.match(line):
            values = line_of_values(file.readline(), float)
            container.wgt = values[0]
            container.dirx = values[1]
            container.diry = values[2]
            container.dirz = values[3]

        elif info_5_re.match(line):  # current position and energy
            values = line_of_values(file.readline(), float)
            container.erg = values[0]
            container.time = values[1]
            container.x = values[2]
            container.y = values[3]
            container.z = values[4]
            container.de_dx = prev_params.de_dx(container)

        elif info_6_re.match(line):
            values = line_of_values(file.readline(), float)
            container.spinx = values[0]
            container.spiny = values[1]
            container.spinz = values[2]

        elif info_7_re.match(line):
            value = int(file.readline())
            container.charge_state = value

        elif prev_params_re.match(line):  # Previous energy and position
            values = line_of_values(file.readline(), float)
            prev_params.set(values)

    container.close()
    print("'{0}' is done!".format(Path(input_file_path).name))
    #  Reopen  root file. This removes the warning from ROOT about a file not being closed
    f = ROOT.TFile(str(container.root_file_path))
    Container.root_files.append(f)
    tree = f.Get(container.tree_name)
    return tree


if __name__ == "__main__":
    test_file_path = "/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/PTRAC.txt"
    phits_to_root(test_file_path)
    tb = ROOT.TBrowser()
    import time
    while True:
        ROOT.gSystem.ProcessEvents()
        time.sleep(0.02)
