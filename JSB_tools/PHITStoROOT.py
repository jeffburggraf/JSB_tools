import ROOT
from pathlib import Path
import os
import re
import numpy as np

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
    def __init__(self, input_file_path, output_file_name=None, max_events=None, tree_name=None):
        self.input_file_path = input_file_path
        self.output_file_name = output_file_name
        self.max_events = max_events

        self.tree_name = "tree" if tree_name is None else tree_name
        self.root_file, self.tree = self.__make_ROOT_file_and_tree__()

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
        self.tree.Branch(b_name, getattr(self, b_name + "_br"),root_dtype)

        if reset is not None:  # some values must be reset to default value at each point in simulation
            self.__br_arrays__.append((getattr(self, b_name + "_br"), reset))
        assert hasattr(self, b_name), "define a getter and setter named '{0}' for Container".format(b_name)

    def __make_ROOT_file_and_tree__(self):
        directory = Path(os.path.dirname(self.input_file_path))
        if self.output_file_name is None:
            new_file_name = os.path.basename(self.input_file_path)
            _m = re.match(r"([^\.]+).*", new_file_name)
            assert _m, "Could not set correct file name using: {0}".format(new_file_name)
            new_file_name = _m.group(1) + ".root"

        new_file_name = directory / new_file_name

        file = ROOT.TFile(str(new_file_name), "RECREATE")

        tree = ROOT.TTree("tree", self.tree_name)
        return file, tree
    
"""prev_params.erg = values[0]
            prev_params.time = values[1]
            prev_params.x = values[2]
            prev_params.y = values[3]
            prev_params.z = values[4]"""
# Todo: I dont think this is useful actually. ???
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

def phits_to_root(input_file_path, output_file_name=None, max_events=None, tree_name=None):
    container = Container(input_file_path, output_file_name, max_events, tree_name )

    ncol_re = re.compile("NCOL=")
    nps_re = re.compile("NOCAS,")
    info_1_re = re.compile("NO,MAT,ITYP")  # cascade ID, Mat, particle, ...
    info_2_re = re.compile("IBLZ1,IBLZ2")  # Cell num, level structure, ...
    info_3_re = re.compile("NAME,NCNT\(1\)")
    info_4_re = re.compile("WT,U,V")
    info_5_re = re.compile("EC,TC,XC")
    info_6_re = re.compile("SPX,SPY")
    info_7_re = re.compile("NZST=")
    
    prev_params_re = re.compile("E,T,X")
    
    prev_params = PrevParameters()

    file = open(container.input_file_path)

    n_events = 0

    def line_of_values(l, map_func=None):
        if map_func is None:
            return l.replace("D", "E").split()
        else:
            return list(map(map_func, l.replace("D", "E").split()))

    while max_events is None or n_events < max_events:
        line = file.readline()
        if line == "":
            break
        if ncol_re.match(line):
            line = file.readline()
            container.ncol = int(line)
            if container.ncol == 4:
                container.is_src = 1

            if container.prev_ncol > 1:  # If previous ncol appearance does not indicate start of calculation
                container.fill()
                n_events += 1

        elif nps_re.match(line):
            line = file.readline()
            container.nps = int(line.split()[0])

        elif info_1_re.match(line):
            values = line_of_values(file.readline())
            container._cas_num = int(values[0])
            container.mat = int(values[1])
            container.par_id = int(values[2])
            zaid = int(values[3])
            container.charge_number = int(values[4])
            container.zaid = zaid//1000 + zaid % 1000000
            container.rest_mass = float(values[-2])

        elif info_2_re.match(line):
            values = line_of_values(file.readline(), int)
            # Todo: Verify that I should use values[1] for cell ID at XC,YC,ZC.

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


if __name__ == "__main__":
    test_file_path = "/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/PTRAC.txt"
    phits_to_root(test_file_path)
    tb = ROOT.TBrowser()
    import time
    while True:
        ROOT.gSystem.ProcessEvents()
        time.sleep(0.02)
