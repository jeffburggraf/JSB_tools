import pickle
import warnings
from JSB_tools import cm_2_best_unit
import matplotlib.pyplot as plt
import numpy as np

from JSB_tools.MCNP_helper.materials import Material, DepletedUranium
from pathlib import Path
from JSB_tools.MCNP_helper.input_deck import InputDeck
from openmc.data import ATOMIC_NUMBER, ATOMIC_SYMBOL
import re
import subprocess
from subprocess import PIPE
from functools import cached_property
import ROOT
from typing import Union
from JSB_tools.MCNP_helper.PTRAC import ptrac2root, TTreeHelper
from JSB_tools import TBrowser


cwd = Path(__file__).parent
G_save_dir = cwd / 'user_saved_data'/'MCIonRanges'
G_save_dir.mkdir(parents=True, exist_ok=True)

cash_paths = {"MCNP": G_save_dir/"mcnp"/'cashed.pickle',
              "PHITS": G_save_dir/"phits"/'cashed.pickle'}

for v in cash_paths.values():
    if not v.parent.exists():
        v.parent.mkdir()


class Base:
    root_files = []

    def __init__(self, projectile: str, material: Material, energy):
        assert isinstance(projectile, str)
        if projectile.lower() == 'proton':
            projectile = "H4"
        self.z = 0
        self.a = 0
        if m := re.match('([A-Za-z]{1,3})-*([0-9]+)', projectile):
            try:
                z = int(ATOMIC_NUMBER[m.groups()[0]])
                a = int(m.groups()[1])
            except KeyError:
                assert False, f"Invalid projectile, {projectile}. Examples: Xe139, H2, He3, H3, Mo105"
            self.z = z
            self.a = a
            # self.zaid = int(1E6 * z + a)
            self.erg_per_a = energy / a

        self._x = ROOT.vector('float')()
        self._y = ROOT.vector('float')()
        self._z = ROOT.vector('float')()
        self._t = ROOT.vector('float')()

        self.tree = None
        self.root_file = None  #

        self.data_dir = cash_paths[self.__class__.__name__].parent
        self.cash_path = cash_paths[self.__class__.__name__]

        if not self.data_dir.exists():
            self.data_dir.mkdir()
        self.save_dir: Union[Path, None] = None  # will be set later

        self.__param_tuples = Base.__get_param_tuple__(projectile, material, energy)

        try:
            with open(self.cash_path, 'rb') as f:
                self.cash = pickle.load(f)
        except FileNotFoundError:
            self.cash = {}

        self.template_path = cwd / 'misc_data' / f'{self.__class__.__name__}_ion_range_template'
        assert self.template_path.exists()

    def _run(self, run_cmd):
        """create input deck and run simulation"""
        cmd = f'cd {self.save_dir};{run_cmd} {self.save_dir.name}'
        command = subprocess.Popen(cmd, shell=True, stdout=PIPE, )
        std_out = str(command.stdout.read())

        if 'Error Message' in std_out:
            self.delete_from_cash()
            assert False, f'Error when executing {run_cmd}:\n{std_out}'

    @staticmethod
    def __get_param_tuple__(projectile: str, material: Material, energy):
        fractions = np.array(list(sorted(material.atom_fractions)), dtype=float)
        fractions *= 1.0 / sum(fractions)
        if material.is_weight_fraction:
            fractions *= -1
        fractions = tuple(fractions)

        zaids = tuple(sorted(material.zaids))
        out = projectile, material.density, zaids, fractions, material.density, energy
        return out

    def delete_from_cash(self):
        try:
            del self.cash[self.__param_tuples]
        except KeyError:
            pass
        try:
            for p in self.save_dir.iterdir():
                p.unlink(missing_ok=True)
            self.save_dir.rmdir()
            self.cash_path.unlink()

        except FileNotFoundError:
            pass

    def create_new_tree(self):
        self.root_file = ROOT.TFile(str(self.save_dir / 'stop_points.root'), 'recreate')
        # Base.root_files.append(f)
        Base.root_files.append(self.root_file)
        self.tree = ROOT.TTree('tree', 'tree')
        self.tree.Branch('x', self._x)
        self.tree.Branch('y', self._y)
        self.tree.Branch('z', self._z)
        self.tree.Branch('t', self._t)

    def get_next_dir(self):
        """Get next unused directory to store the path in self.save_dir attribute
           Also create the tree"""
        cls_name = self.__class__.__name__.lower()
        paths = [p.name for p in self.data_dir.iterdir() if re.match(f'{cls_name}_[0-9]+', p.name) and p.is_dir()]
        i = 0
        while (path_name := f'{cls_name}_{i}') in paths:
            i += 1
        self.save_dir = self.data_dir/path_name
        self.save_dir.mkdir()
        self.create_new_tree()
        return path_name

    def load_from_cash(self):
        """Load from cash and set save dir.
        Otherwise raise FIleNotFound"""
        try:
            self.save_dir = self.cash[self.__param_tuples]
            root_file_path = self.save_dir/'stop_points.root'
            if not root_file_path.exists():
                warnings.warn(f"Cash existed but directory did not.")
                self.delete_from_cash()
                raise FileNotFoundError
            root_file = ROOT.TFile(str(root_file_path))
            Base.root_files.append(root_file)
            self.tree = root_file.Get('tree')
            try:  # test for valid tree
                if self.tree is None:
                    warnings.warn("Root file existed but couldn't load tree")
                    raise FileNotFoundError
                if not hasattr(self.tree, 'GetEntries'):
                    raise FileNotFoundError
                if self.tree.GetEntries() == 0:
                    warnings.warn("Empty tree")
                    raise FileNotFoundError
                else:
                    print(f"loaded tree with {self.tree.GetEntries()} entries")
            except FileNotFoundError:
                self.delete_from_cash()
                raise
        except (KeyError, FileNotFoundError):
            raise FileNotFoundError

    def fill(self, x, y, z, t):
        self._x.push_back(x)
        self._y.push_back(y)
        self._z.push_back(z)
        self._t.push_back(t)
        self.tree.Fill()
        self._x.clear()
        self._y.clear()
        self._z.clear()
        self._t.clear()

    def set_cash(self):
        self.cash[self.__param_tuples] = self.save_dir
        with open(self.cash_path, 'wb') as f:
            pickle.dump(self.cash, f)

        with open(self.save_dir/"params.txt", 'w') as f:  # just for human readability
            f.write(str(self.__param_tuples))

    @cached_property
    def raw_z_data(self):
        a = []
        n_entries = self.tree.GetEntries()
        for i in range(n_entries):
            self.tree.GetEntry(i)
            a.append(self.tree.z[0])
        return np.array(a)

    @cached_property
    def mean_range(self):
        return np.mean(self.raw_z_data)

    @cached_property
    def max_range(self):
        return np.max(self.raw_z_data)

    @cached_property
    def min_range(self):
        return np.min(self.raw_z_data)

    def plot(self, ax=None, bins='auto', xtitle=None, units='cm', **ax_kwargs):
        if ax is None:
            ax = plt.subplot()
            _min = self.min_range
            _max = self.max_range
        else:
            _min = min([ax.get_xlim()[0], self.min_range])
            _max = max([ax.get_xlim()[1], self.max_range])
        _min *= 0.9
        _max *= 1.1
        y, bin_edges = np.histogram(self.raw_z_data, bins)
        x = [(x1 + x2)/2 for x1, x2 in zip(bin_edges[:-1], bin_edges[1:])]
        # x, units = cm_2_best_unit(x)

        ax.plot(x, y, **ax_kwargs)
        # ax.set_xlabel(f'{xtitle} [{units}]')
        ax.set_ylabel('counts')
        ax.legend()
        return ax

    @staticmethod
    def MCNP_PHITS_Compare(projectile: str, material: Material, energy, nps=5000, overwrite=False, bins='auto'):
        phits = PHITS(projectile, material, energy, nps, overwrite)
        mcnp = MCNP(projectile, material, energy, nps, overwrite)
        ax = phits.plot(bins=bins, label='PHITS')
        mcnp.plot(ax=ax, bins=bins, label='MCNP')
        return ax


class PHITS(Base):
    def __init__(self, projectile: str, material: Material, energy, nps=5000, overwrite=False):
        super().__init__(projectile, material, energy)
        try:
            kfcode = {'electron': '11', 'positron': '-11', 'h1': '2212'}[projectile.lower()]

        except KeyError:  # this mean nucleus, or invalid projectile
            kfcode = 1000000 * self.z + self.a

        try:
            self.ityp = {"proton": '1', 'h1': '1', 'electron': '12', 'positron': '13', 'h2': '15', 'h3': '16',
                         'he3': '17', 'he4': '18'}[projectile.lower()]
        except KeyError:
            self.ityp = '19'

        try:
            self.load_from_cash()
            if overwrite:
                self.delete_from_cash()
                raise FileNotFoundError
        except FileNotFoundError:
            try:
                self.get_next_dir()
                i = InputDeck.phits_input_deck(self.template_path, self.save_dir.parent, gen_run_script=False)
                i.write_inp_in_scope(locals(), new_file_name=self.save_dir.name)

                self._run('phits.sh')
                self.save_data()  # todo
                self.tree.Write()
                self.set_cash()

            except Exception as e:
                self.delete_from_cash()
                raise

    def save_data(self):
        f_path = self.save_dir/"ptrac"
        assert f_path.exists(), f"No PTRAC file found for {self.__class__.__name__} simulation! Simulation had an error." \
                                f"\n File path: {f_path}"

        with open(f_path) as f:
            term_waiting = False
            while (line := f.readline()) != '':
                if line[:5] == 'NCOL=':
                    if f.readline() == ' 11\n':  # check for ncol == 11 (term by erg cut off)
                        f.readline()
                        line = f.readline()
                        if line.split()[2] == self.ityp:
                            # is charged particle
                            term_waiting = True
                        else:
                            assert False, f'Expected ityp: {self.ityp}, actual ityp: {line.split()[2]}'
                elif term_waiting:

                    if line[:7] == 'EC,TC,X':  # Then next line is termination position and time
                        line = f.readline()
                        _, t, x, y, z,  = tuple(map(float, line.replace('D', 'E').split()))
                        self.fill(x, y, z, t)
                        term_waiting = False

        if self.tree is not None and self.tree.GetEntries() == 0:
            warnings.warn("TTree was empty! Something went wrong. Not cashing result.")
            self.delete_from_cash()
        self.set_cash()


class MCNP(Base):
    def __init__(self, projectile: str, material: Material, energy, nps=5000, overwrite=False):

        super().__init__(projectile, material, energy)
        try:
            mode = {"He4": 'a', "He3": "s", 'H3': 't', "H2": "d", "H1": 'h', 'electron': 'e'}[projectile]
            zaid = mode
        except KeyError:  # this mean nucleus, or invalid projectile
            zaid = 1000*self.z + self.a
            mode = "#"
        efac = 0.98
        if mode == 'e':
            phys_card = f"PHYS:e 7j 0 5j {efac}"
        elif mode == 'h':
            phys_card = f"PHYS:{mode} 13j {efac}"
        elif mode == '#':
            phys_card = f'PHYS:{mode} {1.1*energy} 12j {efac}'
        else:
            assert False
        # material.mat_kwargs["HSTEP"] = "200"

        try:
            self.load_from_cash()
            if overwrite:
                self.delete_from_cash()
                raise FileNotFoundError
        except FileNotFoundError:
            try:
                self.get_next_dir()
                i = InputDeck.mcnp_input_deck(self.template_path, self.save_dir.parent, gen_run_script=False)
                i.write_inp_in_scope(locals(), new_file_name=self.save_dir.name)

                self._run('mcnp6 i=')
                self.save_data()  # todo
                self.tree.Write()
                self.set_cash()

            except Exception as e:
                self.delete_from_cash()
                raise

    def save_data(self):
        f_path = self.save_dir/"ptrac"
        assert f_path.exists(), "No PTRAC file found! Simulation had an error"
        p = ptrac2root(f_path)

        ptrac2root_path = self.save_dir/'ptrac.root'
        tree = TTreeHelper(ptrac2root_path)
        for t in tree:
            if t.is_term:
                self.fill(t.x, t.y, t.z, t.time)
        # ptrac2root_path.unlink()
        self.root_file.cd()
        self.set_cash()



# m = DepletedUranium()
# gas = Material.gas(['He', 'Ar'], atom_fractions=[1, 1], pressure=1.4, mat_kwargs={'HSTEP': '40'})
# p = MCNP('Xe139', gas, 70, overwrite=True)
# p.plot()
# p2 = Base.MCNP_PHITS_Compare('Xe139', gas, 70, overwrite=True)

# p2.plot()
# plt.show()