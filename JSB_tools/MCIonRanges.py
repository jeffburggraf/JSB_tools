import pickle
import warnings

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
        cmd = f'cd {self.save_dir.relative_to(cwd)};{run_cmd} {self.save_dir.name}'
        command = subprocess.Popen(cmd, shell=True, stdout=PIPE, )
        std_out = str(command.stdout.read())

        if 'Error Message' in std_out:
            self.delete_from_cash()
            assert False, f'Error when executing {run_cmd}:\n{std_out}'

    @staticmethod
    def __get_param_tuple__(projectile: str, material: Material, energy):
        fractions = np.array(list(sorted(material._zaid_proportions)), dtype=float)
        fractions *= 1.0 / sum(fractions)
        if material.is_weight_fraction:
            fractions *= -1
        fractions = tuple(fractions)

        zaids = tuple(sorted(material._zaids))
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
        """Get next unused directory to store result.
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
        """Load from cahs and set save dir.
        Otherwise raise FIleNotFound"""
        try:
            self.save_dir = self.cash[self.__param_tuples]
            root_file_path = self.save_dir/'stop_points.root'
            if not root_file_path.exists():
                warnings.warn("Cash existed but directory did not")
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


class MCNP(Base):
    def __init__(self, projectile: str, material: Material, energy, nps=5000, overwrite=False):

        super().__init__(projectile, material, energy)
        try:
            mode = {"He4": 'a', "He3": "s", 'H3': 't', "H2": "d", "H": 'h', 'electron': 'e'}[projectile]
            zaid = mode
        except KeyError:  # this measn nucleus, or invalid projectile
            zaid = 1000*self.z + self.a
            mode = "#"
        if mode == 'e':
            phys_card = "PHYS:e 7j 0 5j 0.98"
        else:
            phys_card = f"PHYS:{mode} 13j 0.98"

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
        ptrac2root_path.unlink()
        self.root_file.cd()
        self.set_cash()



# mat = Material.gas(['U'], atom_fractions=[1,1], pressure=1.4)
p = MCNP('Xe139', DepletedUranium(), 70, overwrite=True)
# p = PHITS('He4', DepletedUranium(), 70)

TBrowser()
