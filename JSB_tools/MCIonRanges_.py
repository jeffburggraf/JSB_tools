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
from JSB_tools.MCNP_helper.PTRAC import ptrac2root, TTreeHelper
from JSB_tools import TBrowser
from openmc.data import Evaluation, Reaction

cwd = Path(__file__).parent
G_save_dir = cwd / 'user_saved_data'/'MCIonRanges'
G_save_dir.mkdir(parents=True, exist_ok=True)


class Base:
    root_files = []

    def __init__(self, projectile: str, material: Material, energy, nps=1E4, overwrite=False):
        assert isinstance(projectile, str)
        # if projectile.lower() == 'electron':
        #     projectile = 12
        #     self.zaid = 12
        #     self.erg_per_a = energy
        #     assert False, "Todo: Electron takes too long to run simulation"
        # else:
        if m := re.match('([A-Za-z]{1,3})-*([0-9]+)', projectile):
            # assert m, f"Invalid projectile, {projectile}. Examples: Xe139, H2, He3, H3, Mo105"
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
        self._tree = None
        fractions = np.array(list(sorted(material._zaid_proportions)), dtype=float)
        fractions *= 1.0/sum(fractions)
        if material.is_weight_fraction:
            fractions *= -1
        fractions = tuple(fractions)

        zaids = tuple(sorted(material._zaids))
        out = projectile, material.density, zaids, fractions, material.density, energy
        self.__param_tuples = out
        self.save_dir = None
        self.data_dir = G_save_dir/self.__cls_name
        if not self.data_dir.exists():
            self.data_dir.mkdir()

    def _run(self, cmd_name):
        """create input deck and run simulation"""
        cmd = f'cd {self.save_dir.relative_to(cwd)};{cmd_name} {self.save_dir.name}'
        command = subprocess.Popen(cmd, shell=True, stdout=PIPE, )
        std_out = str(command.stdout.read())
        print(std_out)
        print(cmd)
        if 'Error Message' in std_out:
            for p in self.save_dir.iterdir():
                p.unlink()
            self.save_dir.rmdir()
            assert False, f'Error when executing {cmd_name}:\n{std_out}'

    def get_next_dir(self):
        """Get next unused directory to store result."""
        cls_name = self.__cls_name.lower()
        paths = [p.name for p in self.data_dir.iterdir() if re.match(f'{cls_name}_[0-9]+', p.name) and p.is_dir()]
        i = 0
        while (path_name := f'{cls_name}_{i}') in paths:
            i += 1
        return path_name

    @property
    def __cls_name(self):
        return self.__class__.__name__.lower()

    @property
    def template_path(self):
        out = cwd/'misc_data'/f'{self.__cls_name}_ion_range_template'
        assert out.exists()
        return out

    @cached_property
    def tree(self):
        if self._tree is not None:
            return self._tree
        f = ROOT.TFile(str(self.save_dir/'stop_points.root'), 'recreate')
        # Base.root_files.append(f)
        Base.root_files.append(f)
        tree = ROOT.TTree('tree', 'tree')
        tree.Branch('x', self._x)
        tree.Branch('y', self._y)
        tree.Branch('z', self._z)
        tree.Branch('t', self._t)
        return tree

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

    def _load_data(self):
        """
        Look for cashed data for the simulation parameters. If found, set self._tree and self.save_dir
        If not found, then create and set self.save_dir, and raise FileNotFoundError.
        Returns: None

        """
        if self.__param_tuples in self._get_cash:
            self.save_dir = self._get_cash[self.__param_tuples]
            print(f"Found cashed in pickle file ")
            saved_path = self.save_dir / 'stop_points.root'
            if saved_path.exists():
                print(f"Cashed file exists at {self.save_dir} ")
                f = ROOT.TFile(str(saved_path))
                Base.root_files.append(f)
                self._tree = f.Get('tree')
                if self._tree.GetEntries() == 0:
                    self.unlink_save_dir()
                    warnings.warn("Cash had invalid Tree. Rerunning.")
                    raise FileNotFoundError
                return
            else:
                self.unlink_save_dir()

        # Didn't find
        self.save_dir = self.data_dir / self.get_next_dir()
        raise FileNotFoundError

    @cached_property
    def _get_cash(self):

        try:
            with open(self.data_dir/'cashed.pickle', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def _set_cash(self, remove_self=False):
        cash = self._get_cash
        if not remove_self:
            cash[self.__param_tuples] = self.save_dir
        with open(self.save_dir.parent/'cashed.pickle', 'wb') as f:
            pickle.dump(cash, f)
        print("saving cash: ", cash)

    def unlink_save_dir(self):
        """Delete cash directory and remove entry from cashed.pickle"""
        for f in self.save_dir.iterdir():
            f.unlink()
            print(f"removing {f}")
        self._set_cash(True)
        self.save_dir.rmdir()


class PHITS(Base):
    def __init__(self, projectile: str, material: Material, energy, nps=1E4, overwrite=False):
        super().__init__(projectile, material, energy, nps, overwrite)
        zaid = 1E6*self.z + self.a
        try:
            self._load_data()
        except FileNotFoundError:
            try:
                i = InputDeck.phits_input_deck(self.template_path, self.save_dir.parent, gen_run_script=False)
                i.write_inp_in_scope(locals(), new_file_name=self.save_dir.name)

                try:
                    self.ityp = {"H1": '1', 'electron': '12', 'positron': '13', 'H2': '15', 'H3': '16', 'He3': '17',
                                 'He4': '18'}[projectile.replace('-', '')]
                except KeyError:
                    self.ityp = '19'

                self._run('phits.sh')
                self.save_data()
                self.tree.Write()

            except Exception as e:
                self.unlink_save_dir()
                raise e

    def save_data(self):
        with open(self.save_dir/'ptrac') as f:
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

        if self.tree is not None and  self.tree.GetEntries()==0:
            warnings.warn("TTree was empty! Something went wrong. Not cashing result.")
            self.unlink_save_dir()
        self._set_cash()


class MCNP(Base):
    def __init__(self, projectile: str, material: Material, energy, nps=100, overwrite=False):
        super().__init__(projectile, material, energy, nps, overwrite)
        try:
            mode = {"He4": 'a', "He3": "s",'H3': 't', "H2":"d", "H": 'h', 'electron': 'e'}[projectile]
            zaid = mode
        except KeyError: # this measn nucleus, or invalid projectile
            zaid = 1000*self.z + self.a
            mode = "#"
        if mode == 'e':
            phys_card = "PHYS:e 7j 0 5j 0.98"
        else:
            phys_card = f"PHYS:{mode} 13j 0.98"

        # mode = "#"  # todo: make this compatible for protons and electrons
        material.mat_kwargs['Hstep'] = "100"
        #  projectile: Must be a for He4, etc. Todo
        try:
            self._load_data()
        except FileNotFoundError:
            try:
                i = InputDeck.mcnp_input_deck(self.template_path, self.save_dir.parent, gen_run_script=False)
                i.write_inp_in_scope(locals(), new_file_name=self.save_dir.name)

                self._run('mcnp6 i=')
                self.save_data()  # todo
                self.tree.Write()

            except Exception as e:
                self.unlink_save_dir()
                raise e

    def save_data(self):
        f_path = self.save_dir/"ptrac"
        ptrac2root_path = self.save_dir/'ptrac.root'
        assert f_path.exists(), "No PTRAC file found! Simulation had an error"
        p = ptrac2root(f_path)
        tree = TTreeHelper(ptrac2root_path)
        for t in tree:
            if t.is_term:
                self.fill(t.x, t.y, t.z, t.time)
        ptrac2root_path.unlink()
        self._set_cash()



# mat = Material.gas(['U'], atom_fractions=[1,1], pressure=1.4)
p = PHITS('He4', DepletedUranium(), 70, nps=10)
# p = PHITS('He4', DepletedUranium(), 70)

TBrowser()
