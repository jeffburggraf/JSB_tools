import pickle
import re
from pathlib import Path
import subprocess
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from os import system
from JSB_tools.nuke_data_tools import CrossSection1D, Nuclide
from openmc.data import ATOMIC_NUMBER, ATOMIC_SYMBOL
import subprocess

talys_dir = Path(__file__).parent/'data'/'talys'

tally_executable = '/Users/burggraf1/Talys/talys'


def run(target_nuclide, projectile, min_erg=0., max_erg=85, erg_step=None,
        fission_yields=False, maxlevelstar=10, maxlevelsres=10, auto_run=True, runnum=0,
        verbose=True,
        **kwargs):
    """
    Args:
        target_nuclide:
        projectile: n, p, d, t, h, a, g representing neutron, proton, deuteron, triton, 3He, alpha and gamma, respectively.
        min_erg: Min energy
        max_erg: Max energy
        erg_step: If None, use default TALYS energy grid.
        fission_yields: Calc. Fission yields?
        maxlevelstar: The number of excited levels of target (starting from gs=0) considered in non-elastic reactions.
        maxlevelsres: The number of excited levels of residue (starting from gs=0) considered in non-elastic reactions.
        auto_run: If False, won't run Talys, but will still prepare input file.
        runnum: Input file is created in within talys/{target}-{projectile}/runnum.
        verbose: If True, print run config to stdout
    Returns:

    """

    m = re.match("([A-Z][a-z]{0,2})-?([0-9]{1,3})(?:_m([0-9]+))?", target_nuclide)
    assert m
    a_symbol = m.groups()[0]
    mass = int(m.groups()[1])

    out_kwargs = {'projectile': projectile, 'element': a_symbol, 'mass': mass}
    iso_state = int(m.groups()[2]) if m.groups()[2] is not None else 0

    if iso_state != 0:
        out_kwargs['Ltarget'] = iso_state
        f_name = f"{a_symbol}{mass}_m{iso_state}-{projectile}"
    else:
        f_name = f"{a_symbol}{mass}-{projectile}"

    if erg_step is None:
        assert isinstance(min_erg, int), "When using default energy grid, min/max must be int"
        assert isinstance(max_erg, int), "When using default energy grid, min/max must be int"
        out_kwargs['energy'] = f'{projectile}{min_erg}-{max_erg}.grid'
    else:
        out_kwargs['energy'] = f'energy {min_erg} {max_erg} {erg_step}'

    if fission_yields:
        out_kwargs['outfy'] = True

    if maxlevelsres + maxlevelstar > 0:
        kwargs['outdiscrete'] = 'y'

        if maxlevelstar > 0:
            kwargs['maxlevelstar'] = maxlevelstar  #

        if maxlevelsres > 0:
            kwargs['maxlevelsres'] = maxlevelsres  #

    for k, v in kwargs.items():
        if isinstance(v, bool):
            v = 'y' if v else 'n'
        out_kwargs[k] = v

    lines = []

    for k, v in out_kwargs.items():
        lines.append(f"{k} {v}")

    assert projectile in ['p', 'n', 'g']

    data_path = talys_dir/f"{f_name}/{runnum}"

    if verbose:
        print(f"Running Talys in {data_path}.\n\tInput keywords:\n\t{out_kwargs}")

    Path.mkdir(data_path, exist_ok=True, parents=True)

    with open(data_path/f_name, 'w') as f:
        f.write('\n'.join(lines))

    cmd = f'cd {data_path};{tally_executable} < {f_name} > output'
    # cmds = [f'cd {data_path}', f'{tally_executable} < {f_name} > output']

    if auto_run:
        # process = subprocess.Popen(cmds) #, stdout=PIPE, stderr=PIPE)
        # stdout, stderr = process.communicate()
        out = system(cmd)
        try:
            with open(data_path / 'output') as f:
                _flag = 0  # 3 make None to begin printing
                while line := f.readline():
                    if "TALYS-error" in line:
                        _flag = None  # will not print till end
                    if _flag is None:
                        raise ValueError(line)
                    else:  # _flag is not None
                        _flag += 1
                        if _flag > 10:  # give up, not an error file
                            break

            meta_path = data_path.parent / 'talys_inputs.pickle'

            try:
                with open(meta_path, 'rb') as f:
                    meta_data = pickle.load(f)
            except FileNotFoundError:
                meta_data = {}
            if verbose:
                if runnum in meta_data and len([_ for _ in data_path.iterdir()]):
                    print(f"Overwrote ../{data_path.relative_to(talys_dir)}")
            meta_data[runnum] = {'input': '\n'.join(lines), 'kwargs': out_kwargs}

            with open(meta_path, 'wb') as f:
                pickle.dump(meta_data, f)

        except FileNotFoundError:
            raise
    else:
        out = None

    return out


def pickle_result(target, projectile):
    data = {}
    path = talys_dir/f'{target}-{projectile}'

    for p in path.iterdir():
        m_ground = re.match('rp[0-9]+\.tot', p.name)
        m_excited = re.match('rp[0-9]+\.L([0-9]+)', p.name)

        if m_ground or m_excited:

            with open(p) as f:
                lines = f.readlines()

            m = re.match('.+Production of +([0-9]+)([A-Z][a-z]{0,2})', lines[0])
            product = f'{m.groups()[1]}{m.groups()[0]}'
            if m_excited:
                product += f'_m{int(m_excited.groups()[0])}'
            if product == target:
                continue

            def get_values():
                if m_excited:
                    return tuple(map(float, line.split()[:2]))
                else:
                    return tuple(map(float, line.split()))

            for i in range(len(lines)):
                if lines[i][0] != '#':
                    break
            lines = lines[i:]
            ergs = []
            xss = []
            for line in lines:
                erg, xs = get_values()
                ergs.append(erg)
                xss.append(xs*1E-3)
            data[product] = xss

    with open(path/'data.pickle', 'wb') as f:
        pickle.dump(data, f)
        pickle.dump(ergs, f)


class ReadResult:
    particles = {'neutron': 'n', 'gamma': 'g', 'alpha': 'a', 'H3': 't', 'He3': 'h', 'H2': 'd', 'proton': 'p'}

    @staticmethod
    def _get_par(par):
        try:
            par = ReadResult.particles[par]
        except KeyError:
            if not (par in ReadResult.particles.values()):
                valid_pars = list(zip(ReadResult.particles.keys(), ReadResult.particles.values()))
                raise ValueError(f"Invalid particle, '{par}'. Valid particles are:\n{valid_pars}")
        return par

    def __init__(self, nuclide_name, projectile, runnum=0):
        """

        Args:
            nuclide_name:
            projectile:
            runnum: Multiple runs with the same nucleus
        """
        projectile = self._get_par(projectile)
        directory = Path(__file__).parent/'data'/'talys'/f'{nuclide_name}-{projectile}'/f'{runnum}'
        self.directory = Path(directory)
        try:
            self.files = list(self.directory.iterdir())
        except FileNotFoundError:
            options = []
            if directory.parent.exists():
                dirs = [f.name for f in directory.parent.iterdir() if re.match("[0-9]+", f.name)]
                options = dirs
            raise FileNotFoundError(f"No Talys run for {projectile} on {nuclide_name}, runnum {runnum}. "
                                    f"Runnum options are {options}")

        self.runnum = runnum

        self.nuclide_name = nuclide_name

        self.projectile = self._get_par(projectile)

        try:
            with open(self.directory.parent/"talys_inputs.pickle", 'rb') as f:
                data = pickle.load(f)[runnum]
            self.input_file = data['input']
            self.input_kwargs = data['kwargs']
        except (KeyError, FileNotFoundError):
            self.input_file = None

    def fission(self, res=None) -> CrossSection1D:
        """

        Args:
            res: A residue nuclei for second/third/.. etc fission. Default is self.nucleus

        Returns:

        """

        if res is None:
            f_name = 'fission.tot'
            fig_label = f"{self.nuclide_name}({self.projectile}, F)"
        else:

            n = Nuclide.from_symbol(res)
            f_name = f'rp{n.Z:0>3}{n.A:0>3}.fis'
            fig_label = f"{self.nuclide_name}({self.projectile}, ({n.name})F)"

        with open(self.directory/f_name) as f:
            ergs, xss = self._read_file_simple(f)

        out = CrossSection1D(ergs, xss, fig_label, self.projectile, 'TALYS')

        return out

    def __repr__(self):
        return " | ".join(f"{k} = {v}" for k, v in self.input_kwargs.items())

    @staticmethod
    def _read_file_simple(buffer, erg_index=0, xs_index=1):
        ergs = []
        xss = []

        for line in buffer:
            if line[0] == '#':
                continue

            for i, v in enumerate(line.split()):
                if i == erg_index:
                    ergs.append(float(v))
                elif i == xs_index:
                    xss.append(float(v)*1E-3)
        return ergs, xss

    def inelastic(self):
        path = self.directory/'nonelastic.tot'

        with open(path) as f:
            ergs, xss = self._read_file_simple(f)

        return CrossSection1D(ergs, xss, f"{self.nuclide_name} tot. inelastic", incident_particle=self.projectile,
                              data_source='TALYS')

    def elastic(self):
        path = self.directory/'elastic.tot'

        with open(path) as f:
            ergs, xss = self._read_file_simple(f)

        return CrossSection1D(ergs, xss, f"{self.nuclide_name} tot. elastic", incident_particle=self.projectile,
                              data_source='TALYS')

    def all_residues(self, a_z_m_cut=None, return_filenames=False) -> List[str]:
        """

        Args:
            a_z_m_cut: python string to be evaluated in scope with nucleus variables mass number as 'a',
                proton number as 'z', and isometric level as 'm'.
            return_filenames: If True, return list of files names. Else, return list of nuclide names,
                e.g. ['U235', 'U235_m1', 'U235_m2'].

        Returns:

        """
        outs = []

        def cut():
            if a_z_m_cut is not None:
                return eval(a_z_m_cut, {'a': a, 'z': z, 'm': m})
            return True

        for path in self.directory.iterdir():
            if match := re.match(r"rp([0-9]{3})([0-9]{3})\.(?:L([0-9]{2}))?", path.name):
                if path.suffix == '.fis':
                    continue
                if return_filenames:
                    out = path.name
                else:
                    z = int(match.groups()[0])
                    a = int(match.groups()[1])

                    out = f"{ATOMIC_SYMBOL[z]}{a}"

                    m = match.groups()[2]

                    if m is not None:
                        out += f'_m{m}'

                if cut():
                    outs.append(out)

        return list(sorted(outs))

    def get_any_xs(self, file_name, erg_index=0, xss_index=1) -> Tuple[list, list]:
        """
        Grab a cross-section by file name.
        Args:
            file_name:
            erg_index:
            xss_index:

        Returns:

        """
        with open(self.directory/file_name) as f:
            return self._read_file_simple(f, erg_index, xss_index)

    def all_inelastic2levels(self, outgoing_par=None, sort_by_total=False):
        """

        Args:
            outgoing_par:
            sort_by_total: If True, sorts by most to least cross-section. Else, sort by level.

        Returns:

        """
        if outgoing_par is None:
            outgoing_par = self.projectile

        levels = []
        for path in self.directory.iterdir():
            if path.is_file():
                if m := re.match(f"{self.projectile}{outgoing_par}.L([0-9]+)", path.name):
                    levels.append(int(m.groups()[0]))

        if not sort_by_total:
            return list(sorted(levels))
        else:
            outs = []
            tots = []

            for l in levels:
                xs = self.inelastic2level(l)
                tot = max(xs.xss)
                index = np.searchsorted(tots, tot)
                outs.insert(index, l)
                tots.insert(index, tot)

            # outs = [k for k in sorted(outs.keys(), key=lambda x: -outs[x])]
            outs = outs[::-1]

            return outs

    def inelastic2level(self, level, outgoing_par=None):
        """
        Direct inelastic excitation to specific level. To get production of a level including de-excitation from higher
         levels, use self.residue_production
        Args:
            level:
            outgoing_par: Defaults to same as incident particle. For other channels, change to, e.g., p, a, h, d.

        Returns:

        """
        assert isinstance(level, int), f'Invalid type, "{type(level)}" of arg `level`.'

        if outgoing_par is None:
            outgoing_par = self.projectile

        f_name = f"{self.projectile}{outgoing_par}.L{level:0>2}"

        with open(self.directory/f_name) as f:
            q_value = None
            threshold = None
            ergs = []
            xss = []

            for line in f.readlines():
                if line[0] == '#':
                    if match := re.match(r'# Q-value += ?([-0-9.E+]+)', line):
                        q_value = float(match.groups()[0])
                    elif match := re.match(r'# *E-threshold *= ?([-0-9.E+]+)', line):
                        threshold = float(match.groups()[0])

                else:
                    erg, xs, _, _ = map(float, line.split())  # branching ratio is given in this case

                    ergs.append(erg)
                    xss.append(xs * 1E-3)

        # f'{self.nucleus}({self.projectile},{outgoing_par}) to L.{level:0>2} (Q={q_value * 1E3:.3e} keV)',
        out = CrossSection1D(ergs, xss, fig_label=f"{self.nuclide_name}: direct to L{level}",
                             incident_particle=self.projectile,
                             data_source='TALYS', q_value=q_value, threshold=threshold)
        return out

    def residue_production(self, res: str, filename_in_label=False, label=None):
        """
        Total cross-section for production of a residue nucleus.

        Args:
            res: string, e.g. 'U235_m1' or 'U235'
            filename_in_label: Include filename in legend label.

        Notes:
            Using "U235: as an example...
            If Talys calculates production of isomers, then res="U235" gives to the sum of all U235 nuclei regardless
                of isomeric state. On the other hand, "U235_m0" will give production yield of ground state specifically.
            The production cross-section of a given excitation level includes feeding from higher levels.
                However, if a level is considered an isomer according to the isomer keyword, than it will not feed
                lower levels. Thus, changing the isomer keyword in Talys input may affect the results for any "U235_mi".

        Returns: CrossSection1D

        """
        match = re.match("([A-Z][a-z]{0,2})-?([0-9]{1,3})(?:_m([0-9]+))?", res)
        symbol = match.groups()[0]
        a = match.groups()[1]
        m = match.groups()[2]
        z = ATOMIC_NUMBER[symbol]
        # res_nuclide_name = f"{symbol}{a}"

        f_name = f"rp{z:0>3}{a:0>3}"

        if m is None:
            f_name += '.tot'
        else:
            m = int(m)
            f_name += f'.L{m:0>2}'

        ergs, xss = [], []

        q_value = None
        level_erg = 0

        with open(self.directory/f_name) as f:
            for line in f.readlines():
                if line[0] == '#':
                    if match := re.match(r'# Q-value +=([-0-9.E+]+)', line):
                        q_value = float(match.groups()[0])
                    elif match := re.match('#.+ ([0-9.]+) MeV', line):
                        level_erg = 1E3 * float(match.groups()[0])

                else:
                    if m is None:
                        erg, xs = map(float, line.split())
                    else:
                        erg, xs, _ = map(float, line.split())  # branching ratio is given in this case

                    ergs.append(erg)
                    xss.append(xs*1E-3)

        if label is None:
            fig_label = f'{self.nuclide_name}({self.projectile},X){res}'
            if level_erg != 0:
                fig_label += f" {level_erg:.3e} keV"

            if filename_in_label:
                fig_label += f'; {f_name}'
        else:
            fig_label = label

        out = CrossSection1D(ergs, xss, fig_label=fig_label,
                             incident_particle=self.projectile,
                             q_value=q_value, level_erg=level_erg)
        return out

    def all_gamma_transitions(self, nucleus=None):
        """
        Return all available gamma transition files.
        Args:
            nucleus: Defaults to target nucleus.

        Returns:

        """
        if nucleus is None:
            nucleus = self.nuclide_name

        n = Nuclide.from_symbol(nucleus)
        z = n.Z
        a = n.A

        options = []
        for p in self.directory.iterdir():
            if m := re.match(f"gam{z:0>3}{a:0>3}L([0-9]+)L([0-9]+)", p.name):
                options.append((int(m.groups()[0]), int(m.groups()[1])))
        options = list(sorted(options))
        return options

    def gamma_production_xs(self, init_level, final_level, nucleus=None, label=None):
        if nucleus is None:
            nucleus = self.nuclide_name

        n = Nuclide.from_symbol(nucleus)
        z = n.Z
        a = n.A
        f_name = f"gam{z:0>3}{a:0>3}L{init_level:0>2}L{final_level:0>2}.tot"
        gamma_erg = None
        final_erg = None
        init_erg = None
        ergs = []
        xss = []

        if not (self.directory/f_name).exists():
            options = ", ".join(map(str, self.all_gamma_transitions(nucleus)))

            raise FileNotFoundError(f"No transition from available from level {init_level} to level {final_level}."
                                    f" Options are:\n{options} ")

        with open(self.directory/f_name) as f:
            for line in f.readlines():
                if line[0] == '#':
                    if m := re.match('# E-initial += +([0-9.E]+) E-final= +([0-9.E]+)', line):
                        init_erg = float(m.groups()[0])*1E3
                        final_erg = float(m.groups()[1])*1E3
                        gamma_erg = init_erg - final_erg
                    continue
                erg, xs = map(float, line.split())
                xs *= 1E-3
                ergs.append(erg)
                xss.append(xs)
        if label is None:
            leg_label = fr"{self.nuclide_name}: L{init_level} $\rightarrow$ L{final_level} $\gamma$ at {gamma_erg:.1f} keV"
        else:
            leg_label = label

        out = CrossSection1D(ergs, xss,
                             fig_label=leg_label,
                             incident_particle=self._get_par(self.projectile))
        return out

    def particle_production(self, par) -> CrossSection1D:
        par = ReadResult._get_par(par)
        path = self.directory/f'{par}prod.tot'
        ergs = []
        xss = []

        with open(path) as f:
            q_value = None

            for line in f.readlines():
                if line[0] == '#':
                    if m := re.match(r'# Q-value +=([-0-9.E+]+)', line):
                        q_value = float(m.groups()[0])
                    continue
                erg, xs, elastic_fraction = tuple(map(float, line.split()))
                ergs.append(erg)
                xss.append(xs*1E-3)

        out = CrossSection1D(ergs, xss, fig_label=f'{self.nuclide_name}; {par} production', incident_particle=self.projectile,
                             data_source='TALYS',
                             q_value=q_value)
        return out


if __name__ == '__main__':

    # rgs = ReadResult('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/data/talys/U235-n')
    # rm1 = ReadResult('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/data/talys/U235_m1-n')
    #
    # xs = rgs.inelastic()
    # ax = xs.plot()
    #
    # xs = rm1.inelastic()
    # xs.plot(ax=ax)
    # plt.show()

    # args = ('N14', 'g')
    target = 'Th232'
    projectile = 'n'
    runnum = None  # None for main folder name
    max_erg = 35

    run('U235', "p", max_erg=25, min_erg=1, isomer=1E-12, fileresidual=True, maxlevelstar=100,
        maxlevelsres=100, runnum=0)
    # run(target, projectile, auto_run=True, max_erg=max_erg,
    #     maxlevelstar=30, maxlevelsres=30, fission='n', runnum=runnum, fileresidual='y',
    #     outlevels='y', isomer=1E-12, maxZ=2, maxN=2, outgamdis='y')

