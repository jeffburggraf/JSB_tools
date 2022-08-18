"""
This file implements automatic execution of SRIM simulations and saves the results to a pickle file.
Saved results can be retrieved by the SRIMTable class which has a few useful functions.

In order to use SRIM you must be on a Windows machine. Installation instructions:
    1. Download SRIM from internet (file will be named e.g. SRIM-2013-Pro.e)
      rename SRIM-2013-Pro.e to SRIM-2013-Pro.exe and move it to JSB_tools/SRIM-2013. Double click to extract.
    4. A bunch of new stuff will appear, among them a directory named 'SRIM-Setup'. Go inside this directory and
        right click '_SRIM-Setup (Right-Click)'. Follow the prompts.
    6. Be done. Things should work now if you have the needed packages installed.

Examples of usage:

    from JSB_tools.MCNP_helper.materials import IdealGasProperties

    # Run SRIM for 1:1 atom ratio of Argon + He at 1.5 bar pressure

    g = IdealGasProperties(['He', 'Ar'])

    density = g.get_density_from_atom_fractions([1, 1], pressure=1.5, )

    run_srim(target_atoms=['He', 'Ar'],fractions=[1,1], density=density, projectile='Xe139', max_erg=120, gas=True)

    # The results can from now on be accessed by doing
    data = find_SRIM_run(target_atoms=['He', 'Ar'], fractions=[1,1], density=density, projectile='Xe139', gas=True)

    # And now plot, if you want
    data.plot_dedx()


"""
from __future__ import annotations
from typing import List, Union, Dict
import warnings
from pathlib import Path
# from mendeleev import element, Isotope
import re
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

srim_dir = Path(__file__).parent / "SRIM-2013"

save_dir = srim_dir / 'srim_outputs'
if not save_dir.exists():
    save_dir.mkdir(exist_ok=True, parents=True)


def _get_file_attribs_tuple(target_atoms, fractions, density, projectile, gas):
    density = f"{density:.4E}"
    arg_sort = np.argsort(target_atoms)
    target_atoms = list(map(lambda x: x[0].upper() + x[1:].lower(), target_atoms))
    target_atoms = np.array(target_atoms)[arg_sort]
    fractions = np.array(fractions)[arg_sort]
    fractions = fractions / sum(fractions)
    fractions = tuple(map(float, [f"{n:.3f}" for n in fractions]))
    target_atoms = tuple(target_atoms)
    projectile = projectile[0].upper() + projectile[1:].lower()
    file_attribs = (target_atoms, fractions, density, projectile, gas)

    return file_attribs


def existing_outputs():
    """
    Find all stored SRIM simulations. Delete/remove invalid entries in outputs.txt
    Returns:

    """
    try:
        with open(save_dir / "outputs.txt") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    out = {}
    name_params = {}
    new_lines = []
    re_write_file = False

    for line in lines:
        _ = line.split('__--__')
        f_name = _[-1].lstrip().rstrip()
        try:
            params = eval(_[0])
        except Exception as e:
            warnings.warn(f"Bad line in outputs.txt. Exception: {e}\nLine:\n{line}")
            continue
        name_params[f_name] = params

        if (save_dir / f_name).exists():
            out[params] = f_name
            new_lines.append(line)
        else:
            re_write_file = True

    if re_write_file or True:
        with open(save_dir / "outputs.txt", 'w') as f:
            for line in new_lines:
                f.write(line)

    return out


def _save_file_attribs(attribs_dict):
    lines = []
    for k, v in attribs_dict.items():
        lines.append(f"{k} __--__ {v}\n")
    with open(save_dir / "outputs.txt", 'w') as f:
        f.writelines(lines)


def _save_output(target_atoms, fractions, density, projectile, gas, save_SR_OUTPUT=False):
    file_attribs = _get_file_attribs_tuple(target_atoms, fractions, density, projectile, gas)
    all_file_attribs = existing_outputs()
    try:
        (save_dir / all_file_attribs[file_attribs]).unlink()
    except (KeyError, FileNotFoundError):
        pass

    files_so_far = [f.name for f in save_dir.iterdir() if re.match(r'out_[0-9]+\.pickle', f.name)]
    i = 0

    while (fname := f"out_{i}.pickle") in files_so_far:
        if fname not in all_file_attribs.values():
            (save_dir / fname).unlink()  # pickle file not in outputs.txt. delete and use this filename
            break
        i += 1

    all_file_attribs[file_attribs] = fname
    _save_file_attribs(all_file_attribs)

    data = []
    ergs = []
    stopping_units = None
    srim_output_path = srim_dir / 'SR Module' / 'SR_OUTPUT.txt'

    if save_SR_OUTPUT:
        import shutil

        i = 0
        _names = map(lambda x: x.name, (srim_dir / 'SR Module').iterdir())
        while (_name := f'SR_OUTPUT{i}.txt') in _names:
            i += 1
        shutil.copy(srim_output_path, srim_output_path.parent / _name)

    with open(srim_output_path) as f:
        line = f.readline()
        while not re.match('(-+ +){2,10}', line):
            line = f.readline()
            if m := re.match(' Stopping Units = +(.+)', line):
                stopping_units = m.groups()[0].rstrip().lstrip()

        assert stopping_units == 'MeV/(mg/cm2)', "Code doesn't accommodate other stopping units. " \
                                                 "If this happens, fix this. "

        def get_value(s):
            try:
                return float(s)
            except ValueError:
                return s

        dist_units = {'A': 1E-8, 'um': 1E-6, 'mm': 0.1, 'cm': 1, "m": 100, "km": 1 * 100 * 1000}
        erg_units = {"eV": 1E-6, "keV": 1E-3, "MeV": 1, "GeV": 1E3}
        while not re.match("-+", line := f.readline()):
            erg, erg_unit, elec, nuclear, range_, range_unit, lon_strag, lon_strag_unit, lat_strag, lat_strag_unit \
                = tuple(map(get_value, line.split()))
            try:
                range_ = range_ * dist_units[range_unit]
                lon_strag = lon_strag * dist_units[lon_strag_unit]
                lat_strag = lat_strag * dist_units[lat_strag_unit]
            except KeyError:
                assert False, f"Encountered an unknown unit (it's on of these: " \
                              f"{range_unit, lat_strag_unit, lon_strag_unit})"
            try:
                erg = erg_units[erg_unit] * erg
            except KeyError:
                assert False, f"Encountered an unknown energy unit {erg_unit} "
            elec *= 1000  # MeV/(mg/cm2) -> MeV/(g/cm2)
            nuclear *= 1000
            entry = {"nuclear": nuclear, "electric": elec, 'range': range_, 'lon_strag': lon_strag,
                     'lat_strag': lat_strag}
            data.append(entry)
            ergs.append(erg)

        with open(save_dir / fname, 'wb') as f:
            pickle.dump(ergs, f)
            pickle.dump(data, f)

        return save_dir/fname


class _SRIMConfig:
    """
    Used for storing and comparing SRIM configurations
    """
    __all_instances: List[_SRIMConfig] = []

    def __init__(self, target_atoms, fractions, density, projectile: Union[str, None], gas: int = 0, srim_file_name=None):
        arg_srt = np.argsort(target_atoms)
        self._target_atoms = np.array([target_atoms[i].lower() for i in arg_srt])
        self.target_atoms = np.array([target_atoms[i] for i in arg_srt])

        fractions = np.array(fractions, dtype=float)[arg_srt]
        fractions /= fractions
        self.fractions = fractions

        self.density = float(density)

        self.projectile = projectile
        if projectile is not None:
            self.projectile = self.projectile.lower()

        self.gas = bool(gas)

        self.srim_file_name = srim_file_name

    def __repr__(self):
        return f'{self.target_atoms}, {self.fractions}, {self.density}, {self.projectile}, {self.gas}'

    def __eq__(self, other):
        assert isinstance(other, _SRIMConfig)

        if not (self.projectile is None or other.projectile is None):
            if not self.projectile == other.projectile:
                return False

        if not len(self._target_atoms) == len(other._target_atoms):
            return False

        if not all(self._target_atoms == other._target_atoms):
            return False

        if not all(np.isclose(self.fractions, other.fractions)):
            return False

        if not self.gas == other.gas:
            return False

        return True

    @staticmethod
    def all_configs():
        if len(_SRIMConfig.__all_instances):
            return _SRIMConfig.__all_instances
        for k, v in existing_outputs().items():
            obj = _SRIMConfig(*k, srim_file_name=v)
            _SRIMConfig.__all_instances.append(obj)
        return _SRIMConfig.__all_instances


def find_all_SRIM_runs(target_atoms, fractions, density, gas: int = 0) -> Dict[str, _SRIMTable]:
    """
    Find all SRIM runs with a given target (but all available projectiles).
    Args:
        target_atoms:
        fractions:
        density:
        gas:

    Returns:
    """

    search_config = _SRIMConfig(target_atoms=target_atoms, fractions=fractions, density=density, projectile=None,
                                gas=gas)  # We now try to find an SRIM run with a configuration matching `search_config`

    candidates = [config for config in _SRIMConfig.all_configs() if search_config == config]  # all available configs
    if not len(candidates):  # No match found
        raise FileNotFoundError(f"No SRIM data with the following configuration:\n{search_config}")

    outs = {}

    for config in candidates:  # group configs by projectile
        try:
            outs[config.projectile].append(config)
        except KeyError:
            outs[config.projectile] = [config]

    for proj, configs in outs.items():
        densities = np.array([config.density for config in configs])
        best: _SRIMConfig = configs[np.argmin(np.abs(densities - density))]
        best_table = _SRIMTable(best.target_atoms, best.fractions, density, best.projectile, best.gas,
                                best.srim_file_name)  # SRIM run with closest matching density.

        if abs((best.density - density))/density > 0.75:
            warnings.warn(f"Could not find an SRIM run with a density similar to the provided density of {density}."
                          f"\nClosest was  {best.density} ")

        outs[proj] = best_table
    return outs


def find_SRIM_run(target_atoms, fractions, density, projectile: Union[str, None], gas: int = 0) -> _SRIMTable:
    """
    Use this to look up the results of past SRIM runs!
    Args:
        target_atoms:
        fractions:
        density:
        projectile:
        gas:

    Returns:

    """
    search_config = _SRIMConfig(target_atoms=target_atoms, fractions=fractions, density=density, projectile=projectile,
                                gas=gas)
    candidates = [config for config in _SRIMConfig.all_configs() if search_config == config]
    if not len(candidates):
        raise FileNotFoundError(f"No SRIM data with the following configuration:\n{search_config}")

    densities = np.array([config.density for config in candidates])
    best: _SRIMConfig = candidates[np.argmin(np.abs(densities - density))]
    out = _SRIMTable(best.target_atoms, best.fractions, density, best.projectile, best.gas, best.srim_file_name)
    return out


class _SRIMTable:
    #  TODO: Make a common de/dx structure that is used by MCNPStoppingPower and SRIM...
    #   Also somehow merge this or something into _SRIMConfig
    def __init__(self,  target_atoms, fractions, density, projectile, gas, file_name=None):
        self.target_atoms = target_atoms
        self.fractions = fractions
        self.density = density
        self.projectile = projectile
        self.gas = gas
        self.file_name = file_name

        with open(save_dir / self.file_name, 'rb') as f:
            self.ergs = pickle.load(f)
            data = pickle.load(f)

        self.electric = []
        self.nuclear = []
        self.ranges = []
        self.lat_straggling = []
        self.lon_straggling = []
        self.total_dedx = []

        for d in data:
            self.electric.append(d['electric'])
            self.nuclear.append(d['nuclear'])
            self.ranges.append(d['range'])
            self.lon_straggling.append(d['lon_strag'])
            self.lat_straggling.append(d['lat_strag'])
            self.total_dedx.append(self.electric[-1] + self.nuclear[-1])

        self.total_dedx = np.array(self.total_dedx)

    def __repr__(self):
        return f'{self.target_atoms}, {self.fractions}, {self.density}, {self.projectile}, {self.gas}'

    @property
    def mat_title(self):
        if len(self.target_atoms) == 1:
            return self.target_atoms[0]
        outs = []
        for a, f in zip(self.target_atoms, self.fractions):
            outs.append(f'{a}({float(f)})')
        return ' '.join(outs)

    def plot_dedx(self, ax=None, use_density=True):
        if ax is None:
            _, ax = plt.subplots()
        title = f"{self.projectile} in {self.mat_title}"
        x = self.ergs
        y = self.total_dedx
        if use_density:
            y *= self.density
        ax.plot(x, y)
        ax.set_xlabel('Energy [MeV]')
        if not use_density:
            ax.set_ylabel('Stopping Power [MeV cm2/g]')
        else:
            ax.set_ylabel('Stopping Power [MeV/cm]')
        ax.set_title(title)
        return ax

    def plot_range(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        title = f"{self.projectile} in {self.mat_title}"
        x = self.ergs
        y = self.ranges
        ax.plot(x, y)
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('range [cm]')
        ax.set_title(title)
        return ax

    def eval_dedx(self, ergs):
        if not hasattr(ergs, '__iter__'):
            ergs = [ergs]
        ergs = np.array(ergs)
        out = interp1d(self.ergs, self.total_dedx, kind='quadratic')(ergs)
        if len(out) == 1:
            return out[0]
        return out


def _check_args(target_atoms, fractions, density, projectile, max_erg, gas=False):
    assert hasattr(target_atoms, '__iter__')
    assert hasattr(fractions, '__iter__'), "Fractions must be an integer"

    assert isinstance(max_erg, (float, int))
    assert isinstance(density, (float, int))

    assert len(target_atoms) == len(fractions)

    bad = (np.where(np.array(fractions) == 0))
    good = (np.where(np.array(fractions) != 0))
    target_atoms = np.array(target_atoms)
    fractions = np.array(fractions)

    if len(bad[0]) > 0:
        warnings.warn(
            f"Removing the following elements due to zero atom fractions provided: {np.array(target_atoms)[bad]}")
    assert len(good) > 0, "No non-zero atom fractions!"
    _target_atoms = target_atoms[good]
    target_atoms = []
    for s in _target_atoms:
        m = re.match(r"([A-Z][a-z]{0,3})-?[0-9]?", s)
        assert m, f"Invalid target particle specification, '{s}'"
        target_atoms.append(m.groups()[0])
    fractions = fractions[good]
    m = re.match('([A-Za-z]{1,3})-*([0-9]+)*', projectile)
    assert m, f"Invalid projectile specification, '{projectile}'. Example: Xe139"
    proj_symbol = m.groups()[0]
    a = m.groups()[1]
    return target_atoms, fractions, density, (proj_symbol, a), max_erg, gas


def run_srim(target_atoms, fractions, density, projectile, max_erg, gas=False, save_SR_OUTPUT=False):
    """
    Run SRIM with the provided configuration and save them to be accessed later via the SRIMTable class.

    Parameters:
        target_atoms: List of element symbols, e.g. ["Ar"]
        fractions: List of atom fractions
        density: density in g/cm3
        projectile: Full symbol of projectile, e.g. "Xe139"
        max_erg: Energy in MeV
        gas: True if gas, else False
        save_SR_OUTPUT: If True, preserve the SRIM output text file. Used for debugging.
    """
    from srim import SR, Ion, Layer
    from plasmapy import particles

    target_atoms, fractions, density, projectile_info, max_erg, gas = \
        _check_args(target_atoms, fractions, density, projectile, max_erg, gas)
    proj_symbol, a = projectile_info

    # Deal with the pesky astropy.Unit object (it has no float method?!?! blasphemy)
    proj_mass = float(re.match('([0-9.]+e[-+0-9]+)', str(particles.Particle(f"{proj_symbol}-{a}").mass)).groups()[0]) \
                / 1.66053906660E-27  # convert kg to amu

    layer_arg = {}
    max_erg *= 1E6

    for s, frac in zip(target_atoms, fractions):
        layer_arg[s] = {"stoich": frac}

    gas = int(gas)
    layer = Layer(layer_arg, density, 1000, phase=int(gas))
    ion = Ion(proj_symbol, max_erg, proj_mass)

    sr = SR(layer, ion, output_type=5)
    sr.run(srim_directory=srim_dir)

    fname = _save_output(target_atoms, fractions, density, projectile, gas, save_SR_OUTPUT=save_SR_OUTPUT)

    return _SRIMTable(target_atoms=target_atoms, fractions=fractions, density=density, projectile=projectile,
                      gas=gas, file_name=fname)


if __name__ == '__main__':
    from JSB_tools.MCNP_helper.materials import IdealGasProperties

    # Run SRIM for 1:1 atom ratio of Argon + He at 1.5 bar pressure

    g = IdealGasProperties(['He', 'Ar'])

    density = g.get_density_from_atom_fractions([1, 1], pressure=1.5, )

    run_srim(target_atoms=['He', 'Ar'], fractions=[1, 1], density=density, projectile='Xe139', max_erg=120, gas=True)

    # Now, the results can from now on be accessed by
    data = find_SRIM_run(target_atoms=['He', 'Ar'], fractions=[1, 1], density=density, projectile='Xe139', gas=True)

    # Plot, if you want
    data.plot_dedx()

    plt.show()
