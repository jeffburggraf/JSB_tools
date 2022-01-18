"""
Runs a SRIM simulation and saves the results to a pickle file.
Pickle file has two dumps:
    1. list of energies
    2. a list of dicts of data for each entry
The dicts have the following format:
    {"nuclear": nuclear, "electric": elec, 'range': range_, 'lon_strag': lon_strag,
                     'lat_strag': lat_strag}


"""
import warnings
from pathlib import Path
# from mendeleev import element, Isotope
import re
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

srim_dir = Path(__file__).parent/"SRIM-2013"

save_dir = srim_dir/'srim_outputs'
if not save_dir.exists():
    save_dir.mkdir(exist_ok=True, parents=True)


def _get_file_attribs_tuple(target_atoms, fractions, density, projectile, gas):
    density = f"{density:.4E}"
    arg_sort = np.argsort(target_atoms)
    target_atoms = np.array(target_atoms)[arg_sort]
    fractions = np.array(fractions)[arg_sort]
    fractions = fractions / sum(fractions)
    fractions = tuple(map(float, [f"{n:.3f}" for n in fractions]))
    target_atoms = tuple(target_atoms)
    file_attribs = (target_atoms, fractions, density, projectile, gas)

    return file_attribs


def existing_outputs():
    try:
        with open(save_dir/"outputs.txt") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    out = {}

    for line in lines:
        _ = line.split('__--__')
        f_name = _[-1].lstrip().rstrip()
        params = eval(_[0])
        if (save_dir/f_name).exists():
            out[params] = f_name
    return out


def _save_file_attribs(attribs_dict):
    lines = []
    for k, v in attribs_dict.items():
        lines.append(f"{k} __--__ {v}\n")
    with open(save_dir/"outputs.txt", 'w') as f:
        f.writelines(lines)


def _save_output(target_atoms, fractions, density, projectile, gas):
    file_attribs = _get_file_attribs_tuple(target_atoms, fractions, density, projectile, gas)
    all_file_attribs = existing_outputs()
    try:
        (save_dir/all_file_attribs[file_attribs]).unlink()
    except (KeyError, FileNotFoundError):
        pass
    files_so_far = [f.name for f in save_dir.iterdir() if re.match('out_[0-9]+\.pickle', f.name)]
    i = 0
    while (fname := f"out_{i}.pickle") in files_so_far:
        i += 1

    all_file_attribs[file_attribs] = fname
    _save_file_attribs(all_file_attribs)

    data = []
    ergs = []

    with open(srim_dir/'SR Module'/'SR_OUTPUT.txt') as f:
        line = f.readline()
        while not re.match('(-+ +){2,10}', line):
            line = f.readline()

        def get_value(s):
            try:
                return float(s)
            except ValueError:
                return s
        dist_units = {'A':1E-8, 'um': 1E-6, 'mm': 0.1, 'cm': 1, "m": 100, "km": 1*100*1000}
        erg_units = {"eV": 1E-6, "keV": 1E-3, "MeV": 1, "GeV": 1E3}
        while not re.match("-+", line := f.readline()):
            erg, erg_unit, elec, nuclear, range_,  range_unit, lon_strag, lon_strag_unit, lat_strag, lat_strag_unit \
                = tuple(map(get_value, line.split()))
            try:
                range_ = range_*dist_units[range_unit]
                lon_strag = lon_strag*dist_units[lon_strag_unit]
                lat_strag = lat_strag*dist_units[lat_strag_unit]
            except KeyError:
                assert False, f"Encountered an unknown unit (it's on of these: " \
                              f"{range_unit, lat_strag_unit, lon_strag_unit})"
            try:
                erg = erg_units[erg_unit]*erg
            except KeyError:
                assert False, f"Encountered an unknown energy unit {erg_unit} "
            elec *= 1000
            nuclear *= 1000
            entry = {"nuclear": nuclear, "electric": elec, 'range': range_, 'lon_strag': lon_strag,
                     'lat_strag': lat_strag}
            data.append(entry)
            ergs.append(erg)

        with open(save_dir/fname, 'wb') as f:
            pickle.dump(ergs, f)
            pickle.dump(data, f)


class SRIMTable:
    def __init__(self, target_atoms, fractions, density, projectile, gas: int=0):
        attribs = _get_file_attribs_tuple(target_atoms, fractions, density, projectile, gas)
        try:
            file_name = existing_outputs()[attribs]
        except KeyError:
            assert False, f"No SRIM data for the following configuration:\n{attribs}\nAdd data on a windows machine" \
                          f"using `run_srim(..)`. "
        with open(save_dir/file_name, 'rb') as f:
            self.ergs = pickle.load(f)
            data = pickle.load(f)
        self.target_atoms = attribs[0]
        self.atom_fractions = attribs[1]
        self.density = attribs[2]
        self.proj = attribs[3]
        self.is_gas = attribs[-1]
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

    @property
    def mat_title(self):
        if len(self.target_atoms) == 1:
            return self.target_atoms[0]
        outs = []
        for a, f in zip(self.target_atoms, self.atom_fractions):
            outs.append(f'{a}({float(f)})')
        return ' '.join(outs)

    def plot_dedx(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        title = f"{self.proj} in {self.mat_title}"
        x = self.ergs
        y = self.total_dedx
        ax.plot(x, y)
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('Stopping Power [MeV cm2/g]')
        ax.set_title(title)
        return ax

    def plot_range(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        title = f"{self.proj} in {self.mat_title}"
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
        warnings.warn(f"Removing the following elements due to zero atom fractions provided: {np.array(target_atoms)[bad]}")
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


def run_srim(target_atoms, fractions, density, projectile, max_erg, gas=False):
    """
    Run SRIM with the provided configuration and save them to be accessed later via the SRIMTable class.

    Parameters:
        target_atoms: List of element symbols, e.g. ["Ar"]
        fractions: List of atom fractions
        density: density in g/cm3
        projectile: Full symbol of projectile, e.g. "Xe139"
        max_erg: Energy in MeV
        gas: True if gas, else False
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
    _save_output(target_atoms, fractions, density, projectile, gas)

    return SRIMTable(target_atoms, fractions, density, projectile, gas)


if __name__ == '__main__':
      # For IAC models
    from JSB_tools.MCNP_helper.materials import _IdealGas

    for n in ['Xe139', 'Kr91', 'Sr94', 'Sb132']:
        run_srim(['C'], [1], 1.7, n, 120)

    # g = _IdealGas(['Ar'])
    # g_ar_he = _IdealGas(['Ar', 'He'])
     # = _IdealGas(['Ar'])
    # for n in ['Xe139', 'Kr91', 'Sr94', 'Sb132']:
    #     run_srim(['U'], [1], 19, n, 120)  # Uranium
    #     for press in [0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.6]:
    #         density = g.get_density_from_atom_fractions([1], pressure=press)
    #         run_srim(['Ar'], [1], density, n, 120, True )  # Pure argon
    #
    #         density_heR = g_ar_he.get_density_from_atom_fractions([1,1], pressure=press)
    #         run_srim(['Ar', 'He'], [1, 1], density, n, 120, True)  # He Ar Mix
    #
    #
    #     for frac in np.arange(0, 0.6, 0.1):
    #         fracs = [0.5-frac, frac]
    #         density = g_ar_he.get_density_from_atom_fractions(fracs, pressure=1)
    #         run_srim(['Ar', 'He'], fracs, density, n, 120, True)
    # atoms = ['Ar40', 'He4']
    # he_fracs = list(np.arange(0, 1.2, 0.2)) +   
    # # he_fracs = [0.5]
    # for he_frac in sorted(he_fracs):
    #     fractions = [he_frac, 1-he_frac]
    #     g = _IdealGas(atoms)
    #     density = g.get_density_from_atom_fractions(fractions, pressure=1.1, )
    #     run_srim(atoms, fractions, density, 'Xe139', 120, True)
    #     run_srim(atoms, fractions, density, 'Kr91', 120, True)
    #     # run_srim()
    # run_srim(['U'], [1], 19.1, "Xe139", 120, False)
    # run_srim(['U'], [1], 19.1, "Kr91", 120, False)

