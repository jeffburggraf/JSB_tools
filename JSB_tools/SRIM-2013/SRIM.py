# from plasmapy import particles
# from astropy.units import quantity
from srim import SR, Ion, Layer, Material, Element
from pathlib import Path
# from mendeleev import element, Isotope
import re
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

cwd = Path(__file__).parent

save_dir = cwd/'srim_outputs'

if not save_dir.exists():
    save_dir.mkdir(exist_ok=True, parents=True)


def _get_file_attribs_tuple(target_atoms, fractions, density, projectile, gas):
    density = f"{density:.4E}"
    arg_sort = np.argsort(target_atoms)
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


def save_file_attribs(attribs_dict):
    lines = []
    for k, v in attribs_dict.items():
        lines.append(f"{k} __--__ {v}\n")
    with open(save_dir/"outputs.txt", 'w') as f:
        f.writelines(lines)


def save_output(target_atoms, fractions, density, projectile, gas):
    file_attribs = _get_file_attribs_tuple(target_atoms, fractions, density, projectile, gas)
    all_file_attribs = existing_outputs()
    try:
        (save_dir/all_file_attribs[file_attribs]).unlink()
    except (KeyError, FileNotFoundError):
        pass
    files_so_far = [f.name for f in save_dir.iterdir() if re.match('.+\.pickle', f.name)]
    i = 0
    while (fname := f"out_{i}.pickle") in files_so_far:
        i += 1

    all_file_attribs[file_attribs] = fname
    save_file_attribs(all_file_attribs)

    data = []
    ergs = []

    with open(cwd/'SR Module'/'SR_OUTPUT.txt') as f:
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
            # print(erg, unit, elec, nuclear, range_,  range_unit, lon_strag, lon_strag_unit, lat_strag, lat_strag_unit)

        with open(save_dir/fname, 'wb') as f:
            pickle.dump(ergs, f)
            pickle.dump(data, f)
            print(data)


class SRIMTable:
    def __init__(self, target_atoms, fractions, density, projectile, gas=False):
        attribs = _get_file_attribs_tuple(target_atoms, fractions, density, projectile, gas)
        try:
            print(existing_outputs())
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

    def save_4_phits(self, file_name, dedx_path=Path.expanduser(Path("~"))/'phits'/'phits'/'data'/'dedx'):
        print(dedx_path)
        #  Make a get_kf_code option in JSB_tools.PHITS_helper
        lines = ['unit = 1'] #, f'kf = {TODO}']

        assert dedx_path.exists()

        return "\n".join(lines)


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
    from plasmapy import particles
    from astropy.units import quantity
    assert len(target_atoms) == len(fractions)
    m = re.match('([A-Za-z]{1,3})-*([0-9]+)*', projectile)
    assert m
    proj_symbol = m.groups()[0]
    a = m.groups()[1]
    proj_mass = particles.Particle(f"{proj_symbol}-{a}").mass/quantity.Quantity(1.66053906660E-27, unit='kg')
    layer_arg = {}
    max_erg *= 1E6

    for s, frac in zip(target_atoms, fractions):
        layer_arg[s] = {"stoich": frac}

    sr = SR(Layer(layer_arg, density, 1000, phase=int(gas)), Ion(proj_symbol, max_erg, proj_mass), output_type=5)
    sr.run(Path(__file__).parent)
    save_output(target_atoms, fractions, density, projectile, gas)
    return SRIMTable(target_atoms, fractions, density, projectile, gas)



if __name__ == '__main__':
    from JSB_tools.MCNP_helper.materials import Material
    s = run_srim(["Si", "N"], [1, 1], 2.253, 'Mo106', 105, gas=False)
    s = run_srim(["Si", "N"], [1, 1], 2.253, 'Ba142', 105, gas=False)
    s = run_srim(["C"], [1], 2.2, 'Mo106', 105, gas=False)
    s = run_srim(["C"], [1], 2.2, 'Ba142', 105, gas=False)

    for frac in np.arange(0,1,0.1):
        m = Material.gas(['He', 'Ar'], atom_fractions=[frac, 1-frac], pressure=1.35, )
        run_srim(['He', 'Ar'], [frac, 1-frac], m.density, 'Xe139', 120)
        run_srim(['He', 'Ar'], [frac, 1-frac], m.density, 'Mo105', 120)
