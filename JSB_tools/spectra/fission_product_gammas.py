import pickle
import re
from JSB_tools.tab_plot import TabPlot
import numpy as np
from JSB_tools.nuke_data_tools.nuclide.fission_yields import FissionYields
from JSB_tools.nuke_data_tools import DecayNuclide, Nuclide
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
cwd = Path(__file__).parent
matplotlib.use('Qt5agg')

year = 365 * 24 * 60**2

HEADERS = ['Source Nuclide', 'Decaying Nuclide', 'Erg [keV]', 'Rel. Intensity', 'Gamma intensity [%]']

lengths = list(map(len, HEADERS))
HEADER = ";".join(HEADERS) + '\n'


def print_gammas2(source_name, daughter_name, erg, rel_intensity, gamma_intensity):
    return (f"{source_name: <{lengths[0]}};{daughter_name: <{lengths[1]}};"
            f"{erg: <{lengths[2]}.2f};{rel_intensity: <{lengths[3]}.3e};"
            f"{gamma_intensity: <{lengths[4]+1}.3e}")


def print_gamms(source_nuclide_name, rel_intensity, g):
    return print_gammas2(source_nuclide_name, g.parent_nuclide_name, g.erg.n, rel_intensity, g.intensity.n)
    # return f"{source_nuclide_name: <8}; {g.parent_nuclide.name: <8}; {g.erg.n: <7.2f}; {rel_intensity: <3.2e}; {g.intensity.n * 100: <4.1f}"


# =====================================
write = input("Write gamma file? (y/n)").lower() == 'y'
# ================================
if write:
    file = open('gammas', 'w')
    file.write(HEADER)
else:
    file = None

lines = {}

for path in (cwd / "peak_easy_gammas").iterdir():
    if m := re.match(r"([A-Z][a-z]*)-([0-9]+)?(.+)\.csv", path.name):
        symbol = m.groups()[0]

        A = int(m.groups()[1])
        try:
            abundance = Nuclide.isotopic_breakdown(symbol)[A]
        except KeyError:
            abundance = 1
        decay_mode = m.groups()[-1]

        source_nuclide = f"{symbol}{A}"
        if decay_mode == "(n,n'g)":
            decaying_daughter = source_nuclide
        elif decay_mode == "(n,g)":
            decaying_daughter = f'{symbol}{A + 1}'
        else:
            assert False, path

        with open(path) as f:
            for line in f.readlines()[3:]:
                erg, intensity, _, _ = line.split(',')
                erg = float(erg)
                try:
                    intensity = float(intensity)
                except ValueError:
                    if intensity.strip() == 'Most Likely':
                        intensity = 1
                    intensity = np.nan

                data = {'target': source_nuclide, 'decaying_daughter': decaying_daughter, 'intensity': intensity, 'abundance': abundance, 'energy': erg}
                if symbol not in lines:
                    lines[symbol] = {}

                try:
                    lines[symbol][A].append(data)
                except KeyError:
                    lines[symbol][A] = [data]

if write:
    with open("neutron_activation_lines.pickle", 'wb') as f:
        pickle.dump(lines, f)

print()



parent = Nuclide("Cf252")

fiss_branch = parent.decay_modes[('sf',)][0].branching_ratio
fiss_hl = np.log(2)/(fiss_branch * parent.decay_rate.n)

dt = 7 * year
yields_cum = FissionYields(parent.name, None, independent_bool=False)
yields_indp = FissionYields(parent.name, None, independent_bool=True)

yields_cum.threshold(1E-5)


decay = DecayNuclide('Cf252', fission_yields=yields_indp)
old_cf_rates = decay(dt, decay_rate=True)


rates = {'fiss. prod': {}}

i=0
fp_gammas = []


max_rate = -np.inf

i = 0
for k, v in old_cf_rates.items():
    for gline in Nuclide(k).decay_gamma_lines:
        gline.rate = v * gline.intensity.n

        max_rate = max(gline.rate, max_rate)

        fp_gammas.append(gline)


fp_gammas = list(sorted(fp_gammas, key=lambda x:-x.rate))

print("FP gammas")
i = 0

for g in fp_gammas:
    frac = g.rate / max_rate

    if frac < 0.5E-1:
        break
    i += 1

    print(i, frac, print_gamms('Fiss. Prod', frac, g))
    if file is not None:
        file.write(print_gamms('Fiss. Prod', frac, g) + "\n")



class Source:
    def __init__(self, src_name):
        self.name = src_name
        self.init_parents = []
        self.init_fracs = []

        self.rates = None
        self.ts = None
        self.max_final_rate = -np.inf

    def set_rates(self, ts):
        self.ts = ts
        self.rates = {}
        self.max_final_rate = -np.inf

        for frac, parent in zip(self.init_fracs, self.init_parents):
            decay = DecayNuclide(parent)

            for k, v in decay(ts, decay_rate=True).items():
                v = frac * v
                try:
                    self.rates[k] += v
                except KeyError:
                    self.rates[k] = v

                self.max_final_rate = max(self.max_final_rate, v[-1])

        self.rates = {k: v for k, v in sorted(self.rates.items(), key=lambda k_v: -k_v[1][-1])}

    def get_gammas(self, frac_cut=None, ages=None):
        gammas_dict = {}
        _max = -np.inf

        if ages is None:
            indices = [-1]
        else:
            indices = np.searchsorted(self.ts, ages)

        for i in indices:
            i = min(len(self.ts) - 1, i)

            for k, v in self.rates.items():
                nuclide = Nuclide(k)

                for gline in nuclide.decay_gamma_lines:
                    rel_rate = v[i] * gline.intensity.n / len(indices)

                    try:
                        gammas_dict[gline].append(rel_rate)
                    except KeyError:
                        gammas_dict[gline] = [rel_rate]

        rel_intensities = [np.max(x) for x in gammas_dict.values()]

        gammas = list(gammas_dict.keys())

        rel_intensities /= max(rel_intensities)

        out = [(g, rel_rate) for g, rel_rate in zip(gammas, rel_intensities)]

        if frac_cut is not None:
            out = [x for x in out if x[-1] > frac_cut]

        out = list(sorted(out, key=lambda x: -x[-1]))

        return out

    def get_gamma_rates(self, min_erg=30):
        out = {}

        for k, v in self.rates.items():
            nuclide = Nuclide(k)
            tot_intensity = sum([g.intensity.n for g in nuclide.decay_gamma_lines if g.erg > min_erg])

            out[k] = v * tot_intensity

        out = {k: v for k, v in sorted(out.items(), key=lambda k_v: -k_v[1][-1])}
        return out

    def add_init_parent(self, name, rel_frac):
        self.init_parents.append(name)
        self.init_fracs.append(rel_frac)

        s = sum(self.init_fracs)
        self.init_fracs = [x/s for x in self.init_fracs]


gammas = {}
tab = TabPlot()

for src_name, parents in {'Cf252': [(0.8, 'Cf252'), (0.045, 'Cf249'), (0.08, 'Cf250'), (0.02, 'Cf251'), (0.03, 'Cf253'), (0.015, 'Cf254')],
                          'U238': [(1, 'U238')],
                          'Th232': [(1, 'Th232')],
                          'Pu239': [(1, 'Pu239')],
                          }.items():

    source = Source(src_name)

    [source.add_init_parent(n_name, frac) for frac, n_name in parents]

    gammas[src_name] = []

    max_rate = -np.inf

    rates_dict = {}

    if src_name == 'Cf252':
        ages = [5 * year, 10 * year, 30 * year]
    else:
        ages = [min(Nuclide(src_name).half_life.n, year * 5)]

    tmin = year

    ts = np.logspace(np.log10(tmin), np.log10(ages[-1]), 1000)
    source.set_rates(ts)

    print(src_name)
    for g, rel_rate in source.get_gammas(1E-3, ages=ages):
        print(f"\t{rel_rate: >6.2e} {print_gamms(src_name, rel_rate, g)}")
        if write:
            file.write(print_gamms(src_name, rel_rate, g) + "\n")

    axs = tab.new_ax(f"{src_name}", 2, 1, sharex="all")

    colors = {}
    for label, v in list(source.rates.items()):
        if v[-1]/source.max_final_rate > 1E-4:
            _handle, = axs[0].plot(ts/year, v, label=label)
            print(_handle.get_color(), max(v))
            colors[label] = _handle.get_color()

    gamma_rates = source.get_gamma_rates()
    for k, color in colors.items():
        v = gamma_rates[k]
        axs[1].plot(ts/year, v, label=k, color=color)

    axs[0].set_ylabel("Decay rates [1/s]")
    axs[1].set_ylabel("Gamma rates [1/s]")
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    axs[1].set_xlabel('Age [yr]')

    axs[0].legend()


if file is not None:
    file.close()


plt.show()