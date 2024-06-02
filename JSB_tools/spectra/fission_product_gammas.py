import pickle
import re

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


gammas = {}
for src_name, parents in {'Cf252': [(0.8, 'Cf252'), (0.1, 'Cf249'), (0.8, 'Cf251')],
                          'U238': [(1, 'U238')],
                          'Th232': [(1, 'Th232')],
                          'Pu239': [(1, 'Pu239')],
                          }.items():
    gammas[src_name] = []

    max_rate = -np.inf

    for rel_frac, parent in parents:
        n = Nuclide(parent)
        dt = min(n.half_life.n, year * 5)

        rates = DecayNuclide(parent, )(dt, decay_rate=1)
        for k, v in rates.items():
            for gline in Nuclide(k).decay_gamma_lines:
                grate = v * gline.intensity
                gline.rate = grate
                max_rate = max(max_rate, grate)
                gammas[src_name].append(gline)

    gammas[src_name] = list(sorted(gammas[src_name], key=lambda x: -x.rate))

    print(src_name)

    for g in gammas[src_name][:]:
        frac = g.rate/max_rate
        frac = frac.n

        if frac < 0.5E-2:
            break
        print("\t", frac, print_gamms(src_name, frac, g))

        if file is not None:
            file.write(print_gamms(src_name, frac, g) + "\n")

if file is not None:
    file.close()


