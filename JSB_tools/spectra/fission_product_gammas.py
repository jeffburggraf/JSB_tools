import numpy as np
from JSB_tools.nuke_data_tools.nuclide.fission_yields import FissionYields
from JSB_tools.nuke_data_tools import DecayNuclide, Nuclide
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib

year = 365 * 24 * 60**2

# =====================================
write = input("Write gamma file? (y/n)").lower() == 'y'
# ================================

if write:
    file = open('gammas', 'w')
    file.write("Source nuclide;\tDecaying Nuclide;\tErg [keV];\tRel. intensity;\tGamma Intensity [%]\n")
else:
    file = None


def print_gamms(source_nuclide_name, rel_intensity, g):
    return f"{source_nuclide_name: <8}; {g.parent_nuclide.name: <8}; {g.erg.n: <7.2f}; {rel_intensity: <3.2e}; {g.intensity.n * 100: <4.1f}"


matplotlib.use('Qt5agg')
cwd = Path(__file__).parent

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

    if frac < 1E-1:
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
                # (1, 'U238'), (1, 'Pu239'), (1, 'Th232')]:
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

        if frac < 3.5E-2:
            break
        print("\t", frac, print_gamms(src_name, frac, g))

        if file is not None:
            file.write(print_gamms(src_name, frac, g) + "\n")

if file is not None:
    file.close()
print()




