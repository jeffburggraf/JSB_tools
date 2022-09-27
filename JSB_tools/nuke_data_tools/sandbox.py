import re
from talys import ReadResult, run, talys_dir
# import matplotlib.pyplot as plt
from JSB_tools import Nuclide, TabPlot
import numpy as np
# from JSB_tools.nuke_data_tools.nudel import LevelScheme
# from openmc.data import ATOMIC_SYMBOL, Evaluation, Reaction
from pathlib import Path


braching_ratio_means = []
nuclides =[]

for p in Path("/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/data/talys").iterdir():
    # if p.name[-2:] != '-n':
    #     continue
    m = re.match(r"([A-Z][a-z]*)([0-9]+)-n", p.name)
    if not m:
        continue
    sym = m.groups()[0]
    a = int(m.groups()[1])
    r = ReadResult(f"{sym}{a}", 'n')
    try:
        xs = r.residue_production(f"{sym}{a + 1}_m0", )
    except FileNotFoundError:
        continue
    nuclides.append(f"{sym}{a}")
    braching_ratio_means.append(np.mean(xs.misc_data['branching_ratios']))

argsrt = np.argsort(braching_ratio_means)[::-1]
nuclides = np.array(nuclides)[argsrt]
braching_ratio_means = np.array(braching_ratio_means)[argsrt]

for n, b in zip(nuclides, braching_ratio_means):
    print(n, b, Nuclide.from_symbol(n).add_neutron(1).human_friendly_half_life())

#
#
# atomic_number_range = [3, 40]
#
# for z in range(*atomic_number_range):
#     if z in [90, 91, 92]:
#         break
#     s_ = ATOMIC_SYMBOL[z]
#
#     a, s = Nuclide.max_abundance_nucleus(s_)
#     for n in Nuclide.get_all_isotopes(s_):
#         # print((talys_dir / f"{n}-n"))
#
#         if Nuclide.natural_abundance(n) > 0.05 and not (talys_dir/f"{n}-n").exists():
#             # print(n)
#             run(n, "n", max_erg=35, min_erg=1, isomer=40E-12, fileresidual=True, maxlevelstar=100,
#                 maxlevelsres=100, runnum=0, parallel=True, maxN=4, maxZ=4)
#
#     if s == "H1":
#         continue
#
#     if s == 'He4':
#         continue
#
#     if s is None:
#         continue
#
#     # run(s, "n", max_erg=35, min_erg=1, isomer=40E-12, fileresidual=True, maxlevelstar=100,
#     #     maxlevelsres=100, runnum=0, parallel=True, maxN=4, maxZ=4)


