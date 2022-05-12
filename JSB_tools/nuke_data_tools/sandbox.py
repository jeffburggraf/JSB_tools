from talys import ReadResult, run, talys_dir
import matplotlib.pyplot as plt
from JSB_tools import Nuclide, TabPlot
import numpy as np
from JSB_tools.nuke_data_tools.nudel import LevelScheme
from openmc.data import ATOMIC_SYMBOL, Evaluation, Reaction

print()
i = 0


atomic_number_range = [3, 40]

for z in range(*atomic_number_range):
    if z in [90, 91, 92]:
        break
    s_ = ATOMIC_SYMBOL[z]

    a, s = Nuclide.max_abundance_nucleus(s_)
    for n in Nuclide.get_all_isotopes(s_):
        # print((talys_dir / f"{n}-n"))

        if Nuclide.natural_abundance(n) > 0.05 and not (talys_dir/f"{n}-n").exists():
            # print(n)
            run(n, "n", max_erg=35, min_erg=1, isomer=40E-12, fileresidual=True, maxlevelstar=100,
                maxlevelsres=100, runnum=0, parallel=True, maxN=4, maxZ=4)

    if s == "H1":
        continue

    if s == 'He4':
        continue

    if s is None:
        continue

    # run(s, "n", max_erg=35, min_erg=1, isomer=40E-12, fileresidual=True, maxlevelstar=100,
    #     maxlevelsres=100, runnum=0, parallel=True, maxN=4, maxZ=4)


