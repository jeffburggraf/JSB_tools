from talys import ReadResult
import matplotlib.pyplot as plt
from JSB_tools import Nuclide, TabPlot
import numpy as np
from JSB_tools.nuke_data_tools.nudel import LevelScheme

level = LevelScheme('U235')

for i in [0, 1]:
    r = ReadResult('U235', 'n', i)
    all_levels = r.all_inelastic2levels()

    alphas = np.linspace(0.1, 1, 3)
    colors = ['blue', 'orange', 'green', 'red', 'black']

    if i == 0:
        fig, axs = plt.subplots(2, 3)

        axs = axs.flatten()
        for i, (l, nudel_l) in enumerate(zip(all_levels, level.levels[1:])):
            ax = axs[i//len(colors)]
            c = colors[i%len(colors)]
            xs = r.inelastic2level(l)
            print(i, f"{xs.misc_data['q_value']*1E3: .2f}", nudel_l)

            xs.plot(ax=ax, c=c)

    _, ax = plt.subplots()
    ax.set_title(r.input_kwargs['isomer'])

    for res in r.all_residues('z==92 and a == 235'):
        r.residue_production(res, True).plot(ax=ax)


plt.show()


