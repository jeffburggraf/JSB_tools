import openmc
from openmc.data import Evaluation, Reaction, IncidentNeutron, IncidentPhoton, ResonancesWithBackground, Tabulated1D, Polynomial
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib
matplotlib.use('Qt5Agg')
from JSB_tools.tab_plot import TabPlot
from JSB_tools.nuke_data_tools import Nuclide

n = 'Co59'
d = {}
for k, v in Nuclide(n).get_incident_neutron_daughters(data_source='endf').items():
    d[k] = {'endf': v.xs}

for k, v in Nuclide(n).get_incident_neutron_daughters(data_source='tendl').items():
    # if k in d
    try:
        d[k]['tendl'] = v.xs
    except KeyError:
        d[k] = {'tendl': v.xs}


tab = TabPlot()

x = np.linspace(0.01, 10, 1000)
tab.fig.suptitle(f"Neutrons on {n}")

for k, d_ in d.items():
    if len(d_) < 2:
        continue

    try:
        ax = tab.new_ax(k)
    except OverflowError:
        tab = TabPlot()
        ax = tab.new_ax(k)

    for data, xs in d_.items():
        y = xs(x)
        if all(y == 0):
            tab.remove_last_button()
            break
        ax.plot(x, xs(x), label=data)
    else:
        ax.legend()
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('xs [b]')
plt.show()

