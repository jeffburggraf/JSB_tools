import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
from JSB_tools.spe_reader import SPEFile
matplotlib.use('Qt5agg')
cwd = Path(__file__).parent

p0 = Path("/Users/burgjs/PycharmProjects/Misc/GammaAngCorr/window_scan_data/1_in_Stationary_-3 to 3 micro seconds.Spe")
# p0 = Path("/Users/burgjs/PycharmProjects/Misc/GammaAngCorr/window_scan_data/1_in_Stationary_-3 to 2 micro seconds.Spe")
p1 = Path("/Users/burgjs/PycharmProjects/Misc/GammaAngCorr/window_scan_data/1_in_Stationary_-3 to 2 micro seconds coincidence.Spe")
spe0 = SPEFile(p0)
spe1 = SPEFile(p1)


ax = spe0.plot_erg_spectrum(label=p0.name)
spe1.plot_erg_spectrum(label=p1.name, ax=ax)
ax.legend()
plt.show()