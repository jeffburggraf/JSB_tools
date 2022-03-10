from abc import ABC, ABCMeta

import numpy as np
from matplotlib import pyplot as plt
from JSB_tools.spe_reader import SPEFile
from JSB_tools import mpl_hist
from JSB_tools.list_reader import MaestroListFile
from JSB_tools.nuke_data_tools import FissionYields
from JSB_tools.MCNP_helper.outp_reader import OutP
from JSB_tools.SRIM import find_SRIM_run
# outp = OutP()
# y = FissionYields('U238', 'proton', [])

s = find_SRIM_run(['U'], [1], 19, 'Xe139')

ax = s.plot_dedx()
ax.plot(s.ergs, s.nuclear)
ax.plot(s.ergs, s.nuclear)

emax = 70

ergs = np.array(s.ergs)
dedx = s.total_dedx

# plt.plot(ergs, 1.0/dedx)

# s.plot_dedx()
plt.show()