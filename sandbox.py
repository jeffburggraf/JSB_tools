from abc import ABC, ABCMeta
from matplotlib import pyplot as plt
from JSB_tools.spe_reader import SPEFile
from JSB_tools import mpl_hist
from JSB_tools.list_reader import MaestroListFile
from JSB_tools.nuke_data_tools import FissionYields
from JSB_tools.MCNP_helper.outp_reader import OutP

outp = OutP()
y = FissionYields('U238', 'proton', [])