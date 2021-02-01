
from JSB_tools.nuke_data_tools import Nuclide

n = Nuclide.from_symbol('U238')
print(n.independent_sf_fission_yield())

#
