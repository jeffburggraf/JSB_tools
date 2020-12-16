from JSB_tools.nuke_data_tools import Nuclide
from matplotlib import pyplot as plt
s = Nuclide.from_symbol('U238')
from uncertainties import unumpy as unp

ergs_n, yield_n = s.independent_neutron_fission_yield('Xe139')
ergs_p, yield_p = s.cumulative_neutron_fission_yield('Xe139')
plt.errorbar(ergs_n, unp.nominal_values(yield_n), unp.std_devs(yield_n),  label='independent yield')
plt.errorbar(ergs_p, unp.nominal_values(yield_p), unp.std_devs(yield_p), label='cumulative yield')
plt.title('X-139 yield from U-238(p,F)')
plt.xlabel('Incident proton energy [MeV]')
plt.ylabel('yield')
plt.legend()
plt.show()