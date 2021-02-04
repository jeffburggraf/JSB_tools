
from openmc.data import FissionProductYields, Evaluation, Reaction
from  openmc.data.ace import get_libraries_from_xsdata
from openmc.data.endf import get_evaluations
from pathlib import Path
from JSB_tools.nuke_data_tools import Nuclide
"""as
 /Users/jeffreyburggraf/Downloads/TENDL2017data/tal2017-g/gxs-162/Ag102g.asc 
 /Users/jeffreyburggraf/Downloads/TENDL2017data/tal2017-g/gxs-162/Ag102mg.asc 
"""

from matplotlib import pyplot as plt
f1 = '/Users/jeffreyburggraf/Downloads/TENDL2019-PROTONS/N/N014/lib/endf/p-N014.tendl'
f2 = '/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/jendl-pd2016.1/N014.dat'
print(get_evaluations(f1))
print(get_evaluations(f2))
e = Evaluation(f2)
# e = Evaluation('/Users/jeffreyburggraf/Downloads/proton_file/N/N014/lib/endf/p-N014.tendl')

Nuclide.from_symbol('N14').get_incident_proton_daughters()['C10'].xs.plot()

print('target: ', e.target)

for mf, mt, _, _ in e.reaction_list:
    print('MF: {}, MT: {}'.format(mf, mt))
r = Reaction.from_endf(e, 5)

print(e.reaction_list)
print(r.products)
plt.figure()
for p in r.products:
    print(p)
    print(p.emission_mode)
    print(p.particle)
    if (p.particle) == 'U238':
        x = p.yield_.x*1E-6
        y = p.yield_.y*1000
        plt.plot(x, y)
        plt.ylabel('mb')
plt.show()
# for _, mt, _, _ in e.reaction_list:
#     r = Reaction.from_endf(e, mt)
#
#     print('TENDLE ACE', mt, r.products)
#
#
# e = Evaluation('/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/ENDF-B-VIII.0_gammas/g-092_U_238.endf')
# for _, mt, _, _ in e.reaction_list:
#     r = Reaction.from_endf(e, mt)
#     print(mt, r.products)
#
