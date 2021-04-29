import numpy as np
from pathlib import Path
from openmc.data import FissionProductYields


y = FissionProductYields('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/GEFY81_n/GEFY_86_217_n.dat')

out = {'ergs':y.energies, 'yields': {}}

for yield_dict in getattr(y, 'cumulative'):
    print('sfgsgg',yield_dict)
    for nuclide_name, yield_ in yield_dict.items():
        print(nuclide_name)
        try:
            entry = out['yields'][nuclide_name]
        except KeyError:
            out['yields'][nuclide_name] = [[], []]
            entry = out['yields'][nuclide_name]
        # print(out)
        # assert False
        entry[0].append(yield_.n)
        entry[1].append(yield_.std_dev)
# print(out)
# print(out['yields'])
print(out['yields']['Cr60'])