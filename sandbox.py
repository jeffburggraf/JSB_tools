import numpy as np
from pathlib import Path
from openmc.data import FissionProductYields


y = FissionProductYields('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/GEFY81_n/GEFY_86_217_n.dat')

a = [1,2,3,4,5,6,8,9,]
b = [-2, -1, 0 ,1 ]
c = [3,4,5,6,7]

print(np.argmax([0,0,0,1,1,1,1,1,1,1,1,1,1]))
