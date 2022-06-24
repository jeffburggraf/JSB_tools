from abc import ABC, ABCMeta
import numpy as np
from matplotlib import pyplot as plt

from JSB_tools.SRIM import find_SRIM_run
from scipy.integrate import quad
from scipy.interpolate import interp1d

data = find_SRIM_run(target_atoms=["U"], fractions=[1], density=20, projectile='Xe139', gas=False)

data.plot_dedx()

c = 1.178E8


interp = interp1d(data.ergs, data.total_dedx*data.density, 'cubic')

def f(erg):
    dedx = interp(erg)
    out = -(1.0/dedx)/(np.sqrt(erg)*c)

    return out

print(quad(f, 70, data.ergs[0],  epsrel=1E-15, epsabs=1E-20))

plt.show()