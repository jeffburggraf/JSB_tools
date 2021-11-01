from abc import ABC, ABCMeta
from matplotlib import pyplot as plt
from JSB_tools.spe_reader import SPEFile
from JSB_tools import mpl_hist
from JSB_tools.list_reader import MaestroListFile

counts = []
ergs = []

with open('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/Nickel/Nickel.Spe') as f:
    lines = f.readlines()
    ch = 0
    for line in lines[lines.index('0 16383\n')+1:]:
        try:
            erg = 2.656425E-001 + ch*1.942057E-001 - ch**2*1.060710E-008
            counts.append(float(line))
            ergs.append(erg)
            ch += 1
        except ValueError:
            break


spe = SPEFile('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/Nickel/Nickel.Spe')
# spe = c()


plt.plot(ergs, counts, marker='d')
mpl_hist(spe.erg_bins, spe.counts, ax=plt)

plt.show()