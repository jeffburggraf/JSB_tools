from JSB_tools import Nuclide

n = Nuclide.from_symbol('Xe139')
print(1/n.decay_rate)
