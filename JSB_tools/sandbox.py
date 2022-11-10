import matplotlib.pyplot as plt
from JSB_tools import TabPlot
from JSB_tools.nuke_data_tools.nuclide import Nuclide

n_endf = Nuclide('Pb208').get_incident_proton_daughters('tendl')['Hg200'].xs.debug_plot()
# n_endf = Nuclide('Pb208').get_incident_proton_daughters('endf')
# n_tendl = Nuclide('Pb208').get_incident_proton_daughters('tendl')
#
# tab = TabPlot()
#
# for n in n_endf.keys():
#     if n in n_tendl:
#         endf = n_endf[n]
#         tendl = n_tendl[n]
#         print(endf is tendl)
#
#         try:
#             ax = tab.new_ax(endf.xs.__fig_label__)
#         except OverflowError:
#             tab = TabPlot()
#             ax = tab.new_ax(endf.xs.__fig_label__)
#
#         endf.xs.plot(ax=ax)
#         tendl.xs.plot(ax=ax)

# n1= Nuclide("Pb203").get_incident_proton_daughters('tendl')['Bi202']
# n1= Nuclide("Pb203").get_incident_proton_daughters('tendl')['Bi204']
# n2= Nuclide("Tl205").get_incident_proton_daughters('tendl')['Pb203']
#
# n1.xs.debug_plot()
# n2.xs.debug_plot()
#
# n1.xs.plot(plot_mts=True)
# n2.xs.plot(plot_mts=True)
plt.show()