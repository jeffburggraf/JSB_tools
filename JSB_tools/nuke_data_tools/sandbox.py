import openmc
from openmc.data import Evaluation, Reaction, IncidentNeutron, IncidentPhoton, ResonancesWithBackground, Tabulated1D, Polynomial
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib
matplotlib.use('Qt5Agg')
from JSB_tools.tab_plot import TabPlot
from JSB_tools.nuke_data_tools import Nuclide

n = 'U231'
d = {}
for k, v in Nuclide(n).get_incident_neutron_daughters(data_source='endf').items():
    d[k] = {'endf': v.xs}

for k, v in Nuclide(n).get_incident_neutron_daughters(data_source='tendl').items():
    # if k in d
    try:
        d[k]['tendl'] = v.xs
    except KeyError:
        d[k] = {'tendl': v.xs}

# d = {k: v for k, v in d.items() if len(v) == 2}

tab = TabPlot()
# tab.fig.set_title()
x = np.linspace(0.01, 10, 1000)
tab.fig.suptitle(f"Neutrons on {n}")

for k, d_ in d.items():
    if len(d_) <2:
        continue

    try:
        ax = tab.new_ax(k)
    except OverflowError:
        tab = TabPlot()
        ax = tab.new_ax(k)

    for data, xs in d_.items():
        y = xs(x)
        if all(y == 0):
            tab.remove_last_button()
            break
        ax.plot(x, xs(x), label=data)
    else:
        ax.legend()
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('xs [b]')
plt.show()
print()
    # xs = v.xs()
# p = '/Users/burgjs/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/nuclide/endf_files/ENDF-B-VIII.0_neutrons/n-092_U_238.endf'
# ev = Evaluation(p)
# mts = [x[1] for x in ev.reaction_list if x[0] == 3]
#
# for mt in mts:
#     r = Reaction.from_endf(ev, mt)
#     for prod in r.products:
#         print()
#     print()

# NUCLIDE_NAME_MATCH = re.compile(
#         "^(?P<s>[A-z]{1,3})(?P<A>[0-9]{1,3})(?:_?(?P<m_e>[me])(?P<iso>[0-9]+))?$")
#
# ev = Evaluation(p)
#
#
# class XSR:
#     def __init__(self, xss, yields, mts):
#         self.xss = xss
#         self.yields = yields
#         self.mts = set(mts)
#
#     @classmethod
#     def from_data(cls, data, MT_xs, product=None, mts=None):
#         """
#
#         Args:
#             data: {product1: {'mts': [...], 'yields': [...]}, ...}
#             MT_xs:  {mt: openmc.CRossSection1D}
#             product:
#             mts:
#
#         Returns:
#
#         """
#         if mts is not None:
#             if not isinstance(mts, list):
#                 mts = [mts]
#
#         xss = []
#         yields = []
#         mts = set()
#
#         for prod, d in data.items():
#             _yields = []
#             _xss = []
#             _mts = []
#             if product is not None:
#                 if prod != product:
#                     continue
#             xs = MT_xs[mt]
#
#             for mt, y_ in zip(d['mts'], d['yields']):
#                 if mts is not None:
#                     if mt not in mts:
#                         continue
#                 _xss.append(xs)
#                 _yields.append(y_)
#                 _mts.append(mt)
#
#             if len(_yields) == 0:
#                 _xss.append(xs)
#                 _yields.append(lambda x: 1)
#                 _mts.append(mt)
#
#     def __call__(self, ergs):
#         ergs = ergs * 1E6
#         out = np.zeros_like(ergs) if hasattr(ergs, '__iter__') else 0
#
#         for xs, ys, mt in zip(self.yields, self.xss, self.mts):
#             out += xs(ergs) * ys(ergs)
#             print(type(xs), type(ys), mt)
#
#         return out
#
#
# if projectile == 'neutron':
#     inn = IncidentNeutron.from_endf(ev)
#     reactions = inn.reactions
# elif projectile == 'gamma':
#     inn = IncidentPhoton.from_endf(ev)
#     reactions = inn.reactions
# else:
#     reactions = {mt: Reaction.from_endf(ev, mt) for _, mt, _, _ in ev.reaction_list}
#
#
# MT_XS_dict = {}
#
# product_yields_dict = {}  # {product1: {'mts': [...], 'yields': []}, ...}
#
# for mt, reaction in reactions.items():
#     try:
#         xs = reaction.xs['0K']
#     except KeyError:
#         continue
#
#     MT_XS_dict[mt] = xs
#
#     for prod in reaction.products:
#         yield_ = prod._yield
#         product_keys = [prod.particle]
#         m = NUCLIDE_NAME_MATCH.match(prod.particle)
#         if m:
#             iso = m['iso']
#             if iso is not None:
#                 product_keys.append(m['s'] + m['A'])
#
#         for k in product_keys:
#             try:
#                 d = product_yields_dict[k]
#             except KeyError:
#                 d = product_yields_dict[k] = {'mts': [], 'yields': []}
#
#             d['yields'].append(yield_)
#             d['mts'].append(mt)
#
# for p, d in product_yields_dict.items():
#     print(p)
#     print('\t' , d['mts'])
#     print('\t' , d['yields'])
#
#
# # ergs = np.logspace(-4, 2, 10000)
#
#
# # tab = TabPlot()
#
#
# # for name, xs in product_xss.items():
# #     print(name, xs.mts)
# #     try:
# #         ax = tab.new_ax(f'{name}')
# #         ax.text(0.1, 0.9, f'MTs: {xs.mts}', transform=ax.transAxes)
# #     except OverflowError:
# #         tab = TabPlot()
# #
# #     ax.plot(ergs, xs(ergs))
# #
# # plt.show()
