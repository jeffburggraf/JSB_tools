import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange
from analysis import Shot
import numba as nb
from numba.typed import List
from JSB_tools import Nuclide, TabPlot
from JSB_tools.spectra import EfficiencyCalMixin
# /opt/anaconda3/envs/Test2/bin/python /Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/sandbox.py
# F=3.3e+19
# /Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/sandbox.py:36: MatplotlibDeprecationWarning: The resize_event function was deprecated in Matplotlib 3.6 and will be removed two minor releases later. Use callbacks.process('resize_event', ResizeEvent(...)) instead.
#   plt.plot(ergs, total_particles*(mean_erg)**-1 * np.e ** (min_erg/mean_erg) * np.e ** (-ergs/mean_erg))
# ~2.46e+00 detected; Po206 (8.8 d) (2.95e+01 nuclei produced); xs1*xs2 = 7.82e-03; (endf, tendl) Pb208 -> Bi207 (32.0 a) -> Po206 (8.8 d); Detections per decay: 0.08342319457965974
eff = EfficiencyCalMixin()
eff.unpickle_efficiency('/Users/burggraf1/PycharmProjects/IACExperiment/efficiencies/eff_main.eff')
atom_densities = {'Pb': 3.3E22}
# ========================================
mean_erg = 20  # 10 for PROBIES
min_erg = 5   # 4 for PROBIES
target = Nuclide.from_symbol("Pb208")
total_particles = 0.7E13  # total # of particles with energy > min_erg. 1E11 for PROBIES
dist = 10E-4  # distance from TNSA source in cm. Add  1E-4 for um
opening_angle = 5 * np.pi/180
max_plots = 10
# ========================================


print(f"F={total_particles/(2 * np.pi * dist ** 2 * (1 - np.cos(opening_angle))):.1e}")

atom_density = atom_densities[target.atomic_symbol]

F2_integral = atom_density * total_particles**2/(2*np.pi * dist * (1 - np.cos(opening_angle)))


ergs = np.linspace(min_erg, 100, 200)

probs = np.e ** (-ergs/mean_erg)

plt.plot(ergs, total_particles*(mean_erg)**-1 * np.e ** (min_erg/mean_erg) * np.e ** (-ergs/mean_erg))
plt.ylabel("Protons/MeV")


probs /= sum(probs)

reactions = {}


for k1, v1 in target.get_incident_proton_daughters(data_source='all').items():
    data1 = v1.xs.data_source

    if v1.Z == target.Z + 1:
        xs1 = sum(probs * v1.xs.interp(ergs))

        for k2, v2 in v1.get_incident_proton_daughters(data_source='all').items():
            data2 = v2.xs.data_source

            if v2.Z == v1.Z + 1:
                xs2 = sum(probs * v2.xs.interp(ergs))

                detection_factor = sum([eff.eval_efficiency(g.erg.n) * g.intensity.n for g in v2.decay_gamma_lines
                                        if g.intensity > 0.1])

                reactions[(k1, k2)] = {'xs1': xs1, 'xs2': xs2, 'data1': data1, 'data2': data2, 'n1': v1, 'n2': v2,
                                       'detection_factor': detection_factor}


reactions = {k: v for k, v in sorted(reactions.items(),
                                     key=lambda x: -x[1]['xs1'] * x[1]['xs2'] * x[1]['detection_factor'])}


t = TabPlot()
n_plots = 0
for k, v in reactions.items():
    n1:Nuclide = v['n1']
    n2:Nuclide = v['n2']

    preint_n1 = f"{n1.name} ({n1.pretty_half_life(include_errors=False)})"
    preint_n2 = f"{n2.name} ({n2.pretty_half_life(include_errors=False)})"

    xs1xs2 = v['xs1'] * v['xs2'] * v['detection_factor']
    # total_nuclei_produced =
    reaction_rate = 1/2 * xs1xs2 * 1E-24**2 * F2_integral
    det_factor = reaction_rate * v['detection_factor']
    # if reaction_rate < 0.01 or :
    #     continue
    if n_plots < max_plots:
        ax = t.new_ax(f"{n1.name}->{n2.name}")
        n1.xs.plot(ax=ax)
        n2.xs.plot(ax=ax)
        ax.plot(ergs, ax.get_ylim()[-1]*probs/max(probs), label='energy dist', ls='--', c='black')
        ax.legend()

        n_plots += 1

        print(f"~{det_factor:.2e} detected; {preint_n2} ({reaction_rate:.2e} nuclei produced); xs1*xs2 = {xs1xs2:.2e}; ({v['data1']}, {v['data2']}) "
              f"{target.name} -> {preint_n1} -> {preint_n2}; Detections per decay: {v['detection_factor']}")

plt.show()