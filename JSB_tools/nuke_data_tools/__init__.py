import matplotlib.pyplot as plt

from JSB_tools.nuke_data_tools.nuclide import Nuclide, DecayNuclide
import warnings

if __name__ == "__main__":
    n = Nuclide('Pt193')
    n.total_xs('neutron').plot()
    # for n, xs in n.get_incident_gamma_daughters().items():
    #     print(xs)

    plt.show()

