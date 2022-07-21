import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from openmc.data import Evaluation, Reaction


def get_xs(eval_, product=None, mt=5, x=100):
    r = Reaction.from_endf(eval_, mt)
    xs = r.xs['0K']
    if x is None:
        x = xs.x
    elif isinstance(x, int):
        x = np.linspace(xs.x[0], xs.x[-1], x)
    else:
        pass

    xs_tot = xs(x)

    if product is not None:
        for p in r.products:
            if p.particle == product:
                break
        else:
            assert False, f"Product, '{product}', not found. Options are {[p.particle for p in r.products]}"

        out = xs_tot * p.yield_(x)
    else:
        out = xs_tot

    return x*1E-6, out



p = Evaluation("/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/ENDF-B-VIII.0_protons/p-006_C_012.endf")

plt.plot(*get_xs(p,'photon', 5))
plt.show()
print()