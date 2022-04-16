from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
from openmc.data import Evaluation, Reaction, Tabulated1D, Product, IncidentNeutron, Decay
from pathlib import Path
from uncertainties import UFloat
from uncertainties import unumpy as unp
import re


neutron_path = '/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/ENDF-B-VIII.0_neutrons/n-092_U_238.endf'
proton_path = '/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/ENDF-B-VIII.0_protons/p-082_Pb_208.endf'
gamma_path = '/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/ENDF-B-VIII.0_gammas/g-018_Ar_040.endf'
proton_path_t = '/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/TENDL2019-PROTONS/Ag/Ag096m/lib/endf/p-Ag096m.tendl'


def mt_func_decorator(f):
    # MTGetter.defined_funcs.append(f.__name__)

    def out(*args, **kwargs):
        self = args[0]
        args = args[1:]
        plot = kwargs.get('plot', False)
        temp = kwargs.get('temp', '0K')
        # todo: What about products.yield?
        r = f(self, *args, **kwargs)
        xs = r.xs[temp]
        x, y, = xs.x*1E-6, xs.y

        if plot:
            yerr = np.zeros_like(y)
            y_plot = y
            if isinstance(plot, plt.Axes):
                ax = [plot]
            else:
                fig, ax = plt.subplots()
                fig.suptitle(f"{self.projectile} from {self.endf_path.name}")

            if len(y):
                if isinstance(y[0], UFloat):
                    yerr = unp.std_devs(y)
                    y_plot = unp.nominal_values(y)

            ax.errorbar(x, y_plot, yerr, label=f.__name__)
            ax.set_ylabel("xs [b]")
            ax.set_xlabel("Energy [MeV]")

            ax.legend()

        return x, y
    return out


class MTGetter:
    pickle_attribs = ['non_elastic', 'elastic', 'total']

    class MissingMTError(ValueError):
        pass

    def __init__(self, path, nuclide_name, projectile, library_name):
        """

        Args:
            path: Path to ENDF text file.
            projectile: Incident projectile name. e.g. 'proton', or 'gamma'. Is used to save and locate pickle files.
        """
        self.endf_path = Path(path)
        self.eval = Evaluation(path)
        self.projectile = projectile
        self.library_name = library_name
        self.nuclide_name = nuclide_name

        self._mts = set([x[1] for x in self.eval.reaction_list])

    def _get_reaction(self, mt):
        if mt not in self._mts:
            raise MTGetter.MissingMTError(f"MT {mt} not found in {self.endf_path}.\nAvailible MTs:\n"
                                          + "\n".join(map(str, self._mts)))

        return Reaction.from_endf(self.eval, mt)

    @mt_func_decorator
    def get_mt(self, mt, plot=False, temp='0K'):
        return self._get_reaction(mt)

    @mt_func_decorator
    def non_elastic(self, plot=False, temp='0K'):
        if self.projectile == 'neutron':
            mt = 4
        else:
            mt = 3

        return self._get_reaction(mt)

    @mt_func_decorator
    def neutron_production(self, plot=False, temp='0K'):
        mt = 201
        return self._get_reaction(mt)

    @mt_func_decorator
    def total(self, plot=False, temp='0K'):
        mt = 1
        return self._get_reaction(mt)

    @mt_func_decorator
    def elastic(self, plot=False, temp='0K'):
        mt = 2
        return self._get_reaction(mt)

    def pickle(self):
        pickle_dir = Path(__file__).parent/'data'/f'incident_{self.projectile}'/self.library_name/'misc_xss'

        if not pickle_dir.exists():
            pickle_dir.mkdir()

        data = {}

        for name in MTGetter.pickle_attribs:
            try:
                x, y = getattr(self, name)
            except MTGetter.MissingMTError:
                continue

            data[name] = x, y

        with open(pickle_dir/self.projectile, 'wb') as f:
            pickle.dump(data, f)





# eg = EvalMT(gamma_path, 'gamma')
# en = EvalMT(neutron_path, 'neutron')
# ep = EvalMT(proton_path, 'proton')
proton_tallys = MTGetter(proton_path_t, 'proton', 'tendl')


proton_tallys.pickle()
# proton_tallys.non_elastic(plot=True)
# proton_tallys.elastic(plot=True)

plt.show()


