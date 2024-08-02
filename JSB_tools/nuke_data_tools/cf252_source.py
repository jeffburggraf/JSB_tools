import numpy as np
from pathlib import Path
import pendulum
from JSB_tools.nuke_data_tools import Nuclide
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5agg')
cwd = Path(__file__).parent


cf252 = Nuclide("Cf252")


class Cf252:
    neutron_nu = {'Cf252': 3.7509, 'Cf250': 3.52, 'Cf254': 3.89, 'Cf246': 3.14}

    def __init__(self, tot_weight,
                 serial_number, cal_date: pendulum.DateTime,
                 init_isotopics=(("Cf252", 1),)):

        AVOGADRO = 6.02214076e23

        self.init_mass = tot_weight
        self.cal_date = cal_date

        grams_per_mole = sum(f * Nuclide(n).grams_per_mole for n, f in init_isotopics)
        n_moles = tot_weight / grams_per_mole

        self.n_init_atoms = {name: frac * n_moles * AVOGADRO for name, frac in init_isotopics}

        self.serial_number = serial_number

        self.atom_fractions = [x[1] for x in init_isotopics]
        self.isotopes = [x[0] for x in init_isotopics]

    def decay_corr(self,  date: pendulum.DateTime, nuclide_name):
        nuclide = Nuclide(nuclide_name)

        dt = (date - self.cal_date).total_seconds()
        decay_corr = 0.5 ** (dt / nuclide.half_life.n)
        return decay_corr

    def get_n_atoms(self, nuclide_name, date: pendulum.DateTime = pendulum.now()):
        if nuclide_name is None:
            out = {}

            for name, v in self.n_init_atoms.items():
                out[name] = self.decay_corr(date, name) * self.n_init_atoms[name]

            return sum(out.values())

        else:
            return self.decay_corr(date, nuclide_name) * self.n_init_atoms[nuclide_name]

    def get_fission_rate(self, nuclide_name=None, date: pendulum.DateTime = pendulum.now()):
        if nuclide_name is None:
            out = sum(self.get_fission_rate(n) for n in self.n_init_atoms.keys())
        else:
            nuclide = Nuclide(nuclide_name)
            branching_ratio = nuclide.decay_modes[('sf',)][0].branching_ratio

            out = self.get_n_atoms(nuclide_name, date) * branching_ratio * nuclide.decay_rate.n

        return out

    def neutrons_per_sec(self, nuclide_name=None, date: pendulum.DateTime = pendulum.now()):
        if nuclide_name is None:
            out = sum(self.get_fission_rate(name, date=date) * Cf252.neutron_nu[name] for name in self.n_init_atoms.keys())
        else:
            out = self.get_fission_rate(nuclide_name, date) * Cf252.neutron_nu[nuclide_name]
        return out


def get_neutron_rate(init_atoms, nuclide_name, cal_date: pendulum.DateTime, meas_date: pendulum.DateTime):
    dt = (meas_date - cal_date).total_seconds()
    nuclide = Nuclide(nuclide_name)

    decay_corr = 0.5 ** (dt/nuclide.half_life.n)

    out = init_atoms * decay_corr * nuclide.get_neutron_emission_rate()

    return out


CF252_SOURCES = {
    'M-4Cf052': Cf252(tot_weight=12.6E-6, serial_number="M-4Cf052", cal_date=pendulum.datetime(2003, 10, 28)),
    'M-4Cf053': Cf252(tot_weight=9.90E-6, serial_number="M-4Cf053", cal_date=pendulum.datetime(2004, 9, 9)),
    'M-4Cf054': Cf252(tot_weight=5.50E-6, serial_number="M-4Cf054", cal_date=pendulum.datetime(2005, 11, 8)),
    '1292-73-1': Cf252(tot_weight=0.546E-6, serial_number="SN-22", cal_date=pendulum.datetime(2003, 9, 24)),
}