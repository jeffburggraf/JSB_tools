import os
import sys
# from .outp_reader import OutP
from warnings import warn
from openmc.data import atomic_weight
import re
from typing import List
import numpy as np
import time
from numbers import Number

class ProgressReport:
    def __init__(self, i_final, sec_per_print=2, i_init=0):
        self.__i_final__ = i_final
        self.__i_init__ = i_init
        self.__sec_per_print__ = sec_per_print
        # self.__i_current__ = i_init
        self.__next_print_time__ = time.time() + sec_per_print
        self.__init_time__ = time.time()
        self.__rolling_average__ = []

    def __report__(self, t_now, i):
        evt_per_sec = (i-self.__i_init__)/(t_now - self.__init_time__)
        self.__rolling_average__.append(evt_per_sec)
        evt_per_sec = np.mean(self.__rolling_average__)
        if len(self.__rolling_average__) >= 5:
            self.__rolling_average__ = self.__rolling_average__[:5]
        evt_remaining = self.__i_final__ - i
        sec_remaining = evt_remaining/evt_per_sec
        sec_per_day = 60**2*24
        days = sec_remaining//sec_per_day
        hours = (sec_remaining % sec_per_day)//60**2
        minutes = (sec_remaining % 60**2)//60
        sec = (sec_remaining % 60)
        msg = " {0} seconds".format(int(sec))
        if minutes:
            msg = " {0} minutes,".format(minutes) + msg
        if hours:
            msg = " {0} hours,".format(hours) + msg
        if days:
            msg = "{0} days,".format(days) + msg
        print(msg + " remaining.", i/self.__i_final__)

    def log(self, i):
        t_now = time.time()
        if t_now > self.__next_print_time__:
            self.__report__(t_now, i)
            self.__next_print_time__ += self.__sec_per_print__


def ROOT_loop():
    try:
        import ROOT
        import time
        while True:
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
    except ModuleNotFoundError:
            warn('ROOT not installed. Cannot run ROOT_loop')


class ChemicalFormula:
    def __init__(self, formula: str):
        self.atomic_weights, self.atom_numbers = [], []
        for m in (re.finditer(r'([A-Z][a-z]*)([0-9]*)', formula)):
            print(m)
            self.atomic_weights.append(atomic_weight(m.groups()[0]))
            n_atoms = m.groups()[1]
            if n_atoms == '':
                n_atoms = 1
            else:
                n_atoms = int(n_atoms)
            self.atom_numbers.append(n_atoms)
        self.atom_numbers = np.array(self.atom_numbers)
        self.atomic_weights = np.array(self.atomic_weights)
        self.total_grams_peer_mole = np.sum(self.atom_numbers*self.atomic_weights)


class Gas:
    R = 8.3144626181

    def __init__(self, list_of_chemical_formulas: List[str]):
        list_of_chemical_formulas = [ChemicalFormula(s) for s in list_of_chemical_formulas]
        self.total_grams_per_mole_list = np.array([a.total_grams_peer_mole for a in list_of_chemical_formulas])

    @staticmethod
    def __temp_and_pressure__(temp, pressure, temp_units, press_units):
        temp_units = temp_units.lower()
        press_units = press_units.lower()
        if temp_units == 'k':
            temp = temp
        elif temp_units == 'c':
            temp = temp + 273.15
        elif temp_units == 'f':
            temp = (temp-32)*5/9+273.15
        else:
            assert False, 'Invalid units for temperature: {}'.format(temp_units)

        if press_units == 'atm':
            pressure = pressure*101325
        elif 'bar' in press_units:
            pressure = 1E5*pressure
        elif press_units == 'p':
            pressure = pressure
        else:
            assert False, 'Invalid units for pressure: {}'.format(press_units)
        return temp, pressure

    def get_density_from_mass_ratios(self, mass_ratios: List[Number], temperature=273.15, temp_units='K', pressure=1,
                                     pressure_units='bars', n_sig_digits=4):
        temperature, pressure = self.__temp_and_pressure__(temperature, pressure, temp_units, pressure_units)
        mass_ratios = np.array(mass_ratios)
        assert len(mass_ratios) == len(self.total_grams_per_mole_list)
        norm = sum(mass_ratios)
        p_over_r_t = pressure/(Gas.R*temperature)
        _x = np.sum((mass_ratios/norm)/self.total_grams_per_mole_list)
        out = 1E-6*p_over_r_t/_x
        fmt = '{' + ':.{}E'.format(n_sig_digits) + '}'
        out = float(fmt.format(out))
        return out

    def get_atom_fractions_from_mass_ratios(self, mass_ratios:List[Number]):
        mass_ratios = np.array(mass_ratios)
        assert len(mass_ratios) == len(self.total_grams_per_mole_list)
        norm = np.sum(mass_ratios/self.total_grams_per_mole_list)
        return mass_ratios/self.total_grams_per_mole_list/norm

