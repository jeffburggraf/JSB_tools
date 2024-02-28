from __future__ import annotations
from datetime import datetime
import re
from JSB_tools.nuke_data_tools import Nuclide
from typing import Union
from uncertainties import ufloat
from typing import Dict
import numpy as np
from uncertainties import ufloat


class CalSource:
    instances: Dict[str, CalSource] = {}

    @classmethod
    def get_source(cls, serial) -> CalSource:
        try:
            return cls.instances[str(serial).lower()]
        except KeyError:
            assert False, f'No source entered for serial, "{serial}"\n. Options:\n{list(CalSource.instances.keys())}'

    def __init__(self, nuclide_name, serial_num: Union[str, int], activity, ref_date: datetime, unit='uci', rel_err=None):
        unit = unit.lower().strip()

        if 'ci' in unit:
            c = 3.7E10
        elif "bq" in unit:
            c = 1.
        else:
            assert False, f'Invalid unit, {unit}'

        if re.match("^u.{2}$", unit):
            c *= 1E-6
        elif re.match("^n.{2}$", unit):
            c *= 1E-9
        elif re.match("^m.{2}$", unit):
            c *= 1E-3
        elif re.match("^k.{2}$", unit):
            c *= 1E3

        self.name = nuclide_name
        self.nuclide = Nuclide(nuclide_name)
        self.serial_num = str(serial_num).lower()
        self.ref_activity = activity * c
        self.ref_date = ref_date

        if rel_err is not None:
            self.ref_activity = ufloat(self.ref_activity, rel_err * self.ref_activity)

        CalSource.instances[self.serial_num] = self

    def get_activity(self, date=datetime.now()):
        dt = (date - self.ref_date).total_seconds()
        hl = self.nuclide.half_life
        correction = 0.5 ** (dt / hl)
        return self.ref_activity * correction

    def get_n_decays(self, duration, start_date=datetime.now()):
        """

        Args:
            duration: Acquisition is seconds
            start_date: For source activity correction.

        Returns:

        """
        dt = (start_date - self.ref_date).total_seconds()
        n_nuclides_ref = self.ref_activity / self.nuclide.decay_rate

        l = self.nuclide.decay_rate

        out = n_nuclides_ref * (np.e**(-l * dt) - np.e**(-l * (dt + duration)))
        return out
        # hl = self.nuclide.half_life
        # n_nuclides_ref = self.ref_activity / self.nuclide.decay_rate
        # n_nuclides_begin = n_nuclides_ref*0.5**(dt/hl)
        #
        # percent_decay = 1 - 0.5**(duration/hl)
        # return n_nuclides_begin*percent_decay

    def __repr__(self):
        return f'{self.name}; {self.get_activity():.3E} Bq'


ARMS_sources = {
    '1930-100-1': CalSource('Ba133', '1930-100-1', 21.11,
                            datetime(2017, 6, 15), unit='uCi', rel_err=0.032/2.3),

    '1930-100-4': CalSource('Co60', '1930-100-4', 21.04,
                            datetime(2017, 6, 15), unit='uCi', rel_err=0.032/2.3),

    '1915-86-3': CalSource('Cs137', '1915-86-3', 20.53,
                           datetime(2017, 2, 15), unit='uCi', rel_err=0.032/2.3),

    '1915-86-1': CalSource('Cs137', '1915-86-1', 20.17,
                           datetime(2017, 2, 15), unit='uCi', rel_err=0.032/2.3),

    '9027-89': CalSource('Eu152', '9027-89', 0.01025,
                         datetime(2020, 5, 1), unit='uCi', rel_err=0.032/2.3),

    '1727-63-1': CalSource('Eu152', '1727-63-1', 20.43,
                           datetime(2014, 4, 1), unit='uCi',  rel_err=0.032/2.3),
}

if __name__ == '__main__':
    n = Nuclide("Y88")
