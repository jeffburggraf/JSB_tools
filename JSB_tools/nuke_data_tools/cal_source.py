from __future__ import annotations
from pendulum import datetime
import pendulum
import re
from JSB_tools.nuke_data_tools import Nuclide
from typing import Union
from typing import Dict
import numpy as np
from uncertainties import ufloat
from JSB_tools.spe_reader import SPEFile


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
            self.ref_activity = ufloat(self.ref_activity, rel_err * self.ref_activity, tag='ref activity')

        CalSource.instances[self.serial_num] = self

    def get_activity(self, date=pendulum.now()):
        dt = (date - self.ref_date).total_seconds()
        hl = self.nuclide.half_life
        correction = 0.5 ** (dt / hl)
        return self.ref_activity * correction

    def get_n_decays(self, duration, start_date=pendulum.now(), make_rate=False):
        """

        Args:
            duration: Acquisition is seconds
            start_date: For source activity correction.
            make_rate:

        Returns:

        """
        dt = (start_date - self.ref_date).total_seconds()
        n_nuclides_ref = self.ref_activity / self.nuclide.decay_rate

        l = self.nuclide.decay_rate

        out = n_nuclides_ref * (np.e**(-l * dt) - np.e**(-l * (dt + duration)))
        if not make_rate:
            return out
        else:
            return out / duration

    def __repr__(self):
        return f'{self.name} ({self.serial_num}); {self.get_activity():.3E} Bq'


class CF252Source:
    pass  # todo


std_3_pct_err = 0.03/2.57583   # converting 3% at 99% interval to sigma

ALL_SOURCES = {
    'ARMS': {
        '1930-100-1': CalSource('Ba133', '1930-100-1', 21.11,
                                datetime(2017, 6, 15), unit='uCi', rel_err=std_3_pct_err),

        '1930-100-2': CalSource('Ba133', '1930-100-2', 20.52,
                                datetime(2017, 6, 15), unit='uCi', rel_err=std_3_pct_err),

        '1930-100-4': CalSource('Co60', '1930-100-4', 21.04,
                                datetime(2017, 6, 15), unit='uCi', rel_err=std_3_pct_err),

        '1930-100-5': CalSource('Co60', '1930-100-5', 19.75,
                                datetime(2017, 6, 15), unit='uCi', rel_err=std_3_pct_err),

        '1915-86-3': CalSource('Cs137', '1915-86-3', 20.53,
                               datetime(2017, 2, 15), unit='uCi', rel_err=std_3_pct_err),

        '1930-6-3': CalSource('Cs137', '1930-6-3', 20.75,
                              datetime(2017, 3, 1), unit='uCi', rel_err=std_3_pct_err),

        '1915-86-1': CalSource('Cs137', '1915-86-1', 20.17,
                               datetime(2017, 2, 15), unit='uCi', rel_err=std_3_pct_err),

        '1727-63-3': CalSource('Cs137', '1727-63-3', 18.80,
                               datetime(2014, 4, 1), unit='uCi', rel_err=std_3_pct_err),

        '1727-63-4': CalSource('Cs137', '1727-63-4', 18.45,
                               datetime(2014, 4, 1), unit='uCi', rel_err=std_3_pct_err),

        '1727-63-2': CalSource('Cs137', '1727-63-2', 18.88,
                               datetime(2014, 4, 1), unit='uCi', rel_err=std_3_pct_err),

        '1294-91-2': CalSource('Cs137', '1294-91-2', 10.73,
                               datetime(2008, 6, 1), unit='uCi', rel_err=0.02),  # no confirmation of uncertainty of activity

        '9027-89': CalSource('Eu152', '9027-89', 0.01025,
                             datetime(2020, 5, 1), unit='uCi', rel_err=std_3_pct_err),

        '1727-63-1': CalSource('Eu152', '1727-63-1', 20.43,
                               datetime(2014, 4, 1), unit='uCi', rel_err=std_3_pct_err)
    }
}


def get_source_serial(spe: SPEFile) -> str:
    m = re.match('.*source +([a-zA-Z0-9-]+)(?:$| |,|;)', spe.description, re.IGNORECASE)
    if not m:
        raise ValueError(f"No source serial number found in SPE description:\n{spe.description}")

    return m.groups()[0]


def get_src_cal(serial, facility='ARMS') -> CalSource:
    cal = ALL_SOURCES[facility][serial]
    return cal


if __name__ == '__main__':
    n = Nuclide("Y88")
