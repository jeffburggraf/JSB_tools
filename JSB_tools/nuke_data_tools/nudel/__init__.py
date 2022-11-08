#  SPDX-License-Identifier: GPL-3.0+
#
# Copyright © 2019 O. Papst.
#
# This file is part of nudel.
#
# nudel is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# nudel is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with nudel.  If not, see <http://www.gnu.org/licenses/>.

"""Python interface for ENSDF nuclear data"""
import re
import warnings
from uncertainties import ufloat, UFloat
from functools import cached_property
import numpy as np

from openmc.data import ATOMIC_NUMBER, ATOMIC_SYMBOL

import JSB_tools.nuke_data_tools.nuclide as nuclide
from JSB_tools.nuke_data_tools.nudel.core import get_active_ensdf
# from JSB_tools.nuke_data_tools.nudel.core import (get_active_ensdf, LevelRecord, GammaRecord, DecayRecord, Quantity)
from JSB_tools.nuke_data_tools.nudel.core import LevelRecord as _LevelRecord
from JSB_tools.nuke_data_tools.nudel.core import GammaRecord as _GammaRecord
from JSB_tools.nuke_data_tools.nudel.core import DecayRecord as _DecayRecord
from JSB_tools.nuke_data_tools.nudel.core import Quantity as _Quantity
from JSB_tools.nuke_data_tools.nudel.core import Nuclide as _NudelNuclide
# from .core import (get_active_ensdf, LevelRecord, GammaRecord, DecayRecord)
# from .core import Nuclide as NudelNuclide
from JSB_tools.nuke_data_tools.nudel.util import Dimension
from typing import List, Dict, Tuple

ensdf = get_active_ensdf()


class Level:
    def __init__(self, nuclide_name, _n_level: _LevelRecord):
        self._n_level: _LevelRecord = _n_level

        self.nuclide_name = nuclide_name
        self.decays: List[_DecayRecord] = _n_level.decays
        self.energy: UFloat = ufloat(_n_level.energy.cast_to_unit('keV').val, _n_level.energy.cast_to_unit('keV').pm)

        self.level_index = _n_level.level_index

        if not np.isnan(_n_level.half_life.val):
            if re.match("[a-zA-Z]?eV", _n_level.half_life.unit.symb):
                erg_w = ufloat(_n_level.half_life.cast_to_unit('eV').val,
                               _n_level.half_life.cast_to_unit('eV').pm)
                hbar = 6.582119E-16  # hbar in eV s
                decay_Rate = 2*erg_w/hbar

                half_life = np.log(2)/decay_Rate
                q = _Quantity(f"{half_life*1E15:.1uS} FS")
                q.approximate = True

                self._n_level.half_life = q

        self.decay_modes = {k: v.val for k, v in _n_level.decay_ratio.items()}

    @cached_property
    def half_life(self) -> UFloat:
        if np.isnan(self._n_level.half_life.val):
            return ufloat(0, 0)
        nominal = self._n_level.half_life.cast_to_unit('s').val
        err = self._n_level.half_life.cast_to_unit('s').pm
        if np.isnan(err):
            err = 0
        return ufloat(nominal, err)

    def pretty_half_life(self, unit=None, errors=True, number_format='.2f'):
        if self._n_level.half_life.unit is None:
            return str(None)

        if unit is None:
            unit = self._n_level.half_life.unit.symb

        q = self._n_level.half_life.cast_to_unit(unit)
        out = str(q)

        if not errors:
            m = re.match("(.+)\([0-9]+\)(.*)", str(q))
            if m:
                out = m.groups()[0] + m.groups()[1]

        return out

    def __repr__(self):
        return f"<Level: {self.nuclide_name}, {self.energy} keV, hl: {self.pretty_half_life()}>"


class LevelScheme:
    """
    A class for accessing nuclear level information such as half-lives, energies, decay probabilities, etc.

    """
    @classmethod
    def from_a_z(cls, a, z):
        return cls(f"{ATOMIC_SYMBOL[z]}{a}")

    def __init__(self, nuclide_name):
        if not (m := nuclide.Nuclide.NUCLIDE_NAME_MATCH.match(nuclide_name)):
            raise ValueError(f"Invalide nuclide_name, {nuclide_name}. Correct examples include 'U238', 'H2'")

        a = int(m.group('A'))
        z = ATOMIC_NUMBER[m.group('s')]
        self.nuclide_name = nuclide_name
        self.nudel_nuclide = _NudelNuclide(a, z)

        self.levels: List[Level] = [Level(nuclide_name, l) for l in self.nudel_nuclide.adopted_levels.levels]

    def find_level(self, erg) -> Tuple[int, Level]:
        assert isinstance(erg, (float, int)), type(erg)

        ergs = np.array([l.energy for l in self.levels])

        i = int(np.argmin(np.abs(ergs - erg)))
        closest_erg = ergs[i]
        if not np.isclose(erg, closest_erg, rtol=1E-1):
            warnings.warn(f"Closest level energy from <{self.nuclide_name}>.find_level({erg}) "
                          f"returned a level with energy { closest_erg}. "
                          f"Level energies in the ENDSF are: {[e.energy for e in self.levels]}")

        return i, self.levels[i]


class Coincidence:
    def __init__(self, nuclide_name, daughter_name=None):
        self.nuclide = nuclide.Nuclide.from_symbol(nuclide_name)
        if daughter_name is None:
            br = -1
            for m in self.nuclide.decay_modes.values():
                if m[-1].branching_ratio > br:
                    daughter_name = m[-1].daughter_name
                    br = m[-1].branching_ratio
        self.levels = {l.level_index: l for l in LevelScheme(daughter_name).levels}
        self.daughter_nuclide = nuclide.Nuclide.from_symbol(daughter_name)

        glines_ergs = [g.erg.n for g in self.nuclide.decay_gamma_lines]

        transitions = {}  # transisions[3][1] is transitions from 3 to 1
        for level in self.levels.values():
            entry = {}
            tot = sum([r.rel_intensity.val for r in level.decays])
            if tot == 0:
                continue
            for r in level.decays:
                if np.isnan(r.rel_intensity.val):
                    continue
                gline = self.nuclide.decay_gamma_lines[np.argmin([abs(r.energy.val - e) for e in glines_ergs])]
                intensity = gline.intensity*(1 + (0 if np.isnan(r.conversion_coeff.val) else r.conversion_coeff.val))
                if not np.isclose(gline.erg.n, r.energy.val):
                    intensity = 0

                entry[r.dest_level.level_index] = {'conditional_prob': r.rel_intensity.val/tot,
                                                   'tot_intensity': intensity}
            if len(entry):
                transitions[level.level_index] = entry

        feeding_probs = np.zeros(len(self.levels))

        for i in range(len(feeding_probs))[::-1]:




            print()



if __name__ == '__main__':

    Coincidence("Ni57")