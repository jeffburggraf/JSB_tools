from __future__ import annotations
import re
import warnings
from uncertainties import ufloat, UFloat
from functools import cached_property
import numpy as np
from JSB_tools.nuke_data_tools.nuclide.atomic_data import ATOMIC_NUMBER, ATOMIC_SYMBOL
import JSB_tools.nuke_data_tools.nuclide as nuclide
from JSB_tools.nuke_data_tools.nudel.core import get_active_ensdf
from JSB_tools.nuke_data_tools.nudel.core import LevelRecord as _LevelRecord
from JSB_tools.nuke_data_tools.nudel.core import GammaRecord as _GammaRecord
from JSB_tools.nuke_data_tools.nudel.core import DecayRecord as _DecayRecord
from JSB_tools.nuke_data_tools.nudel.core import Quantity as _Quantity
from JSB_tools.nuke_data_tools.nudel.core import Nuclide as _NudelNuclide
from JSB_tools.nuke_data_tools.nudel.util import Dimension
from typing import List, Dict, Tuple

ensdf = get_active_ensdf()


class GammaRecord:
    def __init__(self, level_scheme: LevelScheme, decay_record: _GammaRecord):
        self.nuclide_name = level_scheme.nuclide_name
        erg = decay_record.energy
        self.energy = ufloat(erg.cast_to_unit('keV').val, erg.cast_to_unit('keV').pm)

        orig_level = decay_record.orig_level

        if orig_level is not None:
            self.orig_level = level_scheme.levels[orig_level.level_index]
        else:
            self.orig_level = None

        dest_level = decay_record.dest_level

        if dest_level is not None:
            self.dest_level = level_scheme.levels[dest_level.level_index]
        else:
            self.dest_level = None

        self.rel_intensity = decay_record.rel_intensity.val

        self.conversion_coeff = decay_record.conversion_coeff.val
        self.gamma_prob = 1.0/(1 + self.conversion_coeff)

    @property
    def half_life(self):
        return self.orig_level.half_life

    def __repr__(self):
        return f"{self.nuclide_name} GammaDecay ({self.rel_intensity}%) ({self.energy} keV, {self.half_life} s) {self.orig_level.energy} to {self.dest_level.energy}"


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
            m = re.match(r"(.+)\([0-9]+\)(.*)", str(q))
            if m:
                out = m.groups()[0] + m.groups()[1]

        return out

    def __repr__(self):
        return f"<Level {self.level_index} of {self.nuclide_name}; {self.energy:.3g} keV; hl: {self.pretty_half_life()}>"


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

        self.gamma_decays: List[GammaRecord] = []

        for level in self.levels:
            gamma_records = [GammaRecord(self, r) for r in level.decays if isinstance(r, _GammaRecord)]
            if not len(gamma_records):
                continue

            rel_intensities = [r.rel_intensity for r in gamma_records]

            if not any([(np.isnan(x) for x in rel_intensities)]):
                tot = sum(rel_intensities)

                if tot > 0:
                    norm = 1.0/tot

                    for r in gamma_records:
                        r.rel_intensity *= norm

            self.gamma_decays.extend(gamma_records)

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
        self.nuclide = nuclide.Nuclide(nuclide_name)
        if daughter_name is None:
            br = -1
            for m in self.nuclide.decay_modes.values():
                if m[-1].branching_ratio > br:
                    daughter_name = m[-1].daughter_name
                    br = m[-1].branching_ratio
        self.levels = {l.level_index: l for l in LevelScheme(daughter_name).levels}
        self.daughter_nuclide = nuclide.Nuclide(daughter_name)

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
    n = nuclide.Nuclide('Xe140')
    n = n.decay_daughters[0]
    l = LevelScheme(n.name)
    decays = []
    for g in l.gamma_decays:
        print(g)
