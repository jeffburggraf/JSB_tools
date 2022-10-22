"""
Helper functions for creating PHITS sources, among other things.
"""
import numpy as np
from typing import List
from JSB_tools.nuke_data_tools import Nuclide
from typing import Union, Sized
from dataclasses import dataclass
from abc import abstractmethod
from numbers import Number


class Distribution:
    @abstractmethod
    def __get_kwargs__(self):
        pass


@dataclass
class CylindricalSource(Distribution):
    """page 77 in PHITS manual"""

    radius: float
    x0: float = 0
    y0: float = 0
    z0: float = 0
    z1: Union[float, None] = None
    dz: Union[float, None] = None
    dir: Union[str, float] = 'all'
    axis_vector = (0, 0, 1)  # todo
    assert axis_vector == (0, 0, 1)

    def __get_kwargs__(self):
        if self.dz is not None:
            assert self.z1 is None, "cannot specify dz and z1. Use one or the other"
            _z1 = self.z0 + self.dz
        else:
            assert self.dz is None and self.z1 is not None, "cannot specify dz and z1. Must use one or the other"
            _z1 = self.z1
            # self.z1 = self.z0 + self.dz
        kwargs = {'s-type': 1, "x0": self.x0, "y0": self.y0, "z0": self.z0, "z1": _z1,
                  "r0": self.radius, "dir": self.dir}

        return kwargs


@dataclass
class PointSource(Distribution):
    x: float = 0
    y: float = 0
    z: float = 0
    dirx: float = 0
    diry: float = 0
    dirz: float = 0

    def __get_kwargs__(self):
        def get_phi(x, y):
            if x == 0:
                if y>0:
                    out = 90
                elif y<0:
                    out = -90
                else:
                    return 'none'
            else:
                out = np.arctan(y/x) - np.sign(x*y)*(x < 0)*np.pi
                out = 180/np.pi*out
            return out
        isotropic = self.dirx == self.diry == self.dirz == 0
        kwargs = {'s-type': 1, "x0": self.x, "y0": self.y, "z0": self.z, "z1": self.z,
                  "r0": 0}
        if not isotropic:
            phi = get_phi(self.dirx, self.diry)
            dir_ = self.dirz/np.linalg.norm((self.dirx, self.diry, self.dirz))
            kwargs["dir"] = dir_
            if isinstance(phi, Number):
                kwargs['phi'] = phi

        return kwargs


def __get_kwargs__(self):
    if self.dz is not None:
        assert self.z1 is None, "cannot specify dz and z1. Use one or the other"
        self.z1 = self.z0 + self.dz
    kwargs = {'s-type': 1, "x0": self.x0, "y0": self.y0, "z0": self.z0, "z1": self.z1,
              "r0": self.radius, "dir": self.dir}

    return kwargs


@dataclass
class GaussianSource(Distribution):
    """page 79 in PHITS manual"""
    proj: Union[Nuclide, str]
    erg_dist: Union[float, Distribution]
    x0: float = 0
    y0: float = 0
    sigma_xy: float = 0  # sigma in the xy plane (radial)
    z0: float = 0
    z1: Union[float, None] = None
    phi: float = 0
    dir: Union[str, float] = 1  # direction cosine relative to the Z axis

    def __get_kwargs__(self):
        if isinstance(self.proj, Nuclide):
            self.proj = f'{self.proj.A}{self.proj.atomic_symbol}'
        if self.z1 is None:
            self.z1 = self.z0
        kwargs = {"proj": self.proj, "s-type": 13, "x0": self.x0, 'y0': self.y0, "r1": 2.355*self.sigma_xy,
                  "z0": self.z0, "z1": self.z1,
                  "phi": self.phi, 'dir': self.dir}

        if isinstance(self.erg_dist, Distribution):
            assert hasattr(self.erg_dist, "__is_energy_dist__")
            assert self.erg_dist.__is_energy_dist__
            kwargs.update(self.erg_dist.__get_kwargs__())
        else:
            kwargs['e0'] = self.erg_dist
        return kwargs

    def source_card(self):
        return "\n".join([f"\t{k} = {v}" for k,v in self.__get_kwargs__().items()])


@dataclass
class GaussianEnergyDistribution(Distribution):
    mean: float
    std: float = None
    min_erg: float = None
    max_erg: float = None
    fwhm: float = None
    __is_energy_dist__ = True

    def __truediv__(self, other):
        new = GaussianEnergyDistribution(self.mean, self.std, self.min_erg, self.max_erg, self.fwhm)
        new /= other
        return new

    def __itruediv__(self, other):
        assert isinstance(other, (float, int))
        if other == 0:
            raise ZeroDivisionError
        other = 1.0/other
        self.__imul__(other)
        return self

    def __mul__(self, other):
        new = GaussianEnergyDistribution(self.mean, self.std, self.min_erg, self.max_erg, self.fwhm)
        new *= other
        return new

    def __imul__(self, other: float):
        assert isinstance(other, Number)
        if self.fwhm is not None:
            self.fwhm *= other
        if self.std is not None:
            self.std *= other
        self.mean *= other

        if self.min_erg is not None:
            self.min_erg *= other
        if self.max_erg is not None:
            self.max_erg *= other
        return self

    def __get_kwargs__(self):
        if self.std is not None:
            self.fwhm = 2.355*self.std
        else:
            assert self.fwhm is not None, 'Must specify either fwhm` or `std`'
            self.std = self.fwhm/2.355

        if self.min_erg is None:
            self.min_erg = self.mean - 5*self.std
        if self.max_erg is None:
            self.max_erg = self.mean + 5*self.std
        kwargs = {'e-type': '2', 'eg0': str(self.mean), "eg1": str(self.fwhm),
                  "eg2": str(self.min_erg), "eg3": str(self.max_erg)}
        return kwargs


@dataclass
class MonoEnergeticDistribution(Distribution):
    erg: float
    __is_energy_dist__ = True

    def __imul__(self, other):
        self.erg *= other
        return self

    def __get_kwargs__(self):
        return {'e0': self.erg}


@dataclass
class GenericEnergyDistribution(Distribution):
    energies: Union[List[float], np.ndarray]
    weights: Union[List[float], np.ndarray]

    def __imul__(self, other):
        self.energies = np.array(self.energies)
        self.energies *= other
        return self

    def __get_kwargs__(self):
        self.weights = np.array(self.weights)
        self.weights /= sum(self.weights)
        assert len(self.weights) == len(self.energies)

        kwargs = {'e-type': 1, 'ne': len(self.weights)}
        for i, (w, erg) in enumerate(zip(self.weights, self.energies)):
            kwargs[f'e({i})'] = erg
            kwargs[f'w({i})'] = w
        return kwargs


class NucleusSource:
    __all_sources__ = []

    @staticmethod
    def all_sources():
        return "\n".join([str(s) for s in NucleusSource.__all_sources__])

    def __init__(self, nuclide_or_name: Union[Nuclide, str], erg_dist: Distribution, spacial_dist: Distribution,
                 src_weight: float = 1):
        NucleusSource.__all_sources__.append(self)
        if not isinstance(nuclide_or_name, Nuclide):
            assert isinstance(nuclide_or_name, str)
            self.nuclide = Nuclide.from_symbol(nuclide_or_name)
        else:
            assert isinstance(nuclide_or_name, Nuclide)
            self.nuclide = nuclide_or_name

        self.erg_dist = erg_dist.__mul__(1.0/self.nuclide.A)  # MeV to MeV/n
        self.spacial_dist = spacial_dist
        self.src_weight = src_weight

    def __str__(self):
        zaid = str(int(1E6*self.nuclide.Z) + self.nuclide.A)
        lines = [f" <source> = {self.src_weight}", f'proj = {zaid}']
        lines.extend([f"{key} = {value}" for key, value in self.spacial_dist.__get_kwargs__().items()])
        lines.extend([f"{key} = {value}" for key, value in self.erg_dist.__get_kwargs__().items()])
        out = "\n    ".join(lines)
        # print(out)
        return out

if __name__ == "__main__":
    n = NucleusSource("Xe139", MonoEnergeticDistribution(70), PointSource())
    print(n)
