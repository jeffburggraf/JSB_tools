"""
For the creation of surfaces and cells in MCNP. Cell/surface numbers can be managed automatically,
or manually specified.
"""

from __future__ import annotations

from abc import ABC

from JSB_tools.MCNP_helper.geometry.geom_core import Cell, Surface, TRCL, get_comment, CellGroup
from typing import Union, List, Dict, Tuple, Sized, Iterable
import numpy as np
from JSB_tools.MCNP_helper.materials import Material as mat
from JSB_tools.MCNP_helper.materials import PHITSOuterVoid

NDIGITS = 7  # Number of significant digits to round all numbers to when used in input decks.


__all__ = ['clear_all', 'clear_all_but_materials', 'clear_cells', 'clear_surfaces']


def clear_cells():
    Cell.clear()


def clear_surfaces():
    Surface.clear()



def clear_all_but_materials():
    clear_cells()
    clear_surfaces()


def clear_all():
    """
    Clears all Cell, Surface, Material, and Tally class variables
    Returns:None

    """
    mat.clear()
    Cell.clear()
    Surface.clear()


class SphereSurface(Surface):
    def __init__(self, radius, x=0, y=0, z=0, surf_name=None, surf_num=None, comment=None):
        super(SphereSurface, self).__init__(surface_number=surf_num, surface_name=surf_name, surface_comment=comment)
        self.radius = radius
        self.x = x
        self.y = y
        self.z = z

    @property
    def volume(self):
        return 4/3*np.pi*self.radius**3

    @property
    def surface_card(self):
        comment = get_comment(self.surface_comment, self.surface_name)
        out = f'{self.surface_number} SPH {self.x:.{NDIGITS}g} {self.y:.{NDIGITS}g} {self.z:.{NDIGITS}g} {self.radius} ' \
              f'{comment}'
        return out


class CuboidSurface(Surface):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, surf_name=None, surf_num=None, comment=None):
        super(CuboidSurface, self).__init__(surface_number=surf_num, surface_name=surf_name, surface_comment=comment)
        for kmin, kmax in zip([xmin, ymin, zmin], [xmax, ymax, zmax]):
            assert kmin != kmax, 'Cuboid with minimum coordinate equals maximum coordinate: " == {}'.format(kmax, kmax)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self._zmin = zmin
        self._zmax = zmax

        if self._zmin > self._zmax:
            self._zmax, self._zmin = self._zmin, self._zmax

    @classmethod
    def from_thickness_and_widths(cls, z0, dz, x_width, y_width, x_center=0, y_center=0, orientation='xyz', surf_name=None, surf_num=None,
                                  comment=None, ):
        """
        Args:
            z0: initial z
            dz: thickness in the z direction
            x_width: full with in the x-axis
            y_width:  full with in the y-axis
            x_center: center of cell in the x axis.
            y_center: center of cell in the y axis.
            orientation: make the direction of the z axis refer to another axis - todo
            surf_name:
            surf_num:
            comment:

        Returns:

        """
        #  todo: orientation
        assert dz != 0
        if dz <= 0:
            dz = abs(dz)
            z0 = z0 - dz

        return CuboidSurface(xmin=x_center-x_width/2, ymin=y_center-y_width/2, zmin=z0, xmax=x_center + x_width/2,
                             ymax=y_center+y_width/2, zmax=z0+dz, surf_name=surf_name, surf_num=surf_num,
                             comment=comment)

    @property
    def dz(self):
        return self._zmax - self._zmin

    @dz.setter
    def dz(self, other):
        self._zmax = self._zmin + other

    @property
    def zmax(self):
        return self._zmax

    @zmax.setter
    def zmax(self, other):
        dz = self.dz
        self._zmax = other
        self._zmin = self.zmax - dz

    @property
    def zmin(self):
        return self._zmin

    @zmin.setter
    def zmin(self, other):
        dz = self.dz
        self._zmin = other
        self._zmax = self.zmin + dz

    @property
    def volume(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)

    @property
    def surface_card(self):
        comment = get_comment(self.surface_comment, self.surface_name)
        out = f'{self.surface_number} RPP {self.xmin:.{NDIGITS}g} {self.xmax:.{NDIGITS}g}  ' \
              f'{self.ymin:.{NDIGITS}g} {self.ymax:.{NDIGITS}g}  {self.zmin:.{NDIGITS}g} {self.zmax:.{NDIGITS}g}' \
              f' {comment}'
        # return '{0} RPP {xmin} {xmax}  {ymin} {ymax}  {zmin} {zmax} {comment}' \
        #     .format(self.surface_number, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, zmin=zmin,
        #             zmax=self.zmax, comment=comment)  # old
        return out


class CuboidCell(Cell, CuboidSurface):
    """
    Examples:
        c1 = CuboidCell(0, 1, 0, 1, 0, 1, importance=('np', 1))  # A void cuboid cell from (0,0,0) to (1,1,1)

        c1.cell_card  # string of the cell card

        c1.surface_card  # string of the surface card

    """
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax=None, dz=None,
                 material: Union[int, mat, PHITSOuterVoid] = 0,
                 cell_name: str = None,
                 importance: Union[None, Tuple[str, int]] = None,
                 cell_num: int = None, cell_comment: str = None,
                 surf_name: str = None, surf_number: int = None,
                 surf_comment: str = None, cell_kwargs=None):
        """
        Args:
            xmin:  min x
            xmax: max x
            ymin: etc...
            ymax: ...
            zmin: ...
            zmax: ...
            dz: If zmax is None, then zmx = zmin + dz
            importance: Cell importance. e.g. ("np", 1) -> neutron and photon importance = 1
            material: MCNP material number
            cell_name:  For use in outpreader.py for looking up cell and tallies by name.
            cell_num: Cell number. If None, automatically choose a cell number.
            cell_comment: Comment for cell
            surf_name:
            surf_number:
            surf_comment:
            cell_kwargs:  Additional keyword arguments to be used in cell card, i.g. vol=1
        """

        assert not (zmax is dz is None), 'Must specify either `zmax` or `dz`.'
        if zmax is None:
            zmax = zmin + dz

        if cell_name is not None:
            if surf_name is None:
                surf_name = cell_name
        if cell_comment is not None:
            if surf_comment is None:
                surf_comment = cell_comment

        super(CuboidCell, self).__init__(
             material=material,
             importance=importance,
             cell_number=cell_num,
             cell_name=cell_name,
             cell_comment=cell_comment,
             cell_kwargs=cell_kwargs
             )
        super(Cell, self).__init__(xmin, xmax, ymin, ymax, zmin, zmax, surf_name=surf_name, surf_num=surf_number,
                                   comment=surf_comment)

    # def copy(self, new_importance: Union[Tuple[int, str], type(None)] = None, new_material=None, new_density=None,
    #          new_cell_name=None, new_cell_num=None, new_cell_comment=None) -> CuboidCell:
    #     pass

    @property
    def volume(self):
        return super(Cell, self).volume

    @property
    def cell_card(self):
        out = super(CuboidCell, self).cell_card
        return out


class RightCylinderSurface(Surface):
    def __init__(self, radius: float, x0: float = 0, y0: float = 0, z0: float = 0,
                 dx: float = 0, dy: float = 0, dz: float = 0,
                 surf_name: str = None, surf_num: Union[int, str] = None,
                 comment: str = None):
        """
        Defines a cylindrical surface who's axis spans from (x0, y0, z0) to (x0+dx, y0+dy, z0+dz), with perpendicular
        radial component of the given radius.
        Args:
            x0:
            y0:
            z0:
            dx:
            dy:
            dz:
            radius:
            surf_name:
            surf_num:
            comment:
        """
        super(RightCylinderSurface, self).__init__(surface_number=surf_num, surface_name=surf_name,
                                                           surface_comment=comment)
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.radius = radius

    @classmethod
    def from_min_max_coords(cls, radius: float, x0: float = 0, y0: float = 0, z0: float = 0,
                            x1: float = 0, y1: float = 0, z1: float = 0,
                            surf_name: str = None, surf_num: Union[int, str] = None,
                            comment: str = None):
        dz = z1-z0
        dy = y1-y0
        dx = x1-x0
        return cls(radius=radius, x0=x0, y0=y0, z0=z0, dx=dx, dy=dy, dz=dz, surf_name=surf_name, surf_num=surf_num,
                   comment=comment)

    def set_z_limits(self, zmin, zmax):
        assert zmin < zmax
        self.dz = zmax-zmin
        self.z0 = zmin

    @property
    def z_mid(self):
        return (self.zmin + self.zmax)/2

    @property
    def zmax(self):
        return self.z0 + self.dz

    @zmax.setter
    def zmax(self, other):
        self.z0 = other-self.dz

    @property
    def zmin(self):
        return self.z0

    @zmin.setter
    def zmin(self, other):
        self.z0 = other

    @property
    def xmax(self):
        return self.x0 + self.dx

    @property
    def ymax(self):
        return self.y0 + self.dy

    @property
    def volume(self):
        dv = np.array([self.dx, self.dy, self.dz])
        return np.pi*self.radius**2*np.linalg.norm(dv)

    @property
    def surface_card(self):
        assert not self.dx == self.dy == self.dz == 0, 'dx, dy, and dz cannot all be zero!'

        comment = get_comment(self.surface_comment, self.surface_name)
        out = f"{self.surface_number} RCC {self.x0:.{NDIGITS}g} {self.y0:.{NDIGITS}g} {self.z0:.{NDIGITS}g}" \
              f"  {self.dx:.{NDIGITS}g} {self.dy:.{NDIGITS}g} {self.dz:.{NDIGITS}g} {self.radius:.{NDIGITS}g}" \
              f" {comment}"
        # out = '{0} RCC {x0} {y0} {z0}  {dx} {dy} {dz} {r} {comment}' \
        #     .format(self.surface_number, x0=self.x0, y0=self.y0, z0=self.z0, dx=self.dx, dy=self.dy,
        #             dz=self.dz, r=self.radius, comment=comment)
        return out


class RightCylinder(Cell, RightCylinderSurface):
    def __init__(self, radius: float,
                 material: Union[int, mat, PHITSOuterVoid] = 0,
                 importance: Union[None, Tuple[str, int]] = None,
                 x0: float = 0, y0: float = 0, z0: float = 0,
                 dx: float = 0, dy: float = 0, dz: float = 0, cell_name: str = None,
                 cell_num: int = None, cell_comment: str = None,
                 surf_number: int = None,
                 surf_comment: str = None, cell_kwargs=None):

        if cell_name is not None:
            surf_name = cell_name
        else:
            surf_name = None
        if cell_comment is not None:
            if surf_comment is None:
                surf_comment = cell_comment
        if dz < 0:
            z0 = z0+dz
            dz = -dz
        if dy < 0:
            y0 = y0+dy
            dy = -dy
        if dx < 0:
            x0 = x0+dx
            dx = -dx

        super(RightCylinder, self).__init__(material=material, importance=importance,
                                            cell_number=cell_num, cell_name=cell_name, cell_comment=cell_comment,
                                            cell_kwargs=cell_kwargs)
        super(Cell, self).__init__(x0=x0, y0=y0, z0=z0, dx=dx, dy=dy, dz=dz, radius=radius, surf_name=surf_name,
                                   surf_num=surf_number, comment=surf_comment)


if __name__ == '__main__':
    m_Ar = mat.gas(['Ar'], [1], pressure=1.4)
    # c = CuboidCell
    c = RightCylinder(1, m_Ar, dy = 21)
    print(c.geometry)
    c2 = c.like_but(density=100, trcl=TRCL())
    print(c2.cell_card)
    print(c.cell_card)