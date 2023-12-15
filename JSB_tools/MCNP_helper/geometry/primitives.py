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

NDIGITS = 10  # Number of significant digits to round all numbers to when used in input decks.


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
    def z1(self):  # max z
        return self.z + self.radius

    @property
    def z0(self):  # min z
        return self.z0 - self.radius

    @property
    def volume(self):
        return 4/3*np.pi*self.radius**3

    @property
    def surface_card(self):
        comment = get_comment(self.surface_comment, self.surface_name)
        out = f'{self.surface_number} SPH {self.x:.{NDIGITS}g} {self.y:.{NDIGITS}g} {self.z:.{NDIGITS}g} {self.radius} ' \
              f'{comment}'
        return out


class SimplePlaneSurface(Surface):
    def __init__(self, loc=0, ax='z', surf_name=None, surf_num=None, comment=None):
        """
        Generate a plane from a location and normal. Needs sognioficant reworking and kscdjgnvbadgf
        Args:
            loc:
            normal_vec_or_str:
            surf_name:
            surf_num:
            comment:
        """
        super(SimplePlaneSurface, self).__init__(surface_number=surf_num, surface_name=surf_name, surface_comment=comment)
        self.mnemonic = f'p{ax.lower()}'
        self.ax = ax
        self.loc = loc

    @property
    def surface_card(self):
        comment = get_comment(self.surface_comment, self.surface_name)
        out = f'{self.surface_number} {self.mnemonic} {self.loc:.{NDIGITS}g} {comment}'
        return out


class SimpleCylinderSurface(Surface):
    def __init__(self, radius, ax='z', surf_name=None, surf_num=None, comment=None):
        """
        Generate a plane from a location and normal. Needs sognioficant reworking and kscdjgnvbadgf
        Args:
            loc:
            ax: Axis of cylinder
        """
        super(SimpleCylinderSurface, self).__init__(surface_number=surf_num, surface_name=surf_name, surface_comment=comment)
        self.mnemonic = f'c{ax.lower()}'
        self.ax = ax
        self.radius = radius

    @property
    def surface_card(self):
        comment = get_comment(self.surface_comment, self.surface_name)
        out = f'{self.surface_number} {self.mnemonic} {self.radius:.{NDIGITS}g} {comment}'
        return out


class SphereCell(Cell, SphereSurface):
    def __init__(self, radius, x=0, y=0, z=0,
                 material: Union[int, mat, PHITSOuterVoid] = 0,
                 importance: Union[None, Tuple[str, int]] = None,
                 cell_name: str = None,
                 cell_num: int = None, cell_comment: str = None,
                 cell_kwargs=None, surf_name=None, surf_num=None, comment=None, **kwargs):

        super(SphereCell, self).__init__(material=material, importance=importance, cell_number=cell_num,
                                         cell_name=cell_name, cell_comment=cell_comment, cell_kwargs=cell_kwargs)

        super(Cell, self).__init__(radius=radius, x=x, y=y, z=z, surf_name=surf_name, surf_num=surf_num,
                                   comment=comment)


class CuboidSurface(Surface):
    @classmethod
    def square(cls, width,
               z0, z1=None, dz=None,
               surf_name: str = None, surf_number: int = None,
               comment: str = None):
        return cls(-width/2, width/2, -width/2, width/2, z0=z0, z1=z1, dz=dz, surf_name=surf_name,
                   surf_number=surf_number, comment=comment)

    def __init__(self, x0, x1, y0, y1, z0, z1=None, dz=None, surf_name=None, surf_number=None, comment=None):
        super(CuboidSurface, self).__init__(surface_number=surf_number, surface_name=surf_name, surface_comment=comment)
        for coord, kmin, kmax in zip(['x', 'y', 'z'], [x0, y0, z0], [x1, y1, z1]):
            assert kmin != kmax, f'Provided min and max {coord} coordinates of Cuboid are equal: ' \
                                 f'"{coord}0={coord}1={kmin}'
        assert not (dz is None and z1 is None), 'Must set either dz or z1!'
        if z1 is None:
            assert dz is not None, 'Must supply either `dz` r `z1`'
            z1 = z0 + dz
        else:
            assert dz is None, 'Cannot supply `z1` and `dz`'
        if z0 > z1:
            z0, z1 = z1, z0
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

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

        return CuboidSurface(x0=x_center - x_width / 2, y0=y_center - y_width / 2, z0=z0, x1=x_center + x_width / 2,
                             y1=y_center + y_width / 2, z1=z0 + dz, surf_name=surf_name, surf_number=surf_num,
                             comment=comment)

    def cross_section_area(self, axis='s'):
        try:
            return self.volume/{'x': self.dx, 'y': self.dz, 'z': self.dz}[axis.lower()]
        except KeyError:
            raise ValueError(f'Invalid axis, "{axis}".')

    @property
    def dx(self):
        return self.x1 - self.x0

    @property
    def dy(self):
        return self.y1 - self.y0

    @property
    def dz(self):
        return self.z1 - self.z0

    @dz.setter
    def dz(self, value):
        raise AttributeError("`dz` cannot be set via property. Use self.set_dz(dz).")

    def set_dz(self, dz, fix_max=False):
        """

        Args:
            dz: Delta in the z-coordinate
            fix_max: If True, move min to alter self.dz, else, move max instead

        Returns:

        """
        if fix_max:
            self.z0 = self.z1 - dz
        else:
            self.z1 = self.z0 + dz

    @property
    def volume(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0) * (self.z1 - self.z0)

    @property
    def surface_card(self):
        comment = get_comment(self.surface_comment, self.surface_name)
        out = f'{self.surface_number} RPP ' \
              f'{self.x0:.{NDIGITS}g} {self.x1:.{NDIGITS}g} ' \
              f'{self.y0:.{NDIGITS}g} {self.y1:.{NDIGITS}g} ' \
              f'{self.z0:.{NDIGITS}g} {self.z1:.{NDIGITS}g}' \
              f' {comment}'
        return out


class CuboidCell(Cell, CuboidSurface):
    """
    Examples:
        c1 = CuboidCell(0, 1, 0, 1, 0, 1, importance=('np', 1))  # A void cuboid cell from (0,0,0) to (1,1,1)

        c1.cell_card  # string of the cell card

        c1.surface_card  # string of the surface card

    """
    @classmethod
    def square(cls, width,
               z0, z1=None, dz=None,
               material: Union[int, mat, PHITSOuterVoid] = 0,
               cell_name: str = None,
               importance: Union[None, Tuple[str, int]] = None,
               cell_num: int = None, cell_comment: str = None,
               surf_name: str = None, surf_number: int = None,
               surf_comment: str = None, cell_kwargs=None):

        return cls(-width/2, width/2, -width/2, width/2, z0=z0, z1=z1, dz=dz, material=material,
                   cell_name=cell_name, importance=importance, cell_num=cell_num, cell_comment=cell_comment,
                   surf_name=surf_name, surf_number=surf_number, surf_comment=surf_comment, cell_kwargs=cell_kwargs)

    def __init__(self, x0, x1, y0, y1,
                 z0, z1=None, dz=None,
                 material: Union[int, mat, PHITSOuterVoid] = 0,
                 cell_name: str = None,
                 importance: Union[None, Tuple[str, int]] = None,
                 cell_num: int = None, cell_comment: str = None,
                 surf_name: str = None, surf_number: int = None,
                 surf_comment: str = None, cell_kwargs=None):
        """
        Args:
            x0:  min x
            x1: max x
            y0: etc...
            y1: ...
            z0: ...
            z1: ...
            dz: If z1 is None, then z1 = z0 + dz
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
        super(Cell, self).__init__(x0, x1, y0, y1, z0, z1=z1, dz=dz, surf_name=surf_name, surf_number=surf_number,
                                   comment=surf_comment)

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

    def set_z_range(self, z0, z1):
        assert z0 < z1
        self.dz = z1-z0
        self.z0 = z0

    @property
    def z_mid(self):
        return (self.z0 + self.z1)/2

    @property
    def z1(self):
        return self.z0 + self.dz

    # def set_dz(self, other, fix_max=False):
    #     """
    #
    #     Args:
    #         other:
    #         fix_max: If True, move min to alter self.dz, else, move max instead
    #
    #     Returns:
    #
    #     """
    #     if fix_max:
    #         self.z0 = self.z1 - other
    #         self.dz = other
    #     else:
    #         self.dz = other

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
        assert not self.dx == self.dy == self.dz == 0, f'dx, dy, and dz cannot all be zero! In Surface {self}'

        comment = get_comment(self.surface_comment, self.surface_name)
        out = f"{self.surface_number} RCC " \
              f"{self.x0:.{NDIGITS}g} {self.y0:.{NDIGITS}g} {self.z0:.{NDIGITS}g}  " \
              f"{self.dx:.{NDIGITS}g} {self.dy:.{NDIGITS}g} {self.dz:.{NDIGITS}g} " \
              f"{self.radius:.{NDIGITS}g}" \
              f" {comment}"

        return out


class RightCylinder(Cell, RightCylinderSurface):
    def __init__(self, radius: float,
                 material: Union[int, mat, PHITSOuterVoid] = 0,
                 importance: Union[None, Tuple[str, int]] = None,
                 x0: float = 0, y0: float = 0, z0: float = 0,
                 dx: float = 0, dy: float = 0, dz: float = 0, cell_name: str = None,
                 cell_num: int = None, cell_comment: str = None,
                 surf_num: int = None,
                 surf_comment: str = None,
                 surf_name: str = None,
                 cell_kwargs=None):

        if surf_name is None:
            surf_name = cell_name

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
                                   surf_num=surf_num, comment=surf_comment)

    @classmethod
    def from_min_max_coords(cls, radius: float, x0: float = 0, y0: float = 0, z0: float = 0,
                            x1: float = 0, y1: float = 0, z1: float = 0,
                            material: Union[int, mat, PHITSOuterVoid] = 0,
                            importance: Union[None, Tuple[str, int]] = None,
                            cell_name: str = None,
                            cell_num: str = None,
                            cell_comment: str = None,
                            surf_name: str = None,
                            surf_num: Union[int, str] = None,
                            comment: str = None,
                            cell_kwargs: dict = None):
        return cls(radius, material=material, importance=importance,
                   x0=x0, dx=x1-x0,
                   y0=y0, dy=y1-y0,
                   z0=z0, dz=z1-z0,
                   cell_name=cell_name, cell_num=cell_num, cell_comment=cell_comment, surf_num=surf_num,
                   surf_name=surf_name, cell_kwargs=cell_kwargs
                   )




if __name__ == '__main__':
    m_Ar = mat.gas(['Ar'], [1], pressure=1.4)
    # c = CuboidCell
    c = RightCylinder(1, m_Ar, dy = 21)
    print(c.geometry)
    c2 = c.like_but(density=100, trcl=TRCL())
    print(c2.cell_card)
    print(c.cell_card)