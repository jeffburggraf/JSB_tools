from __future__ import annotations
from typing import Tuple, List, Dict, Union, Iterable, Sized
from abc import abstractmethod, ABC
import numpy as np
from numbers import Number
import re

def get_rotation_matrix(theta, vx=0, vy=0, vz=1, _round=3):
    """Return the rotation matrix for a rotation around an arbitrary axis. theta is in degrees.

    """
    v = np.array([vx, vy, vz], dtype=float)
    v /= np.linalg.norm(v)
    vx, vy, vz = v
    out = [[((-1 + vz ** 2) * (-1 + vy ** 2 + vz ** 2) * (
            1 - vy ** 2 - vz ** 2 + (vy ** 2 + vz ** 2) * np.cos((np.pi * theta) / 180.))) / (
            vx ** 2 * (vx ** 2 + vy ** 2)), (
            -(vy * (-1 + vy ** 2 + vz ** 2)) + vy * (-1 + vy ** 2 + vz ** 2) * np.cos(
            (np.pi * theta) / 180.) - vx * vz * np.sin((np.pi * theta) / 180.)) / vx, (
            -(vz * (-1 + vy ** 2 + vz ** 2)) + vz * (-1 + vy ** 2 + vz ** 2) * np.cos(
            (np.pi * theta) / 180.) + vx * vy * np.sin((np.pi * theta) / 180.)) / vx], [(-(
            vy * (-1 + vy ** 2 + vz ** 2)) + vy * (-1 + vy ** 2 + vz ** 2) * np.cos(
            (np.pi * theta) / 180.) + vx * vz * np.sin((np.pi * theta) / 180.)) / vx, vy ** 2 - (-1 + vy ** 2) * np.cos(
            (np.pi * theta) / 180.), ((-1 + vz ** 2) * (
            vy * vz * (-1 + np.cos((np.pi * theta) / 180.)) + vx * np.sin((np.pi * theta) / 180.))) / (
            vx ** 2 + vy ** 2)],
            [(-(vz * (-1 + vy ** 2 + vz ** 2)) + vz * (-1 + vy ** 2 + vz ** 2) * np.cos(
            (np.pi * theta) / 180.) - vx * vy * np.sin((np.pi * theta) / 180.)) / vx,
            vy * vz - vy * vz * np.cos((np.pi * theta) / 180.) - (
            (-1 + vy ** 2 + vz ** 2) * np.sin((np.pi * theta) / 180.)) / vx,
            vz ** 2 - (-1 + vz ** 2) * np.cos((np.pi * theta) / 180.)]]
    for i in [0,1,2]:
        for j in [0,1,2]:
            out[i][j] = round(out[i][j], _round)
    return out


class MCNPNumberMapping(dict):
    """Used for automatically assigning numbers to MCNP cells and surfaces. """

    def __init__(self, class_name, starting_number: int):
        self.class_name: type = class_name
        self.starting_number: int = starting_number
        super(MCNPNumberMapping, self).__init__()  # initialize empty dict
        self.__auto_picked_numbers__ = []  # numbers that have been chosen automatically.

    def get_number_auto(self):
        if len(self) == 0:
            num = self.starting_number
        else:
            num = list(sorted(self.keys()))[-1] + 1
        self.__auto_picked_numbers__.append(num)
        return num

    def names_used(self, skip_item):
        return [o.name for o in self.values() if o is not skip_item]

    def __setitem__(self, number, item):
        assert hasattr(item, 'name')
        assert hasattr(item, 'number')
        assert isinstance(number, (int, type(None))), '{} number must be an integer'.format(self.class_name)

        if number is not None:  # do not pick number automatically
            if number in self.keys():  # there is a numbering conflict, not allowed in mcnp.
                conflicting_item = self[number]
                if number in self.__auto_picked_numbers__:  # can fix the numbering conflict by changing a number
                    self.__auto_picked_numbers__.remove(number)
                    new_number = self.get_number_auto()
                    self[new_number] = conflicting_item  # re-assign old cell to new number
                    conflicting_item.__number__ = new_number

                else:  # cannot fix the numbering conflict, raise an Exception
                    raise Exception('{} number {} has already been used (by {})'.format(self.class_name, number,
                                                                                        conflicting_item))
        else:
            number = self.get_number_auto()

        super(MCNPNumberMapping, self).__setitem__(number, item)  # re-assign current cell to number
        item.__number__ = number
        if isinstance(item.name, str):
            assert len(item.name) > 0, 'Blank name used in {} {}'.format(self.class_name, item)
        if item.name is not None:
            if item.name in self.names_used(item):
                raise Exception('{} name `{}` has already been used.'.format(self.class_name, item.name))
        else:
            item.name = None


def __get_comment__(comment, name):
    if name is not None:
        return_comment = " name: {}".format(name)
    else:
        return_comment = ''
    if comment is not None:
        return_comment = '{} {}'.format(comment, return_comment)

    if len(return_comment):
        return_comment = " $ " + return_comment
    return return_comment


class Surface(ABC, GeomSpec):
    all_surfs: MCNPNumberMapping = MCNPNumberMapping('Surface', 1)

    def __init__(self, surf_name=None, surf_num=None, comment=None):
        self.name = surf_name
        self.number = surf_num
        super().__init__(self)

        self.comment = comment
        Surface.all_surfs[surf_num] = self

    @abstractmethod
    def surface_card(self):
        pass


class CuboidSurface(Surface, GeomSpec):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, surf_name=None, surf_num=None, comment=None):
        super(CuboidSurface, self).__init__(surf_name, surf_num, comment)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    @property
    def volume(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)

    @property
    def surface_card(self):
        comment = __get_comment__(self.comment, self.name)
        return '{0} RPP {xmin} {xmax}  {ymin} {ymax}  {zmin} {zmax} {comment}' \
            .format(self.number, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, zmin=self.zmin,
                    zmax=self.zmax, comment=comment)


class Cell(ABC, GeomSpec):
    all_cells: MCNPNumberMapping = MCNPNumberMapping('Cell', 10)

    def __init__(self,
                 importance: Tuple[int, str],
                 material: Union[int, None] = None,
                 density: Union[float, None] = None,
                 cell_name: Union[str, None] = None,
                 cell_num: int = None,
                 cell_comment: Union[str, None] = None):

        self.name = cell_name
        if self.name is not None:
            assert isinstance(self.name, str)
        self.number = cell_num
        super().__init__(self)
        Cell.all_cells[cell_num] = self
        self.density = density
        if isinstance(self.density, float):
            self.density = abs(self.density)

        self.material = material
        self.importance = importance
        self.comment = cell_comment
        self.kwargs: Dict[str, str] = {}
        self.__geom_spec__ = None

        assert (len(importance) == 2) and isinstance(importance[1], str), \
            "improper specification of `importance argument`. Correct example: (True, 'pn')"

    def set_name(self, name):
        self.name = name

    def set_comment(self, comment):
        self.comment = comment

    def set_density_in_g_per_cm3(self, d):
        self.density = -d

    def set_void(self):
        self.density = 0

    def set_geometry(self, obj):
        self.__geom_spec__ = str(obj)

    @abstractmethod
    def cell_card(self):
        if self.density is None:
            assert (self.material in [0, None]), '`density` was not specified in cell {}, therefore `material` must be ' \
                                                 'specified as 0, or left as the default (None), not "{}"'\
                                                  .format(self, self.material)
            material = 0
        else:
            assert self.material is not None, 'In cell {}, a density was specified but a material was not! This is not'\
                                              ' allowed in MCNP'.format(self)
            assert self.material != 0, 'Material of "0" not allowed for non-void cells'
            material = self.material

        out = '{} {}'.format(self.number, material)
        if self.density is not None:
            out += ' {}'.format(self.density)
        imp = 'imp:{}={}'.format(','.join(self.importance[1]), int(self.importance[0]))
        out += ' {} {}'.format(self.__geom_spec__, imp)
        for key, arg in self.kwargs.items():
            out += ' {}={}'.format(key, arg)
        comment = __get_comment__(self.comment, self.name)
        return out + comment

    def __repr__(self):
        return '<MCNP cell, name: "{}";  cell number: {}>'.format(self.name, self.number)


class CuboidCell(Cell):
    def __init__(self, surface: CuboidSurface, importance: Tuple[int, str], material: int = 0,
                 density: Union[type(None), float] = None, cell_name=None, cell_num=None, comment=None):
        super(CuboidCell, self).__init__(importance=importance,
                                         material=material,
                                         density=density,
                                         cell_name=cell_name,
                                         cell_num=cell_num,
                                         cell_comment=comment)
        self.surface: CuboidSurface = surface

    def copy(self, new_importance: Union[Tuple[int, str], type(None)] = None, new_material=None, new_density=None,
             new_cell_name=None, new_cell_num=None, new_cell_comment=None) -> CuboidCell:
        if new_importance is None:
            new_importance = self.importance
        if new_material is None:
            new_material = self.material
        if new_density is None:
            new_density = self.density
        new_cell = CuboidCell(self.surface, importance=new_importance, material=new_material, density=new_density,
                              cell_name=new_cell_name,
                              cell_num=new_cell_num,  comment=new_cell_comment)

        assert self.__geom_spec__ is not None, 'call Cell.set_geom_spec before trying to copy'
        new_cell.__geom_spec__ = self.__geom_spec__

        return new_cell

    @classmethod
    def from_coordinates(cls, xmin, xmax, ymin, ymax, zmin, zmax, importance: Tuple[int, str], material=0,
                         density=None, cell_name=None, cell_num=None, cell_comment=None, surf_name=None,
                         surf_number=None, surf_comment=None) -> CuboidCell:

        surface = CuboidSurface(xmin, xmax, ymin, ymax, zmin, zmax, surf_name=surf_name, surf_num=surf_number,
                                comment=surf_comment)

        out = cls(surface, importance=importance, material=material, density=density, cell_name=cell_name,
                  cell_num=cell_num, comment=cell_comment)
        return out

    def offset(self, offset_vector: Sized[float]):
        #  Todo: use like but here (if all changes are none)!

        assert isinstance(offset_vector, Iterable)
        assert hasattr(offset_vector, '__len__')
        assert len(offset_vector) == 3

        self.kwargs['trcl'] = '({} {} {})'.format(*offset_vector)

    @property
    def volume(self):
        return self.surface.volume

    @property
    def cell_card(self):
        out = super(CuboidCell, self).cell_card()
        return out





if __name__ == '__main__':
    def test_geom_spec():
        cell1 = CellSpec(89)
        surf1 = SurfaceSpec(1)
        surf2 = SurfaceSpec(2)
        surf3 = SurfaceSpec(3)

        s1 = ~cell1 & surf1 & surf2
        s2 = ~cell1 | surf1 & surf2
        print(s1)
        print(s2)
    c1 = CuboidCell.from_coordinates(-1,1, -1,1 ,-1,1, (1, 'np'), 1000 )
    c99 = CuboidCell.from_coordinates(-10,10, -10,10 ,-10,10, (1, 'np'), 1000)
    c1.set_geometry(c1 & ~c2)

    c2 = c1.offset([1,0, 0])



    # c0 = CuboidCell.from_coordinates(-1,1, -1,1 ,-1,1, (1, 'np'), material=1000, density=1.2, cell_comment=None, cell_num=13, cell_name='c',
    #                                 surf_comment='fuck', surf_name='surf camsd')
    # c2 = c.copy()
    # c2.offset([1,1,1])
    #
    #
    # for k, v in Cell.all_cells.items():
    #     print(k, v, '\n',v.cell_card)
    #
    # for k, v in Surface.all_surfs.items():
    #     print(k, v, '\n', v.surface_card)


