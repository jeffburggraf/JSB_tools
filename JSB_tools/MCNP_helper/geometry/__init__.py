from __future__ import annotations
from typing import Union, List, Dict, Tuple, Sized, Iterable
from abc import ABC, abstractmethod
from JSB_tools.MCNP_helper.geometry.__geom_helpers__ import get_rotation_matrix, GeomSpecMixin, BinaryOperator
from warnings import warn
from numbers import Number
import numpy as np


class MCNPNumberMapping(dict):
    """Used for automatically assigning numbers to MCNP cells and surfaces. """

    def __init__(self, class_name, starting_number: int):
        self.class_name: type = class_name
        self.starting_number: int = starting_number
        super(MCNPNumberMapping, self).__init__()  # initialize empty dict
        self.__auto_picked_numbers__ = []  # numbers that have been chosen automatically.

    def number_getter(self, item):
        if self.class_name == 'Cell':
            return getattr(item, 'cell_number')
        elif self.class_name == 'Surface':
            return getattr(item, 'surface_number')
        else:
            assert False

    def number_setter(self, item, num):
        if self.class_name == 'Cell':
            return setattr(item, 'cell_number', num)
        elif self.class_name == 'Surface':
            return setattr(item, 'surface_number', num)
        else:
            assert False

    def get_number_auto(self):
        if len(self) == 0:
            num = self.starting_number
        else:
            num = list(sorted(self.keys()))[-1] + 1
        self.__auto_picked_numbers__.append(num)
        return num

    def names_used(self):
        return [o.__name__ for o in self.values() if o.__name__ is not None]

    def __setitem__(self, number, item):
        assert isinstance(number, (int, type(None))), '{} number must be an integer'.format(self.class_name)

        if isinstance(item.__name__, str):
            assert len(item.__name__) > 0, 'Blank name used in {} {}'.format(self.class_name, item)
        if item.__name__ is not None:
            if item.__name__ in self.names_used():
                raise Exception('{} name `{}` has already been used.'.format(self.class_name, item.__name__))
        else:
            item.__name__ = None

        if number is not None:  # do not pick number automatically
            if number in self.keys():  # there is a numbering conflict, not allowed in MCNP. Try to resolve conflict
                conflicting_item = self[number]  # Item with numbering conflict.
                if number in self.__auto_picked_numbers__:  # Can we fix the numbering conflict by changing a number?
                    # Yes, we can, because the conflicting instance's number was not user chosen, but this one was.
                    self.__auto_picked_numbers__.remove(number)
                    new_number = self.get_number_auto()
                    self[new_number] = conflicting_item  # re-assign old cell to new number, resolving the conflict
                    self.number_setter(conflicting_item, new_number)  # change

                else:  # cannot fix the numbering conflict, raise an Exception
                    raise Exception('{} number {} has already been used (by {})'.format(self.class_name, number,
                                                                                        conflicting_item))
        else:
            number = self.get_number_auto()

        super(MCNPNumberMapping, self).__setitem__(number, item)  # re-assign current cell to number
        self.number_setter(item, number)  # set the cell or surf instance's number attribute to the correct value


def get_comment(comment, name):
    if name is not None:
        return_comment = " name: {}".format(name)
    else:
        return_comment = ''
    if comment is not None:
        return_comment = '{} {}'.format(comment, return_comment)

    if len(return_comment):
        return_comment = " $ " + return_comment
    return return_comment


class Surface(ABC, GeomSpecMixin):
    all_surfs = MCNPNumberMapping('Surface', 1)

    def __init__(self, surface_number: float = None,
                 surface_name: Union[str, None] = None,
                 surface_comment: Union[str, None] = None):
        self.__name__ = surface_name
        self.surface_number = surface_number
        self.surface_comment = surface_comment
        Surface.all_surfs[self.surface_number] = self
        super().__init__()

    @property
    def surface_name(self):
        return self.__name__
    #
    # @property
    # def surface_number(self):
    #     return self.__number__

    @abstractmethod
    def surface_card(self):
        pass


class CuboidSurface(Surface):
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
        comment = get_comment(self.surface_comment, self.surface_name)
        return '{0} RPP {xmin} {xmax}  {ymin} {ymax}  {zmin} {zmax} {comment}' \
            .format(self.surface_number, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, zmin=self.zmin,
                    zmax=self.zmax, comment=comment)


class TRCL:
    def __init__(self, offset_vector=(0, 0, 0), rotation_theta=0., rotation_axis=(0, 0, 1), _round=3):
        """Creates TRCL cards for MCNP. rotation_theta is in degrees."""
        assert hasattr(offset_vector, '__iter__'), 'Invalid offset vector, {}'.format(offset_vector)
        assert len(offset_vector) == 3, 'Invalid offset vector, {}'.format(offset_vector)
        assert isinstance(rotation_theta, Number), 'Invalid rotation_theta, {}'.format(rotation_theta)
        assert hasattr(rotation_axis, '__iter__'), 'Invalid rotation_axis, {}'.format(rotation_axis)
        assert len(rotation_axis) == 3, 'Invalid rotation_axis, {}'.format(rotation_axis)
        self.offset_vector = offset_vector
        if rotation_axis == (0, 0, 1) and rotation_theta == 0:
            self.rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        else:
            self.rotation_matrix = get_rotation_matrix(rotation_theta, *rotation_axis, _round=_round)

    def __str__(self) -> str:
        offset = ' '.join(map(str, self.offset_vector))
        matrix = '  '.join([row for row in [' '.join(map(str, a)) for a in self.rotation_matrix]])
        return '{}   {}'.format(offset, matrix)


class Cell(GeomSpecMixin):
    all_cells = MCNPNumberMapping('Cell', 10)

    def __init__(self, importance: Tuple[str, int],
                 material: int = 0,
                 density: Union[float, type(None)] = None,
                 geometry: Union[type(None), GeomSpecMixin, BinaryOperator, str] = None,
                 cell_number:  float = None,
                 cell_name: Union[str, None] = None,
                 cell_comment: Union[str, type(None)] = None,
                 mcnp_kwargs: Dict = None):
        self.__name__ = cell_name
        self.cell_number = cell_number
        self.importance = importance
        self.material = material
        self.geometry = None
        self.density = density
        self.geometry = geometry
        if mcnp_kwargs is not None:
            self.mcnp_kwargs = mcnp_kwargs
        else:
            self.mcnp_kwargs = {}
        self.cell_comment = cell_comment
        Cell.all_cells[cell_number] = self
        GeomSpecMixin.__init__(self)
        self.__like_but_kwargs__ = {}
        self.__like_but_number__ = None

    @property
    def cell_name(self) -> str:
        return self.__name__

    def like_but(self, importance: Tuple[str, int]=None,
                 material: int = 0,
                 density: Union[float, type(None)] = None,
                 cell_number:  float = None,
                 cell_name: Union[str, None] = None,
                 cell_comment: Union[str, type(None)] = None,
                 trcl: TRCL = None) -> Cell:
        like_but_kwargs = {}
        if density is None:
            density = self.density
        else:
            like_but_kwargs['rho'] = density

        if importance is None:
            importance = self.importance
        else:
            like_but_kwargs['imp'] = self.__get_imp_str__(importance)

        if material is None:
            material = self.material
        else:
            like_but_kwargs['MAT'] = material

        if trcl is not None:
            like_but_kwargs['TRCL'] = str(trcl)

        new_cell = Cell(importance=importance, material=material, density=density, geometry=None,
                        cell_number=cell_number, cell_name=cell_name, cell_comment=cell_comment)
        new_cell.__like_but_kwargs__ = like_but_kwargs
        new_cell.__like_but_number__ = self.cell_number
        return new_cell

    def __build_like_but_cell_card__(self):
        assert self.__like_but_number__ is not None
        assert len(self.__like_but_kwargs__) != 0
        kwargs = ['{} = {}'.format(k, v) for k, v in self.__like_but_kwargs__.items()]
        kwargs = ' '.join(kwargs)
        return 'LIKE {} BUT {}'.format(self.__like_but_number__, kwargs)

    def __get_imp_str__(self, imp=None):

        if imp is None:
            imp = self.importance
        assert hasattr(imp, '__iter__') and len(imp) == 2 and isinstance(imp[0], str), \
            'Importance, "{}", not specified properly. Correct example: ("np", 1)'.format(imp)

        return 'imp:{}={}'.format(','.join(imp[0]), int(imp[1]))

    def __build_cell_card__(self):
        out = '{} {}'.format(self.cell_number, self.material)
        if self.density is not None:
            out += ' {}'.format(self.density)
        imp = self.__get_imp_str__()
        # if not from_str:
        #     assert self.geometry is not None,\
        #         'Attempted to access the cell card of a cell for which geometry\nhas not been specified!'
        if self.geometry is None:
            warn('Cell card of cell {} was accessed with out specifying geometry.')
        out += ' {} {}'.format(self.geometry, imp)
        for key, arg in self.mcnp_kwargs.items():
            out += ' {}={}'.format(key, arg)
        comment = get_comment(self.cell_comment, self.cell_name)
        return out + comment

    @property
    def cell_card(self):
        if self.density is None or self.density == 0:
            assert (self.material in [0, None]), \
                '`density` was not specified for cell with {} {}, therefore `material` must be ' \
                'specified as 0, or left as the default (None)'\
                .format('name' if self.cell_name is not None else 'number',
                        self.cell_name if self.cell_name is not None else self.cell_number)
            self.material = 0
            if self.density == 0:
                self.density = None
        else:
            assert self.material is not None, 'In cell {}, a density was specified but a material was not! This is not'\
                                              ' allowed in MCNP'.format(self)
            assert self.material != 0, 'Material of "0" not allowed for non-void cells'
            self.material = self.material
        if len(self.__like_but_kwargs__) == 0:
            return self.__build_cell_card__()
        else:
            return self.__build_like_but_cell_card__()

    def set_geometry(self, geom: Union[str, GeomSpecMixin, BinaryOperator]):
        if isinstance(geom, (GeomSpecMixin, BinaryOperator)):
            self.geometry = geom.__to_str__()
        else:
            self.geometry = str(geom)
        return self.geometry

    def __str__(self):
        return self.cell_card
