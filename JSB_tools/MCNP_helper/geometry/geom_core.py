from __future__ import annotations

import re
from typing import Union, List, Dict, Tuple, Sized, Iterable, Type
from abc import ABC, abstractmethod, ABCMeta
from JSB_tools.MCNP_helper.geometry import get_rotation_matrix, GeomSpecMixin, BinaryOperator,\
    get_comment, MCNPNumberMapping
from warnings import warn
from numbers import Number
import numpy as np
from JSB_tools.MCNP_helper.materials import Material, PHITSOuterVoid
from pathlib import Path
import io
"""
Base class definitions for geometry primitives defined in primitives.py
"""
#  todo: In outp_reader, return a Cell instance (the one defnied here). Impliment this in the Cell class,
#   e.g. a from_outp method


class Surface(GeomSpecMixin, metaclass=ABCMeta):
    all_surfs = MCNPNumberMapping('Surface', 1)

    @staticmethod
    def global_zmax():
        """
        Return the largest .zmax attribute of all surfaces.
        Returns:

        """
        _max = None
        for c in Surface.all_surfs.values():
            try:
                if _max is None:
                    _max = c.zmax
                else:
                    if c.zmax > _max:
                        _max = c.zmax
            except AttributeError:
                continue
        return _max

    @staticmethod
    def clear():
        Surface.all_surfs = MCNPNumberMapping('Surface', 1)

    def __init__(self, surface_number: float = None,
                 surface_name: Union[str, None] = None,
                 surface_comment: Union[str, None] = None):
        self.__name__ = surface_name
        self.surface_number = surface_number
        self.surface_comment = surface_comment
        Surface.all_surfs[self.surface_number] = self
        self.surface = self
        super().__init__()
        self.enabled = True

    @property
    def surface_name(self):
        return self.__name__

    @staticmethod
    def get_all_surface_cards():
        outs = []
        for surf in Surface.all_surfs.values():
            if surf.enabled:
                outs.append(surf.surface_card)
        return '\n'.join(outs)

    @property
    @abstractmethod
    def surface_card(self):
        pass

    def delete_surface(self):
        raise NotImplementedError("Don't do this! Modify the features of an existing surface instead. ")

        # if self.surface_number in Surface.all_surfs.auto_picked_numbers:
        #     Surface.all_surfs.auto_picked_numbers.remove(self.surface_number)
        #
        # del Surface.all_surfs[self.surface_number]


class TRCL:
    def __init__(self, offset_vector=(0, 0, 0), rotation_theta=0., rotation_axis=(0, 0, 1), _round=3):
        """Creates TRCL cards for MCNP. rotation_theta is in degrees.
        MCNP will first rotate the cell, AND THEN apply the translate operation.
        """

        assert hasattr(offset_vector, '__iter__'), '`offset_vector` must be an iterator, not "{}"'.format(offset_vector)
        assert len(offset_vector) == 3, f'`offset_vector` length must be 3, not {len(offset_vector)}: "{offset_vector}"'
        assert isinstance(rotation_theta, Number), f'`rotation_theta` must be a number, not "{rotation_theta}"'
        assert hasattr(rotation_axis, '__iter__'), 'Invalid rotation_axis, {}'.format(rotation_axis)
        assert len(rotation_axis) == 3, 'Invalid rotation_axis, {}'.format(rotation_axis)
        self.offset_vector = np.array(offset_vector)
        if rotation_axis == (0, 0, 1) and rotation_theta == 0:
            self.rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        else:
            self.rotation_matrix = get_rotation_matrix(rotation_theta, *rotation_axis, _round=_round)

    def cell_form(self):
        return f'({self})'

    def card_form(self):
        assert False, 'Todo'

    def __str__(self) -> str:
        offset = ' '.join(map(str, self.offset_vector))
        matrix = '  '.join([row for row in [' '.join(map(str, a)) for a in self.rotation_matrix]])
        return '{}   {}'.format(offset, matrix)


class Cell(GeomSpecMixin):
    """
    Methods:
        cell_card: Property returning the cell card.
        surface_card: Property returning the surface card
        like_but: Create a new cell using MCNP's LIKE BUT feature.
    """
    all_cells: Dict[int, Cell] = MCNPNumberMapping('Cell', 10)

    @staticmethod
    def clear():
        Cell.all_cells = MCNPNumberMapping('Cell', 10)

    def delete_cell(self):
        raise NotImplementedError("No, bad idea. Don't delete Cells, modify the features of an existing cell instead."
                                  "You can do this in every case.")

    def enable(self):
        self.enabled = True
        try:
            self.surface.enabled = True  # for macros (sub classes of Cell) which create surfaces automatically.
        except AttributeError:
            pass

    def disable(self):
        self.enabled = False
        try:
            self.surface.enabled = False  # for macros (sub classes of Cell) which create surfaces automatically.
        except AttributeError:
            pass

    def __repr__(self):
        return f'Cell {self.cell_number}; {self.cell_card}'

    def __init__(self, material: Union[int, Material, PHITSOuterVoid] = 0,
                 geometry: Union[type(None), GeomSpecMixin, BinaryOperator, str] = None,
                 importance: Tuple[str, int] = None,
                 cell_number:  float = None,
                 cell_name: Union[str, None] = None,
                 cell_comment: Union[str, type(None)] = None,
                 cell_kwargs: Dict = None,
                 trcl: Union[None, TRCL] = None,
                 **kwargs):
        """
        MCNP cell assistant.
        Args:
            material: MCNP material number or Material instance
            geometry: Cell geom. spec. Can use Surfaces/Cells with operators ~, &, +, and -.
            importance: Cell importance. e.g. ("np", 1) -> neutron and photon importance = 1
            cell_number: Cell number. If None, automatically choose a cell number.
            cell_name: Name tagging for use in outpreader.py for looking up cells, surfaces, and tallies by name.
            cell_comment: Comment for cell
            cell_kwargs: Additional keyword arguments to be used in cell card, e.g. vol=1
            trcl: TRCL instance.
        """

        self.cell = self
        self.__name__ = cell_name
        self.cell_number = cell_number
        if importance is not None:
            assert hasattr(importance, '__iter__') and len(importance) == 2, "Format for specifying importance is, " \
                                                                             "for example, ('np', 1)"
            self.importance = [importance[0], importance[1]]
            self.importance[0] = self.importance[0].replace(' ', '')

            self.importance[0] = self.importance[0].replace(',', '')  # remove commas added by the user.
        else:
            self.importance = importance

        self.material = material
        if not isinstance(self.material, (int, Material, PHITSOuterVoid)):
            assert material in ['0', 0, None], "`material` argument must be either a Material instance, or, a" \
                                               " PHITSVoid instance or 0/None for void."

        self.__geometry__ = geometry
        if cell_kwargs is not None:
            assert 'trcl' not in cell_kwargs and 'TRCL' not in cell_kwargs,\
                'To use trcl, use pass a TRCL instance to the `trcl` argument'
            self.cell_kwargs = cell_kwargs
        else:
            self.cell_kwargs = {}
        self.cell_comment = cell_comment
        self.trcl = trcl
        if 'from_outp' not in kwargs:
            self.__from_outp__ = False  # For cells reconstructed from MCNP outp
            Cell.all_cells[cell_number] = self
        else:
            self.__from_outp__ = True
        GeomSpecMixin.__init__(self)
        self.__like_but_kwargs__ = {}
        self.__like_but_number__ = None

        if self.importance is None and not self.__from_outp__:
            warn(f'Importance of cell {self.cell_name if self.cell_name is not None else self.cell_number}'
                 f' is not specified')

        self.enabled = True

    @property
    def cell_mass(self):
        if isinstance(self.density, Number) and hasattr(self, "volume"):
            return self.density*self.volume
        else:
            return None

    def set_volume_kwarg(self, vol: float):
        """
        Set volume to be used internally by MCNP for this cell. Set to 1 to normalize volume out of
                tallies (e.g. to get an F4 tally_n in units of track length per source particle)

        Returns: None
        """
        assert 'vol' not in self.cell_kwargs, '"vol" keyword argument already in use for this cell!'
        self.cell_kwargs['VOL'] = str(vol)

    @staticmethod
    def get_all_cell_cards():
        """
        Returns:
            A string of all cell cards created in current session to be used in an MCNP input deck
        """
        outs = []
        for cell in sorted(Cell.all_cells.values(), key=lambda x: x.cell_number):
            if cell.enabled:
                outs.append(cell.cell_card)
        return '\n'.join(outs)

    @property
    def cell_name(self) -> str:
        return self.__name__

    def offset(self, offset_vector: Sized[float]):
        """
        Offsets the current cell. Does not create another cell! Use CuboidCell.like_but for that.
        Args:
            offset_vector: Vector defining the translation.

        Returns: None
        """

        assert isinstance(offset_vector, Iterable)
        assert hasattr(offset_vector, '__len__')
        assert len(offset_vector) == 3
        offset_vector = np.array(offset_vector)

        if self.trcl is None:
            self.trcl = TRCL(offset_vector)
        else:
            self.trcl.offset_vector += offset_vector

    @property
    def density(self):
        if "RHO" in self.__like_but_kwargs__:
            return self.__like_but_kwargs__["RHO"]
        else:
            return self.material.density

    def like_but(self, trcl: TRCL,
                 importance: Tuple[str, int] = None,
                 material: Union[int, Material] = None,
                 density: Union[float, type(None)] = None,
                 cell_number:  float = None,
                 cell_name: Union[str, None] = None,
                 cell_comment: Union[str, type(None)] = None,
                 ) -> Cell:
        """
        Create a new cell using MCNP's LIKE BUT feature. Leave any argument as None to leave it unchanged.
        Args:
            trcl:  An instance of the TRCL class. Translated and rotates cell.
            importance: new cell importance
            material: new cell material number or a JSB_tools.MCNP_helper.materials.Material instance.
            density:  new cell density
            cell_number:  new cell number (leave None for automatic)
            cell_name:  new cell name
            cell_comment:  New cell's comment

        Returns:
            Cell: new cell identical to original except for some select changes.

        """
        like_but_kwargs = {}
        if importance is density is material is trcl is None:
            assert False, "Like but used, but no changes were made (all args were None)"

        if material is None:
            material = self.material
        else:
            if isinstance(material, Material):
                density = material.density
                like_but_kwargs['MAT'] = material.mat_number
            else:
                assert isinstance(material, (str, int))
                if isinstance(material, str):
                    material = int(material)
                like_but_kwargs['MAT'] = material

        if density is not None:
            like_but_kwargs['RHO'] = density
            assert isinstance(material, Material), 'Attempted to change density using "like but" on a void cell.'
            if density > 0:
                warn('Positive densities in MCNP are interpreted as atoms per barn cm. Use negative number for grams'
                     ' per cm3')

        if importance is None:
            importance = self.importance
        else:
            like_but_kwargs['IMP'] = self.__get_imp_str__(importance)

        new_cell = Cell(importance=importance, material=material, geometry=None,
                        cell_number=cell_number, cell_name=cell_name, cell_comment=cell_comment)
        new_cell.__like_but_kwargs__ = like_but_kwargs
        new_cell.__like_but_number__ = self.cell_number
        assert isinstance(trcl, TRCL)
        new_cell.trcl = trcl
        return new_cell

    def __build_like_but_cell_card__(self):
        assert self.__like_but_number__ is not None
        assert len(self.__like_but_kwargs__) != 0
        kwargs = ['{} = {}'.format(k, v) for k, v in self.__like_but_kwargs__.items()]
        if self.trcl is not None:  # todo: this is ugly. Incorporate it differently
            kwargs += ['TRCL = {}'.format(self.trcl.cell_form())]
        kwargs = ' '.join(kwargs)
        return f'{self.cell_number} LIKE {self.__like_but_number__} BUT {kwargs}'

    @staticmethod
    def __get_imp_str__(imp=None):
        if imp is None:
            return ''
        assert hasattr(imp, '__iter__') and len(imp) == 2 and isinstance(imp[0], str), \
            'Importance, "{}", not specified properly. Correct example: ("np", 1)'.format(imp)

        return 'imp:{}={}'.format(','.join(imp[0]), int(imp[1]))

    @property
    def geometry(self):
        if self.__geometry__ is None:
            if isinstance(self, Surface):
                return -self
            else:
                warn(f'No geometry for cell, {self.__name__ if self.__name__ is not None else self.cell_number}'
                     ', returning None')
                return None
        else:
            return self.__geometry__

    @geometry.setter
    def geometry(self, value):
        self.__geometry__ = value

    def __build_cell_card__(self):
        if isinstance(self.material, PHITSOuterVoid):
            out = f'{self.cell_number} -1'
        elif isinstance(self.material, (int, type(None))):
            out = f'{self.cell_number} 0'
        else:
            out = f'{self.cell_number} {self.material.mat_number} -{abs(self.material.density)}'

        imp = self.__get_imp_str__(self.importance)

        out += f' {self.geometry} {imp}'
        for key, arg in self.cell_kwargs.items():
            out += ' {}={}'.format(key, arg)
        if self.trcl is not None:  # todo: this is ugly. Incorporate it differently
            out += ' TRCL = {} '.format(self.trcl.cell_form())
        comment = get_comment(self.cell_comment, self.cell_name)
        return out + comment

    @property
    def cell_card(self):
        if len(self.__like_but_kwargs__) == 0:
            return self.__build_cell_card__()
        else:
            return self.__build_like_but_cell_card__()

    def __str__(self):
        return self.cell_card


class CellGroup:
    def __init__(self, *cells: Union[Cell, Surface]):
        self.cells = list(cells)

    def disable(self):
        for c in self.cells:
            c.enabled = False

    def enable(self):
        for c in self.cells:
            c.enabled = True

    def add_cells(self, *cells):
        for x in cells:
            assert isinstance(x, (Cell, Surface)),\
                'Only Cell/Surface instances can be used in CellGroup, not "{}"'.format(type(x))
            number = x.surface_number if isinstance(x, Surface) else x.cell_number
            name = x.surface_name if isinstance(x, Surface) else x.cell_name
            assert x not in self.cells, 'Cell (#{}, name: {} added twice to cell group.'.format(number, name)
        self.cells.extend(cells)

    def remove_cell(self, cell_obj: Union[Cell, Surface]):
        self.cells.remove(cell_obj)

    def __invert__(self):
        geom: Union[Cell, Surface] = None
        assert len(self.cells) > 0, 'No cells in group!'

        def f(cells):
            nonlocal geom
            if len(cells) == 0:
                return
            else:
                cell = cells[0]
                if isinstance(cell, Surface):
                    additional_geom = +cell.surface
                elif isinstance(cell, Cell):
                    additional_geom = ~cell
                else:
                    assert False
                if geom is None:
                    geom = additional_geom
                else:
                    geom = geom & additional_geom
                f(cells[1:])

        f(self.cells)
        return geom


if __name__ == '__main__':
    p = '/Users/burggraf1/PycharmProjects/PHELIX/IAC/2021Sim/0_inp/outp'
    p = Cell.from_outp(p, cell_name='    Convertor, 0.3 cm       ')
    print(p)
