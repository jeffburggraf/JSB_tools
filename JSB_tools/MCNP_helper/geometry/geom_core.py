from __future__ import annotations
from typing import Union, List, Dict, Tuple, Sized, Iterable, Type
from abc import ABC, abstractmethod
from JSB_tools.MCNP_helper.geometry.__geom_helpers__ import get_rotation_matrix, GeomSpecMixin, BinaryOperator,\
    get_comment
from warnings import warn
from numbers import Number
import numpy as np
from JSB_tools.MCNP_helper.materials import Material
from JSB_tools.MCNP_helper.geometry.__geom_helpers__ import MCNPNumberMapping
from JSB_tools.MCNP_helper.input_deck import NDIGITS


class Surface(ABC, GeomSpecMixin):
    all_surfs = MCNPNumberMapping('Surface', 1)

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
        super().__init__()

    @property
    def surface_name(self):
        return self.__name__

    @staticmethod
    def get_all_surface_cards():
        outs = []
        for surf in Surface.all_surfs.values():
            outs.append(surf.surface_card)
        return '\n'.join(outs)

    @abstractmethod
    def surface_card(self):
        pass


class Tally:
    all_f4_tallies = MCNPNumberMapping("F4Tally", 1)

    @staticmethod
    def clear():
        Tally.all_f4_tallies = MCNPNumberMapping("F4Tally", 1)

    @staticmethod
    def get_all_tally_cards():
        outs = []
        for tally in Tally.all_f4_tallies.values():
            outs.append(tally.tally_card)
        return '\n'.join(outs)


class F4Tally(Tally):
    def __init__(self, cell: Cell, particle: str, tally_number=None, tally_name=None, tally_comment=None):
        """
        Args:
            cell: Cell instance for which the tally will be applied
            particle: MCNP particle designator
            tally_number: Must end in a 4. Or, just leave as None and let the code pick for you
            tally_name:  Used in JSB_tools.outp_reader to fetch tallies by name.
            tally_comment:
        """
        self.cell = cell
        assert isinstance(particle, str), 'Particle argument must be a string, e.g. "n", or, "p", or, "h"'
        self.erg_bins_array = None
        self.tally_number = tally_number
        if self.tally_number is not None:
            assert str(self.tally_number)[-1] == '4', 'F4 tally number must end in a "4"'
        self.__name__ = tally_name
        self.tally_comment = tally_comment
        Tally.all_f4_tallies[self.tally_number] = self
        self.__modifiers__ = []

        assert isinstance(particle, str), '`particle` argument must be a string.'
        particle = particle.lower()
        for key, value in {'photons': 'p', 'protons': 'h', 'electrons': 'e', 'neutrons': 'n',
                           'alphas': 'a'}.items():
            if particle == key or particle == key[:-1]:
                particle = value
                break

        self.particle = particle

    @property
    def name(self):
        return self.__name__

    def add_fission_rate_multiplier(self, mat: int) -> None:
        """
        Makes this tally a fission rate tally [fissions/(src particle)/cm3] for neutrons and protons
        Args:
            mat: Material number

        Returns: None

        """
        mod = 'FM{} -1 {mat} -2'.format(self.mcnp_tally_number, mat=mat)
        self.__modifiers__.append(mod)

    @property
    def mcnp_tally_number(self):
        return int(str(self.tally_number) + '4')

    def set_erg_bins(self, erg_min=None, erg_max=None, n_erg_bins=None, erg_bins_array=None, _round=3):
        if erg_bins_array is not None:
            assert hasattr(erg_bins_array, '__iter__')
            assert len(erg_bins_array) >= 2
            assert erg_min == erg_max == n_erg_bins, 'Can either specify energy bins by an array of values, or by ' \
                                                     'erg_min, erg_max, and the number of bins, n_erg_bins.'
            self.erg_bins_array = erg_bins_array

        else:
            assert erg_bins_array is None, "Can't specify bins by using an array and a min, max, and number of bins." \
                                           "Set erg_bins_array to None."
            self.erg_bins_array = np.linspace(erg_min, erg_max, n_erg_bins)
        self.erg_bins_array = np.array([round(x, _round) for x in self.erg_bins_array])

    @property
    def tally_card(self):
        comment = get_comment(self.tally_comment, self.__name__)
        out = 'F{num}:{par} {cell} {comment}'.format(num=self.mcnp_tally_number, par=self.particle,
                                                     cell=self.cell.cell_number, comment=comment)
        if self.erg_bins_array is not None:
            out += '\nE{num} {bins}'.format(num=self.mcnp_tally_number, bins=' '.join(map(str, self.erg_bins_array)))

        out += '\n'.join(self.__modifiers__)
        return out


class TRCL:
    def __init__(self, offset_vector=(0, 0, 0), rotation_theta=0., rotation_axis=(0, 0, 1), _round=3):
        """Creates TRCL cards for MCNP. rotation_theta is in degrees."""
        assert hasattr(offset_vector, '__iter__'), 'Invalid offset vector, {}'.format(offset_vector)
        assert len(offset_vector) == 3, 'Invalid offset vector, {}'.format(offset_vector)
        assert isinstance(rotation_theta, Number), 'Invalid rotation_theta, {}'.format(rotation_theta)
        assert hasattr(rotation_axis, '__iter__'), 'Invalid rotation_axis, {}'.format(rotation_axis)
        assert len(rotation_axis) == 3, 'Invalid rotation_axis, {}'.format(rotation_axis)
        self.offset_vector = np.array(offset_vector)
        if rotation_axis == (0, 0, 1) and rotation_theta == 0:
            self.rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        else:
            self.rotation_matrix = get_rotation_matrix(rotation_theta, *rotation_axis, _round=_round)

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
    all_cells = MCNPNumberMapping('Cell', 10)

    @staticmethod
    def clear():
        Cell.all_cells = MCNPNumberMapping('Cell', 10)

    def __init__(self, material: Union[int, Material] = 0,
                 density: Union[float, type(None)] = None,
                 geometry: Union[type(None), GeomSpecMixin, BinaryOperator, str] = None,
                 importance: Tuple[str, int] = None,
                 cell_number:  float = None,
                 cell_name: Union[str, None] = None,
                 cell_comment: Union[str, type(None)] = None,
                 cell_kwargs: Dict = None,
                 trcl: Union[None, TRCL] = None):
        """
        Args:
            importance: Cell importance. e.g. ("np", 1) -> neutron and photon importance = 1
            material: MCNP material number
            density:  Density in g/cm3
            cell_name:  Name tagging for use in outpreader.py for looking up cells, surfaces, and tallies by name.
            cell_number: Cell number. If None, automatically choose a cell number.
            cell_comment: Comment for cell
            cell_kwargs:  Additional keyword arguments to be used in cell card, i.g. vol=1
            trcl: TRCL instance.
        """
        self.__name__ = cell_name
        self.cell_number = cell_number
        if importance is not None:
            assert hasattr(importance, '__iter__') and len(importance) == 2, "Format for specifying importance is, " \
                                                                             "for example, ('np', 1)"
            self.importance = importance[0].replace(',', ''), importance[1]  # remove commas added by the user.
        else:
            self.importance = importance

        self.material = material
        self.geometry = None
        if isinstance(self.material, Material):
            if density is not None:
                warn('Material instances passed to `material` argument along with a `density`.\n'
                     'Using density from the Material object, and ignoring the `density` argument.')
            self.density = -abs(material.density)
        else:
            self.density = density
        if self.density is not None and self.density > 0:
            warn('Positive densities in MCNP are interpreted as atoms per barn cm. Use negative number for grams'
                 'per cm3')

        self.geometry = geometry
        if cell_kwargs is not None:
            assert 'trcl' not in cell_kwargs and 'TRCL' not in cell_kwargs,\
                'To use trcl, use pass a TRCL instance to the `trcl` argument'
            self.cell_kwargs = cell_kwargs
        else:
            self.cell_kwargs = {}
        self.cell_comment = cell_comment
        self.trcl = trcl
        Cell.all_cells[cell_number] = self
        GeomSpecMixin.__init__(self)
        self.__like_but_kwargs__ = {}
        self.__like_but_number__ = None

    def set_volume_kwarg(self, vol: float):
        """
        Set volume to be used internally by MCNP for this cell. Set to 1 to normalize volume out of
                tallies (e.g. to get an F4 tally in units of track length per source particle)

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

        if density is None:
            density = self.density
        else:
            like_but_kwargs['RHO'] = density
            if density > 0:
                warn('Positive densities in MCNP are interpreted as atoms per barn cm. Use negative number for grams'
                     'per cm3')

        if importance is None:
            importance = self.importance
        else:
            like_but_kwargs['IMP'] = self.__get_imp_str__(importance)

        new_cell = Cell(importance=importance, material=material, density=density, geometry=None,
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
            kwargs += ['TRCL = {}'.format(self.trcl)]
        kwargs = ' '.join(kwargs)
        return 'LIKE {} BUT {}'.format(self.__like_but_number__, kwargs)

    def __get_imp_str__(self, imp=None):

        if imp is None:
            imp = self.importance
        if imp is None:
            return ''
        assert hasattr(imp, '__iter__') and len(imp) == 2 and isinstance(imp[0], str), \
            'Importance, "{}", not specified properly. Correct example: ("np", 1)'.format(imp)

        return 'imp:{}={}'.format(','.join(imp[0]), int(imp[1]))

    def __build_cell_card__(self):
        if isinstance(self.material, int):
            mat = self.material
        elif isinstance(self.material, Material):
            mat = self.material.mat_number
        else:
            assert False, 'Invalid material type, {}'.format(type(self.material))

        out = '{} {}'.format(self.cell_number, mat)

        if self.density is not None:
            out += ' {}'.format(self.density)
        imp = self.__get_imp_str__()
        if self.geometry is None:
            warn('\nCell card of cell {} was accessed with out specifying geometry.'
                 .format(self.cell_name if self.cell_name is not None else self.cell_number))
        out += ' {} {}'.format(self.geometry, imp)
        for key, arg in self.cell_kwargs.items():
            out += ' {}={}'.format(key, arg)
        if self.trcl is not None:  # todo: this is ugly. Incorporate it differently
            out += ' TRCL = {} '.format(self.trcl)
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
        self.geometry = geom
        # if isinstance(geom, (GeomSpecMixin, BinaryOperator)):
        #     self.geometry = geom.__to_str__()
        # else:
        #     self.geometry = str(geom)
        # return self.geometry

    def __str__(self):
        return self.cell_card


class CellGroup:
    def __init__(self, *cells: Cell):
        self.cells = list(cells)

    def add_cells(self, *cells):
        for x in cells:
            assert isinstance(x, Cell), 'Only Cell instances can be used in CellGroup, not "{}"'.format(type(x))
            assert x not in self.cells, 'Cell (#{}, name: {} added twice to cell group.'.format(x.cell_number,
                                                                                                x.cell_name)
        self.cells.extend(cells)

    def remove_cell(self, cell_obj: Cell):
        self.cells.remove(cell_obj)

    def __invert__(self):
        geom = None
        assert len(self.cells) > 0, 'No cells in group!'

        def f(cells):
            nonlocal geom
            if len(cells) == 0:
                return
            else:
                cell = cells[0]
                if geom is None:
                    geom = ~cell
                else:
                    geom = geom & ~cell
                f(cells[1:])

        f(self.cells)
        return geom


class CellGroup:
    def __init__(self, *cells: Union[Cell, Surface]):
        self.cells = list(cells)

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
        geom = None
        assert len(self.cells) > 0, 'No cells in group!'

        def f(cells):
            nonlocal geom
            if len(cells) == 0:
                return
            else:
                cell = cells[0]
                if isinstance(cell, Surface):
                    additional_geom = +cell
                else:
                    additional_geom = ~cell
                if geom is None:
                    geom = additional_geom
                else:
                    geom = geom & additional_geom
                f(cells[1:])

        f(self.cells)
        return geom
