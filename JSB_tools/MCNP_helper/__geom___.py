from __future__ import annotations
from JSB_tools.MCNP_helper.__geom_helpers__ import MCNPNumberMapping, GeomSpecMixin, BinaryOperator, get_comment
from typing import Union, List, Dict, Tuple, Sized, Iterable
from abc import ABC, abstractmethod

class Surface(ABC, GeomSpecMixin):
    all_surfs = MCNPNumberMapping('Surface', 1)

    def __init__(self, surface_number: float = None,
                 surface_name: Union[str, None] = None,
                 surface_comment: Union[str, None] = None):
        self.__name__ = surface_name
        self.__number__ = surface_number
        self.surface_comment = surface_comment
        Surface.all_surfs[self.__number__] = self

        super().__init__()

    @property
    def surface_name(self):
        return self.__name__

    @property
    def surface_number(self):
        return self.__number__

    @abstractmethod
    def surface_card(self):
        pass




class Cell(GeomSpecMixin):
    all_cells = MCNPNumberMapping('Cell', 10)

    def __init__(self, importance: Tuple[str, int],
                 material: int = 0,
                 density: Union[float, type(None)] = None,
                 geometry: Union[type(None), GeomSpecMixin, BinaryOperator, str] = None,
                 cell_number:  float = None,
                 cell_name: Union[str, None] = None,
                 cell_comment: Union[str, type(None)] = None,
                 mcnp_kwargs: Dict = {}):
        self.__name__ = cell_name
        self.__number__ = cell_number
        self.importance = importance
        self.material = material
        self.geometry = None
        self.density = density
        self.geometry = geometry
        self.mcnp_kwargs = mcnp_kwargs
        self.cell_comment = cell_comment
        Cell.all_cells[cell_number] = self
        GeomSpecMixin.__init__(self)

    @property
    def cell_number(self):
        return self.__number__

    @property
    def cell_name(self):
        return self.__name__

    @property
    def cell_card(self):
        if self.density is None or self.density == 0:
            assert (self.material in [0, None]), \
                '`density` was not specified for cell with {} {}, therefore `material` must be ' \
                'specified as 0, or left as the default (None)'\
                .format('name' if self.cell_name is not None else 'number',
                        self.cell_name if self.cell_name is not None else self.cell_number)
            material = 0
            if self.density == 0:
                self.density = None
        else:
            assert self.material is not None, 'In cell {}, a density was specified but a material was not! This is not'\
                                              ' allowed in MCNP'.format(self)
            assert self.material != 0, 'Material of "0" not allowed for non-void cells'
            material = self.material

        out = '{} {}'.format(self.cell_number, material)
        if self.density is not None:
            out += ' {}'.format(self.density)
        imp = 'imp:{}={}'.format(','.join(self.importance[0]), int(self.importance[1]))
        # if not from_str:
        #     assert self.geometry is not None,\
        #         'Attempted to access the cell card of a cell for which geometry\nhas not been specified!'
        out += ' {} {}'.format(self.geometry, imp)
        for key, arg in self.mcnp_kwargs.items():
            out += ' {}={}'.format(key, arg)
        comment = get_comment(self.cell_comment, self.cell_name)
        return out + comment

    def set_geometry(self, geom: Union[str, GeomSpecMixin, BinaryOperator]):
        self.geometry = str(geom)

    def __str__(self):
        return self.cell_card





c1 = CuboidCell.from_coordinates(0,1,0,1, 0,1, ('np', 1))
c2 = CuboidCell.from_coordinates(0,1,0,1, 0,1, ('np', 1))
print(-c1)



