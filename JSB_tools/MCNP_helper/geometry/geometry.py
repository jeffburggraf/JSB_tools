from __future__ import annotations
from JSB_tools.MCNP_helper.geometry.__init__ import Cell, Surface, TRCL
from JSB_tools.MCNP_helper.geometry.__init__ import get_comment
from typing import Union, List, Dict, Tuple, Sized, Iterable

"""
For the creation of surfaces and cells in MCNP. Cell/surface numbers can be managed automatically, 
or manually specified.
"""


class CuboidSurface(Surface):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, surf_name=None, surf_num=None, comment=None):
        super(CuboidSurface, self).__init__(surface_number=surf_num, surface_name=surf_name, surface_comment=comment)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        for kmin, kmax in zip([xmin, ymin, zmin], [xmax, ymax, zmax]):
            assert kmin < kmax, 'Cuboid with minimum coordinate >= maximum coordinate: {} is >= {}'.format(kmax, kmax)

    @property
    def volume(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)

    @property
    def surface_card(self):
        comment = get_comment(self.surface_comment, self.surface_name)
        return '{0} RPP {xmin} {xmax}  {ymin} {ymax}  {zmin} {zmax} {comment}' \
            .format(self.surface_number, xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, zmin=self.zmin,
                    zmax=self.zmax, comment=comment)


class CuboidCell(Cell, CuboidSurface):
    """
    Examples:
        c1 = CuboidCell(0, 1, 0, 1, 0, 1, importance=('np', 1))  # A void cuboid cell from (0,0,0) to (1,1,1)

        c1.cell_card  # string of the cell card

        c1.surface_card  # string of the surface card


        # You can also specify density and material

        c2 = CuboidCell(0, 1, 0, 1, 0, 1, importance=('np', 1), density=1, material=1000)

    """
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax,
                 importance: Tuple[str, int], material: int = 0,
                 density: float = None, cell_name: str = None,
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
            importance: Cell importance. e.g. ("np", 1) -> neutron and photon importance = 1
            material: MCNP material number
            density:  Density in g/cm3
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
             importance=importance,
             material=material,
             density=density,
             cell_number=cell_num,
             cell_name=cell_name,
             cell_comment=cell_comment,
             cell_kwargs=cell_kwargs
             )
        super(Cell, self).__init__(xmin, xmax, ymin, ymax, zmin, zmax, surf_name=surf_name, surf_num=surf_number,
                                   comment=surf_comment)

        self.geometry = -self

    def copy(self, new_importance: Union[Tuple[int, str], type(None)] = None, new_material=None, new_density=None,
             new_cell_name=None, new_cell_num=None, new_cell_comment=None) -> CuboidCell:
        pass

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

        self.cell_kwargs['trcl'] = '({} {} {})'.format(*offset_vector)

    @property
    def volume(self):
        return self.volume

    @property
    def cell_card(self):
        out = super(CuboidCell, self).cell_card
        return out


c1 = CuboidCell(0, 1, 0, 1, 0, 1, ('np', 1), cell_num=10000, surf_comment='omg', surf_name='i am a surface ane')
c2 = CuboidCell(0, 1, 0, 1, 0, 1, ('np', 1), cell_num=2)
c3 = c2.like_but(trcl=TRCL([1,1,0], rotation_theta = 180))

c99 = CuboidCell(0, 1, 0, 1, 0, 1, ('np', 1))

# print(c1.surface_card)
# print(c1.cell_card)

print(-0 == 0)
print(c3.cell_card)
print(c1 | ~(c2 & (-c99 | c1)), 'ggg ')
print(1.0==1)