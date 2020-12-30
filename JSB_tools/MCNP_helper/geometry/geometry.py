from __future__ import annotations
from JSB_tools.MCNP_helper.geometry.__init__ import Cell, Surface
from JSB_tools.MCNP_helper.geometry.__init__ import get_comment
from typing import Union, List, Dict, Tuple, Sized, Iterable





# class LikeButCell(Cell):
#     def like_but(self, trcl: TRCL = None, material=None, density=None, geometry=None, importance=None):
#         'RHO -> density'
#         if density is None:


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
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax,
                 importance: Tuple[str, int], material=0,
                 density=None, cell_name=None,
                 cell_num=None, cell_comment=None,
                 surf_name=None, surf_number=None,
                 surf_comment=None, cell_kwargs=None):
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
             mcnp_kwargs=cell_kwargs
             )
        super(Cell, self).__init__(xmin, xmax, ymin, ymax, zmin, zmax, surf_name=surf_name, surf_num=surf_number,
                                   comment=surf_comment)

        self.geometry = -self

    def copy(self, new_importance: Union[Tuple[int, str], type(None)] = None, new_material=None, new_density=None,
             new_cell_name=None, new_cell_num=None, new_cell_comment=None) -> CuboidCell:
        pass

    def offset(self, offset_vector: Sized[float]):
        #  Todo: use like but here (if all changes are none)!

        assert isinstance(offset_vector, Iterable)
        assert hasattr(offset_vector, '__len__')
        assert len(offset_vector) == 3

        self.mcnp_kwargs['trcl'] = '({} {} {})'.format(*offset_vector)

    @property
    def volume(self):
        return self.volume

    @property
    def cell_card(self):
        out = super(CuboidCell, self).cell_card
        return out

c1 = CuboidCell(0,1, 0,1, 0,1, ('np', 1), cell_num=10000, surf_comment='omg', surf_name='i am a surface ane')
c2 = CuboidCell(0,1, 0,1, 0,1, ('np', 1), cell_num=2)
c99 = CuboidCell(0,1, 0,1, 0,1, ('np', 1))

print(c1.surface_card)
print(c1.cell_card)
print(c1 | ~(c2 & (-c99 | c1)), 'ggg ')