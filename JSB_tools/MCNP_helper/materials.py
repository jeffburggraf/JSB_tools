from JSB_tools.MCNP_helper.geometry.geom_core import MCNPNumberMapping
from typing import Dict, Union
from JSB_tools.nuke_data_tools import Nuclide

class Material:
    all_materials = MCNPNumberMapping('Material', 1000, 1000)

    def __init__(self, density: float, mat_number: int = None, name: str = None, mat_kwargs: Dict[str, str] = None):
        self.mat_number = mat_number
        self.__name__ = name
        Material.all_materials[self.mat_number] = self
        self.density = density
        self.__zaids = []
        self.__zaid_proportions = []

        if mat_kwargs is None:
            self.mat_kwargs = {}
        else:
            self.mat_kwargs = mat_kwargs

        self.is_weight_fraction = None

    def add_zaid(self, zaid_or_nuclide: Union[Nuclide, int], fraction, is_weight_fraction=False):
        if self.is_weight_fraction is None:
            self.is_weight_fraction = is_weight_fraction
        if isinstance(zaid_or_nuclide, int):
            pass
        elif isinstance(zaid_or_nuclide, Nuclide):
            zaid_or_nuclide = 1000*zaid_or_nuclide.Z + zaid_or_nuclide.A
        else:
            assert False, 'Incorrect type, "{}", passed in `zaid_or_nuclide` argument.'.format(type(zaid_or_nuclide))

        self.__zaids.append(zaid_or_nuclide)
        self.__zaid_proportions.append(fraction)

    @property
    def mat_card(self, mat_kwargs=None) -> str:
        if mat_kwargs is not None:
            self.mat_kwargs.update(mat_kwargs)

        outs = ['M{}  $ density = {}'.format(self.mat_number, self.density)]
        for n, zaid in zip(self.__zaid_proportions, self.__zaids):
            outs.append('     {} {}'.format(zaid, '-{}'.format(n) if self.is_weight_fraction else n))
        outs.extend(["     {} = {}".format(k, v) for k, v in self.mat_kwargs.items()])
        return '\n'.join(outs)

    @property
    def name(self):
        return self.__name__


class Uranium(Material):
    pass


class Tungsten(Material):
    pass