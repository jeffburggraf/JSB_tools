"""
Quickly generate MCNP input decks of sphere of material with a particle source at center.
"""
from JSB_tools.MCNP_helper import InputDeck, Cell, Surface
from JSB_tools.MCNP_helper.geometry.primitives import SphereCell
from JSB_tools.MCNP_helper.materials import Material
from pathlib import Path
import numpy as np


class AttenuationResult:
    def __init__(self):
        pass


def gamma_attenuation(material: Material, max_radius, max_erg, min_erg=0, erg_bin_width=1, par='p'):
    """
    All energies are in KeV
    Todo
    Args:
        material:
        max_radius:
        max_erg:
        min_erg:
        erg_bin_width:
        par:

    Returns:

    """
    pass


mat = Material(3.5, )
mat.add_zaid(6013, 1)
par = 'p'
nps = 1E5
layers = 1
max_r = 1
erg_bin_width = 10
max_erg = 2
min_erg = .7

ergs = np.arange(min_erg, max_erg + erg_bin_width, erg_bin_width)

prev_cell = None
for r in np.linspace(0, max_r, layers+1)[1:]:
    cell = SphereCell(r, material=mat, importance=(par, 1))

    if prev_cell is None:
        pass
    else:
        cell.geometry = +prev_cell & -cell

    prev_cell = cell

Cell(geometry=+cell, importance=(par, 0))


i = InputDeck.mcnp_input_deck(Path(__file__).parent/'attenuation.inp')
i.write_inp_in_scope(globals())