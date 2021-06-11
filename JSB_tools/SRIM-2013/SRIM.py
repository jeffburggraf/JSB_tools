from plasmapy import particles
from astropy.units import quantity
from srim import SR, Ion, Layer, Material, Element
from pathlib import Path
from mendeleev import element, Isotope
import re

# Construct a 3MeV Nickel ion
# ion = Ion('Ni', energy=3.0e6)

# element()


def run_srim(target_atoms, fractions, density, projectile, max_erg, gas=False):
    """
    Parameters:
        target_atoms: List of element symbols, e.g. ["Ar"]
        fractions: List of atom fractions
        density: density in g/cm3
        projectile: Full symbol of projectile, e.g. "Xe139"
        max_erg: Energy in MeV
        gas: True is gas, else False
    """
    assert len(target_atoms) == len(fractions)

    m = re.match('([A-Za-z]{1,3})-*([0-9]+)*', projectile)
    assert m
    proj_symbol = m.groups()[0]
    a = m.groups()[1]
    proj_mass = particles.Particle(f"{proj_symbol}-{a}").mass/quantity.Quantity(1.66053906660E-27, unit='kg')
    layer_arg = {}

    for s, frac in zip(target_atoms, fractions):
        layer_arg[s] = {"stoich": frac}

    sr = SR(Layer(layer_arg, density, 1000, phase=int(gas)), Ion(proj_symbol, max_erg, proj_mass), output_type=5)
    sr.run(Path(__file__).parent)


run_srim(['Ar', 'He'], [1, 1], 0.00126, 'Xe139', 100E6, gas=True)
