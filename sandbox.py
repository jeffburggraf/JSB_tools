from JSB_tools.MCNP_helper.materials import Material

for percent in [0.1, 0.2, 0.3, 0.4, 0.4, 0.6, 0.7, 0.8, 0.9, 1]:
    mat = Material.gas(['He', 'Ar'], atom_fractions=[1-percent, percent], )