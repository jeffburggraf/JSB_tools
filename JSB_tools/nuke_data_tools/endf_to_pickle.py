from __future__ import annotations
import pickle
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER, FissionProductYields
from openmc.data import Reaction, Decay
from pathlib import Path
import re
from typing import Dict, List
from JSB_tools.nuke_data_tools import NUCLIDE_INSTANCES, Nuclide, DECAY_PICKLE_DIR, PHOTON_PICKLE_DIR,\
    PROTON_PICKLE_DIR,CrossSection1D, ActivationReactionContainer, SF_YIELD_PICKLE_DIR
from warnings import warn
cwd = Path(__file__).parent


parent_data_dir = cwd / 'endf_files'
decay_data_dir = parent_data_dir / 'decay'
proton_padf_data_dir = parent_data_dir / 'PADF_2007' / 'Files'
proton_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_protons'
photon_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_gammas'
#  Download SF yields from https://www.cenbg.in2p3.fr/GEFY-GEF-based-fission-fragment,780
sf_yield_data_dir = parent_data_dir / 'gefy81_s'
neutron_fission_yield_data_dir = parent_data_dir / 'gefy81_n'


def pickle_decay_data():
    directory = decay_data_dir  # Path to downloaded ENDF decay data
    assert directory.exists()
    for file_path in directory.iterdir():
        file_name = file_path.name
        _m = re.match(r"dec-[0-9]{3}_(?P<S>[A-Za-z]{1,2})_(?P<A>[0-9]+)(?:m(?P<M>[0-9]+))?\.endf", file_name)
        if _m:
            a = int(_m.group("A"))
            _s = _m.group("S")  # nuclide symbol, e.g. Cl, Xe, Ar
            m = _m.group("M")
            if m is not None:
                m = int(m)
            parent_nuclide_name = "{0}{1}{2}".format(_s, a, "" if m is None else "_m{0}".format(m))
        else:
            continue

        if parent_nuclide_name in NUCLIDE_INSTANCES:
            parent_nuclide = NUCLIDE_INSTANCES[parent_nuclide_name]
        else:
            parent_nuclide = Nuclide(parent_nuclide_name, __internal__=True)
            NUCLIDE_INSTANCES[parent_nuclide_name] = parent_nuclide

        openmc_decay = Decay(Evaluation(file_path))
        daughter_names = [mode.daughter for mode in openmc_decay.modes]
        for daughter_nuclide_name in daughter_names:
            if daughter_nuclide_name in NUCLIDE_INSTANCES:
                daughter_nuclide = NUCLIDE_INSTANCES[daughter_nuclide_name]
            else:
                daughter_nuclide = Nuclide(daughter_nuclide_name, __internal__=True)
                NUCLIDE_INSTANCES[daughter_nuclide_name] = daughter_nuclide

            if daughter_nuclide_name != parent_nuclide_name:
                daughter_nuclide.__decay_parents_str__.append(parent_nuclide_name)
                parent_nuclide.__decay_daughters_str__.append(daughter_nuclide_name)

        print("Preparing data for {0}".format(parent_nuclide_name))
        parent_nuclide.__set_data_from_open_mc__(openmc_decay)

    for nuclide_name in NUCLIDE_INSTANCES.keys():
        with open(DECAY_PICKLE_DIR/(nuclide_name + '.pickle'), "wb") as pickle_file:
            print("Writing data for {0}".format(nuclide_name))
            pickle.dump(NUCLIDE_INSTANCES[nuclide_name], pickle_file)

    with open(DECAY_PICKLE_DIR/'quick_nuclide_lookup.pickle', 'wb') as f:
        data = {}
        for name, nuclide in NUCLIDE_INSTANCES.items():
            key = nuclide.A, nuclide.Z, nuclide.half_life
            data[key] = name
        pickle.dump(data, f)


# modularize the patch work of reading PADF and ENDF-B-VIII.0_protons data.
class ProtonENDFFile:
    def __init__(self, padf_directory, endf_b_directory):
        self.nuclide_name_and_file_path = {}

        for path in Path(padf_directory).iterdir():
            f_name = path.name
            _m = re.match("([0-9]{4,6})(?:M([0-9]))?", f_name)
            if _m:
                zaid = int(_m.groups()[0])
                if _m.groups()[1] is not None:
                    isometric_state = int(_m.groups()[1])
                else:
                    isometric_state = 0
                z = zaid // 1000
                a = zaid % 1000
                nuclide_name = self.get_name_from_z_a_m(z, a, isometric_state)
                self.nuclide_name_and_file_path[nuclide_name] = path

        for path in Path(endf_b_directory).iterdir():
            f_name = path.name
            _m = re.match(r"p-([0-9]+)_([A-Za-z]{1,2})_([0-9]+)\.endf", f_name)
            if _m:
                z = _m.groups()[0]
                a = _m.groups()[2]
                nuclide_name = self.get_name_from_z_a_m(z, a, 0)
                self.nuclide_name_and_file_path[nuclide_name] = path

    @staticmethod
    def get_name_from_z_a_m(z, a, m):
        z, a, m = map(int, [z, a, m])
        if z == 0:
            symbol = "Nn"
        else:
            symbol = ATOMIC_SYMBOL[z]
        symbol += str(a)
        if m != 0:
            symbol += "_m{0}".format(m)
        return symbol


def pickle_proton_activation_data():
    assert PROTON_PICKLE_DIR.exists()
    all_reactions = {}
    files = ProtonENDFFile(padf_directory=proton_padf_data_dir, endf_b_directory=proton_enfd_b_data_dir)

    for nuclide_name, f_path in files.nuclide_name_and_file_path.items():
        ActivationReactionContainer.set(all_reactions, nuclide_name, f_path, 'proton')
        # print('Reading data from {}'.format(nuclide_name))
        # if nuclide_name in all_reactions:
        #     reaction = all_reactions[nuclide_name]
        # else:
        #     reaction = ActivationReactionContainer(nuclide_name)
        #     all_reactions[nuclide_name] = reaction
        #
        # e = Evaluation(f_path)
        # for heavy_product in Reaction.from_endf(e, 5).products:
        #     heavy_product_name = heavy_product.particle
        #     if heavy_product_name == "photon":
        #         continue
        #     if heavy_product_name == "neutron":
        #         heavy_product_name = "Nn1"
        #     xs_fig_label = "{0}(p,X){1}".format(nuclide_name, heavy_product_name)
        #     xs = CrossSection1D(heavy_product.yield_.x / 1E6, heavy_product.yield_.y, xs_fig_label, 'proton')
        #     reaction.product_nuclide_names_xss[heavy_product_name] = xs
        #     if heavy_product_name in all_reactions:
        #         daughter_reaction = all_reactions[heavy_product_name]
        #     else:
        #         daughter_reaction = ActivationReactionContainer(heavy_product_name)
        #         all_reactions[heavy_product_name] = daughter_reaction
        #     daughter_reaction.parent_nuclide_names.append(nuclide_name)

    for nuclide_name, reaction in all_reactions.items():
        pickle_file_name = PROTON_PICKLE_DIR/(nuclide_name + ".pickle")
        with open(pickle_file_name, "bw") as f:
            print('Creating and writing {}'.format(f.name))
            pickle.dump(reaction, f)


#  Special case implimented in pickle_proton_fission_data() for proton induced fission cross-sections.
#  As of now, they must be manually copy-pasted :(
#  See 'JSB_tools/nuke_data_tools/endf_files/FissionXS/Proton/readme' for instructions.
proton_fission_xs = {}


def pickle_proton_fission_data():
    for file in (parent_data_dir/'FissionXS'/'Proton').iterdir():
        _m = re.match(r'([A-Z][a-z]{0,2})-([0-9]{1,3})\(P,F\)', file.name)
        if _m:
            a = _m.groups()[1]
            s = _m.groups()[0]
            reaction_name = file.name
            nuclide_name = '{0}{1}'.format(s, a)
            _m = re.compile('# +([A-z])eV +')

            with open(file) as f:
                lines = f.readlines()
            for index, line in enumerate(lines):
                if _m.match(line):
                    if _m.match(line).groups()[0] != 'M':
                        assert False, 'Invalid units of energy. Edit code to scale tlo MeV'
                    break
            else:
                assert False, 'Invalid file "{}" for induced fission. Did not find line matching "# +([A-z])eV +"'\
                    .format(file.name)
            ergs = []
            xss = []
            _m = re.compile(' +[0-9][0-9.e+-]* +[0-9][0-9.e+-]*')
            for line in lines[index+1:]:

                if _m.match(line):
                    erg, xs = line.split()
                    ergs.append(float(erg))
                    xss.append(float(xs))
            xs_obj = CrossSection1D(ergs, xss,reaction_name, 'proton')
            proton_fission_xs[nuclide_name] = xs_obj

    if len(proton_fission_xs):
        for n_name, data in proton_fission_xs.items():
            with open(PROTON_PICKLE_DIR/'fission'/'{}.pickle'.format(n_name), 'wb') as f:
                pickle.dump(data, f)


def pickle_photon_fission_data():
    photo_fission_data = {}
    for file in photon_enfd_b_data_dir.iterdir():
        _m = re.match(r'g-([0-9]{3})_([A-Z,a-z]+)_([0-9]{3})\.endf', file.name)
        if _m:
            a = _m.groups()[2]
            symbol = _m.groups()[1]
            nuclide_name = '{0}{1}'.format(symbol, a)
            ev = Evaluation(file)
            xs = Reaction.from_endf(ev, 18).xs
            if len(xs):
                fission_xs = list(Reaction.from_endf(ev, 18).xs.values())[0]
                xs = CrossSection1D(fission_xs.x/1E6, fission_xs.y, '{0}{1}(G,F)'.format(symbol, a), 'photon')
                photo_fission_data[nuclide_name] = xs
            else:
                continue

    for nuclide_name, xs in photo_fission_data.items():
        with open(PHOTON_PICKLE_DIR/'fission'/'{0}.pickle'.format(nuclide_name), 'wb') as f:
            pickle.dump(xs, f)


def pickle_photon_activation_data():
    assert PHOTON_PICKLE_DIR.exists()

    all_reactions = {}

    for file_path in photon_enfd_b_data_dir.iterdir():
        _m = re.match(r'g-([0-9]{3})_([A-Z,a-z]+)_([0-9]{3})\.endf', file_path.name)
        if _m:
            a = int(_m.groups()[2])
            symbol = _m.groups()[1]
            nuclide_name = '{0}{1}'.format(symbol, a)
            ActivationReactionContainer.set(all_reactions, nuclide_name, file_path, 'photon')

    for nuclide_name, reaction in all_reactions.items():
        pickle_file_name = PHOTON_PICKLE_DIR/(nuclide_name + ".pickle")
        if len(reaction) == 0:  # this is probably not needed
            continue
        with open(pickle_file_name, "bw") as f:
            print('Creating and writing {}'.format(f.name))
            pickle.dump(reaction, f)


def pickle_sf_yields():
    for f_path in sf_yield_data_dir.iterdir():
        _m = re.match('GEFY_([0-9]+)_([0-9]+)_s.dat', f_path.name)
        if _m:
            z = int(_m.groups()[0])
            a = int(_m.groups()[1])
            e_symbol = ATOMIC_SYMBOL[z]
            try:
                print('Pickling SF data for {}, z={}, a={} from file {}'.format(e_symbol, z, a, f_path))
                y = FissionProductYields(str(f_path))
            except KeyError:
                warn('Failed to load SF data from "{}" in "{}"'.format(e_symbol, f_path.name))
                continue

            nuclide_name = y.nuclide['name']

            with open(SF_YIELD_PICKLE_DIR/'cumulative'/'{}.pickle'.format(nuclide_name), 'wb') as f:
                pickle.dump(y.cumulative[0], f)
            with open(SF_YIELD_PICKLE_DIR/'independent'/'{}.pickle'.format(nuclide_name), 'wb') as f:
                pickle.dump(y.independent[0], f)


proton_mass = 938.272
neutron_mass = 939.565


def get_proton_equiv_fission_erg(n: Nuclide, proton_erg):
    print(Nuclide.from_Z_A_M(n.Z + 1, n.A - 1).name)
    'eproton - rm[0, 1] + rm[1, 1] + rm[Z, A] - rm[1 + Z, -1 + A] + \
 rm[1 + Z, A] - rm[1 + Z, 1 + A]'
    _m = re.match('([A-zA-Z]+)([0-9]+).*', n.name)
    assert _m
    z, a = map(int, (ATOMIC_NUMBER[_m.groups()[0]], _m.groups()[1]))
    return proton_erg - neutron_mass + proton_mass \
           + Nuclide.get_mass_in_mev_per_c2(z=z, a=a) \
           - Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a - 1) \
           + Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a) \
           - Nuclide.get_mass_in_mev_per_c2(z=z + 1, a=a + 1)


def pickle_neutron_yields():
    for f_path in neutron_fission_yield_data_dir.iterdir():
        _m = re.match('GEFY_([0-9]+)_([0-9]+)_n.dat', f_path.name)
        if _m:
            z = int(_m.groups()[0])
            a = int(_m.groups()[1])
            e_symbol = ATOMIC_SYMBOL[z]
            try:
                print('Pickling n-induced fission data for {}, z={}, a={} from file {}'.format(e_symbol, z, a, f_path))
                y = FissionProductYields(str(f_path))
            except KeyError:
                warn('Failed to load n_induced fission data from "{}" in "{}"'.format(e_symbol, f_path.name))
                continue
            for e in y.energies:
                print(e)
            break


def pickle_all_nuke_data():
    pickle_neutron_yields()
    # pickle_sf_yields()
    # pickle_decay_data()
    # pickle_proton_activation_data()
    # pickle_proton_fission_data()  # pickle proton fission data in a special way due to compatibility issues with EDNF6
    # pickle_photon_fission_data()
    # pickle_photon_activation_data()
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    pickle_all_nuke_data()

