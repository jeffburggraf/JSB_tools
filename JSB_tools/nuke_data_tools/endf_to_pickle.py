from __future__ import annotations
import pickle
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
from openmc.data import Reaction, Decay
from pathlib import Path
import re
from JSB_tools.nuke_data_tools import NUCLIDE_INSTANCES, Nuclide, DECAY_PICKLE_DIR, PROTON_PICKLE_DIR, _Reaction,\
    CrossSection1D


cwd = Path(__file__).parent

decay_data_dir = cwd/'endf_files'/'decay'
proton_padf_data_dir = cwd/'endf_files'/'PADF_2007'/'Files'
proton_enfd_b_data_dir = cwd/'endf_files'/'ENDF-B-VIII.0_protons'


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

        parent_nuclide.__set_data_from_open_mc__(openmc_decay)
        print("Preparing data for {0}".format(parent_nuclide_name))

    for nuclide_name in NUCLIDE_INSTANCES.keys():
        with open(DECAY_PICKLE_DIR/(nuclide_name + '.pickle'), "wb") as pickle_file:
            print("Writing data for {0}".format(nuclide_name))
            pickle.dump(NUCLIDE_INSTANCES[nuclide_name], pickle_file)


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


def pickle_proton_data():
    assert PROTON_PICKLE_DIR.exists()
    i = 0
    all_reactions = {}
    files = ProtonENDFFile(padf_directory=proton_padf_data_dir, endf_b_directory=proton_enfd_b_data_dir)

    for nuclide_name, f_path in files.nuclide_name_and_file_path.items():
        print('Reading data from {}'.format(nuclide_name))
        if nuclide_name in all_reactions:
            reaction = all_reactions[nuclide_name]
        else:
            reaction = _Reaction(nuclide_name)
            all_reactions[nuclide_name] = reaction

        e = Evaluation(f_path)
        for heavy_product in Reaction.from_endf(e, 5).products:
            heavy_product_name = heavy_product.particle
            if heavy_product_name == "photon":
                continue
            if heavy_product_name == "neutron":
                heavy_product_name = "Nn1"
            xs_fig_label = "{0}(p,X){1}".format(nuclide_name, heavy_product_name)
            xs = CrossSection1D(heavy_product.yield_.x / 1E6, heavy_product.yield_.y, xs_fig_label, 'proton')
            reaction.product_nuclide_names_xss[heavy_product_name] = xs
            if heavy_product_name in all_reactions:
                daughter_reaction = all_reactions[heavy_product_name]
            else:
                daughter_reaction = _Reaction(heavy_product_name)
                all_reactions[heavy_product_name] = daughter_reaction
            daughter_reaction.parent_nuclide_names.append(nuclide_name)
        i += 1

    for nuclide_name, reaction in all_reactions.items():
        pickle_file_name = PROTON_PICKLE_DIR/(nuclide_name + ".pickle")
        with open(pickle_file_name, "bw") as f:
            print('Creating and writing {}'.format(f.name))
            pickle.dump(reaction, f)


def pickle_all_nuke_data():
    # pickle_decay_data()
    pickle_proton_data()


if __name__ == '__main__':
    # pickle_all_nuke_data()
    print(list(Nuclide.from_symbol('C10').get_incident_proton_parents().values())[0].xs)