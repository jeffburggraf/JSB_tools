import re
from pathlib import Path

pwd = Path(__file__).parent


DECAY_PICKLE_DIR = pwd/'data'/'nuclides'  # rel. dir. of pickled nuclide data (half lives, ect)
LEVEL_PICKLE_DIR = pwd/'data'/'nuclides'/'levels'  # rel. dir. of pickled nuclear level data (from ENDSF)

PROTON_PICKLE_DIR = pwd / "data" / "activation" / "proton"  # rel. dir. of pickled proton/fission data
GAMMA_PICKLE_DIR = pwd / "data" / "activation" / "gamma"  # rel. dir. of pickled photon/fission  data
NEUTRON_PICKLE_DIR = pwd / "data" / "activation" / "neutron"  # rel. dir. of pickled neutron activation/fission data

FISS_YIELDS_PATH = pwd/'data'/'fiss_yields'


for d in [DECAY_PICKLE_DIR, LEVEL_PICKLE_DIR, PROTON_PICKLE_DIR, GAMMA_PICKLE_DIR, NEUTRON_PICKLE_DIR,
          FISS_YIELDS_PATH]:
    d.mkdir(exist_ok=True)


parent_data_dir = pwd / 'endf_files'  # ENDF file directory


# **************************************************************************************
# *********** Below is the specification of for downloaded nuclear libraries ***********
# **************************************************************************************
decay_data_dir = parent_data_dir / 'decay2020'

fission_yield_dirs = {'sf': {'endf': parent_data_dir/'ENDF-B-VIII.0_sfy',
                             'gef': parent_data_dir/'GEFY61data'/'gefy61_sfy'},

                      'neutron': {'endf': parent_data_dir/'ENDF-B-VIII.0_nfy',
                                  'ukfy': parent_data_dir/'UKFY42data'/'ukfy4_2n',
                                   'gef': parent_data_dir/'GEFY61data'/'gefy61_nfy'},

                      'gamma': {'ukfy': parent_data_dir/'UKFY41data'/'ukfy4_1g'},

                      'proton': {'ukfy': parent_data_dir/'UKFY41data'/'ukfy4_1p'},

                      'alpha': {'ukfy': parent_data_dir/'UKFY41data'/'ukfy4_1a'}}


activation_directories = {'neutron': {'endf': parent_data_dir / 'ENDF-B-VIII.0_neutrons',
                                      'tendl': parent_data_dir/'TENDL-n'},

                          'gamma': {'endf': parent_data_dir/'ENDF-B-VIII.0_gammas',
                                     'tendl': parent_data_dir/'TENDL-g'},

                          'proton': {'endf': parent_data_dir/'ENDF-B-VIII.0_protons',
                                     'tendl': parent_data_dir/'TENDL-p'}}


if __name__ == '__main__':
    pass

