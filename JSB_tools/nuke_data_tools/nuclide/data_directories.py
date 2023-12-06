import re
from pathlib import Path

cwd = Path(__file__).parent
pickle_dir = Path(__file__).parent/'pickled_data'


all_particles = ['gamma', 'electron', 'proton', 'neutron', 'deuteron', 'triton', '3He',  'alpha']

DECAY_PICKLE_DIR = pickle_dir/'nuclides'  # rel. dir. of pickled nuclide data (half lives, ect)
LEVEL_PICKLE_DIR = pickle_dir/'nuclides'/'levels'  # rel. dir. of pickled nuclear level data (from ENDSF)

FISS_YIELDS_PATH = pickle_dir/'fiss_yields'


for d in [DECAY_PICKLE_DIR, LEVEL_PICKLE_DIR, FISS_YIELDS_PATH]:
    d.mkdir(exist_ok=True)

endf_data_dir = cwd / 'endf_files'  # ENDF file directory


# **************************************************************************************
# *********** Below is the specification of downloaded nuclear libraries ***********
# **************************************************************************************
decay_data_dir = endf_data_dir / 'decay2020'

fission_yield_dirs = {'sf': {'endf': endf_data_dir / 'ENDF-B-VIII.0_sfy',
                             'gef': endf_data_dir / 'GEFY61data' / 'gefy61_sfy'},

                      'neutron': {'endf': endf_data_dir / 'ENDF-B-VIII.0_nfy',
                                  'ukfy': endf_data_dir / 'UKFY42data' / 'ukfy4_2n',
                                   'gef': endf_data_dir / 'GEFY61data' / 'gefy61_nfy'},

                      'gamma': {'ukfy': endf_data_dir / 'UKFY41data' / 'ukfy4_1g'},

                      'proton': {'ukfy': endf_data_dir / 'UKFY41data' / 'ukfy4_1p'},

                      'alpha': {'ukfy': endf_data_dir / 'UKFY41data' / 'ukfy4_1a'}}


activation_directories = {'neutron': {'endf': endf_data_dir / 'ENDF-B-VIII.0_neutrons',
                                      'tendl': endf_data_dir / 'TENDL-n'},

                          'gamma': {'endf': endf_data_dir / 'ENDF-B-VIII.0_gammas',
                                     'tendl': endf_data_dir / 'TENDL-g'},

                          'proton': {'endf': endf_data_dir / 'ENDF-B-VIII.0_protons',
                                     'tendl': endf_data_dir / 'TENDL-p'}}


if __name__ == '__main__':
    pass

