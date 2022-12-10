import re
from pathlib import Path

pwd = Path(__file__).parent

# Done change the values below.
DECAY_PICKLE_DIR = pwd/'data'/'nuclides'  # rel. dir. of pickled nuclide data (half lives, ect)
LEVEL_PICKLE_DIR = pwd/'data'/'nuclides'/'levels'  # rel. dir. of pickled nuclide data (half lives, ect)
PROTON_PICKLE_DIR = pwd / "data" / "incident_proton"  # rel. dir. of pickled proton/fission data
GAMMA_PICKLE_DIR = pwd / "data" / "incident_gamma"  # rel. dir. of pickled photon/fission  data
NEUTRON_PICKLE_DIR = pwd / 'data' / 'incident_neutron'  # rel. dir. of pickled neutron activation/fission data

FISS_YIELDS_PATH = pwd/'data'/'fiss_yields'
SF_YIELD_PICKLE_DIR = pwd/'data'/'SF_yields'
NEUTRON_F_YIELD_PICKLE_DIR = pwd / 'data' / 'neutron_fiss_yields'


parent_data_dir = pwd / 'endf_files'  # ENDF file directory


# **************************************************************************************
# *********** Below is the specification of for downloaded nuclear libraries ***********
# **************************************************************************************
decay_data_dir = parent_data_dir / 'ENDF-decay'
proton_enfd_b_data_dir = parent_data_dir / 'ENDF-protons'
neutron_enfd_b_data_dir = parent_data_dir / 'ENDF-neutrons'
gamma_endf_dir = parent_data_dir / 'ENDF-gammas'

gamma_jendl_dir = parent_data_dir / 'JENDL-gammas'


proton_fiss_yield_data_dir_ukfy = parent_data_dir / 'UKFY-pfy'
gamma_fiss_yield_data_dir_ukfy = parent_data_dir / 'UKFY-gfy'
neutron_fission_yield_data_dir_ukfy = parent_data_dir / 'UKFY-nfy'
# alpha_fiss_yield_data_dir_ukfy = parent_data_dir / 'UKFY-afy'
# deuterium_fiss_yield_data_dir_ukfy = parent_data_dir / 'UKFY4-dfy'

neutron_fission_yield_data_dir_endf = parent_data_dir / 'ENDF-nfy'

sf_yield_data_dir_gef = parent_data_dir / 'GEFY_sfy'
neutron_fission_yield_data_dir_gef = parent_data_dir / 'GEFY_nfy'


proton_tendl_data_dir = parent_data_dir/'TENDL-protons'
neutron_tendl_data_dir = parent_data_dir/'TENDL-neutrons'
gamma_tendl_dir = parent_data_dir / 'TENDL-gammas'
# =====================================================================================================
# ===============================End specification  ====================
# =====================================================================================================


def get_neutron_endf_dir(sym, a, m=0):
    from openmc.data import ATOMIC_NUMBER
    z = ATOMIC_NUMBER[sym]
    out = f"n-{z:0>3}_{sym}_{a:0>3}"
    if m != 0:
        out += f'm{m}'
    return neutron_enfd_b_data_dir/f"{out}.endf"


all_dirs = {"neutron": {'endf':{'path': neutron_enfd_b_data_dir,
                                'match': re.compile(r'n-([0-9]{3})_([A-Z,a-z]+)_([0-9]{3})\.endf'),
                                'get': get_neutron_endf_dir},

                        'tendl': {'path': neutron_tendl_data_dir,
                                  'match': re.compile(r'([A-Z][a-z]{0,2})([0-9]+)[mnopg]\.asc')}
                        },
            'proton': {'endf': {'path': proton_enfd_b_data_dir,
                                'match': re.compile("p-([0-9]+)_([A-Za-z]{1,2})_([0-9]+)\.endf")},
                       'tendl': {'path': proton_tendl_data_dir},

                       },
            'gamma': {'endf': gamma_endf_dir, 'tendl':gamma_tendl_dir}}



if __name__ == '__main__':
    p = (get_neutron_endf_dir('Co', 58, 1))
    print(p, p.exists())

