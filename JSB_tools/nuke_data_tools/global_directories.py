import re
from pathlib import Path

pwd = Path(__file__).parent

DECAY_PICKLE_DIR = pwd/'data'/'nuclides'  # rel. dir. of pickled nuclide data (half lives, ect)
PROTON_PICKLE_DIR = pwd / "data" / "incident_proton"  # rel. dir. of pickled proton/fissionXS activation data
GAMMA_PICKLE_DIR = pwd / "data" / "incident_gamma"  # rel. dir. of pickled photon/fissionXS activation data
NEUTRON_PICKLE_DIR = pwd / 'data' / 'incident_neutron'  # rel. dir. of pickled neutron activation/fissionXS data

FISS_YIELDS_PATH = pwd/'data'/'fiss_yields'
SF_YIELD_PICKLE_DIR = pwd/'data'/'SF_yields'
NEUTRON_F_YIELD_PICKLE_DIR = pwd / 'data' / 'neutron_fiss_yields'


parent_data_dir = pwd / 'endf_files'  # ENDF file directory

#  Below are instructions for downloading all the required data. As the names of the downloaded folders may change
#  as libraries are updated, make sure the changes are reflected in the below directories.

# download decay data from https://fispact.ukaea.uk/nuclear-data/downloads/. You should find a file titled
# 'general decay data' in the list.
#  Second set of decay files (JEFF) can be found at  https://www.nndc.bnl.gov/endf/b8.0/download.html
#  See endf_files/decay/readme.md for more information.

# =====================================================================================================
# ===============================Begin specification of paths to nuclear libraries ====================
# =====================================================================================================
decay_data_dir = parent_data_dir / 'decay'

# Download Proton Activation Data File from: https://www-nds.iaea.org/padf/
proton_padf_data_dir = parent_data_dir / 'PADF_2007'


#  Down load the data for the below at: https://www.nndc.bnl.gov/endf/b8.0/download.html
proton_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_protons'
neutron_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_neutrons'
neutron_fission_yield_data_dir_endf = parent_data_dir / 'ENDF-B-VIII.0_nfy'
neutron_fission_yield_data_dir_ukfy = parent_data_dir / 'UKFY42data'/'ukfy4_2n'
tendl_2019_proton_dir = parent_data_dir/'TENDL2019-PROTONS'


# download below at https://www-nds.iaea.org/photonuclear/  Replaced previous ENDF-B-VIII.0 Library.
gamma_endf_dir = parent_data_dir / 'iaea-pd2019'
gamma_tendl_dir = parent_data_dir / 'TENDL-g'

#  Download SF yields (gefy model) from https://www.cenbg.in2p3.fr/GEFY-GEF-based-fission-fragment,780
sf_yield_data_dir = parent_data_dir / 'gefy81_s'
neutron_fission_yield_data_dir_gef = parent_data_dir / 'gefy81_n'

#  Download proton induced fissionXS yields from https://fispact.ukaea.uk/nuclear-data/downloads/
proton_fiss_yield_data_dir_ukfy = parent_data_dir / 'UKFY41data' / 'ukfy4_1p'
gamma_fiss_yield_data_dir_ukfy = parent_data_dir / 'UKFY41data' / 'ukfy4_1g'
alpha_fiss_yield_data_dir_ukfy = parent_data_dir / 'UKFY41data' / 'ukfy4_1a'
deuterium_fiss_yield_data_dir_ukfy = parent_data_dir / 'UKFY41data' / 'ukfy4_1d'
neutron_tendl_data_dir = parent_data_dir/'TENDL2021-NEUTRONS/gendf-1102'

proton_tendl_data_dir = parent_data_dir/'TENDL2019-PROTONS'
# =====================================================================================================
# ===============================End specification of paths to nuclear libraries ====================
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
                       'padf':{'path':proton_padf_data_dir,
                               'match':re.compile("([0-9]{2})([0-9]{3})(M[0-9])?")}
                       },
            'gamma': {'endf': gamma_endf_dir, 'tendl':gamma_tendl_dir}}



if __name__ == '__main__':
    p = (get_neutron_endf_dir('Co', 58, 1))
    print(p, p.exists())

