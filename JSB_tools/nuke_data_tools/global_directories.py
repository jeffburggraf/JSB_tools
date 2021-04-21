from pathlib import Path

pwd = Path(__file__).parent

DECAY_PICKLE_DIR = pwd/'data'/'nuclides'  # rel. dir. of pickled nuclide data (half lives, ect)
PROTON_PICKLE_DIR = pwd / "data" / "incident_proton"  # rel. dir. of pickled proton/fission activation data
GAMMA_PICKLE_DIR = pwd / "data" / "incident_photon"  # rel. dir. of pickled photon/fission activation data
NEUTRON_PICKLE_DIR = pwd / 'data' / 'incident_neutron'  # rel. dir. of pickled neutron activation/fission data

FISS_YIELDS_PATH = pwd/'data'/'fiss_yields'
SF_YIELD_PICKLE_DIR = pwd/'data'/'SF_yields'
NEUTRON_F_YIELD_PICKLE_DIR = pwd / 'data' / 'neutron_fiss_yields'


parent_data_dir = pwd / 'endf_files'  # ENDF file directory

#  Below are instructions for downloading all the required data. As the names of the downloaded folders may change
#  as libraries are updated, make sure the changes are reflected in the below directories.

# download decay data from https://fispact.ukaea.uk/nuclear-data/downloads/. You should find a file titled
# 'general decay data' in the list.
#  Second set of decay files (JEFF) can be found at  https://www.nndc.bnl.gov/endf/b8.0/download.html
#  See endf_files/decay/readme for more information.
decay_data_dir = parent_data_dir / 'decay'

# Download Proton Activation Data File from: https://www-nds.iaea.org/padf/
proton_padf_data_dir = parent_data_dir / 'PADF_2007'

#  Down load the data for the below at: https://www.nndc.bnl.gov/endf/b8.0/download.html
proton_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_protons'
gamma_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_gammas'
neutron_fission_yield_data_dir_endf = parent_data_dir / 'ENDF-B-VIII.0_nfy'
tendl_2019_proton_dir = parent_data_dir/'TENDL2019-PROTONS'

#  Download SF yields (gefy model) from https://www.cenbg.in2p3.fr/GEFY-GEF-based-fission-fragment,780
sf_yield_data_dir = parent_data_dir / 'gefy81_s'
neutron_fission_yield_data_dir_gef = parent_data_dir / 'gefy81_n'

#  Download proton induced fission yields from https://fispact.ukaea.uk/nuclear-data/downloads/
proton_fiss_yield_data = parent_data_dir/'UKFY41data'/'ukfy4_1p'