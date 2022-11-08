from __future__ import annotations
import pickle
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, FissionProductYields
from openmc.data import Reaction, Decay
from pathlib import Path
import re
from typing import Union, List
import marshal
from JSB_tools.nuke_data_tools. nuclide import Nuclide
from warnings import warn
from JSB_tools.nuke_data_tools.nuclide.data_directories import GAMMA_PICKLE_DIR,\
     PROTON_PICKLE_DIR, NEUTRON_PICKLE_DIR, FISS_YIELDS_PATH, DECAY_PICKLE_DIR
from JSB_tools.nuke_data_tools.nuclide.cross_section import CrossSection1D, ActivationReactionContainer
from data_directories import parent_data_dir, decay_data_dir, proton_enfd_b_data_dir,\
    gamma_endf_dir, neutron_fission_yield_data_dir_endf, neutron_fission_yield_data_dir_gef, sf_yield_data_dir_gef,\
    proton_fiss_yield_data_dir_ukfy, gamma_fiss_yield_data_dir_ukfy, neutron_enfd_b_data_dir, gamma_tendl_dir,\
    neutron_fission_yield_data_dir_ukfy, neutron_tendl_data_dir, proton_tendl_data_dir
from JSB_tools import ProgressReport
from multiprocessing import Process
cwd = Path(__file__).parent


instructions = """
Set the appropriate paths in JSB_tools/nuke_data_tools/nuclide/data_directories.py

Some websites for downloading data as of 2022 are shown below. Unfortunately, these links change often. 
    Photo-nuclear data: 
        https://www-nds.iaea.org/photonuclear/
    
    Fission yields, nuclide decay data, and more from the UK atomic energy agency:
        https://fispact.ukaea.uk/nuclear-data/downloads/
        
    TENDL data:
        https://tendl.web.psi.ch/tendl_2021/tar.html
        
    ENDF data:
        https://www.nndc.bnl.gov/endf-b8.0/download.html
    
"""


def iter_paths(path):
    path = Path(path)
    if path.is_dir():
        for sub_path in path.iterdir():
            yield from iter_paths(sub_path)
    else:
        yield path


def _run(data_directory, incident_projectile, library_name, _pickle, paths=None):  # for parallel
    if paths is None:
        paths = [p for p in iter_paths(data_directory)]

    p = ProgressReport(len(paths), 5)
    i = 0

    for file_path in paths:
        try:
            i += 1
            ActivationReactionContainer.from_endf(file_path, incident_projectile, library_name)
        except ActivationReactionContainer.EvaluationException:  # invalid/not an ENDF file. Continue
            continue

        p.log(i, f'{library_name}-{incident_projectile}')

    if _pickle:
        ActivationReactionContainer.pickle_all(incident_projectile, library_name)


def run(data_directory, incident_projectile, library_name, _pickle=True, parallel=False,
        paths: Union[List[str], None] = None):
    """
    Loops through directory containing nuclear cross-section data, loads, and saves data to pickle files.
    Args:
        data_directory: Path to directory containing nuclear data (you should be able to just download and
            supply the path to the downloaded directory (unzipped)).

        incident_projectile: e.g. 'gamma', or 'proton'

        library_name: e.g. 'endf', or 'tendl'.

        _pickle: If true, save data to pickle file.

        parallel: start in other process

        paths: Only run for these explicit paths to endf files.

    Returns:

    """
    if (not data_directory.exists()) or len(list(data_directory.iterdir())) == 0:
        data_directory.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"No data in {data_directory}. See instructions below on how to add the data.\n"
                                f"\{instructions}")

    if parallel:
        proc = Process(target=_run, args=(data_directory, incident_projectile, library_name, _pickle, paths))
        proc.start()

    else:
        _run(data_directory, incident_projectile, library_name, _pickle, paths)


def quick_nuclide_lookup():

    with open(DECAY_PICKLE_DIR / 'quick_nuclide_lookup.pickle', 'wb') as f:
        data = {}

        for path in DECAY_PICKLE_DIR.iterdir():
            if path.is_file():
                if Nuclide.NUCLIDE_NAME_MATCH.match(path.name):
                    nuclide = Nuclide(path.stem)
                    key = nuclide.A, nuclide.Z, nuclide.half_life, nuclide.isometric_state
                    data[key] = nuclide.name

        pickle.dump(data, f)


#  Special case implemented in pickle_proton_fission_xs_data() for proton induced fission cross-sections.
#  As of now, they must be manually copy-pasted :(
#  See 'JSB_tools/nuke_data_tools/endf_files/FissionXS/Proton/readme.md' for instructions.
proton_fission_xs = {}


def pickle_proton_fission_xs_data():
    """
    Special case due to bugs in ENDF6 format for proton fission xs
    Returns:

    """
    for file in (parent_data_dir/'proton_fission_xs').iterdir():
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
                assert False, 'Invalid file "{}" for induced fission xs. Did not find line matching "# +([A-z])eV +"'\
                    .format(file.name)
            ergs = []
            xss = []
            _m = re.compile(' +[0-9][0-9.e+-]* +[0-9][0-9.e+-]*')
            for line in lines[index+1:]:

                if _m.match(line):
                    erg, xs = line.split()
                    ergs.append(float(erg))
                    xss.append(float(xs))
            xs_obj = CrossSection1D(ergs, xss, reaction_name, 'proton')
            proton_fission_xs[nuclide_name] = xs_obj

    if len(proton_fission_xs):
        save_path = (PROTON_PICKLE_DIR /'endf'/'fission_xs')
        save_path.mkdir(parents=True, exist_ok=True)

        for n_name, data in proton_fission_xs.items():
            with open(save_path/'{}.pickle'.format(n_name), 'wb') as f:
                pickle.dump(data, f)


def pickle_proton_activation_data(pickle=True, endf=True, tendl=True, parallel=False, paths=None):

    if endf:
        run(proton_enfd_b_data_dir, 'proton', 'endf', _pickle=pickle, parallel=parallel, paths=paths)

    if tendl:
        run(proton_tendl_data_dir, 'proton', 'tendl', _pickle=pickle, parallel=parallel, paths=paths)


def pickle_neutron_activation_data(pickle=True, endf=True, tendl=True, parallel=False, paths=None):

    if endf:
        run(neutron_enfd_b_data_dir, 'neutron', 'endf', _pickle=pickle, parallel=parallel, paths=paths)

    if tendl:
        run(neutron_tendl_data_dir, 'neutron', 'tendl', _pickle=pickle, parallel=parallel, paths=paths)


def pickle_gamma_activation_data(pickle=True, endf=True, tendl=True, parallel=False, paths=None):

    if endf:
        run(gamma_endf_dir, 'gamma', 'endf', _pickle=pickle, parallel=parallel, paths=paths)

    if tendl:
        run(gamma_tendl_dir, 'gamma', 'tendl', _pickle=pickle, parallel=parallel, paths=paths)


def _run_fission_yield_pickle(raw_dir, particle, library):
    def build_data(y: FissionProductYields):
        _data = {'independent': {},
                 'cumulative': {}}
        _l = len(y.energies)   # number of energies
        for attrib in ['independent', 'cumulative']:
            out = _data[attrib]
            for index, yield_dict in enumerate(getattr(y, attrib)):  # one `index` for each energy in y.energies
                for nuclide_name, yield_ in yield_dict.items():
                    try:
                        entry = out[nuclide_name]
                    except KeyError:
                        out[nuclide_name] = [[0]*_l, [0]*_l]  # sometimes entries are missing. This sets those to 0.
                        entry = out[nuclide_name]
                    entry[0][index] = yield_.n
                    entry[1][index] = float(yield_.std_dev)
        _ergs = list(map(float, y.energies*1E-6))
        return _ergs, _data

    save_dir = FISS_YIELDS_PATH / particle / library

    for x in iter_paths(raw_dir):
        try:
            openmc_yield = FissionProductYields(x)
            parent_symbol = openmc_yield.nuclide['name']

        except (KeyError, ValueError):
            warn('Unable to load fission yield from file {}'.format(x))
            continue

        ergs, data = build_data(openmc_yield)

        for yield_type, data in data.items():  # yield_type: cumulative or independent
            if not (save_dir / yield_type).exists():
                Path.mkdir(save_dir / yield_type, parents=True)
            f_path = save_dir / str(yield_type) / str(parent_symbol + '.marshal')

            with open(f_path, 'wb') as f:
                marshal.dump(ergs, f)
                marshal.dump(data, f)

            print('Written Fission yields for {}'
                  .format(Path(f_path.parents[2].name) / f_path.parents[1].name / f_path.parent.name / f_path.name))


def pickle_fission_product_yields(particles=None, libraries=None, parallel=False):
    """
    To add more fission yield sources/ search for the tag 'Add here for fiss yield' and make the needed changes.
    To selectively update one type of fission yieds, look at the helpers variable below.

    Marshal'd fission yield data is a dict of the form:
        1st dump is energies:
            List[float]
        2nd dump is yield data:
            {'Xe139':[list[float], list[float]],  # where the first list is yield, and the second is yield uncertainty
                 Ta185: .... and so on,
                 ...,
            }

    """

    for raw_dir in (Path(__file__).parent/'endf_files').iterdir():
        if raw_dir.is_dir():
            if _m := re.match("([A-z]+)-(.)fy", raw_dir.name):
                try:
                    particle = {'n': 'neutron', 'p': 'proton', 's': 'sf', 'g': 'gamma'}[_m.groups()[1]]
                except KeyError:
                    continue

                library = _m.groups()[0].lower()

                if libraries is not None:
                    if library not in libraries:
                        continue

                if particles is not None:
                    if particle not in particles:
                        continue

                if not parallel:
                    _run_fission_yield_pickle(raw_dir, particle, library)
                else:
                    proc = Process(target=_run_fission_yield_pickle, args=(raw_dir, particle, library))
                    proc.start()


for _directory in [DECAY_PICKLE_DIR, GAMMA_PICKLE_DIR, PROTON_PICKLE_DIR, FISS_YIELDS_PATH, NEUTRON_PICKLE_DIR]:
    if _directory == DECAY_PICKLE_DIR:
        if len(list(DECAY_PICKLE_DIR.iterdir())) == 0:
            warn('Decay data directory empty.\n'
                 'Cannot pickle nuclear data until pickle_decay_data() has been run.\n'
                 'Running pickle_decay_data() now')
            # pickle_decay_data()
    if not _directory.exists():
        print(f'Creating {_directory}')
        _directory.mkdir()


def pickle_all_nuke_data():  # todo: paralellize this
    pickle_fission_product_yields()
    # pickle_decay_data()
    # pickle_proton_activation_data()
    # pickle_proton_fission_xs_data()
    # pickle_gamma_fission_xs_data()
    # pickle_gamma_activation_data()

    pass


def debug_nuclide(n: str, library="ENDF"):
    library = library.upper()
    assert library in ['ENDF', 'JEFF']

    m = re.match("([A-Z][a-z]{0,2})([0-9]+)", n)
    assert m, n

    symbol = m.groups()[0]
    a = f"{m.groups()[1]:0>3}"
    path = decay_data_dir/f'decay_{library}'

    if library == 'ENDF':
        m_str = f"dec-[0-9]{{3}}_{symbol}_{a}.endf"
    else:
        m_str = n
    matcher = re.compile(m_str)

    for p in path.iterdir():
        fname = p.name
        if matcher.match(fname):
            endf_path = p
            break
    else:
        raise FileNotFoundError(f"No ENDF file found for {n}")

    e = Evaluation(endf_path)
    d = Decay(e)

    for m in d.modes:
        print('\t', m)

    for spec_type, spectra in d.spectra.items():
        print(spec_type, spectra)


if __name__ == '__main__':
    pass
    # pickle_fission_product_yields('neutron')
    # pickle_gamma_activation_data(parallel=True)
    # pickle_neutron_activation_data(tendl=True, endf=False, parallel=True)
    pickle_proton_activation_data(tendl=True, parallel=False, paths=['/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/nuclide/endf_files/TENDL-protons/p-U238.tendl'])