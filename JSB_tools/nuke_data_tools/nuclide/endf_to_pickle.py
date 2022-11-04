from __future__ import annotations

import datetime
import pickle
import warnings
import matplotlib.pyplot as plt
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER, FissionProductYields
from openmc.data import Reaction, Decay
from pathlib import Path
import re
import marshal
from typing import Dict, List, TypedDict
from JSB_tools.nuke_data_tools. nuclide import Nuclide, DecayMode, _DiscreteSpectrum
from warnings import warn
from JSB_tools.nuke_data_tools.nuclide.data_directories import GAMMA_PICKLE_DIR,\
     PROTON_PICKLE_DIR, NEUTRON_PICKLE_DIR, FISS_YIELDS_PATH
from JSB_tools.nuke_data_tools.nuclide.cross_section import CrossSection1D, ActivationReactionContainer
from uncertainties import ufloat
from numbers import Number
import numpy as np
from data_directories import parent_data_dir, decay_data_dir, proton_padf_data_dir, proton_enfd_b_data_dir,\
    gamma_endf_dir, neutron_fission_yield_data_dir_endf, neutron_fission_yield_data_dir_gef, sf_yield_data_dir,\
    proton_fiss_yield_data_dir_ukfy, gamma_fiss_yield_data_dir_ukfy, neutron_enfd_b_data_dir, gamma_tendl_dir,\
    neutron_fission_yield_data_dir_ukfy, tendl_2019_proton_dir, neutron_tendl_data_dir, proton_tendl_data_dir
from uncertainties import UFloat
import uncertainties.unumpy as unp
from JSB_tools import ProgressReport, TabPlot
cwd = Path(__file__).parent

# parent_data_dir = cwd / 'endf_files'

#  Below are instructions for downloading all the required data. As the names of the downloaded folders may change
#  as libraries are updated, make sure the changes are reflected in the below directories.

# download decay data from https://fispact.ukaea.uk/nuclear-data/downloads/. You should find a file titled
# 'general decay data' in the list.
#  Second set of decay files (JEFF) can be found at  https://www.nndc.bnl.gov/endf/b8.0/download.html
#  See endf_files/decay/readme.md for more information.
# decay_data_dir = parent_data_dir / 'decay'

# Download Proton Activation Data File from: https://www-nds.iaea.org/padf/
# proton_padf_data_dir = parent_data_dir / 'PADF_2007'

#  Down load the data for the below at: https://www.nndc.bnl.gov/endf/b8.0/download.html
# proton_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_protons'
# gamma_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_gammas'
# neutron_fission_yield_data_dir_endf = parent_data_dir / 'ENDF-B-VIII.0_nfy'

#  Download SF yields (gefy model) from https://www.cenbg.in2p3.fr/GEFY-GEF-based-fission-fragment,780
# sf_yield_data_dir = parent_data_dir / 'gefy81_s'
# neutron_fission_yield_data_dir_gef = parent_data_dir / 'gefy81_n'

#  Download proton induced fissionXS yields from https://fispact.ukaea.uk/nuclear-data/downloads/
# proton_fiss_yield_data = parent_data_dir/'UKFY41data'/'ukfy4_1p'
#  ==============

_seconds_in_day = 24*60**2
_seconds_in_month = _seconds_in_day*30
_seconds_in_year = _seconds_in_month*12

hl_match_1 = re.compile(r'Parent half-life: +([0-9.E+-]+) ([a-zA-Z]+) ([0-9A-Z]+).*')
hl_match_2 = re.compile(r'T1\/2=([0-9.decE+-]+) ([A-Z]+).*')
stable_match = re.compile('Parent half-life: STABLE.+')


def decay_evaluation_score(decay1):
    if decay1 is None:
        return 0
    try:
        return len(decay1.spectra['gamma']['discrete'])
    except KeyError:
        return 0.5


def quick_nuclide_lookup():

    with open(DECAY_PICKLE_DIR / 'quick_nuclide_lookup.pickle', 'wb') as f:
        data = {}

        for path in DECAY_PICKLE_DIR.iterdir():
            if path.is_file():
                if Nuclide.NUCLIDE_NAME_MATCH.match(path.name):
                    nuclide = Nuclide.from_symbol(path.stem)
                    key = nuclide.A, nuclide.Z, nuclide.half_life, nuclide.isometric_state
                    data[key] = nuclide.name

        pickle.dump(data, f)


#  Special case implemented in pickle_proton_fission_xs_data() for proton induced fission cross-sections.
#  As of now, they must be manually copy-pasted :(
#  See 'JSB_tools/nuke_data_tools/endf_files/FissionXS/Proton/readme.md' for instructions.
proton_fission_xs = {}


def pickle_proton_fission_xs_data():
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
                assert False, 'Invalid file "{}" for induced fissionXS. Did not find line matching "# +([A-z])eV +"'\
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
        assert (PROTON_PICKLE_DIR / 'fissionXS').exists()
        for n_name, data in proton_fission_xs.items():
            with open(PROTON_PICKLE_DIR/'fissionXS'/'{}.pickle'.format(n_name), 'wb') as f:
                pickle.dump(data, f)


def pickle_gamma_neutron_fission_xs_data():

    loop_data = {'neutron': {"src_path": neutron_enfd_b_data_dir,
                             "match": re.compile(r"n-([0-9]{3})_([A-Z,a-z]+)_([0-9]{3})\.endf"),
                             "data": {},
                             'save_dir': NEUTRON_PICKLE_DIR},

                 'gamma': {"src_path": gamma_endf_dir,
                           "match": re.compile(r"g_([0-9]{1,3})-([A-Za-z]+)-([0-9]{0,3})_[0-9]+\.endf"),
                           "data": {},
                           "save_dir": GAMMA_PICKLE_DIR}
                 }

    for par, data in loop_data.items():
        result_dict = data['data']
        save_dir = data['save_dir']

        for filepath in data['src_path'].iterdir():
            if data['match'].match(filepath.name):
                ev = Evaluation(filepath)
                nuclide_name = ev.gnd_name
                try:
                    xs = Reaction.from_endf(ev, 18).xs
                except KeyError:  # OpenMC bug for some neutron ENDF files
                    continue
                if len(xs):
                    fission_xs = list(Reaction.from_endf(ev, 18).xs.values())[0]
                    xs_fig_label = f'{nuclide_name}({par[0].upper()},F)'
                    xs = CrossSection1D(fission_xs.x / 1E6, fission_xs.y, xs_fig_label, par)
                    result_dict[nuclide_name] = xs

        for nuclide_name, xs in result_dict.items():
            assert (save_dir / 'fissionXS').exists()
            with open(save_dir / 'fissionXS' / '{0}.pickle'.format(nuclide_name), 'wb') as f:
                pickle.dump(xs, f)


def pickle_proton_activation_data(pickle=True, endf=True, tendl=True):

    if endf:
        p = ProgressReport(len(list(proton_enfd_b_data_dir.iterdir())), 5)
        i = 0
        for path in Path(proton_enfd_b_data_dir).iterdir():
            f_name = path.name

            _m = re.match(r"p-([0-9]+)_([A-Za-z]{1,2})_([0-9]+)\.endf", f_name)
            if _m:
                ActivationReactionContainer.from_endf(path, 'proton', 'endf')
            i += 1
            p.log(i, 'Protons ENDF')

    if tendl:
        p = ProgressReport(len(list(proton_tendl_data_dir.iterdir())), 5)
        i = 0
        for path in Path(proton_tendl_data_dir).iterdir():
            if re.match(r'p-[A-Z][a-z]*[0-9]+[mnopqrst]?\.tendl', path.name):
                ActivationReactionContainer.from_endf(path, 'proton', 'tendl')
            i += 1
            p.log(i, 'Protons TENDL')

    if pickle:
        ActivationReactionContainer.pickle_all('proton')


def pickle_neutron_activation_data(pickle=True, endf=True, tendl=True):

    if endf:
        p = ProgressReport(len(list(neutron_enfd_b_data_dir.iterdir())), 5)
        i = 0
        for file_path in neutron_enfd_b_data_dir.iterdir():
            _m = re.match(r'n-([0-9]{3})_([A-Z,a-z]+)_([0-9]{3})\.endf', file_path.name)
            if _m:
                ActivationReactionContainer.from_endf(file_path, 'neutron', 'endf')
        i += 1
        p.log(i, 'Neutrons ENDF')

    if tendl:
        p = ProgressReport(len(list(neutron_tendl_data_dir.iterdir())), 5)
        i = 0
        for file_path in neutron_tendl_data_dir.iterdir():
            _m = re.match(r'([A-Z][a-z]{0,2})([0-9]+)[mnop]?g\.asc', file_path.name)
            if _m:
                ActivationReactionContainer.from_endf(file_path, 'neutron', 'tendl')
            i += 1
            p.log(i, 'Neutrons TENDL')

    if pickle:
        ActivationReactionContainer.pickle_all('neutron')


def pickle_gamma_activation_data(pickle=True, endf=True, tendl=True):

    if endf:
        p = ProgressReport(len(list(gamma_endf_dir.iterdir())), 5)
        i = 0
        for file_path in gamma_endf_dir.iterdir():
            _m = re.match(r'g_(?P<Z>[0-9]+)-(?P<S>[A-Z][a-z]{0,2})-(?P<A>[0-9]+)_[0-9]+.+endf', file_path.name)
            if _m:
                ActivationReactionContainer.from_endf(file_path, 'gamma', 'endf')
            i += 1
            p.log(i, 'Gammas ENDF')

    if tendl:
        p = ProgressReport(len(list(gamma_tendl_dir.iterdir())), 5)
        i = 0
        for file_path in gamma_tendl_dir.iterdir():
            if m := re.match("g-([A-Z][a-z]*)([0-9]+)([m-z]{0,1})\.tendl", file_path.name):
                ActivationReactionContainer.from_endf(file_path, 'gamma', 'tendl')

            i += 1
            p.log(i, 'Gammas TENDL')

    if pickle:
        ActivationReactionContainer.pickle_all('gamma')


class Helper:
    def __init__(self, raw_data_dir: Path, new_data_dir: Path, file_name_converter):
        """

        Args:
            raw_data_dir: dir of downloaded data
            new_data_dir: Dir to save data
            file_name_converter: Two options:
                1. A str representing a regex that has 2-3 capturing groups that are, in order, Z, A, <m>, where m is
                    optional.
                2. A callable that takes filename and return nuclide symbol.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.new_data_dir = Path(new_data_dir)

        if isinstance(file_name_converter, str):
            _compilied = re.compile(file_name_converter)

            def file_name_converter(s):
                if m := _compilied.match(s):
                    if len(m.groups()) > 1:
                        z = int(m.groups()[0])
                        a = int(m.groups()[1])
                        try:
                            m = m.groups()[2]
                            if m is None:
                                m = 0
                        except IndexError:
                            m = 0
                        symbol = ATOMIC_SYMBOL[z] + str(a) + ("_m{}".format(m) if m else '')
                        return symbol

            self.file_name_converter = file_name_converter

        else:
            assert hasattr(file_name_converter, '__call__')
            self.file_name_converter = file_name_converter

    def get_valid(self):
        """Generator that loops through source directory and returns dict of nuclide name and data path"""
        for f_path in self.raw_data_dir.iterdir():
            f_name = f_path.name
            symbol = self.file_name_converter(f_name)
            if symbol is not None:
                yield {'parent_symbol': symbol, 'f_path': f_path}


def ukfy_file2symbol(f_name):
    if m := re.match('([A-Z][a-z]{0,2}[0-9]{1,3})([mnop]?)', f_name):
        s = m.groups()[0]
        if m.groups()[1]:
            s += f"_m{'mnop'.index(m.groups()[1])+1}"
        return s


def pickle_fission_product_yields():
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

    #   To add more fissionXS yield source, add the data source directory, write directory,
    #   and the regex to extract Z,A and M (All is done via a Helper instance)
    # Add here for fiss yield
    neutron_yield_marshal_path_endf = FISS_YIELDS_PATH/'neutron'/'endf'
    neutron_yield_marshal_path_gef = FISS_YIELDS_PATH/'neutron'/'gef'
    neutron_yield_marshal_path_ukfy = FISS_YIELDS_PATH/'neutron'/'ukfy'
    sf_yield_marshal_path_gef = FISS_YIELDS_PATH/'SF'/'gef'
    proton_yield_marshal_path_ukfy = FISS_YIELDS_PATH/'proton'/'ukfy'
    gamma_yield_marshal_path_ukfy = FISS_YIELDS_PATH/'gamma'/'ukfy'

    for _dir in [neutron_yield_marshal_path_endf, neutron_yield_marshal_path_gef, neutron_yield_marshal_path_ukfy,
                 sf_yield_marshal_path_gef, proton_yield_marshal_path_ukfy, gamma_yield_marshal_path_ukfy]:
        Path.mkdir(_dir.parent, parents=True, exist_ok=True)

    # Add here for fiss yield
    # helpers = [Helper(neutron_fission_yield_data_dir_endf,
    #                   neutron_yield_marshal_path_endf,
    #                   'nfy-([0-9]+)_[a-zA-Z]+_([0-9]+)(?:m([0-9]))*')]
    helpers = [
                Helper(neutron_fission_yield_data_dir_gef,
                       neutron_yield_marshal_path_gef,
                       'GEFY_([0-9]+)_([0-9]+)_n.dat'),
               Helper(neutron_fission_yield_data_dir_endf,
                      neutron_yield_marshal_path_endf,
                      'nfy-([0-9]+)_[a-zA-Z]+_([0-9]+)(?:m([0-9]))*'),
               Helper(neutron_fission_yield_data_dir_ukfy,
                      neutron_yield_marshal_path_ukfy,
                      ukfy_file2symbol),
               Helper(sf_yield_data_dir,
                      sf_yield_marshal_path_gef,
                      'GEFY_([0-9]+)_([0-9]+)_s.dat'),
               Helper(proton_fiss_yield_data_dir_ukfy,
                      proton_yield_marshal_path_ukfy,
                      ukfy_file2symbol),
               Helper(gamma_fiss_yield_data_dir_ukfy,
                      gamma_yield_marshal_path_ukfy,
                      ukfy_file2symbol)
               ]
    #
    for helper in helpers:
        for x in helper.get_valid():
            f_path = x['f_path']
            parent_symbol = x['parent_symbol']

            try:
                openmc_yield = FissionProductYields(f_path)
            except KeyError:
                warn('Unable to load fissionXS yield from file {}'.format(f_path))
                continue

            ergs, data = build_data(openmc_yield)
            for yield_type, data in data.items():  # yield_type: cumulative or independent
                if not (helper.new_data_dir/yield_type).exists():
                    Path.mkdir(helper.new_data_dir/yield_type, parents=True)
                f_path = helper.new_data_dir/str(yield_type)/str(parent_symbol + '.marshal')

                with open(f_path, 'wb') as f:
                    marshal.dump(ergs, f)
                    marshal.dump(data, f)

                print('Written Fission yields for {}'
                      .format(Path(f_path.parents[2].name)/f_path.parents[1].name/f_path.parent.name/f_path.name))


for _directory in [DECAY_PICKLE_DIR, GAMMA_PICKLE_DIR, PROTON_PICKLE_DIR, FISS_YIELDS_PATH, NEUTRON_PICKLE_DIR]:
    if _directory == DECAY_PICKLE_DIR:
        if len(list(DECAY_PICKLE_DIR.iterdir())) == 0:
            warn('Decay data directory empty.\n'
                 'Cannot pickle nuclear data until pickle_decay_data() has been run.\n'
                 'Running pickle_decay_data() now')
            pickle_decay_data()
    if not _directory.exists():
        print(f'Creating {_directory}')
        _directory.mkdir()
for _directory in [PROTON_PICKLE_DIR/'fissionXS', GAMMA_PICKLE_DIR/'fissionXS', NEUTRON_PICKLE_DIR/'fissionXS']:
    if not Path.exists(_directory):
        Path.mkdir(_directory)


def pickle_all_nuke_data():
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
    print(f'Reading {endf_path}')
    e = Evaluation(endf_path)
    d = Decay(e)
    print("Decay modes:")
    for m in d.modes:
        print('\t', m)

    for spec_type, spectra in d.spectra.items():
        print(spec_type, spectra)


if __name__ == '__main__':
    pass
    print()
    # quick_nuclide_lookup()
    # pickle_neutron_activation_data()
    # pickle_gamma_activation_data()
    # pickle_decay_data(pickle_data=False)
    # pickle_fission_product_yields()
    # pickle_proton_activation_data(True, True, False)
    # pickle_neutron_activation_data(True, True, False)
    # pickle_gamma_neutron_fission_xs_data()
    # pickle_neutron_activation_data()
# pickle_gamma_fission_xs_data
