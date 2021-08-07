from __future__ import annotations

import datetime
import pickle

import matplotlib.pyplot as plt
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER, FissionProductYields
from openmc.data import Reaction, Decay
from pathlib import Path
import re
import marshal
from typing import Dict, List, TypedDict
from JSB_tools.nuke_data_tools import NUCLIDE_INSTANCES, Nuclide, DECAY_PICKLE_DIR, GAMMA_PICKLE_DIR,\
     PROTON_PICKLE_DIR, NEUTRON_PICKLE_DIR, CrossSection1D, ActivationReactionContainer,\
     GammaLine, DecayMode, FISS_YIELDS_PATH, _DiscreteSpectrum
from warnings import warn
from uncertainties import ufloat
from numbers import Number
import numpy as np
from global_directories import parent_data_dir, decay_data_dir, proton_padf_data_dir, proton_enfd_b_data_dir,\
    gamma_enfd_b_data_dir, neutron_fission_yield_data_dir_endf, neutron_fission_yield_data_dir_gef, sf_yield_data_dir,\
    proton_fiss_yield_data_dir_ukfy, gamma_fiss_yield_data_dir_ukfy  # Add here for fiss yield

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


def get_error_from_error_in_last_digit(x: Number, last_digit_err: Number):
    """Convert `x` to a ufloat, where `x` is a number whose uncertainty in the last digit is given by `last_digit_err`
        """
    assert isinstance(x, Number)
    assert isinstance(last_digit_err, Number)
    x_str = str(x)
    if 'e' in x_str.lower():
        index = x_str.index('e')
        x_str, exponent = x_str[:index], int(x_str[index+1:])
    else:
        exponent = 0

    if '.' not in x_str:
        x_str += '.'
    n_digits_after_zero = len(x_str[x_str.index('.'):]) - 1
    err_scale = 10**-n_digits_after_zero
    err = float(last_digit_err)*err_scale
    x = float(x_str)
    return ufloat(x, err)*10**exponent


def get_hl_from_ednf_file(path_to_file):
    """
    Used to correct for bug in openmc where the ENDF evaluation of some stable isotopes return a half life of zero.
    Also, very very long half-lives, e.g. 1E12 years, were also returning a half life of zero.
    """

    with open(path_to_file) as f:
        _line = None
        for line in f:
            if m := hl_match_1.match(line):
                units = m.groups()[1]
                hl = float(m.groups()[0])
                err = m.groups()[-1]
                _line = line
                break
            elif m := hl_match_2.match(line):
                units = m.groups()[1]
                hl = float(m.groups()[0])
                _line = line
                break
            elif stable_match.match(line):
                hl = ufloat(np.inf, 0)
                return hl
        else:
            assert False, 'Failed to read half life in file: {}'.format(path_to_file.name)
    if units == 'Y':
        scale = _seconds_in_year
    else:
        assert False, 'Haven\'t implemented the units "{}". This is an easy fix.'.format(units)

    if err == 'GT' or err == 'GE':
        err = hl
    else:
        try:
            err = float(err)
        except ValueError:
            warn('\nUnknown specification of half-life error: "{}", in line: "{}", in file: "{}"'
                 .format(err, _line, path_to_file.name))
            err = hl
        else:
            hl_out = get_error_from_error_in_last_digit(hl, err)
            return scale * hl_out

    return ufloat(hl*scale, err*scale)


def decay_evaluation_score(decay1):
    if decay1 is None:
        return 0
    try:
        return len(decay1.spectra['gamma']['discrete'])
    except KeyError:
        return 0.5


def set_nuclide_attributes(_self: Nuclide, open_mc_decay):
    _self.half_life = open_mc_decay.half_life
    _self.mean_energies = open_mc_decay.average_energies
    _self.spin = open_mc_decay.nuclide["spin"]
    if np.isinf(_self.half_life.n) or open_mc_decay.nuclide["stable"]:
        _self.is_stable = True
        _self.half_life = ufloat(np.inf, 0)
    else:
        _self.is_stable = False
    _self.decay_radiation_types.extend(open_mc_decay.spectra.keys())
    for mode in open_mc_decay.modes:
        decay_mode = DecayMode(mode, _self.half_life)
        try:
            _self.decay_modes[tuple(mode.modes)].append(decay_mode)
        except KeyError:
            _self.decay_modes[tuple(mode.modes)] = [decay_mode]
    l = len(_self.decay_modes)
    for key, value in list(_self.decay_modes.items())[:]:
        _self.decay_modes[key] = list(sorted(_self.decay_modes[key], key=lambda x: -x.branching_ratio))
    assert len(_self.decay_modes) == l


def pickle_spectra(nuclide: Nuclide, open_mc_decay: Decay):
    """
    Pickle decay data for each radiation type.
    Returns:

    """
    for spectra_mode, spec_data in open_mc_decay.spectra.items():
        spectra = _DiscreteSpectrum(nuclide, spec_data)
        spectra.__pickle__()


def pickle_decay_data(pickle_data=True, nuclides_to_process=None):
    """
    Pickles nuclide properties into ../data/nuclides/x.pickle
    Writes   __fast__gamma_dict__.marshal, which can be used to quickly look up decays by decay energy.
    data structure of __fast__gamma_dict__:
        {
         g_erg_1: ([name1, name2, ...], [intensity1, intensity2, ...], [half_life1, half_life2, ...]),
         g_erg_2: (...)
         }
    Args:
        pickle_data: If false, don't save to pickle files.
        nuclides_to_process: If None, process all data. Otherwise, should be a list of nuclide names,
            e.g. ["Na22", "U240"].

    Returns:

    """
    if nuclides_to_process is not None:
        assert hasattr(nuclides_to_process, '__iter__')
        assert isinstance(nuclides_to_process[0], str)
    directory_endf = decay_data_dir/'decay_ENDF'  # Path to downloaded ENDF decay data
    directory_jeff = decay_data_dir/'decay_JEFF'  # Path to downloaded ENDF decay data

    assert directory_endf.exists(), 'Create the following directory: {}'.format(directory_endf)
    assert directory_jeff.exists(), 'Create the following directory: {}'.format(directory_jeff)
    openmc_decays = {}

    jeff_decay_file_match = re.compile('([A-Z][a-z]{0,2})([0-9]{1,3})([mnop]*)')
    endf_decay_file_match = re.compile(r'dec-[0-9]{3}_(?P<S>[A-Za-z]{1,2})_(?P<A>[0-9]+)(?:m(?P<M>[0-9]+))?\.endf')

    for endf_file_path in directory_endf.iterdir():
        file_name = endf_file_path.name
        if _m := endf_decay_file_match.match(file_name):
            a = int(_m.group("A"))
            _s = _m.group("S")  # nuclide symbol, e.g. Cl, Xe, Ar
            m = _m.group("M")
            if m is not None:
                m = int(m)
            nuclide_name = "{0}{1}{2}".format(_s, a, "" if m is None else "_m{0}".format(m))
            if nuclides_to_process is not None and nuclide_name not in nuclides_to_process:
                continue

            print('Reading ENDSF decay data from {}'.format(endf_file_path.name))
            d = Decay(Evaluation(endf_file_path))

            if d.nuclide["stable"]:
                half_life = ufloat(np.inf, 0)
            elif d.half_life.n == 0:
                half_life = get_hl_from_ednf_file(endf_file_path)
            else:
                half_life = d.half_life
            openmc_decays[nuclide_name] = {'endf': d, 'jeff': None, 'half_life': half_life}
        else:
            continue
    #
    for jeff_file_path in directory_jeff.iterdir():
        file_name = jeff_file_path.name
        if match := jeff_decay_file_match.match(file_name):
            s = match.groups()[0]
            a = int(match.groups()[1])
            m = match.groups()[2]
            m = {'': 0, 'm': 1, 'n': 2, 'o': 3}[m]
            if m != 0:
                nuclide_name = '{}{}_m{}'.format(s, a, m)
            else:
                nuclide_name = '{}{}'.format(s, a)
            if nuclides_to_process is not None and nuclide_name not in nuclides_to_process:
                continue
            print('Reading JEFF decay data from {}'.format(jeff_file_path.name))

            eval = Evaluation(jeff_file_path)
            d = Decay(eval)
            if nuclide_name in openmc_decays:
                openmc_decays[nuclide_name]['jeff'] = d
            else:
                openmc_decays[nuclide_name] = {'endf': None, 'jeff': d, 'half_life': d.half_life}

    for parent_nuclide_name, openmc_dict in openmc_decays.items():
        jeff_score = decay_evaluation_score(openmc_dict['jeff'])
        endf_score = decay_evaluation_score(openmc_dict['endf'])
        if jeff_score > endf_score:
            openmc_decay = openmc_dict['jeff']
        else:
            openmc_decay = openmc_dict['endf']

        openmc_decay.half_life = openmc_dict['half_life']

        if parent_nuclide_name in NUCLIDE_INSTANCES:
            parent_nuclide = NUCLIDE_INSTANCES[parent_nuclide_name]
        else:
            parent_nuclide = Nuclide(parent_nuclide_name, __internal__=True)
            NUCLIDE_INSTANCES[parent_nuclide_name] = parent_nuclide

        daughter_names = [mode.daughter for mode in openmc_decay.modes]
        for daughter_nuclide_name in daughter_names:
            # if (nuclides_to_process is not None) and (daughter_nuclide_name not in nuclides_to_process):
            #     continue
            if daughter_nuclide_name in NUCLIDE_INSTANCES:
                daughter_nuclide = NUCLIDE_INSTANCES[daughter_nuclide_name]
            else:
                daughter_nuclide = Nuclide(daughter_nuclide_name, __internal__=True)
                NUCLIDE_INSTANCES[daughter_nuclide_name] = daughter_nuclide

            if daughter_nuclide_name != parent_nuclide_name:
                daughter_nuclide.__decay_parents_str__.append(parent_nuclide_name)
                parent_nuclide.__decay_daughters_str__.append(daughter_nuclide_name)

        set_nuclide_attributes(parent_nuclide, openmc_decay)
        pickle_spectra(parent_nuclide, openmc_decay)

    if pickle_data:
        for nuclide_name in NUCLIDE_INSTANCES.keys():
            with open(DECAY_PICKLE_DIR/(nuclide_name + '.pickle'), "wb") as pickle_file:
                print("Writing decay data for {0}".format(nuclide_name))
                pickle.dump(NUCLIDE_INSTANCES[nuclide_name], pickle_file)

        with open(DECAY_PICKLE_DIR/'quick_nuclide_lookup.pickle', 'wb') as f:
            data = {}
            for name, nuclide in NUCLIDE_INSTANCES.items():
                key = nuclide.A, nuclide.Z, nuclide.half_life
                data[key] = name
            pickle.dump(data, f)

        # data structure of d: {g_erg: ([name1, name2, ...], [intensity1, intensity2, ...], [half_life1, half_life2, ...])}

        if nuclides_to_process is None:  # None means all here.
            d = {}
            for nuclide in NUCLIDE_INSTANCES.values():
                for g in nuclide.decay_gamma_lines:
                    erg = g.erg.n
                    intensity: float = g.intensity.n
                    hl = nuclide.half_life.n
                    try:
                        i = len(d[erg][1]) - np.searchsorted(d[erg][1][::-1], intensity)
                        # the below flow control is to avoid adding duplicates.
                        for __name, __intensity in zip(d[erg][0], d[erg][1]):
                            if __name == nuclide.name and __intensity == intensity:
                                break
                        else:
                            d[erg][0].insert(i, nuclide.name)
                            d[erg][1].insert(i, intensity)
                            d[erg][2].insert(i, hl)

                    except KeyError:
                        d[erg] = ([nuclide.name], [intensity], [hl])

            with open(DECAY_PICKLE_DIR / '__fast__gamma_dict__.marshal', 'wb') as f:
                print("Writing quick gamma lookup table...")
                marshal.dump(d, f)


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


def pickle_gamma_fission_xs_data():
    photo_fission_data = {}
    for file in gamma_enfd_b_data_dir.iterdir():
        _m = re.match(r'g-([0-9]{3})_([A-Z,a-z]+)_([0-9]{3})\.endf', file.name)
        if _m:
            a = _m.groups()[2]
            symbol = _m.groups()[1]
            nuclide_name = '{0}{1}'.format(symbol, a)
            ev = Evaluation(file)
            xs = Reaction.from_endf(ev, 18).xs
            if len(xs):
                fission_xs = list(Reaction.from_endf(ev, 18).xs.values())[0]
                xs_fig_label = '{0}{1}(G,F)'.format(nuclide_name, a)
                xs = CrossSection1D(fission_xs.x/1E6, fission_xs.y, xs_fig_label, 'photon')
                photo_fission_data[nuclide_name] = xs
            else:
                continue

    for nuclide_name, xs in photo_fission_data.items():
        assert (GAMMA_PICKLE_DIR / 'fissionXS').exists()
        with open(GAMMA_PICKLE_DIR / 'fissionXS' / '{0}.pickle'.format(nuclide_name), 'wb') as f:
            pickle.dump(xs, f)


def pickle_proton_activation_data():
    assert PROTON_PICKLE_DIR.exists()
    files = ProtonENDFFile(padf_directory=proton_padf_data_dir, endf_b_directory=proton_enfd_b_data_dir)

    for nuclide_name, f_path in files.nuclide_name_and_file_path.items():
        ActivationReactionContainer.from_endf(f_path, nuclide_name, 'proton')

    for nuclide_name, reaction in ActivationReactionContainer.all_instances['proton'].items():
        pickle_file_name = PROTON_PICKLE_DIR/(nuclide_name + ".pickle")
        with open(pickle_file_name, "bw") as f:
            print('Creating and writing {}'.format(f.name))
            pickle.dump(reaction, f)


def pickle_gamma_activation_data():
    assert GAMMA_PICKLE_DIR.exists()

    for file_path in gamma_enfd_b_data_dir.iterdir():
        _m = re.match(r'g-([0-9]{3})_([A-Z,a-z]+)_([0-9]{3})\.endf', file_path.name)
        if _m:
            a = int(_m.groups()[2])
            symbol = _m.groups()[1]
            nuclide_name = '{0}{1}'.format(symbol, a)
            ActivationReactionContainer.from_endf(file_path, nuclide_name, 'gamma')

    for nuclide_name, reaction in ActivationReactionContainer.all_instances['gamma'].items():
        pickle_file_name = GAMMA_PICKLE_DIR / (nuclide_name + ".pickle")
        if len(reaction) == 0:  # this is probably not needed
            continue
        with open(pickle_file_name, "bw") as f:
            print('Creating and writing {}'.format(f.name))
            pickle.dump(reaction, f)


class Helper:
    def __init__(self, raw_data_dir: Path, new_data_dir: Path, file_name_converter):
        """

        Args:
            raw_data_dir: dir of downloaded data
            new_data_dir: Dir to save data
            matcher: a re.compiled object that has 2-3 capturing groups that are, in order, Z, A, <m>, where m is
             optional.
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
                        symbol = ATOMIC_SYMBOL[z] + str(a) + ("m{}".format(m) if m else '')
                        return symbol

            self.file_name_converter = file_name_converter

        else:
            assert hasattr(file_name_converter, '__call__')
            self.file_name_converter = file_name_converter

    def get_valid(self):
        """Generator that loops through source directory and returns dict of nuclide name and data path"""
        for f_path in self.raw_data_dir.iterdir():
            f_name = f_path.name
            if symbol := self.file_name_converter(f_name):
                if symbol is not None:
                    yield {'parent_symbol': symbol, 'f_path': f_path}



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
    sf_yield_marshal_path_gef = FISS_YIELDS_PATH/'SF'/'gef'
    proton_yield_marshal_path_ukfy = FISS_YIELDS_PATH/'proton'/'ukfy'
    gamma_yield_marshal_path_ukfy = FISS_YIELDS_PATH/'gamma'/'ukfy'

    for _dir in [neutron_yield_marshal_path_endf, neutron_yield_marshal_path_gef, sf_yield_marshal_path_gef,
                 proton_yield_marshal_path_ukfy]:
        if not _dir.parent.exists():
            Path.mkdir(_dir.parent)
        if not _dir.exists():
            Path.mkdir(_dir)

    # Add here for fiss yield
    # helpers = [Helper(neutron_fission_yield_data_dir_endf,
    #                   neutron_yield_marshal_path_endf,
    #                   'nfy-([0-9]+)_[a-zA-Z]+_([0-9]+)(?:m([0-9]))*')]
    helpers = [Helper(neutron_fission_yield_data_dir_gef,
                      neutron_yield_marshal_path_gef,
                      'GEFY_([0-9]+)_([0-9]+)_n.dat'),
               Helper(neutron_fission_yield_data_dir_endf,
                      neutron_yield_marshal_path_endf,
                      'nfy-([0-9]+)_[a-zA-Z]+_([0-9]+)(?:m([0-9]))*'),
               Helper(sf_yield_data_dir,
                      sf_yield_marshal_path_gef,
                      'GEFY_([0-9]+)_([0-9]+)_s.dat'),
               Helper(proton_fiss_yield_data_dir_ukfy,
                      proton_yield_marshal_path_ukfy,
                      lambda x: x),
               Helper(gamma_fiss_yield_data_dir_ukfy,
                      gamma_yield_marshal_path_ukfy,
                      lambda x: x)
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
        print('\t',m)

    for spec_type, spectra in d.spectra.items():
        print(spec_type, spectra)
        # for
    # print(d.spectra)

if __name__ == '__main__':
    pass
    import JSB_tools.nuke_data_tools
    # JSB_tools.nuke_data_tools.DEBUG = True
    debug_nuclide("Cd109", library="Jeff")
    # pickle_decay_data(pickle_data=True, nuclides_to_process=None)
    # n = Nuclide.from_symbol("Na22")
    # print(n.positron_intensity)
    #



