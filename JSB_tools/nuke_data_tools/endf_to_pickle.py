from __future__ import annotations
import pickle
from openmc.data.endf import Evaluation
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER, FissionProductYields
from openmc.data import Reaction, Decay
from pathlib import Path
import re
import marshal
from typing import Dict, List
from JSB_tools.nuke_data_tools import NUCLIDE_INSTANCES, Nuclide, DECAY_PICKLE_DIR, PHOTON_PICKLE_DIR,\
    PROTON_PICKLE_DIR,CrossSection1D, ActivationReactionContainer, SF_YIELD_PICKLE_DIR, NEUTRON_F_YIELD_PICKLE_DIR_GEF,\
    NEUTRON_F_YIELD_PICKLE_DIR_ENDF
from warnings import warn
from uncertainties import ufloat
from numbers import Number
import numpy as np


cwd = Path(__file__).parent

parent_data_dir = cwd / 'endf_files'

# download decay data from https://fispact.ukaea.uk/nuclear-data/downloads/. You should find a file titled
# 'general decay data' in the list.
decay_data_dir = parent_data_dir / 'decay'

# doenload the below at: https://www-nds.iaea.org/padf/
proton_padf_data_dir = parent_data_dir / 'PADF_2007' / 'Files'

#  Down load the data for the below at: https://www.nndc.bnl.gov/endf/b8.0/download.html
proton_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_protons'
photon_enfd_b_data_dir = parent_data_dir / 'ENDF-B-VIII.0_gammas'
neutron_fission_yield_data_dir_endf = parent_data_dir / 'ENDF-B-VIII.0_nfy'

#  Download SF yields (gefy model) from https://www.cenbg.in2p3.fr/GEFY-GEF-based-fission-fragment,780
sf_yield_data_dir = parent_data_dir / 'gefy81_s'
neutron_fission_yield_data_dir_gef = parent_data_dir / 'gefy81_n'

for _directory in [DECAY_PICKLE_DIR, PHOTON_PICKLE_DIR, PROTON_PICKLE_DIR,
                  SF_YIELD_PICKLE_DIR, NEUTRON_F_YIELD_PICKLE_DIR_GEF]:
    if not _directory.exists():
        _directory.mkdir()

_seconds_in_day = 24*60**2
_seconds_in_month = _seconds_in_day*30
_seconds_in_year = _seconds_in_month*12

hl_match_1 = re.compile('Parent half-life: +([0-9.E+-]+) ([a-zA-Z]+) ([0-9A-Z]+).*')
hl_match_2 = re.compile('T1\/2=([0-9.decE+-]+) ([A-Z]+).*')
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
    print(x_str)
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
        return 0


def pickle_decay_data():
    directory_1 = decay_data_dir  # Path to downloaded ENDF decay data
    directory_2 = decay_data_dir/'decay_2020'  # Path to downloaded ENDF decay data

    assert directory_1.exists()
    assert directory_2.exists()
    for file_path in directory_1.iterdir():
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
        if parent_nuclide.name == 'n1':
            print('n1 n1 n1: ', file_path)

        if openmc_decay.nuclide["stable"]:
            openmc_decay.half_life = ufloat(np.inf, 0)
        elif openmc_decay.half_life.n == 0:
            openmc_decay.half_life = get_hl_from_ednf_file(file_path)
            print('My half life: {}'.format(openmc_decay.half_life))

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


#  Special case implemented in pickle_proton_fission_data() for proton induced fission cross-sections.
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
            xs_obj = CrossSection1D(ergs, xss, reaction_name, 'proton')
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
                xs_fig_label = '{0}{1}(G,F)'.format(nuclide_name, a)
                xs = CrossSection1D(fission_xs.x/1E6, fission_xs.y, xs_fig_label, 'photon')
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

            cumulative_dir = SF_YIELD_PICKLE_DIR/'cumulative'
            independent_dir = SF_YIELD_PICKLE_DIR/'independent'

            if not cumulative_dir.exists():
                cumulative_dir.mkdir()

            if not independent_dir.exists():
                independent_dir.mkdir()

            with open(cumulative_dir/'{}.pickle'.format(nuclide_name), 'wb') as f:
                pickle.dump(y.cumulative[0], f)
            with open(independent_dir/'{}.pickle'.format(nuclide_name), 'wb') as f:
                pickle.dump(y.independent[0], f)


def pickle_gef_neutron_yields():
    ergs_flag = False  # only write energies once.

    for f_path in neutron_fission_yield_data_dir_gef.iterdir():
        _m = re.match('GEFY_([0-9]+)_([0-9]+)_n.dat', f_path.name)
        if _m:
            z = int(_m.groups()[0])
            a = int(_m.groups()[1])
            e_symbol = ATOMIC_SYMBOL[z]

            try:
                print('Pickling neutron-induced fission data for {}, z={}, a={} from file {}'.format(e_symbol, z, a, f_path))
                y = FissionProductYields(str(f_path))
            except KeyError:
                warn('Failed to load neutron-induced fission data from "{}" in "{}"'.format(e_symbol, f_path.name))
                continue

            data = {'independent': {}, 'cumulative': {}}

            for yield_type in data.keys():
                parent_directory = NEUTRON_F_YIELD_PICKLE_DIR_GEF / yield_type
                if not parent_directory.exists():
                    parent_directory.mkdir()

                for i, erg in enumerate(y.energies):
                    # for product, yield_ in y.independent[i].items():
                    for product, yield_ in getattr(y, yield_type)[i].items():
                        assert isinstance(product, str)
                        try:
                            data[yield_type][product]['ergs'].append(float(erg))
                            data[yield_type][product]['yield'].append(float(yield_.n))
                            data[yield_type][product]['yield_err'].append(float(yield_.std_dev))
                        except KeyError:
                            data[yield_type][product] = {'ergs': [erg], 'yield': [yield_.n],
                                                         'yield_err': [float(yield_.std_dev)]}

            for yield_type, yield_data in data.items():
                for daughter_nuclide in yield_data.keys():
                    for key in yield_data[daughter_nuclide].keys():
                        if key == 'ergs':
                            if not ergs_flag:
                                ergs_flag = True
                                with open(NEUTRON_F_YIELD_PICKLE_DIR_GEF / 'yield_ergs.pickle', 'wb') as f:
                                    erg_data = np.array(yield_data[daughter_nuclide]['ergs'])*1E-6  # eV -> MeV
                                    pickle.dump(erg_data, f)
                        else:
                            # set yield and yield_error
                            yield_data[daughter_nuclide][key] = list(yield_data[daughter_nuclide][key])

                    del yield_data[daughter_nuclide]['ergs']
                with open(NEUTRON_F_YIELD_PICKLE_DIR_GEF / yield_type / (y.nuclide['name'] + '.marshal'), 'wb') as f:
                    marshal.dump(yield_data, f)


def pickle_endf_neutron_yields():
    for f_path in neutron_fission_yield_data_dir_endf.iterdir():
        _m = re.match('nfy-([0-9]+)_([a-zA-Z]+)_([0-9]+)(m[0-9])*', f_path.name)
        if _m:
            s = _m.groups()[1] + _m.groups()[2]
            if _m.groups()[-1] is not None:
                s += '_' + _m.groups()[-1]
            y = FissionProductYields(f_path)
            ergs = y.energies/1E6
            for yield_type in ['independent', 'cumulative']:
                data = {'ergs': [float(e) for e in ergs]}
                new_f_path = Path(str(NEUTRON_F_YIELD_PICKLE_DIR_ENDF/yield_type/s) + '.marshal')
                if not new_f_path.parent.exists():
                    new_f_path.parent.mkdir()
                for yields in getattr(y, yield_type):
                    for product, yield_ in yields.items():
                        if product not in data:
                            data[product] = {'yield': [yield_.n], 'yield_err': [float(yield_.std_dev)]}
                        else:
                            data[product]['yield'].append(yield_.n)
                            data[product]['yield_err'].append(float(yield_.std_dev))
                with open(new_f_path, 'wb') as f:
                    marshal.dump(data, f)





def pickle_all_nuke_data():
    # pickle_gef_neutron_yields()
    # pickle_endf_neutron_yields()
    # pickle_sf_yields()
    pickle_decay_data()
    # pickle_proton_activation_data()
    # pickle_proton_fission_data()  # pickle proton fission data in a special way due to compatibility issues with EDNF6
    # pickle_photon_fission_data()
    # pickle_photon_activation_data()

    pass

r = re.compile('([A-Z][a-z]{0,2})([0-9]{1,3})([mnop]*)')

# all_ns = [n.name for n in Nuclide.get_all_nuclides()]
openmc_decays = {}

new_ns = list()
openmc_decays = {}

# print('# of paths: {}'.format(len(paths)))
for path in (decay_data_dir/'decay_2020').iterdir():
    if m := r.match(path.name):
        s = m.groups()[0]
        a = int(m.groups()[1])
        m = m.groups()[2]
        z = ATOMIC_NUMBER[s]
        m = {'':0, 'm':1, 'n':2, 'o':3}[m]
        e = Evaluation(path)
        if m != 0:
            name = '{}{}_m{}'.format(s, a, m)
        else:
            name = '{}{}'.format(s, a)
        if name in openmc_decays:
            openmc_decays[name][1] = Decay(e)
        else:
            openmc_decays[name] = [None, Decay(e)]

for path in decay_data_dir.iterdir():
    if m := re.match(r"dec-[0-9]{3}_(?P<S>[A-Za-z]{1,2})_(?P<A>[0-9]+)(?:m(?P<M>[0-9]+))?\.endf", path.name):
        s = m.group('S')
        a = int(m.group('A'))
        m = m.group('M')
        if m:
            name = '{}{}_m{}'.format(s, a, m)
        else:
            name = "{}{}".format(s, a)
        e = Evaluation(path)

        if name in openmc_decays:
            openmc_decays[name][0] = Decay(e)
        else:
            openmc_decays[name] = [Decay(e), None]

for name, decays in openmc_decays.items():
    score_old = decay_evaluation_score(decays[0])
    score_new = decay_evaluation_score(decays[1])
    msg = str([score_old, score_new])
    if score_new > score_old:
        msg += ' New'
    elif score_new == score_old:
        msg += 'Tie'
    else:
        msg += 'Old'
    print(name, msg)
#         d_new = Decay(e)
#         old_path = '/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/decay/'
#         z = str(z)
#         old_path += 'dec-{}_{}_{}.endf'.format(z if len(z) == 3 else '0' + z, s, a if len(a) == 3 else '0' + a)
#         old_path = Path(old_path)
#         if old_path.exists():
#             e = Evaluation(old_path)
#             d_old = Decay(e)
#             # print()
#             c = True
#             if 'gamma' in d_old.spectra and d_old.spectra['gamma']['continuous_flag'] == 'discrete':
#                 l_old = len(d_old.spectra['gamma']['discrete'])
#                 # print('old: ',l_old,  d_old.spectra['gamma'])
#
#
#                 if 'gamma' in d_new.spectra and d_new.spectra['gamma']['continuous_flag'] == 'discrete':
#                     l_new = len(d_new.spectra['gamma']['discrete'])
#                     # print('new: ',l_new,  d_new.spectra['gamma'])
#                     if l_new != l_old:
#                         print('new: ', l_new, d_new.spectra['gamma'])
#                         print('old: ', l_old, d_old.spectra['gamma'])
#
#                     no_new = False
#                 else:
#                     print('Yes old no new: ', old_path)
#                 # print('new: None')
#                 pass
#
#
#
#                 # if no_new:
#                 #     print ('########')
#             # else:
#             #     print('old: None')
#             # print()
#
#
#         new_ns.append(n.name)
# print('Number of matches: ', len(new_ns))
# # all_ns = list(sorted(all_ns))
# # new_ns = list(sorted(new_ns))
# # print(all_ns)
# # print(new_ns)
# # for n in all_ns:
# #     if n not in new_ns:
# #         print('omg?!?!?!?, ', n)



if __name__ == '__main__':
    pass
    import matplotlib.pyplot as plt
    # pickle_all_nuke_data()

    # e = Evaluation('/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/endf_files/decay/decay_2020/C010')
    # d =Decay.from_endf(e)
    # print(d.spectra)


