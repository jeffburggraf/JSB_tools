from __future__ import annotations
import pickle
from JSB_tools import ProgressReport
from openmc.data.endf import Evaluation
from openmc.data import Reaction, Decay
from pathlib import Path
import re
import marshal
from warnings import warn
from uncertainties import ufloat, ufloat_fromstr
from endf_to_pickle import iter_paths
import numpy as np
from data_directories import decay_data_dir, DECAY_PICKLE_DIR
import JSB_tools.nuke_data_tools.nuclide as nuclide_module


cwd = Path(__file__).parent

_seconds_in_day = 24*60**2
_seconds_in_month = _seconds_in_day*30
_seconds_in_year = _seconds_in_month*12

hl_match_1 = re.compile(r'Parent half-life: +([0-9.E+-]+) ([a-zA-Z]+) ([0-9A-Z]+).*')
hl_match_2 = re.compile(r'T1\/2=([0-9.decE+-]+) ([A-Z]+).*')
stable_match = re.compile('Parent half-life: STABLE.+')


def get_hl_from_ednf_file(path_to_file):
    """
    Used to correct for bug in openmc where the ENDF evaluation of some stable isotopes return a half life of zero.
    Also, very long half-lives, e.g. 1E12 years, were also returning a half life of zero.
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
            hl_out = ufloat_fromstr(f"{hl}({err})")
            # hl_out = get_error_from_error_in_last_digit(hl, err)
            return scale * hl_out

    return ufloat(hl*scale, err*scale)


def set_nuclide_attributes(_self: nuclide_module.Nuclide, ev: Evaluation, open_mc_decay: Decay):
    _self.half_life = open_mc_decay.half_life
    _self.mean_energies = open_mc_decay.average_energies
    _self.spin = open_mc_decay.nuclide["spin"]

    _self.excitation_energy = ev.target['excitation_energy'] * 1E-3  # eV -> keV
    _self.nuclear_level = ev.target['state']
    _self.fissionable = ev.target['fissionable']

    if np.isinf(_self.half_life.n) or open_mc_decay.nuclide["stable"]:
        _self.is_stable = True
        _self.half_life = ufloat(np.inf, 0)
    else:
        _self.is_stable = False

    for mode in open_mc_decay.modes:
        decay_mode = nuclide_module.DecayMode(mode, _self.half_life)
        try:
            _self.decay_modes[tuple(mode.modes)].append(decay_mode)
        except KeyError:
            _self.decay_modes[tuple(mode.modes)] = [decay_mode]

    l = len(_self.decay_modes)

    for key, value in list(_self.decay_modes.items())[:]:
        _self.decay_modes[key] = list(sorted(_self.decay_modes[key], key=lambda x: -x.branching_ratio))

    assert len(_self.decay_modes) == l


def pickle_decay_data(pickle_nuclides=True, pickle_spectra=True, nuclides_to_process=None):
    """
    Pickles nuclide properties into ../data/nuclides/x.pickle
    Writes   __fast__gamma_dict__.marshal, which can be used to quickly look up decays by decay energy.
    data structure of __fast__gamma_dict__:
        {
         g_erg_1: ([name1, name2, ...], [intensity1, intensity2, ...], [half_life1, half_life2, ...]),
         g_erg_2: (...)
         }
    Args:
        pickle_nuclides: If false, don't save to pickle files.
        pickle_spectra:
        nuclides_to_process: If None, process all data. Otherwise, should be a list of nuclide names,
            e.g. ["Na22", "U240"].

    Returns:

    """
    if nuclides_to_process is not None:
        assert hasattr(nuclides_to_process, '__iter__')
        assert isinstance(nuclides_to_process[0], str)
        pickle_nuclides = pickle_spectra = False  # Don't save to pickle file in this case.

    directory_endf = decay_data_dir  # Path to downloaded ENDF decay data

    prog = ProgressReport(len(list(directory_endf.iterdir())), 5)

    if not directory_endf.exists() or len(list(directory_endf.iterdir())) == 0:
        raise FileNotFoundError('Download ENDF decay data and place it into {}'.format(directory_endf))

    openmc_decays = {}

    i = 0
    # for endf_file_path in directory_endf.iterdir():
    for endf_file_path in iter_paths(directory_endf):
        i += 1
        prog.log(i)

        try:
            e = Evaluation(endf_file_path)

        except (KeyError, ValueError):
            print(f"Failed for {endf_file_path.name}")
            continue

        nuclide_name = e.gnd_name

        if nuclides_to_process is not None and nuclide_name not in nuclides_to_process:
            continue

        d = Decay(e)

        if d.nuclide["stable"]:
            half_life = ufloat(np.inf, 0)
        elif d.half_life.n == 0:
            half_life = get_hl_from_ednf_file(endf_file_path)
        else:
            half_life = d.half_life

        openmc_decays[nuclide_name] = {'evaluation': e, 'decay': d, 'half_life': half_life}

    for parent_nuclide_name, openmc_dict in openmc_decays.items():
        openmc_decay = openmc_dict['decay']
        openmc_evaluation = openmc_dict['evaluation']

        openmc_decay.half_life = openmc_dict['half_life']

        if parent_nuclide_name in nuclide_module.Nuclide.all_instances:
            parent_nuclide = nuclide_module.Nuclide.all_instances[parent_nuclide_name]
        else:
            parent_nuclide = nuclide_module.Nuclide(parent_nuclide_name, _default=True)
            nuclide_module.Nuclide.all_instances[parent_nuclide_name] = parent_nuclide

        daughter_names = [mode.daughter for mode in openmc_decay.modes]

        for daughter_nuclide_name in daughter_names:
            if daughter_nuclide_name == parent_nuclide_name:
                continue

            if daughter_nuclide_name in nuclide_module.Nuclide.all_instances:
                daughter_nuclide = nuclide_module.Nuclide.all_instances[daughter_nuclide_name]
            else:
                daughter_nuclide = nuclide_module.Nuclide(daughter_nuclide_name, _default=True)
                nuclide_module.Nuclide.all_instances[daughter_nuclide_name] = daughter_nuclide

            if daughter_nuclide_name != parent_nuclide_name:
                daughter_nuclide.__decay_parents_str__.append(parent_nuclide_name)
                parent_nuclide.__decay_daughters_str__.append(daughter_nuclide_name)

        set_nuclide_attributes(parent_nuclide, openmc_evaluation, openmc_decay)

        if pickle_spectra:
            for spectra_mode, spec_data in openmc_decay.spectra.items():
                spectra = nuclide_module._DiscreteSpectrum(parent_nuclide, spec_data)
                spectra.__pickle__()

    if pickle_nuclides:
        for nuclide_name in nuclide_module.Nuclide.all_instances.keys():
            if nuclides_to_process is not None and nuclide_name not in nuclides_to_process:
                continue
            with open(DECAY_PICKLE_DIR/(nuclide_name + '.pickle'), "wb") as pickle_file:
                print("Writing decay data for {0}".format(nuclide_name))
                pickle.dump(nuclide_module.Nuclide.all_instances[nuclide_name], pickle_file)

        if nuclides_to_process is None:  # None means all here.
            d = {}
            for nuclide in nuclide_module.Nuclide.all_instances.values():
                for g in nuclide.decay_gamma_lines:
                    erg = g.erg.n
                    intensity: float = g.intensity.n
                    hl = nuclide.half_life.n
                    try:
                        i = len(d[erg][1]) - np.searchsorted(d[erg][1][::-1], intensity)
                        # the below flow_pat control is to avoid adding duplicates.
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


if __name__ == '__main__':
    pickle_decay_data(True, False, nuclides_to_process=None)