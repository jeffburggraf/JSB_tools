from numpy import random
import numpy as np
from matplotlib import pyplot as plt
from typing import List
from uncertainties import ufloat_fromstr, ufloat


def eff(erg):
    return np.e**(-13.82 + 4.8*np.log(erg) - 0.49*np.log(erg)**2)


class Transition:
    """
    Attributes:
        start_level: the initial excitation energy of this transition in KeV

        end_level: the final excitation energy of this transition in KeV

        q_value: The energy released in the transition.

        gamma_intensity: Gamma's emitted per decay via this transition.

        decay_channel: b+/EC, b-, alpha, etc.

        decay_by_gamma_prob: Probability that a gamma is emitted during this transition
            (instead of "internal conversion")

        transition_intensity: Chance of this transition in the decay pof the parent.

        nuclide_name: Name of parent nuclei

        feeding_intensity: The change that self.start_level is the excited state populated immediately after parent
            decay.
    """
    # decays = {}

    def __init__(self, nuclide_name, start_level, end_level, conversion_coeff, gamma_intensity, decay_type):
        self.start_level = start_level
        self.end_level = end_level
        self.q_value = self.start_level - self.end_level
        self.gamma_intensity = gamma_intensity
        self.decay_channel = decay_type

        self.decay_by_gamma_prob = 1.0 / (1 + conversion_coeff)
        self.transition_intensity = gamma_intensity / self.decay_by_gamma_prob
        self.nuclide_name = nuclide_name

        self.feeding_intensity = 0

        # try:
        #     Transition.decays[nuclide_name].append(self)
        # except KeyError:
        #     Transition.decays[nuclide_name] = [self]

    def __repr__(self):
        return f"Nuclear level transition ({self.decay_channel}):\n" \
               f"\tfrom {self.start_level} KeV to {self.end_level} KeV\n" \
               f"\tAbs Y intensity: {100 * self.gamma_intensity: .2f}%\n"\
               f"\tfeeding intensity: {100*self.feeding_intensity:.2f}%;\n" \
               f"\tTransitions per decay: {100*self.transition_intensity:.2f}%\n"


class Levels:
    def __init__(self, f_path, nuclide_name, parent_energy=0):
        with open(f_path) as f:
            lines = f.readlines()
        header = [x.rstrip().lstrip() for x in lines[0].split(',')]
        # header[header.index('isospin')+1] = "Q unc"  # from hear through "[..., 'electric quadrupole [b]', 'unc']" are bs. delete.
        erg_index = header.index('energy')
        gamma_intensity_index = header.index('intensity_%')
        start_level_index = header.index('start level energy  [keV]')
        end_level_index = header.index('end level energy  [keV]')
        conversion_coeff_index = header.index('conversion coeff.')
        decay_index = header.index('decay')

        parent_erg_index = header.index('parent energy  [keV]')

        self.gamma_ergs = []
        self.gamma_intensities = []
        self.start_levels = []
        self.end_levels = []
        self.gamma_probs = []
        self.test = []
        self.transition_intensities = []

        self.transitions: List[Transition] = []

        def get_data(line):
            line = line.rstrip().lstrip()
            if not line:
                return None
            data = line.split(',')

            try:
                _parent_energy = float(data[parent_erg_index])
                if parent_energy != _parent_energy:
                    return None
                start_level = float(data[start_level_index])
                end_level = float(data[end_level_index])
                gamma_intensity = float(data[gamma_intensity_index])
                gamma_intensity_unc = data[gamma_intensity_index+1].rstrip()
                if gamma_intensity_unc == '':
                    gamma_intensity_unc = 0
                gamma_intensity = ufloat_fromstr(f"{gamma_intensity}({gamma_intensity_unc})")/100

                decay_type = data[decay_index]
                conversion_coeff = data[conversion_coeff_index].rstrip()
                conversion_coeff_unc = data[conversion_coeff_index+1].rstrip()

                if conversion_coeff_unc == '':
                    conversion_coeff_unc = 0
                if conversion_coeff == '':
                    conversion_coeff = ufloat(0, 0)
                else:
                    conversion_coeff = ufloat_fromstr(f"{conversion_coeff}({conversion_coeff_unc})")
            except ValueError as e:
                return None
            self.transitions.append(Transition(nuclide_name, start_level, end_level,
                                               conversion_coeff, gamma_intensity, decay_type))

        for line in lines[1:]:
            get_data(line)

        self.level_population_probs = {}

        for index, transition in enumerate(self.transitions):

            start_level = transition.start_level
            try:
                self.level_population_probs[start_level] += transition.transition_intensity

            except KeyError:
                self.level_population_probs[start_level] = transition.transition_intensity

            for other_transition in self.transitions:
                if other_transition is transition:
                    continue
                if other_transition.end_level == transition.start_level and\
                        other_transition.decay_channel == transition.decay_channel:
                    self.level_population_probs[start_level] -= other_transition.transition_intensity

        for transition in self.transitions:
            transition.feeding_intensity = self.level_population_probs[transition.start_level]

        self.transitions = list(sorted(self.transitions, key=lambda x: -x.gamma_intensity))

        # for t in self.transitions:
        #     print(t)

        coinc_emissions = {}
        for t_high in self.transitions:
            for t_low in self.transitions:
                if t_high.start_level <= t_low.start_level:
                    continue
                #  P(L|H) = P(H|L)*P(L)/P(H)
                #  P(H|L) = P(L|H)*P(H)/P(L)
                elif t_low.decay_channel == t_high.decay_channel and t_high.end_level == t_low.start_level:
                    # prob_low_gamma_given_pop = t_low.gamma_intensity/sum(t.transition_intensity for t in self.transitions if t.start_level == t_low.start_level)
                    prob_low_given_high = t_low.gamma_intensity/sum(t.transition_intensity for t in self.transitions if t.start_level == t_low.start_level)
                    prob_high = t_high.gamma_intensity
                    prob_low = t_low.gamma_intensity
                    phigh_given_low = t_high.gamma_intensity*prob_low_given_high/prob_low
                    print(t_high, t_low, 'prob_low_given_high: ', prob_low_given_high,
                          '\nprob_low_given_high/prob_low', prob_low_given_high/prob_low)
                    print()

                    try:
                        coinc_emissions[t_high.q_value]['coinc'].append((t_low.q_value, prob_low_given_high))
                    except KeyError:
                        coinc_emissions[t_high.q_value] = {'intensity': prob_high}
                        coinc_emissions[t_high.q_value]['coinc'] = [(t_low.q_value, prob_low_given_high)]

                    try:
                        coinc_emissions[t_low.q_value]['coinc'].append((t_high.q_value, phigh_given_low))
                    except KeyError:
                        coinc_emissions[t_low.q_value] = {'intensity': t_low.gamma_intensity}
                        coinc_emissions[t_low.q_value]['coinc'] = [(t_high.q_value, phigh_given_low)]

                        # coinc_emissions[t_high.q_value]['tot'] = prob_low_given_high


                    # coinc_emissions[f'P({t_low.q_value:.2f}|{t_high.q_value:.2f})'] = low_level_branching_ratio
                    # coinc_emissions[f'P({t_high.q_value:.2f}|{t_low.q_value:.2f})'] =

        coinc_emissions = {k:v for k, v in sorted(coinc_emissions.items(), key=lambda x: -x[1]['intensity'])}
        for k,v in coinc_emissions.items():
            coincidences = v['coinc']
            tot = sum([x[1] for x in coincidences]).n
            printable_coinc = [f"{erg} KeV at {100*p.n:.2f}%" for erg, p in coincidences]
            print(k, f"intensity: {100*v['intensity']}, tot coinc.: {tot}; coincidences: {printable_coinc}")

    def __len__(self):
        return len(self.end_levels)


l = Levels('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/data/gamma_levels/Y88','Y88')
# l = Levels('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/data/gamma_levels/Co57','Co57')
# l = Levels('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/data/gamma_levels/Eu152','eu152')
# print(l.co)
