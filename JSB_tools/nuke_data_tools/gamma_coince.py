"""
Calculate the rate of coincident gammas emitted from nuclei.
"""
import re
from numpy import random
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Dict
from uncertainties import ufloat_fromstr, ufloat
from uncertainties.core import UFloat
from pathlib import Path


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
               f"\tfrom {self.start_level} KeV to {self.end_level} KeV (Q:{self.q_value})\n" \
               f"\tAbs Y intensity: {100 * self.gamma_intensity: .2f}%\n"\
               f"\tfeeding intensity: {100*self.feeding_intensity:.2f}%;\n" \
               f"\tTransitions per decay: {100*self.transition_intensity:.2f}%\n"


class Levels:
    """
    Calculates the joint probability distribution of gamma-gamma pairs.

    The class loads a nuclear levels CSV file that can be downloaded from the following, link under
    the "Decay Radiation" tab:
        https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html
    Once the "Decay Radiation" tab is opened, click the "CSV" icon at the top of the "Gammas" section, which will
    begin a download automatically. Save the CSV file to the following path:
        JSB_tools/nuke_data_tools/data/gamma_levels/<nuclide_name>
    , where nuclide_name is, e.g., "Eu152".

    Attributes:
        coinc_emissions: Joint probability distribution of gamma-gamma pairs. It is a dictionary with keys equal to
            gamma energies. The following format is used:
            {gamma_erg_0: {"intensity": I_0, "coinc": [(gamma_erg_1, P[g_1|g_0]), (gamma_erg_2 P[g_2|g_0])]}}
            where,
                g_0: Refers to the event of the gamma with energy in the corresponding dictionary key
                gamma_erg_0: The aforementioned dictionary key, equal to the energy of g_0
                I_0: Singles intensity of the g_0

                g_1: Refers to the event of a particular gamma with energy gamma_erg_1 being emitted
                gamma_erg_1: The energy of g_1
                P[g_1|g_0]): The probability of g_1 being emitted given that g_0 was emitted.

                g_2: Same as above.

        level_population_probs: Probability that a given daughter level is populated during decay.

        transitions: List of Transition instances

    """
    def __init__(self, nuclide_name, parent_energy=0):
        """

        Args:
            nuclide_name: e.g. "Y88"
            parent_energy: e.g. if the first excited state is desired, set to 1.
        """
        self.nuclide_name = nuclide_name
        m = re.match('([A-Z][a-z]{0,2}[0-9]{1,3})(_m[0-9])*', nuclide_name)
        assert m, f'Invalid nuclide name, {nuclide_name}. Correct example: Xe139, Co57, Ba155_m1'

        f_path = Path(__file__).parent/'data'/'gamma_levels'/m.groups()[0]
        if not f_path.exists():
            print(f_path)
            raise FileNotFoundError(f"Cannot find levels data for {nuclide_name}.\n"
                                    f"See class doc strong on how to download the data.")
        with open(f_path) as f:
            lines = f.readlines()
        header = [x.rstrip().lstrip() for x in lines[0].split(',')]
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

        self.level_population_probs: Dict[float, UFloat] = {}

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

        self.coinc_emissions = {}
        for t_high in self.transitions:
            for t_low in self.transitions:
                if t_high.start_level <= t_low.start_level:
                    continue
                elif t_low.decay_channel == t_high.decay_channel and t_high.end_level == t_low.start_level:
                    # prob_low_given_high = t_low.gamma_intensity/sum(t.transition_intensity for t in self.transitions if
                    #                                                 t.start_level == t_low.start_level)
                    # Below is the probability that the given gamma is emitted given the it's excited level is excited.

                    prob_low_given_high = t_low.decay_by_gamma_prob * t_low.transition_intensity / sum(
                        t.transition_intensity for t in self.transitions if
                        t.start_level == t_low.start_level)
                    prob_high = t_high.gamma_intensity
                    prob_low = t_low.gamma_intensity
                    phigh_given_low = prob_high*prob_low_given_high/prob_low
                    # print(t_high, t_low, 'prob_low_given_high: ', prob_low_given_high,
                    #       '\nprob_low_given_high/prob_low', prob_low_given_high/prob_low)
                    # print()

                    try:
                        self.coinc_emissions[t_high.q_value]['coinc'].append((t_low.q_value, prob_low_given_high))
                    except KeyError:
                        self.coinc_emissions[t_high.q_value] = {'g_intensity': prob_high,
                                                                't_intensity': t_low.transition_intensity}
                        self.coinc_emissions[t_high.q_value]['coinc'] = [(t_low.q_value, prob_low_given_high)]

                    try:
                        self.coinc_emissions[t_low.q_value]['coinc'].append((t_high.q_value, phigh_given_low))
                    except KeyError:
                        self.coinc_emissions[t_low.q_value] = {'g_intensity': t_low.gamma_intensity,
                                                               't_intensity': t_low.transition_intensity}
                        self.coinc_emissions[t_low.q_value]['coinc'] = [(t_high.q_value, phigh_given_low)]

        self.coinc_emissions = {k: v for k, v in sorted(self.coinc_emissions.items(), key=lambda x: -x[1]['g_intensity'])}

    def print_coinc(self, probability_cut_off: float = 0):
        """
        Args:
            probability_cut_off: Any coinc or abs intensity below this is not printed.
        Returns:

        """
        print(f'Coincident gammas from {self.nuclide_name}')
        for k,v in self.coinc_emissions.items():
            if v['g_intensity'] < probability_cut_off:
                continue
            coincidences = v['coinc']
            tot = sum([x[1] for x in coincidences]).n
            coinc_header = "Eg [KeV]  P[g_i|g_0]"
            printable_coinc = "\n".join([f"\t\t{erg: <8.2f}  {100*p.n:.2f}%" for erg, p in coincidences if p>= probability_cut_off])
            print(f"\t{k:.3f} KeV; "
                  f"gamma intensity: {100*v['g_intensity'].n:.2g}%; "
                  f"transition intensity: {100*v['t_intensity'].n:.2g}%; "
                  f"tot coinc.: {tot:.3f}\n\t\t{coinc_header}\n{printable_coinc}\n")

    def __len__(self):
        return len(self.end_levels)

    def get_coince(self, e_gamma) -> list:
        keys = np.array(list(self.coinc_emissions.keys()))
        i = np.argmin(np.abs(keys-e_gamma))
        return self.coinc_emissions[keys[i]]['coinc']


if __name__ == '__main__':

    l = Levels('Xe139')
    l.print_coinc(probability_cut_off=0.0)

