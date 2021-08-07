from numpy import random
import numpy as np
from matplotlib import pyplot as plt
from typing import List

def eff(erg):
    return np.e**(-13.82 + 4.8*np.log(erg) - 0.49*np.log(erg)**2)


class Transition:
    decays = {}

    def __init__(self, nuclide_name, start_level, end_level, conversion_coeff, gamma_intensity):
        self.start_level = start_level
        self.end_level = end_level
        self.gamma_erg = self.start_level - self.end_level
        self.gamma_intensity = gamma_intensity

        self.decay_by_gamma_prob = 1.0 / (1 + conversion_coeff)
        self.transition_intensity = gamma_intensity / self.decay_by_gamma_prob
        self.nuclide_name = nuclide_name

        self.population_prob = 0

        try:
            Transition.decays[nuclide_name].append(self)
        except KeyError:
            Transition.decays[nuclide_name] = [self]

    def __repr__(self):
        return f"from {self.start_level} to {self.end_level}; populated {100*self.population_prob}% of decays;" \
               f" Abs Y intensity: {100*self.gamma_intensity}"


class Levels:
    def __init__(self, f_path, nuclide_name, parent_energy=0):
        with open(f_path) as f:
            lines = f.readlines()
        header = [x.rstrip().lstrip() for x in lines[0].split(',')]
        # header[header.index('isospin')] = "Q"
        # header[header.index('isospin')+1] = "Q unc"  # from hear through "[..., 'electric quadrupole [b]', 'unc']" are bs. delete.
        erg_index = header.index('energy')
        gamma_intensity_index = header.index('intensity_%')
        start_level_index = header.index('start level energy  [keV]')
        end_level_index = header.index('end level energy  [keV]')
        conversion_coeff_index = header.index('conversion coeff.')

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
                gamma_intensity = float(data[gamma_intensity_index])/100
                erg = float(data[erg_index])
                conversion_coeff = data[conversion_coeff_index].rstrip()

                if conversion_coeff == '':
                    conversion_coeff = 0
                else:
                    conversion_coeff = float(conversion_coeff)
                gamma_prob = 1.0/(1+conversion_coeff)
            except ValueError:
                return None
            self.transitions.append(Transition(nuclide_name, start_level, end_level,
                                               conversion_coeff, gamma_intensity))
            # self.gamma_ergs.append(erg)
            # self.gamma_intensities.append(gamma_intensity)
            # self.start_levels.append(start_level)
            # self.end_levels.append(end_level)
            # self.gamma_probs.append(gamma_prob)
            # self.transition_intensities.append(gamma_intensity/gamma_prob)
        #
        #
        # arg_sort = np.argsort(self.transition_intensities)[::-1]
        # self.start_levels = np.array(self.start_levels)[arg_sort]
        # self.end_levels = np.array(self.end_levels)[arg_sort]
        # self.transition_intensities = np.array(self.transition_intensities)[arg_sort]
        # self.gamma_probs = np.array(self.gamma_probs)[arg_sort]
        # self.gamma_ergs = np.array(self.gamma_ergs)[arg_sort]
        # self.gamma_intensities = np.array(self.gamma_intensities)[arg_sort]

        # P(a|b)=P(ab)/P(b)
        for line in lines:
            get_data(line)

        self.level_population_probs = {}

        for index, transition in enumerate(self.transitions):
            # print('Start end levels: ', self.start_levels[i1], self.end_levels[i1],
            #       " ... Y erg", self.gamma_ergs[i1],
            #       "abs transition intensity: ", self.transition_intensities[i1]
            #       )
            # transition = self.transitions[i1]
            start_level = transition.start_level
            # print(start_level, self.level_population_probs)
            try:
                self.level_population_probs[start_level] += transition.transition_intensity
                if start_level == 1085.8408:
                    print(f"Adding {transition.transition_intensity} to {start_level} to get {self.level_population_probs[start_level]}")

            except KeyError:
                if start_level == 1085.8408:
                    print(f"Beginning {start_level} at {transition.transition_intensity}")
                self.level_population_probs[start_level] = transition.transition_intensity
                # print(f"Adding the following to {start_level}: {transition} ")

            for other_transition in self.transitions[index+1:]:
                if other_transition is transition:
                    continue

                elif other_transition.end_level == transition.start_level:
                    self.level_population_probs[start_level] -= other_transition.transition_intensity
                    if start_level == 1085.8408:
                        print(f"Subtracting {other_transition.transition_intensity} from {start_level} to get {self.level_population_probs[start_level]}")

        for k, v in self.level_population_probs.items():
            print(k, 100*v)
        for t in self.transitions:
            print(t)

    def __len__(self):
        return len(self.end_levels)


# l = Levels('/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/data/gamma_levels/Y88','Y88')
l = Levels('/Users/burggraf1/Documents/Eu152GammaLevels.csv','eu152')

