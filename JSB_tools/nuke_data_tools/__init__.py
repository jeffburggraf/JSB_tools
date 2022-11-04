# from __future__ import annotations
# import pickle
# import numpy as np
# import warnings
# from matplotlib import pyplot as plt
# import re
# from pathlib import Path
# from warnings import warn
# from uncertainties import ufloat, UFloat
# from typing import Union, List, Dict, Tuple
# from global_directories import DECAY_PICKLE_DIR, PROTON_PICKLE_DIR, GAMMA_PICKLE_DIR, \
#     NEUTRON_PICKLE_DIR, FISS_YIELDS_PATH
# from uncertainties import nominal_value
# from datetime import datetime, timedelta
from logging import warning as warn

def s():









    #  Units

    #  Note to myself: Pickled nuclear data is on personal SSD. Update this regularly!
    #  Todo:
    #   * make cross section pulls be implemented in a nuke_data.cross_secion file. Let the endf to pickle also be
    #     implemented there
    #   * Make a Nuclide.fromreaction('parent_nuclide_name', inducing particle, daughter_nuclide_name )
    #   * Add documentation, and exception messages, to explain where the data can be downloaded and how to regenerate
    #     the pickle files.
    #   * Get rid of <additional_nuclide_data> functionality. too complex and stupid


    NUCLIDE_INSTANCES = {}  # Dict of all Nuclide class objects created. Used for performance enhancements and for pickling
    PROTON_INDUCED_FISSION_XS1D = {}  # all available proton induced fissionXS xs. lodaed only when needed.
    PHOTON_INDUCED_FISSION_XS1D = {}  # all available gamma induced fissionXS xs. lodaed only when needed.
    NEUTRON_INDUCED_FISSION_XS1D = {}  # all available neutron induced fissionXS xs. lodaed only when needed.


    # global variable for the bin-width of xs interpolation
    XS_BIN_WIDTH_INTERPOLATION = 0.1

    # Some additional nuclide info that aren't in ENDSFs
    additional_nuclide_data = {"In101_m1": {"half_life": ufloat(10, 5)},
                               "Lu159_m1": {"half_life": ufloat(10, 5)},
                               "Rh114_m1": {"half_life": ufloat(1.85, 0.05), "__decay_daughters_str__": "Pd114"},
                               "Pr132_m1": {"half_life": ufloat(20, 5), "__decay_daughters_str__": "Ce132"}}












    def decay_default_func(nuclide_name):
        """
        For the trivial case of stable nuclides.
        Also used for nuclides with no data

        Args:
            nuclide_name:

        Returns:

        """
        def func(ts, scale=1, decay_rate=False, *args, **kwargs):
            if decay_rate:
                scale = 0

            if hasattr(ts, '__iter__'):
                out = {nuclide_name: scale*np.ones_like(ts)}
            else:
                out = {nuclide_name: scale}

            return out

        return func


    class DecayNuclide:
        """
        This class's __call__ method solves the following problem:
            Starting with some amount (see init_quantity) of a given unstable nuclide, what amount of the parent and its
             progeny nuclides remain after time t?

        This is done by solving the diff. eq: y' == M.y, where M is a matrix derived from decay rates/branching ratios.
        Either y or y' can be returned (see documentation in __call__).

        Use driving_term arg for the case of nuclei being generated at a constant rate (in Hz), e.g. via a beam.
        A negative driving_term can be used, however, not that if the number of parent nuclei goes negative
        the solution is unphysical.

        Spontaneous fission products are not (yet) included in decay progeny.

        Solution is exact in the sense that the only errors are from machine precision.

        The coupled system of linear diff. egs. is solved "exactly" by solving the corresponding eigenvalue problem
            (or inhomogeneous system of lin. diff. eq. in the case of a nonzero driving_term).

        Args:
            nuclide_name: Name of parent nuclide.

            init_quantity: Unitless scalar representing the number (or amount) of the parent nuclei at t = 0
                i.e. the initial condition.

            init_rate: Similar to init_quantity, except specify a initial rate (decays/second).
                Only one of the two init args can be used

            driving_term: Set to non-zero number to have the parent nucleus be produced at a constant rate (in Hz).

            fiss_prod: If True, decay fission products as well.

            fiss_prod_thresh: Threshold of a given fission product yield (rel to highest yielded product).
                Between 0 and 1.0. 0 means accept all.

            max_half_life: A given decay chain will halt after reaching a nucleus with half life (in sec.) above this value.

        Returns:
            A function that takes a time (or array of times), and returns nuclide fractions at time(s) t.
            Return values of this function are of form: Dict[nuclide_name: Str, fractions: Union[np.ndarray, float]]
            See 'return_func' docstring below for more details.

        """

        def __init__(self, nuclide_name: str, init_quantity=1., init_rate=None, driving_term=0.,
                     fiss_prod=False, fiss_prod_thresh=0, max_half_life=None):
            nuclide = Nuclide.from_symbol(nuclide_name)
            self.nuclide_name = nuclide_name
            self.driving_term = driving_term

            if (not nuclide.is_valid) or nuclide.is_stable:
                self.return_default_func = True
            else:
                self.return_default_func = False
                # return decay_default_func(nuclide_name)

            if fiss_prod:
                from JSB_tools.nuke_data_tools.fission_yields import FissionYields
                assert ('sf',) in nuclide.decay_modes, f"Cannot include fission product on non-SF-ing nuclide, {nuclide}"
                fission_yields = FissionYields(nuclide.name, None, independent_bool=True)
                fission_yields.threshold(fiss_prod_thresh)
            else:
                fission_yields = None

            assert isinstance(init_quantity, (float, int, type(None))), "`init_quantity` must be a float or int"
            assert isinstance(init_rate, (float, int, type(None))), "`init_rate` must be a float or int"
            assert not init_quantity is init_rate is None, "Only one of the two init args can be used"
            if init_rate is not None:
                init_quantity = init_rate/nuclide.decay_rate.n

            self.column_labels = [nuclide_name]  # Nuclide names corresponding to lambda_matrix.
            self.lambda_matrix = [[-nuclide.decay_rate.n]]  # Seek solutions to F'[t] == lambda_matrix.F[t]

            completed = set()

            def loop(parent_nuclide: Nuclide, decay_modes):
                if not len(decay_modes):  # or parent_nuclide.name in _comp:  # stable nuclide. Terminate recursion.
                    return

                if parent_nuclide.name in completed:  # this decay chain has already been visited. No need to repeat.
                    return

                # Loop through all decay channels
                for mode_name_tuple, modes in decay_modes.items():
                    if not len(modes):
                        continue

                    if mode_name_tuple == ('sf',):
                        if fiss_prod:
                            fiss_branching = modes[0].branching_ratio

                            modes = []  # new modes that mimics structure of typical decay but for all fission products.

                            for fp_name, y in fission_yields.yields.items():
                                _mode = type('', (), {})()
                                _mode.parent_name = parent_nuclide.name
                                _mode.daughter_name = fp_name
                                _mode.branching_ratio = y*fiss_branching.n
                                modes.append(_mode)
                        else:
                            continue

                    # A given decay channels (e.g. beta- -> gs or 1st excited state, also fission) can have multiple
                    # child nuclides, so loop through them all.
                    for mode in modes:

                        parent_index = self.column_labels.index(mode.parent_name)

                        child_nuclide = Nuclide.from_symbol(mode.daughter_name)

                        child_lambda = child_nuclide.decay_rate.n

                        try:
                            # index of row/column for child nuclide in lambda matrix.
                            child_index = self.column_labels.index(mode.daughter_name)
                            child_row = self.lambda_matrix[child_index]

                        except ValueError:  # First time encountering this nuclide. Add new row/column to lambda-matrix.
                            self.column_labels.append(mode.daughter_name)
                            child_index = len(self.column_labels) - 1

                            for _list in self.lambda_matrix:
                                _list.append(0)  # add another column to maintain an nxn matrix

                            child_row = [0]*len(self.lambda_matrix[-1])  # create source(/sink) vector for current daughter nucleus

                            child_row[child_index] = -child_lambda  # Set entry for decay of child (diagonal term).

                            self.lambda_matrix.append(child_row)  # finally add new row to matrix.

                        # Do not use += below. The parent feeding rate is a constant no matter how many times the same
                        # parent/daughter combo is encountered.
                        child_row[parent_index] = mode.branching_ratio.n*parent_nuclide.decay_rate.n  # parent feeding term

                        if (max_half_life is not None) and child_nuclide.half_life.n > max_half_life:
                            continue  # don't worry about daughter bc nucleus decays too slow according to `max_half_life`
                        else:
                            loop(child_nuclide, child_nuclide.decay_modes)  # recursively loop through all daughters

                completed.add(parent_nuclide.name)  # Add parent to list of completed decay chains to avoid repeats

            loop(nuclide, nuclide.decay_modes)  # initialize recursion.

            self.lambda_matrix = np.array(self.lambda_matrix)

            self.eig_vals, self.eig_vecs = np.linalg.eig(self.lambda_matrix)

            self.eig_vecs = self.eig_vecs.T

            b = [init_quantity] + [0.] * (len(self.eig_vals) - 1)

            if self.driving_term != 0:
                # coefficients of the particular solution (which will be added to homo. sol.)
                self.particular_coeffs = np.linalg.solve(-self.lambda_matrix, [self.driving_term] + [0.] * (len(self.eig_vals) - 1))
            else:
                self.particular_coeffs = np.zeros_like(self.eig_vals)  # No driving term. Will have no effect in this case.

            self.coeffs = np.linalg.solve(self.eig_vecs.T, b - self.particular_coeffs)  # solve for initial conditions



if __name__ == "__main__":pass
    # print(Nuclide.NUCLIDE_NAME_MATCH.match("U238").groups('A'))
    # n = Nuclide.from_symbol("C13")
    # print(Nuclide.from_symbol('U235').decay_daughters)
    # times = np.linspace(0, Nuclide.from_symbol('U235').half_life.n*3, 30)
    # # f = decay_nuclide('U235')
    # # print(f(times))
    # f0 = decay_nuclide('U235', init_quantity=10)
    # f1 = decay_nuclide('U235', init_quantity=10)
    # y0 = f0(times)
    # y1 = f1(times)
    # fig, (ax0, ax1) = plt.subplots(1, 2)
    # for k in y0:
    #     if sum(y0[k]) < 1E-7:
    #         continue
    #     ax0.plot(times, unp.nominal_values(y0[k]), label=k)
    #     ax1.plot(times, unp.nominal_values(y1[k]), label=k)
    # ax1.legend()
    # ax0.legend()
    # #
    # plt.show()
    # print()

    # talys_calculation('C13', 'g')
    # f = decay_nuclide('Xe139', True)
