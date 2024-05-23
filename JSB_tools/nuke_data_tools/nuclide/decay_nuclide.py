import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
from JSB_tools.nuke_data_tools import Nuclide
from JSB_tools.nuke_data_tools.nuclide.fission_yields import FissionYields
from uncertainties import UFloat
from scipy.integrate import odeint
from warnings import warn
from typing import Callable, Dict


matplotlib.use('Qt5agg')
cwd = Path(__file__).parent


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
            out = {nuclide_name: scale * np.ones_like(ts)}
        else:
            out = {nuclide_name: scale}

        return out

    return func


class DecayNuclide:
    """
    This class's __call__ method solves the following problem:
        Starting with some amount (see init_quantity) of a given unstable nuclide, what amount of the parent and its
         progeny nuclides remain after time t?

    This is done by solving the diff. eq: y' == M.y, where M is a matrix derived from decay gamma_rates/branching ratios.
    Either y or y' can be returned (see documentation in __call__).

    Use driving_term arg for the case of nuclei being generated at a constant rate (in Hz), e.g. via a beam.
    A negative driving_term can be used, however, note that if the number of parent nuclei goes negative
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
            Note:

        fiss_prod: If True, decay fission products as well.

        fiss_prod_thresh: Threshold of a given fission product yield (rel to highest yielded product).
            Between 0 and 1.0. 0 means accept all.

        max_half_life: A given decay chain will halt after reaching a nucleus with half life (in sec.) above this value.

    Returns:
        A function that takes a time (or array of times), and returns nuclide fractions at time(s) t.
        Return values of this function are of form: Dict[nuclide_name: Str, fractions: Union[np.ndarray, float]]
        See 'return_func' docstring below for more details.

    """
    @property
    def nuclide(self):
        return Nuclide(self.nuclide_name)

    def __init__(self, nuclide_name: str, init_quantity=1., init_rate=None,
                 fission_yields: FissionYields = None, max_half_life=None):
        nuclide = Nuclide(nuclide_name)
        self.nuclide_name = nuclide_name

        if (not nuclide.is_valid) or nuclide.is_stable:
            self.return_default_func = True
        else:
            self.return_default_func = False

        if fission_yields is not None:
            assert ('sf',) in nuclide.decay_modes, f"Cannot include fission product on non-SF-ing nuclide, {nuclide}"
            # fission_yields = FissionYields(nuclide.name, None, independent_bool=True)
            # fission_yields.threshold(fiss_prod_thresh)
        # else:
        #     fission_yields = None

        assert isinstance(init_quantity, (float, int, type(None))), "`init_quantity` must be a float or int"
        assert isinstance(init_rate, (float, int, type(None))), "`init_rate` must be a float or int"
        assert not init_quantity is init_rate is None, "Only one of the two init args can be used"
        if init_rate is not None:
            init_quantity = init_rate / nuclide.decay_rate.n

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
                    if fission_yields is not None:
                        fiss_branching = modes[0].branching_ratio

                        modes = []  # new modes that mimics structure of typical decay but for all fission products.

                        for fp_name, y in fission_yields.yields.items():
                            _mode = type('', (), {})()
                            _mode.parent_name = parent_nuclide.name
                            _mode.daughter_name = fp_name
                            _mode.branching_ratio = y * fiss_branching.n
                            modes.append(_mode)
                    else:
                        continue

                # A given decay channels (e.g. beta- -> gs or 1st excited state, also fission) can have multiple
                # child nuclides, so loop through them all.
                for mode in modes:

                    parent_index = self.column_labels.index(mode.parent_name)

                    child_nuclide = Nuclide(mode.daughter_name)

                    if child_nuclide.half_life is None:
                        continue
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

                        child_row = [0] * len(self.lambda_matrix[-1])  # create source(/sink) vector for current daughter nucleus

                        child_row[child_index] = -child_lambda  # Set entry for decay of child (diagonal term).

                        self.lambda_matrix.append(child_row)  # finally add new row to matrix.

                    # Do not use += below. The parent feeding rate is a constant no matter how many times the same
                    # parent/daughter combo is encountered.
                    child_row[parent_index] = mode.branching_ratio.n * parent_nuclide.decay_rate.n  # parent feeding term

                    if (max_half_life is not None) and child_nuclide.half_life.n > max_half_life:
                        continue  # don't worry about daughter bc nucleus decays too slow according to `max_half_life`
                    else:
                        loop(child_nuclide, child_nuclide.decay_modes)  # recursively loop through all daughters

            completed.add(parent_nuclide.name)  # Add parent to list of completed decay chains to avoid repeats

        loop(nuclide, nuclide.decay_modes)  # initialize recursion.

        self.lambda_matrix = np.array(self.lambda_matrix)

        self.eig_vals, self.eig_vecs = np.linalg.eig(self.lambda_matrix)

        self.eig_vecs = self.eig_vecs.T

        self.init_conditions = np.array([init_quantity] + [0.] * (len(self.eig_vals) - 1))

        # if self.driving_term != 0:
        self.coeffs = None  # use odeint

        # else:
        #     self.coeffs = np.linalg.solve(self.eig_vecs.T, self.init_conditions)  # solve for initial conditions

    def __call__(self, ts, scale=1, decay_rate=False, driving_term=None, threshold=None, plot=False) -> Dict[str, np.ndarray]:
        """
        Evaluates the diff. eq. solution for all daughter nuclides at the provided times (`ts`).
        Can determine, as a function of time, the quantity of all decay daughters or the decay rate.
        This is controlled by the `decay_rate` argument.
        Args:
            ts: Times at which to evaluate the specified quantity

            scale: Scalar to be applied to the yield of all daughters.

            decay_rate: If True, return decays per second instead of fraction remaining. I.e. return y[i]*lambda_{i}

            threshold: Fraction of the total integrated yield below which solution arent included. None includes all.

            plot: Plot for debugging.

        Returns: dict of the form, e.g.:
            {'U235': [1., 0.35, 0.125],
             'Pb207': [0, 0.64, 0.87],
              ...}
        """
        if hasattr(scale, '__iter__'):
            assert len(scale) == 1
            scale = scale[0]

        if isinstance(scale, UFloat):
            if self.coeffs is None:
                raise ValueError("Cannot pass errors through solution when driving term is present. ")

            warn("`scale` argument is a UFloat. This reduces performance by a factor of ~1000. `")

        if self.return_default_func:
            return decay_default_func(self.nuclide_name)(ts, scale=scale, decay_rate=decay_rate)

        if threshold is not None:
            raise NotImplementedError("Todo")

        if hasattr(ts, '__iter__'):
            iter_flag = True
        else:
            iter_flag = False
            ts = np.array([ts])

        if not isinstance(ts, np.ndarray):
            ts = np.array(ts)

        #
        # yields = [np.sum([c * vec * np.e ** (val * t) for c, vec, val in
        #                   zip(self.coeffs, self.eig_vecs, self.eig_vals)], axis=0) for t in ts]  # old

        if driving_term is None:
            if self.coeffs is None:
                self.coeffs = np.linalg.solve(self.eig_vecs.T, self.init_conditions)  # solve for initial conditions

            yields = np.matmul((self.coeffs.reshape((len(self.coeffs), 1)) * self.eig_vecs).T,
                               np.e ** (self.eig_vals.reshape((len(self.coeffs), 1)) * ts))  # New
        else:
            driving_func = None
            driving = np.zeros(len(self.eig_vals))

            if isinstance(driving_term, (int, float, UFloat)):
                if isinstance(driving_term, UFloat):
                    rel_err = driving_term.std_dev
                    warn('Errors not currently propagated through solution! todo')

                driving[0] = driving_term

            elif isinstance(driving_term, Callable):
                driving[0] = 1
                driving_func = driving_term
            else:
                raise ValueError

            def J(y, t):
                return self.lambda_matrix

            def func(y, t):
                out = np.matmul(self.lambda_matrix, y)
                if driving_func is not None:
                    out += driving * driving_func(t)
                else:
                    out += driving

                return out

            yields, infodict = odeint(func, self.init_conditions, t=ts, Dfun=J, full_output=True)

            yields = yields.transpose()

        if not decay_rate:
            out = {name: scale * yield_ for name, yield_ in zip(self.column_labels, yields)}
        else:
            out = {name: scale * yield_ * lambda_ for name, yield_, lambda_ in
                   zip(self.column_labels, yields, np.abs(np.diagonal(self.lambda_matrix)))}

        if not iter_flag:
            for k, v in out.items():
                out[k] = v[0]

        if plot:
            # if not (plot is True)
            assert iter_flag, 'Cannot plot for only one time'
            plt.figure()
            for k, v in out.items():
                plt.plot(ts, v, label=k)
            if decay_rate:
                plt.ylabel("Decays/s")
            else:
                plt.ylabel("Rel. abundance")
            plt.xlabel('Time [s]')
            plt.legend()

        return out
