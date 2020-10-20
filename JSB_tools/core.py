import numpy as np
from warnings import warn
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc
from pathlib import Path
import sys
import time

try:
    import soerp
except ModuleNotFoundError as e:
    warn("No soerp module available. Analytical 2nd order error propagation not available.")
try:
    import mcerp
    from mcerp import UncertainFunction
except ModuleNotFoundError as e:
    warn("No mcerp module available. Analytical monte carlo error propagation not available.")

from matplotlib import pyplot as plt
import time
import scipy.stats
from numbers import Number

class ProgressReport:
    def __init__(self, i_final, sec_per_print=2, i_init=0):
        self.__i_final__ = i_final
        self.__i_init__ = i_init
        self.__sec_per_print__ = sec_per_print
        # self.__i_current__ = i_init
        self.__next_print_time__ = time.time() + sec_per_print
        self.__init_time__ = time.time()
        self.__rolling_average__ = []

    def __report__(self, t_now, i):
        evt_per_sec = (i-self.__i_init__)/(t_now - self.__init_time__)
        self.__rolling_average__.append(evt_per_sec)
        evt_per_sec = np.mean(self.__rolling_average__)
        if len(self.__rolling_average__) >= 5:
            self.__rolling_average__ = self.__rolling_average__[:5]
        evt_remaining = self.__i_final__ - i
        sec_remaining = evt_remaining/evt_per_sec
        sec_per_day = 60**2*24
        days = sec_remaining//sec_per_day
        hours = (sec_remaining % sec_per_day)//60**2
        minutes = (sec_remaining % 60**2)//60
        sec = (sec_remaining % 60)
        msg = " {0} seconds".format(int(sec))
        if minutes:
            msg = " {0} minutes,".format(minutes) + msg
        if hours:
            msg = " {0} hours,".format(hours) + msg
        if days:
            msg = "{0} days,".format(days) + msg
        print(msg + " remaining.", i/self.__i_final__)

    def log(self, i):
        t_now = time.time()
        if t_now > self.__next_print_time__:
            self.__report__(t_now, i)
            self.__next_print_time__ += self.__sec_per_print__



p = ProgressReport(24*60*59.85)
t0 = time.time()
for i in range(60*59):
    p.log(i)
    time.sleep(1)

class UncertainValue:
    def __init__(self, distribution, mc_samples=10000, tag=None):
        self.mc_samples = mc_samples
        self.tag = tag
        self.__hist__ = None  # Binned data for calculating probabilities.

        if isinstance(distribution, scipy.stats._distn_infrastructure.rv_frozen):
            mcerp.npts = mc_samples  # set the number of samples for monte carlo error propagation
            self.__uv__ = mcerp.uv(distribution, tag)

        elif isinstance(distribution, AffineScalarFunc):
            self.__uv__ = distribution

        elif hasattr(distribution, "__iter__"):  # arbitrary distribution from list of points
            assert all([isinstance(i, Number) for i in distribution])
            self.__uv__ = mcerp.UncertainFunction(distribution)

        elif isinstance(distribution, mcerp.UncertainFunction):
            self.__uv__ = distribution

    @property
    def __ufloat__(self):
        if isinstance(self.__uv__, AffineScalarFunc):
            return self.__uv__
        elif isinstance(self.__uv__, UncertainFunction):
            return ufloat(self.mean, self.std_dev)

    @property
    def __mcerp__(self):
        if isinstance(self.__uv__, AffineScalarFunc):
            return mcerp.uv(scipy.stats.norm(loc=self.mean, scale=self.std_dev))
        elif isinstance(self.__uv__, UncertainFunction):
            return self.__uv__

    @property
    def mc_points(self):
        if isinstance(self.__uv__, AffineScalarFunc):
            return scipy.stats.norm(self.__uv__.n, self.__uv__.std_dev).rvs(self.mc_samples)
        else:
            return self.__uv__._mcpts

    @property
    def mean(self):
        if isinstance(self.__uv__, AffineScalarFunc):
            return self.__uv__.n
        else:
            return self.__uv__.mean

    @property
    def std_dev(self):
        if isinstance(self.__uv__, AffineScalarFunc):
            return self.__uv__.std_dev
        elif isinstance(self.__uv__, UncertainFunction):
            return self.__uv__.std

    def prob(self, x):  # Probability density of variable at point x
        if isinstance(self.__uv__, AffineScalarFunc):
            pass  # Todo
        if self.__hist__ is None:
            self.__hist__ = np.histogram(self.mc_points, bins=14)
        bins = self.__hist__[1]
        probs = self.__hist__[0]
        b_width =bins[1] - bins[0]
        n_counts = self.__hist__[0]
        probs = probs/np.sum(probs)/b_width
        print("tot: ", np.sum(probs*b_width))
        print(bins, probs)
        plt.errorbar(0.5*(bins[:-1]+bins[1:]), n_counts, yerr=np.sqrt(n_counts))
        return np.interp(x, 0.5*(bins[:-1]+bins[1:]), probs)
        return probs[(np.abs(self.__hist__[1]-x)).argmin()]

    def __copy__(self):
        if isinstance(self.__uv__, UncertainFunction):
            return UncertainValue(self.__uv__._mcpts, self.mc_samples, self.tag)
        elif isinstance(self.__uv__, AffineScalarFunc):
            return UncertainValue(self.__uv__.__copy__(), self.mc_samples, self.tag)
        else:
            assert False

    def __convert__(self, other):
        if isinstance(other, AffineScalarFunc):
            other = UncertainValue(other)

        if isinstance(other, UncertainValue):
            if isinstance(self.__uv__, AffineScalarFunc):
                return other.__ufloat__
            elif isinstance(self.__uv__, UncertainFunction):
                return other.__mcerp__
        else:
            return other

    def __add__(self, other):
        new = self.__copy__()
        new += other
        print("1: ", type(new))
        return new

    def __iadd__(self, other):
        self.__uv__ += self.__convert__(other)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        new = self.__copy__()
        new -= other
        return new

    def __isub__(self, other):
        self.__uv__ -= self.__convert__(other)
        return self

    def __rsub__(self, other):
        new = self.__copy__()
        new.__uv__ = self.__convert__(other) - new
        return new

    def __truediv__(self, other):
        new = self.__copy__()
        new /= other
        return new

    def __itruediv__(self, other):
        self.__uv__ /= self.__convert__(other)
        return self

    def __mul__(self, other):
        new = self.__copy__()
        new *= other
        return new

    def __imul__(self, other):
        self.__uv__ *= self.__convert__(other)
        return self

    def __pow__(self, power, modulo=None):
        new = self.__copy__()
        new.__ipow__(power)
        return new

    def __ipow__(self, other):
        self.__uv__ **= self.__convert__(other)
        return self

    def __repr__(self):
        return "({0} +/- {1})".format(self.mean, self.std_dev)


class Normal(UncertainValue):
    def __init__(self, mean, sigma, tag=None, method="linear", mc_samples=2000):
        assert method in ["linear", "mcerp"], "Method must be 'linear' for linear propagation, " \
                                             "or 'mcerp' for monte carlo propagation (slower)."
        if method == "linear":
            super().__init__(ufloat(mean, sigma, tag=tag), tag=tag, mc_samples=mc_samples)
        else:
            super().__init__(scipy.stats.norm(loc=mean, scale=sigma), tag=tag, mc_samples=mc_samples)


class Poisson(UncertainValue):
    def __init__(self, mean, tag=None, mc_samples=10000):
        super().__init__(scipy.stats.poisson(mean), tag=tag, mc_samples=mc_samples)


if __name__ == "__main__":
    a = Poisson(5)
    print(a.prob(6))
    # print(a, np.sqrt(5))

    plt.show()


