import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from lmfit.models import GaussianModel
from scipy.stats import norm
from JSB_tools.TH1 import TH1F
from uncertainties import UFloat, ufloat
import uncertainties.unumpy as unp
from lmfit.model import save_modelresult, load_modelresult
from pathlib import Path
import re
from scipy.interpolate import interp1d
from JSB_tools.TH1 import rolling_MAD, rolling_median
from abc import abstractmethod, ABCMeta
from lmfit.model import ModelResult


class FitBase(metaclass=ABCMeta):
    # def __new__(cls, *args, **kwargs):

    def __init__(self, x, y, yerr):
        assert hasattr(x, '__iter__')
        assert hasattr(y, '__iter__')
        assert len(x) == len(y)

        if any([isinstance(i, UFloat) for i in y]):
            assert yerr is None, "y values are UFloats, so `yerr` must not be supplied as an arg!"
            yerr = unp.std_devs(y)
            y = unp.nominal_values(y)
        if yerr is None:
            yerr = np.zeros(len(x))
        self.x = np.array(x)
        self.y = np.array(y)
        self.yerr = np.array(yerr)
        self.__weights__ = 1.0/np.where(self.yerr != 0, self.yerr, 1)

        self.fit_result: ModelResult = None

    def save(self, f_name):
        path = Path(__file__).parent/"user_saved_data"/'fits'/(f_name+'.lmfit')
        if path.exists():
            warnings.warn(f"Fit with name '{f_name}' already saved. Overwriting")
        try:
            save_modelresult(self.fit_result, path)
            # with open(path, 'wb') as f:
            #     pickle.dump(self.fit_result, f)
        except AttributeError:
            raise AttributeError('Invalid sub class. Missing "fit_result" attribute')

    @abstractmethod
    def model_func(self, *args):
        pass

    @classmethod
    def load(cls, f_name):
        path = Path(__file__).parent/"user_saved_data"/'fits'/(f_name +'.lmfit')
        assert path.exists(), f"No fit result saved to\n{path}"
        out: FitBase = cls.__new__(cls)
        out.fit_result: ModelResult = load_modelresult(path, {"model_func": out.model_func})
        try:
            out.x = out.fit_result.userkws['x']
        except KeyError:
            raise KeyError("The loaded lmfit.model.ModelResult had no 'x' value to extract. ")
        if out.fit_result.weights is not None:
            out.yerr = 1.0/out.fit_result.weights
        else:
            out.yerr = np.zeros_like(out.x)
        out.y = out.fit_result.data
        out.params = out.fit_result.params
        out.set_params()
        return out

    def set_params(self):
        for k, v in self.params.items():
            setattr(self, k, ufloat(float(v), v.stderr))

    @property
    def fit_y_err(self):
        return self.fit_result.eval_uncertainty()

    @property
    def fit_y(self):
        return self.fit_result.eval()

    def eval_func(self, x=None):
        if x is None:
            return unp.uarray(self.fit_y, self.fit_y_err)
        else:
            y = interp1d(self.x, self.fit_y, kind='quadratic', bounds_error=False)(x)
            yerr = interp1d(self.x, self.fit_y_err, kind='quadratic', bounds_error=False)(x)
            # y = np.interp(x, self.x, self.fit_y)
            # yerr = np.interp(x, self.x, self.fit_y_err)
            return unp.uarray(y, yerr)

    def plot_fit(self, ax=None, fit_x=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()
        points_line = ax.errorbar(self.x, self.y, self.yerr, label='data', ls='None', marker='o')
        if fit_x is None:
            fit_x = np.linspace(self.x[0], self.x[-1], len(self.x)*4)
        fit_y = self.eval_func(fit_x)
        fit_err = unp.std_devs(fit_y)
        fit_y = unp.nominal_values(fit_y)
        fit_line = ax.plot(fit_x, unp.nominal_values(fit_y), ls='--')[0]
        fill_poly = ax.fill_between(fit_x, fit_y-fit_err, fit_y+fit_err, alpha=0.7, label='Fit')
        print(points_line)
        ax.legend([points_line, (fill_poly, fit_line)], ["data", "Fit"])
        return ax

    def __repr__(self):
        self.params.pretty_print()
        return ""


def get_cut_select(x, min_, max_):
    return np.where((x >= min_) & (x <= max_))


class PeakFit(FitBase):
    @classmethod
    def from_hist(cls, hist: TH1F, peak_center_guess, sigma_guess=None, window=None):
        if not hist.is_density:
            warnings.warn('Histogram supplied to PeakFit is may not be a density. Divide by bin values to correct.')
        return cls(peak_center_guess=peak_center_guess, x=hist.bin_centers, y=hist.nominal_bin_values,
                   yerr=hist.bin_std_devs, sigma_guess=sigma_guess, window=window)

    @staticmethod
    def gen_fake_data(center=0, n_counts=10000, bg_const=10, bg_linear=5, sigma=1, xrange=(-15, 15), nbins=100):
        x = np.linspace(xrange[0], xrange[1], nbins)
        data = sigma * np.random.normal(size=n_counts) + center
        bg_data = []
        for _x, bg in zip(x, np.random.poisson(bg_const, nbins)):
            a = (_x-x[0])/(x[-1]-x[0])
            bg = int(bg + a*bg_linear)

            bg_data.extend(_x*np.ones(bg))
        data = np.concatenate([data, bg_data])
        y = np.histogram(data, bins=nbins, range=xrange)[0]
        return x, y

    def __cut__(self, min_, max_):
        """
        Cut based on x value
        Args:
            min_:
            max_:

        Returns:

        """
        s = get_cut_select(self.x, min_, max_)

        self.x = self.x[s]
        self.y = self.y[s]
        self.yerr = self.yerr[s]
        self.bg_est = self.bg_est[s]
        self.__weights__ = self.__weights__[s]

    @staticmethod
    def model_func(x, center, amp, sigma, bg):
        return amp * norm(loc=center, scale=sigma).pdf(x) + bg

    def __init__(self, peak_center_guess, x, y, yerr=None, window_width=None):
        """
        Fit a peak near `peak_center_guess`.
        It's recommended to supply the whole spectrum so that the code can use broader spectrum in order to estimate
        background.
        Args:
            peak_center_guess:
            x:
            y:
            yerr:
            window_width: Should be about 3x the width (base to base) of peak to be fit
        """

        super().__init__(x, y, yerr)
        if window_width is None:
            warnings.warn("No `window_width` arg provided. Using 20 as window_width. See __init__ docs")
            window_width = 20
        else:
            window_width = window_width//2  # full width -> half width

        self.bg_est = rolling_median(window_width=window_width, values=self.y)
        self.__cut__(peak_center_guess - window_width, peak_center_guess + window_width)
        bg_guess = np.mean(self.bg_est)
        bg_subtracted = self.y - self.bg_est
        plt.plot(self.x, bg_subtracted, label='bg sub')
        guess_model = GaussianModel()
        params = guess_model.guess(data=self.y, x=self.x)  # guess parameters by fitting simple gaussian
        guess_model.fit(x=self.x, data=bg_subtracted, params=params, weights=self.__weights__)
        params.add('bg', bg_guess)
        params['sigma'].min = 1E-4
        params['amp'] = params['amplitude']
        del params['amplitude']

        model = Model(self.model_func)
        self.fit_result = model.fit(x=self.x, data=self.y, params=params,  weights=self.__weights__, scale_covar=False)
        self.params = self.fit_result.params
        self.amp: UFloat = None  # are set later
        self.center: UFloat = None
        self.sigma: UFloat = None
        self.fwhm: UFloat = None
        self.bg: UFloat = None
        self.set_params()



x, y = PeakFit.gen_fake_data(0, 1000, 10,30, xrange=[-50,50], nbins=156)
y += PeakFit.gen_fake_data(23, 200, xrange=[-50, 50], nbins=156)[1]
y += np.random.randint(0, 20,size= len(y))
# plt.plot(x, y)
p = PeakFit(23, x, y, yerr=np.sqrt(y))

p.save('test')

p: PeakFit = p.load('test')
p.plot_fit()

plt.show()