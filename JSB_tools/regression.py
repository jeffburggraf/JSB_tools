import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from lmfit.models import GaussianModel
from scipy.stats import norm
from uncertainties import UFloat, ufloat
import uncertainties.unumpy as unp
from lmfit.model import save_modelresult, load_modelresult
from pathlib import Path
import re
from scipy.interpolate import interp1d
from JSB_tools import rolling_median
from abc import abstractmethod, ABCMeta
from lmfit.model import ModelResult
from lmfit.models import PolynomialModel
from lmfit import Parameters, fit_report
from uncertainties.umath import log as ulog
from scipy.odr import Model as ODRModel
from scipy.odr import RealData, ODR, polynomial, Output
from scipy.odr.models import _poly_fcn
from lmfit import minimize
from typing import List, Union


def independent_var_force_iterator(f):
    def q(*args, **kwargs):
        x = kwargs.get('x')
        if x is not None and not hasattr(x, '__len__'):
            kwargs['x'] = [x]
            return f(*args, **kwargs)[0]

        return f(*args, **kwargs)
    return q


class FitBase(metaclass=ABCMeta):
    @property
    def params(self):
        if hasattr(self, '__params__'):
            return self.__params__
        return self.fit_result.params

    def __init__(self, x, y, yerr):
        assert hasattr(x, '__iter__')
        assert hasattr(y, '__iter__')
        assert len(x) == len(y)
        if any([isinstance(i, UFloat) for i in y]):
            assert yerr is None, "y values are UFloats, so `yerr` must not be supplied as an arg!"
            yerr = unp.std_devs(y)
            y = unp.nominal_values(y)
        if yerr is None:
            yerr = np.ones(len(x))
        self.x = np.array(x)
        self.y = np.array(y)
        self.yerr = np.array(yerr)

        # For zero errors (inf weight), use a large number
        self.__weights__ = 1.0/np.where(self.yerr != 0, self.yerr, 1E-12)

        arg_sort = np.argsort(self.x)
        self.x = self.x[arg_sort]
        self.y = self.y[arg_sort]
        self.yerr = self.yerr[arg_sort]
        self.__weights__ = self.__weights__[arg_sort]

        self.fit_result: ModelResult = None

    def save(self, f_name, directory=None):
        if directory is None:
            directory = Path(__file__).parent

        path = directory/"user_saved_data"/'fits'/f'{f_name}_{type(self).__name__}.lmfit'
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
    def load(cls, f_name, directory=None):
        if directory is None:
            directory = Path(__file__).parent

        path = directory/"user_saved_data"/'fits'/f'{f_name}_{cls.__name__}.lmfit'

        # path = Path(__file__).parent/"user_saved_data"/'fits'/(f_name +'.lmfit')
        assert path.exists(), f"No fit result saved to\n{path}"
        out: FitBase = cls.__new__(cls)
        out.fit_result = load_modelresult(path, {"model_func": out.model_func})
        try:
            out.x = out.fit_result.userkws['x']
        except KeyError:
            raise KeyError("The loaded lmfit.model.ModelResult had no 'x' value to extract. ")
        if out.fit_result.weights is not None:
            out.yerr = 1.0/out.fit_result.weights
        else:
            out.yerr = np.zeros_like(out.x)
        out.y = out.fit_result.data
        out.set_params()
        return out

    def set_params(self):
        for k, v in self.params.items():
            if v.stderr is None:
                setattr(self, k, ufloat(float(v), 0))
            else:
                setattr(self, k, ufloat(float(v), abs(v.stderr)))

    @independent_var_force_iterator
    def eval_fit_error(self, x=None, params=None):
        args = {}
        if params is not None:
            args['params'] = params
        else:
            args['params'] = self.params
        if x is not None:
            args['x'] = x
        return self.fit_result.eval_uncertainty(**args)

    @independent_var_force_iterator
    def eval_fit_nominal(self, x=None, params=None):
        args = {}
        if params is not None:
            args['params'] = params
        else:
            args['params'] = self.params
        if x is not None:
            args['x'] = x
        return self.fit_result.eval(**args)

    def eval_fit(self, x=None, params=None):
        return unp.uarray(self.eval_fit_nominal(x=x, params=params), self.eval_fit_error(x=x, params=params))
        # if x is None:
        #     return unp.uarray(self.fit_y, self.fit_y_err)
        # else:
        #     y = interp1d(self.x, self.fit_y, kind='quadratic', bounds_error=False)(x)
        #     yerr = interp1d(self.x, self.fit_y_err, kind='quadratic', bounds_error=False)(x)
        #     # y = np.interp(x, self.x, self.fit_y)
        #     # yerr = np.interp(x, self.x, self.fit_y_err)
        #     return unp.uarray(y, yerr)

    def plot_fit(self, ax=None, params: Parameters = None, fit_x=None, label=None, marker='.', upsampling=10, color=None
                ):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        if label is None:
            label = 'Data'
        points_line = ax.errorbar(self.x, self.y, self.yerr, label=label, ls='None', marker=marker, zorder=0, c=color)
        if color is None:
            color = points_line[0].get_color()

        if fit_x is None:
            fit_x = np.linspace(self.x[0], self.x[-1], len(self.x)*upsampling)
        fit_y = self.eval_fit(x=fit_x, params=params)
        fit_err = unp.std_devs(fit_y)
        fit_y = unp.nominal_values(fit_y)

        fit_line = ax.plot(fit_x, unp.nominal_values(fit_y), ls='--', color=color)[0]
        fill_poly = ax.fill_between(fit_x, fit_y-fit_err, fit_y+fit_err, alpha=0.7, label='Fit', color=color)
        ax.legend([points_line, (fill_poly, fit_line)], ["data", "Fit"])
        return ax

    def __repr__(self):
        return fit_report(self.fit_result, min_correl=0.5)
        # self.params.pretty_print()
        # return ""


class PolyFit(FitBase):
    @staticmethod
    def model_func(x, params):
        return PolynomialModel().eval(x=x, params=params)

    def __init__(self, x, y, yerr=None, order=1):
        super().__init__(x, y, yerr)
        model = PolynomialModel(degree=order)
        params = model.guess(data=self.y, x=self.x, weights=self.__weights__)
        self.fit_result = model.fit(x=x, data=self.y, params=params, weights=self.__weights__, scale_covar=False)

    @property
    def coeffs(self):
        return [ufloat(float(p), p.stderr) for p in self.params.values()]


class LinearFit(PolyFit):
    def __init__(self, x, y, yerr=None):
        super().__init__(x, y, yerr, order=1)


class PeakFit(FitBase):
    def get_cut_select(x, min_, max_):
        return np.where((x >= min_) & (x <= max_))

    @classmethod
    def from_hist(cls, hist, peak_center_guess, fix_center=False, fix_sigma=None, window_width=None):
        if not hist.is_density:
            warnings.warn('Histogram supplied to PeakFit is may not be a density. You must divide by bin values '
                          'to get the correct answer.')
        return cls(peak_center_guess=peak_center_guess, x=hist.bin_centers, y=hist.nominal_bin_values,
                   yerr=hist.bin_std_devs, fix_center=fix_center, fix_sigma=fix_sigma, window_width=window_width)

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
        s = PeakFit.get_cut_select(self.x, min_, max_)

        self.x = self.x[s]
        self.y = self.y[s]
        self.yerr = self.yerr[s]
        self.bg_est = self.bg_est[s]
        self.__weights__ = self.__weights__[s]

    @staticmethod
    def model_func(x, center, amp, sigma, bg):
        return amp * norm(loc=center, scale=sigma).pdf(x) + bg

    def __init__(self, peak_center_guess, x, y, yerr=None, fix_center=False, fix_sigma=None, window_width=None,
                 make_density=False):
        """
        Fit a peak near `peak_center_guess`.
        It's recommended to supply the whole spectrum so that the code can use broader spectrum in order to estimate
        background.
        Args:
            peak_center_guess:
            x:
            y:
            yerr:
            fix_center: If True, fix center to `peak_center_guess`
            window_width: Should be about 3x the width (base to base) of peak to be fit
            make_density: Divide y values by x[i]-x[i-1]
        """
        super().__init__(x, y, yerr)
        if make_density:
            b_widths = [x2-x1 for x1, x2 in zip(self.x[:-1], self.x[1:])]
            b_widths += [b_widths[-1]]
            b_widths = np.array(b_widths)
            self.y /= b_widths
        if window_width is None:
            warnings.warn("No `window_width` arg provided. Using 20 as window_width. See __init__ docs")
            window_width = 20
        else:
            window_width = window_width//2  # full width -> half width

        self.bg_est = rolling_median(window_width=window_width, values=self.y)
        self.__cut__(peak_center_guess - window_width, peak_center_guess + window_width)
        bg_guess = np.mean(self.bg_est)
        bg_subtracted = self.y - self.bg_est

        guess_model = GaussianModel()
        params = guess_model.guess(data=self.y, x=self.x)  # guess parameters by fitting simple gaussian
        guess_model.fit(x=self.x, data=bg_subtracted, params=params, weights=self.__weights__)
        params.add('bg', bg_guess)
        params['sigma'].min = x[len(x)//2] - x[len(x)//2-1]
        params['sigma'].max = (x[-1] - x[0])*3
        params['sigma'].value = np.std(x)

        params['amp'] = params['amplitude']
        del params['amplitude']
        if fix_center:
            params['center'].set(value=peak_center_guess, vary=False)
        if fix_sigma is not None:
            params['sigma'].set(value=fix_sigma, vary=False)

        model = Model(self.model_func)
        self.fit_result = model.fit(x=self.x, data=self.y, params=params,  weights=self.__weights__, scale_covar=False)

        self.amp: ufloat = None  # are set later
        self.center: ufloat = None
        self.sigma: ufloat = None
        self.fwhm: ufloat = None
        self.bg: ufloat = None
        self.set_params()

    def fit_report(self):
        return self.fit_result.fit_report()


class LogPolyFit(FitBase):
    """
    A fit of the form:
        e^(c0 + c1 log(x) + c2 log(x)^2 + ... + cn log(x)^n)

    This model is good for fitting efficiency curves from HPGe detectors.
    Errors in model are not calculated from Jacobian matrix, as this leads to gross over-estimation.
    Instead, the model errors are interpolated from the data errors.
    Thus you should only use this when the goal is a simple fit to the data that behaves well under this model
     (asymptotically) rather than hypothesis testing. i.e., don't expect the coefficients to have "physical meaning".
    """
    @staticmethod
    def model_func(x, **params):
        out = np.zeros(len(x), dtype=np.float)
        for power, (_, coeff) in enumerate(params.items()):
            out += coeff*np.log(x)**power
        out = np.e**out
        return out

    def __init__(self, x, y, yerr=None, order=3, fix_coeffs: Union[List[int], None, str] = None):
        """
        Log poly fit.
        Args:
            x: array
            y: array or unp.uarray
            yerr: array or unp.uarray
            order: Order of polynomial
            fix_coeffs: Sometimes it is helpful to fix some coeffs to the value determined from the initial
                fit (i.e. the guess) in Log-Log space (i.e. when the fit is linear).
                This can help reduce issues during the final non-linear fit.
                first.

        """
        super().__init__(x, y, yerr)
        assert all([_ > 0 for _ in self.y]),  "All 'y' values must be greater than zero due to the log function!"
        assert all([_ > 0 for _ in self.x]),  "All 'x' values must be greater than zero due to the log function!"
        # _y = unp.uarray(self.y, self.yerr)

        log_y = [ulog(ufloat(_, _err)) for _, _err in zip(self.y, self.yerr)]
        log_y_error = np.array([_.std_dev for _ in log_y])
        log_y_error = np.where(log_y_error > 0, log_y_error, 1)
        log_y = np.array([_.n for _ in log_y])
        log_x = np.log(self.x)

        model_temp = PolynomialModel(degree=order)
        params = model_temp.guess(log_y, x=log_x, weights=1.0/log_y_error)
        _ = model_temp.fit(log_y, params=params, x=log_x, weights=1.0/log_y_error, scale_covar=True)

        if fix_coeffs is not None:
            assert hasattr(fix_coeffs, '__iter__'), '`fix_coeffs` must be a list of coeffs'
            if fix_coeffs == 'all':
                fix_coeffs = range(order)
            if len(fix_coeffs):
                assert max(fix_coeffs) <= order, "`fix_coeffs` was given a value above the fit order! " \
                                                 "(coeff. doesn't exist)"
                assert all([isinstance(a, int) for a in fix_coeffs]), "`fix_coeffs` all must be integers.!"
                for c in fix_coeffs:
                    params[f'c{c}'].vary = False
        model = Model(self.model_func)
        self.fit_result = model.fit(self.y, params=params, x=self.x, weights=self.__weights__, scale_covar=True,
                                    verbose=True, )

        self.coefs = [ufloat(float(p), p.stderr) for p in self.params.values()]

    @staticmethod
    def print_model(model_result: ModelResult):
        out = f"{model_result.params['c0'].value:.2f}" + "".join(
            [f"{'+' if p.value > 0 else ''}{p.value:.2f}*log(erg)^{i + 1}" for i, p in
             enumerate(list(model_result.params.values())[1:])])
        out = f"e^({out})"
        return out

    @independent_var_force_iterator
    def eval_fit_nominal(self, x=None, params=None):
        if params is None:
            params = self.params
        if x is None:
            x = self.x

        return self.model_func(x, **params)


class ODRBase(metaclass=ABCMeta):
    def __init__(self, x, y, xerr, yerr):
        if any(isinstance(xi, UFloat) for xi in x):
            xerr = np.array([xi.std_dev if isinstance(xi, UFloat) else 0 for xi in x])
            x = np.array([xi.n if isinstance(xi, UFloat) else xi for xi in x])
        if any(isinstance(yi, UFloat) for yi in y):
            yerr = np.array([yi.std_dev if isinstance(yi, UFloat) else 0 for yi in y])
            y = np.array([yi.n if isinstance(yi, UFloat) else yi for yi in y])
        args = {}
        if yerr is not None:
            assert not any([isinstance(_, UFloat) for _ in yerr]), "Ufloats in yerr"
            args["sy"] = yerr
        else:
            yerr = np.zeros_like(x)
        if xerr is not None:
            assert not any([isinstance(_, UFloat) for _ in xerr]), "Ufloats in xerr"
            args["sx"] = xerr
        else:
            xerr = np.zeros_like(x)

        self.x = x
        self.y = y
        self.yerr = yerr
        self.xerr = xerr

        self.data = RealData(self.x, self.y, **args)

    @property
    def __beta__(self):
        """Represents the coeffs"""
        return unp.uarray(self.fit_result.beta, self.fit_result.sd_beta)

    @property
    def fit_result(self) -> Output:
        if hasattr(self, '__odr__'):
            return self.__odr__.output
        else:
            assert hasattr(self, '__fit_result__'), "Invalid subclass, missing __odr__ attribute"
            return self.__fit_result__

    def eval_fit(self, x=None, params=None):
        if x is None:
            x = self.x
        if params is None:
            params = self.__beta__
        return self.model_func(params, x)

    @property
    def odr_obj(self):
        #  __odr__
        # make so this can only be gotten once?
        pass

    @odr_obj.setter
    def odr_obj(self, value):
        # Make it so this can only be set once?
        pass

    @abstractmethod
    def model_func(self, b, x):
        pass

    def save(self, fname, extra__=None):
        fname = (fname + "_ODR"+'.lmfit')
        path = Path(__file__).parent/"user_saved_data"/'fits'/fname
        if path.exists():
            warnings.warn(f"Fit named '{fname} already exists!. Overwriting.")
        with open(path, 'wb') as f:
            pickle.dump(self.fit_result, f)
            pickle.dump((self.x, self.y, self.xerr, self.yerr), f)
            if extra__ is not None:
                pickle.dump(extra__, f)

    @classmethod
    def load(cls, fname):
        fname = (fname + "_ODR" + '.lmfit')
        path = Path(__file__).parent / "user_saved_data" / 'fits' / fname
        out = PolyFitODR.__new__(cls)
        if not path.exists():
            raise FileNotFoundError(f"No fit named {fname}")
        with open(path, 'rb') as f:
            out.__fit_result__ = pickle.load(f)
            out.x, out.y, out.xerr, out.yerr = pickle.load(f)
            try:
                extra_attribs = pickle.load(f)
            except EOFError:
                extra_attribs = {}
            for k, value in extra_attribs.items():
                setattr(out, k, value)

        return out

    def plot_fit(self, x=None, ax=None, xlabel=None, ylabel=None, mpl_args=None):
        # if ax is None:
        #     plt.figure()
        #     ax = plt.gca()
        # if mpl_args is None:
        #     mpl_args = {}
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        points_line = ax.errorbar(self.x, self.y, xerr=self.xerr, yerr=self.yerr, label="Data", ls='None', marker='o')
        if x is None:
            x = np.linspace(self.x[0], self.x[-1], len(self.x)*4)

        y_fit = self.eval_fit(x=x)
        y_fit_err = unp.std_devs(y_fit)
        y_fit = unp.nominal_values(y_fit)
        fit_line = ax.plot(x, y_fit, label="Fit")[0]
        fill_poly = ax.fill_between(x, y_fit-y_fit_err, y_fit+y_fit_err, alpha=0.7, label='Fit')
        ax.legend([points_line, (fill_poly, fit_line)], ["data", "Fit"])

        return ax


class PolyFitODR(ODRBase):
    def __init__(self, x, y, xerr=None, yerr=None, order=2):
        super().__init__(x, y, xerr, yerr)
        self.__odr__ = ODR(self.data, polynomial(order))
        self.__odr__.run()
        self._powers = np.asarray(order)

    def save(self, fname):
        super(PolyFitODR, self).save(fname=fname, extra__={"_powers": self._powers})

    @property
    def coeffs(self):
        return self.__beta__

    def model_func(self, b, x):
        return _poly_fcn(b, x, self._powers)

    def __repr__(self):
        self.fit_result.pprint()
        return ""


class MaximumLikelyHoodBase:
    def __init__(self, x, y, xerr, yerr):
        assert hasattr(x, '__iter__')
        assert hasattr(y, '__iter__')
        assert len(x) == len(y)
        if any([isinstance(i, UFloat) for i in y]):
            assert yerr is None, "y values are UFloats, so `yerr` must not be supplied as an arg!"
            yerr = unp.std_devs(y)
            y = unp.nominal_values(y)
        if any([isinstance(i, UFloat) for i in x]):
            assert xerr is None, "y values are UFloats, so `yerr` must not be supplied as an arg!"
            xerr = unp.std_devs(x)
            x = unp.nominal_values(x)
        if yerr is None:
            yerr = np.zeros(len(x))
        if xerr is None:
            xerr = np.zeros(len(x))
        self.x = np.array(x)
        self.y = np.array(y)
        self.yerr = np.array(yerr)
        self.xerr = xerr


class ExponentialMLL:
    def __init__(self, times, max_time=None, weights=None):
        if weights is None:
            weights = np.ones_like(times)
        self.times = times

        # if times_noise is None:
        #     assert weights_noise is None
        #     times_noise = weights_noise = np.array([])
        if weights is None:
            weights = np.ones_like(weights)

        if max_time is None:
            max_time = max(times)

        self.max_time = max_time
        self.weights = weights

    @property
    def hl(self):
        return np.log(2)/self.params.valuesdict()['_lambda']

    @staticmethod
    def likelihood(params, x, _weights=None, max_time=None):
        a = max_time
        parvals = params.valuesdict()
        times = x
        _lambda = parvals['_lambda']
        # noise = parvals['noise']

        # probs = (noise + _lambda*np.e**-(times*_lambda))/(1 - np.e**(-(a*_lambda)) + a*noise)
        probs = ( _lambda*np.e**-(times*_lambda))
        print('probs :', probs)
        print("times: ",times)
        print("params", params, _lambda)
        print()
        llh = np.sum(_weights*np.log(probs))
        print(llh)
        # print(llh)
        return -llh

    def estimate(self, lambda_guess=None, noise_guess=None):
        if lambda_guess is None:
            lambda_guess = np.sum(self.weights)/sum(self.weights*self.times)
            print('Lambda guess', lambda_guess)
        if noise_guess is None:
            noise_guess = 1E-10
        self.params = Parameters()
        self.params.add('_lambda', lambda_guess,min=0, max=max(self.times))
        # self.params.add('noise', noise_guess, min=0, max=sum(self.times))
        m = minimize(self.likelihood, self.params, args=(self.times,),
                     kws={'_weights': self.weights, 'max_time': self.max_time}, method='cobyla')
        print(m.params)




if __name__ == '__main__':
    from TH1 import TH1F
    hl = 40
    n = 0
    data_true = np.random.exponential(hl/np.log(2), 500)+20
    noise = np.random.uniform(0, max(data_true), int(max(data_true)*n))
    h_data = TH1.TH1F.from_raw_data(data_true)

    h_noise = TH1.TH1F.from_raw_data(noise, bins=h_data.__bin_left_edges__)
    h_tot = h_data + h_noise
    h_tot.plot(leg_label="total")
    h_noise.plot(leg_label="noise")
    h_data.plot(leg_label="Signal")
    plt.legend()
    times_meas = data_true
    print("lambda true: ", np.log(2)/hl)

    m = ExponentialMLL(h_tot.bin_centers, weights=h_tot.nominal_bin_values)
    m.estimate()


    plt.show()

    pass
    # from JSB_tools.nuke_data_tools.gamma_spec import PrepareGammaSpec
    # def ch_2_erg(ch):
    #     return 0.08874085 + ch*0.55699971
    #
    # def erg_2_ch(erg):
    #     return (erg-0.08874085)/0.55699971
    # n_chanels = 4000
    # N = 20000
    #
    # c = PrepareGammaSpec(n_chanels)
    #
    # ergs = np.array([59.9, 88.4, 122, 166, 392, 514, 661, 898, 1173, 1332, 1835])
    # effs = np.array([0.06, 0.1, 0.144, 0.157, 0.1, 0.07, 0.05, 0.04, 0.03, 0.027, 0.018])
    # channels = np.arange(n_chanels)
    #
    # counts = np.zeros(n_chanels)
    # counts += np.random.poisson(10, len(counts))
    # fake_ergs = [20, 230, 450, 500, 700, 1230, 1600]
    # true_counts = [N]*len(fake_ergs)
    # channel_guesses = []
    # for erg in fake_ergs:
    #     index_center = erg_2_ch(erg)
    #     channel_guesses.append(index_center)
    #
    #     for i in np.random.normal(index_center, 10, int(N*np.interp(erg, ergs, effs))):
    #         i = int(i)
    #         counts[i] += 1
    #
    # plt.plot(channels, counts)
    # c.add_peaks_4_calibration(counts, channel_guesses, fake_ergs, true_counts, fit_width=50, plot=False)
    # c.compute_calibration()
    # # c.plot_erg_spectrum()
    # c.erg_calibration.plot_fit()
    # c.eff_fit.plot_fit()
    # c.save_calibration('die')
    # c2= PrepareGammaSpec.load_calibration('die')
    # c2.eff_fit.plot_fit()
    # c2.erg_calibration.plot_fit()
    # plt.show()
    # # plt.show()
