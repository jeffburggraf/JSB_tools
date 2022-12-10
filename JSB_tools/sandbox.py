import pickle
import numpy as np
import matplotlib.pyplot as plt
from JSB_tools import TabPlot, mpl_style, mpl_hist
from JSB_tools.nuke_data_tools.nuclide import Nuclide
from lmfit import Minimizer
from math import gamma
from lmfit import Parameters, fit_report, printfuncs
import lmfit
from JSB_tools.spectra.time_depend import multi_guass_fit
from typing import Union, List
from scipy.fft import fft, fftfreq, fftshift
from lmfit.models import GaussianModel
from jax import hessian
import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy.stats import norm as sci_norm
from uncertainties import ufloat

def log_n_factorial(n):
    # i0 = np.searchsorted(ns, 0.2)
    if n > 0.2:
        return -n + n*np.log(n) + np.log(n*(1 + 4*n*(1 + 2*n)))/6. + np.log(np.pi)/2.
    else:
        return -0.577215 * n + 0.822467 * n ** 2

# def model(pars, x):
#     amplitude = pars['amplitude']
#     center = pars['center']
#     sigma = pars['sigma']
#     bg = pars['bg']
#     # out = amplitude * norm.pdf(x, loc=center, scale=sigma) + bg


def get_grid(param_name, result_minimizer):
    indx_shift = result_minimizer.var_names.index(param_name)
    grid = np.unique(result_minimizer.brute_grid[indx_shift].ravel())
    return grid



def gaus_fit(bins, counts, center_guesses, scales: Union[np.ndarray, float] = 1, sigma_guesses=None, center_fixed_in_binQ: Union[List[bool], bool] = False,
             share_sigma=True, fix_sigmas: bool = False, fix_centers: bool = False, fix_bg: float = None,
             fit_buffer_window: Union[int, None] = 5, center_playN=4, **kwargs):
    """

    Args:
        bins:
        counts:
        center_guesses:
        scales:
        sigma_guesses:
        center_fixed_in_binQ:
        share_sigma:
        fix_sigmas:

        fix_centers:

        fix_bg:

        fit_buffer_window:

        center_playN: The number of bins that defins the allowab le range for a given peak center
        **kwargs:

    Returns:

    """

    def model(params):
        amplitude = params['amplitude']
        center = params['center']
        sigma = params['sigma']
        bg = params['bg']

        rates = bin_widths * amplitude * sci_norm.pdf(x, loc=center, scale=sigma)

        rates += bg
        return rates

    def logProb(params: Union[dict, Parameters]):
        model_rates = model(params)
        sel = jnp.where(model_rates > 0)
        model_rates = model_rates[sel]
        out = -jnp.sum(model_rates) + jnp.sum(counts[sel] * jnp.log(model_rates))
        return -out

    def _logProb(args_dict):
        amplitude = args_dict['amplitude']
        center = args_dict['center']
        sigma = args_dict['sigma']
        bg = args_dict['bg']
        model_rates = bg + bin_widths * amplitude * norm.pdf(x, loc=center, scale=sigma)
        sel = jnp.where(model_rates > 0)
        model_rates = model_rates[sel]
        out = -jnp.sum(model_rates) + jnp.sum(counts[sel] * jnp.log(model_rates))
        return -out

    bin_widths = bins[1:] - bins[:-1]
    mean_bwidth = np.mean(bin_widths)
    x = 0.5 * (bins[1:] + bins[:-1])

    assert len(counts) == len(x)

    ax = mpl_hist(bins, counts)
    ax.set_ylabel("Counts")
    ax.set_xlabel("x")

    I = np.sum(bin_widths * counts)

    bg_guess = np.median(counts)
    amp_guess = I - np.sum(bg_guess * bin_widths)
    sigma_guess = np.mean(bin_widths)

    center_min, center_max = center_guesses[0] - mean_bwidth * center_playN, \
                             center_guesses[0] + mean_bwidth * center_playN
    init_params = Parameters()
    init_params.add('amplitude', value=amp_guess, min=np.sqrt(bg_guess), max=I,)
                    # brute_step=amp_guess/100)  # , min=np.sqrt(bg_guess))
    init_params.add('center', value=center_guesses[0], min=center_min, max=center_max)
    init_params.add('sigma', value=sigma_guess, min=0.9 * sigma_guess, max=(x[-1] - x[0])*0.75)# brute_step=mean_bwidth*0.5)
    init_params.add('bg', value=bg_guess, min=np.sqrt(bg_guess)/10, max=max(counts),)  # brute_step=max(counts)/20)

    ax.plot(x, model(init_params), label='initial model')

    fitter = Minimizer(logProb, init_params)

    result = fitter.minimize(method='nelder', params=init_params)  # brute_step
    print(result)

    print(fit_report(result))

    fit_params = result.params

    H = hessian(_logProb)({k: v.value for k, v in fit_params.items()})
    m = np.array([[x for x in v.values()] for v in H.values()])
    errs = [np.sqrt(np.abs(f)) for f in np.diag(np.linalg.inv(m))]

    for i, (k, v) in enumerate(fit_params.items()):
        param = ufloat(v.value, errs[i])
        v.stderr = errs[i]
        print(k, param)

    print(f"init log_prob: {logProb(init_params)}")
    print(f"Final log_prob: {logProb(fit_params)}")

    ax.plot(x, model(fit_params), label='fit')
    ax.legend()

    plt.show()


np.random.seed(0)
# # ==============================
bins = np.linspace(0, 100, 89)
x = 0.5 * (bins[1:] + bins[:-1])
amp = 95000
bf_frac = 240
center, center_guess = 16.5, 20
sigma, sigma_guess = 10, 1
bg_guess_scale = 0.6
#==================================

data = np.random.normal(center, sigma, amp)
data = np.concatenate([data, np.random.uniform(bins[0], bins[-1], int(bf_frac * amp))])

y, bins = np.histogram(data, bins)

nbins_filter = 100/np.median(y)
bwidth = x[1] - x[0]

gaus_fit(bins, y, [center])

plt.show()

