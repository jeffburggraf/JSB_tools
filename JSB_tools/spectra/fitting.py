# import pickle
import lmfit
import numpy as np
import matplotlib.pyplot as plt
from JSB_tools import TabPlot, mpl_style, mpl_hist
# from JSB_tools.nuke_data_tools.nuclide import Nuclide
# from lmfit import Minimizer
# from math import gamma
from lmfit import Parameters, fit_report, printfuncs
# import lmfit
# from JSB_tools.spectra.time_depend import multi_guass_fit
from typing import Union, List, Dict
# from lmfit.models import GaussianModel
import numba as nb
# import numba.types as nb_types
# from numba.typed import Dict as NumbaDict
# from numba.typed import List as NumbaList
# from jax import hessian
import jax.numpy as jnp
import timeit
import jax
from jax.scipy.optimize import minimize as jax_minimize
# from scipy.stats import norm as sci_norm
# from uncertainties import ufloat
from scipy.integrate import trapezoid
from collections import namedtuple
from scipy.optimize import minimize
from jax.config import config
config.update_data_plot("jax_enable_x64", True)


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


def make_params(bins, counts, center_guesses, fix_centersQ: Union[bool, List[bool]] = False,
                fix_bg: Union[float, None] = None, sigma_guesses: Union[List[float], float, None] = None,
                effs: Union[np.ndarray, float] = 1.0):
    bins = np.asarray(bins)

    if hasattr(effs, '__iter__'):
        assert len(effs) == len(counts)
        # effs = np.ones(len(counts), dtype=np.float)

    if sigma_guesses is not None and not hasattr(sigma_guesses, '__iter__'):
        sigma_guesses = [sigma_guesses] * len(center_guesses)

    if not hasattr(fix_centersQ, '__iter__'):
        assert isinstance(fix_centersQ, bool)
        fix_centersQ = [fix_centersQ] * len(center_guesses)

    center_is = np.searchsorted(bins, center_guesses, side='right') - 1

    bin_widths = np.asarray(bins[1:] - bins[:-1])
    counts_density = counts/bin_widths

    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    I = trapezoid(counts_density, bin_centers)

    def get_baseline(l):
        percentiles = np.linspace(0, 50, 50)
        return np.median(np.percentile(l, percentiles))

    if fix_bg is None:
        bg_guess = get_baseline(counts_density)
    else:
        assert isinstance(fix_bg, float)
        bg_guess = fix_bg

    n_half = len(counts_density)//2
    slopiness = (get_baseline(counts_density[n_half:]) - get_baseline(counts_density[:n_half]))
    slope_guess = slopiness/(bins[-1] - bins[-0])

    print(f'slope_guess: {slope_guess}')
    print(f'slopiness: {slopiness}')
    print(f'bg_guess: {bg_guess}')
    bg_guess -= slopiness/2  # account for over estimation of constant bg due to slope

    # bg_guess = np.percentile(counts_density, 45) if fix_bg is None else fix_bg  # in density units

    amps_sum = I - (bg_guess * (bins[-1] - bins[0]) + slopiness/2)
    amp_guesses = amps_sum * (counts[center_is]/np.sum(counts[center_is]))
    center_guesses = bin_centers[center_is]

    # bg_guess = np.percentile(counts_density, 100 * amps_sum / I / 2)

    params = Parameters()
    params.add('bg', value=bg_guess, vary=fix_bg is None, min=min(counts_density), max=max(counts_density))
    params.add('slope', value=slope_guess)

    for i in range(len(center_guesses)):
        amp_guess = amp_guesses[i]
        if sigma_guesses is None:
            max_guess = (np.mean(counts_density[center_is[i] - 2: center_is[i] + 2]) - bg_guess)
            sig_guess = amp_guess * 0.398942/max_guess
        else:
            sig_guess = sigma_guesses[i]

            max_guess = amp_guess/(np.sqrt(2 * np.pi) * sig_guess)

        # params.add(f'_{i}_amplitude', value=amp_guess, min=0, max=I)
        params.add(f'_{i}_max', value=max_guess, min=0)
        params.add(f'_{i}_center', value=center_guesses[i], vary=not fix_centersQ[i], min=bins[0], max=bins[-1])
        # params.add(f'_{i}_sigma', expr=f'_{i}_amplitude*0.398942/_{i}_max')
        params.add(f'_{i}_sigma', value=sig_guess)
        params.add(f'_{i}_amplitude', expr=f'_{i}_max * 2.50663 * _{i}_sigma')

    if not len(counts) == len(bin_centers):
        raise ValueError("length of bins must be len(counts) + 1")

    out_cls = namedtuple("Params", ['params', 'counts_density', 'bin_centers', 'bin_widths', 'bins'])

    # _, axs = plt.subplots(1, 2)
    # _x = np.linspace(0, 100, 100)
    # _y = np.percentile(counts_density, _x)
    # axs[0].plot(_x, _y)
    # mpl_hist(bins, counts_density, ax=axs[1])
    # axs[0].axhline(true_bg, label='Bg True', lw=1)
    # axs[0].axhline(bg_guess, label='Bg guess', c='orange', lw=1)
    # axs[0].legend()

    return out_cls(params, counts_density, bin_centers, bin_widths, bins)


_K = 1.0/np.sqrt(2 * np.pi)


# @jax.jit
# def norm(x: np.ndarray, amp: float, mu: float, sig: float):
#     return amp * _K/sig * np.e ** -(0.5 * ((mu - x)/sig)**2)
@jax.jit
def norm(x: np.ndarray, max: float, mu: float, sig: float):
    return max * np.e ** -(0.5 * ((mu - x)/sig)**2)


def isolate_variable(f, params, indep_par_index, func_args):
    params = params.copy()
    def g(_xx):
        if hasattr(_xx,'__iter__'):
            return np.array([g(x) for x in _xx])
        ps = params.copy()
        ps[indep_par_index] = _xx
        return f(ps, *func_args)
    return g


def gaus_fit(bins, counts, center_guesses, jit=True, fix_centersQ: Union[bool, List[bool]] = False,
             scales: Union[np.ndarray, float] = 1, sigma_guesses=None, center_fixed_in_binQ: Union[List[bool], bool] = False,
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

        center_playN: The number of bins that defines the allowable range for a given peak center
        **kwargs:

    Returns:

    """
    global norm

    if not isinstance(counts[0], int):
        if not all([int(counts[i]) == counts[i] for i in range(len(counts))]):
            raise ValueError("`counts` argument must be an array of integers.")

    counts = np.asarray(counts, dtype=int)
    # jnp = np

    def model(params: np.ndarray, names_indices, n_peaks):
        rates = jnp.zeros(len(x), dtype=float)

        bg = bin_widths * (params[names_indices['bg']]) + bin_widths * params[names_indices['slope']] * (x - x[0])

        for i in range(n_peaks):
            # amplitude = params[names_indices[f'_{i}_amplitude']]
            center = params[names_indices[f'_{i}_center']]
            sigma = params[names_indices[f'_{i}_sigma']]
            _max = params[names_indices[f'_{i}_max']]
            # sigma = amplitude*0.398942/_max
            # rates += bin_widths * norm(x, amplitude, center, sigma)
            rates += bin_widths * norm(x, _max, center, sigma)

        return rates + bg

    def logProb(params: jnp.ndarray, names_indices, n_peaks, scale=1):
        model_rates = model(params, names_indices, n_peaks)
        # sel = jnp.where(model_rates > 0)
        # model_rates = jnp.where(model_rates>0, model_rates, 0)
        # model_rates = model_rates[sel]
        # assert all(model_rates > 0)
        out = -jnp.sum(model_rates) + jnp.sum(counts * jnp.log(model_rates))
        return -out/scale

    if jit:
        norm = jax.jit(norm)
        model = jax.jit(model, static_argnames='n_peaks')
        logProb = jax.jit(logProb, static_argnames='n_peaks')

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    mpl_hist(bins, counts, label='True', ax=ax)
    n_peaks = len(center_guesses)

    params_info = make_params(bins, counts, center_guesses)
    bin_widths = params_info.bin_widths
    x = params_info.bin_centers
    bins = params_info.bins

    init_params = np.array([v for v in params_info.params.values()])
    names_indices = {param.name: i for i, (name, param) in enumerate(params_info.params.items()) if param.vary}
    # init_logProb = abs(logProb(init_params, names_indices, n_peaks, 1))
    init_logProb = 1

    init_y = model(init_params, names_indices, n_peaks)

    mpl_hist(bins, init_y, ax=ax, label='Init model')
    calls_per_sec = 2000/timeit.timeit('logProb(init_params, names_indices, n_peaks)', number=2000, globals=locals())
    print(f"calls_per_sec: {calls_per_sec}")
    # res = jax_minimize(logProb, init_params, args=(names_indices, n_peaks), method='BFGS')
    res = minimize(logProb, init_params, args=(names_indices, n_peaks, init_logProb), method='Nelder-Mead') #, jac='3-point')
    # res = lmfit.minimize(logProb, params_info.params, args=[names_indices, n_peaks])
    # final_params = res.x
    final_params = res.x

    H = jax.hessian(logProb)(final_params, names_indices, n_peaks, init_logProb)
    m = np.array([[x for x in v.values()] for v in H.values()])
    errs = {k: np.sqrt(np.abs(f)) for k, f in zip(H.keys(), np.diag(np.linalg.inv(m)))}

    print('name        \tfinal\tinitial')
    for k, i in names_indices.items():
        # print(f'{k: <13}\t{final_params[v]:.2e} +/- {par_errs[k]}\t{init_params[v]:.2e}')
        # final_param_val =
        print(f'{k: <13}\t{final_params[i]:.2e} \t{init_params[i]:.2e}')
    print(res)
    # res = minimize(logProb, init_params, )
    # fitter = Minimizer(logProb_lmfit, init_params)
    #
    # result = fitter.minimize(method='nelder', params=init_params)  # brute_step
    # final_params = result.params

    ax.plot(x, model(final_params, names_indices, n_peaks), label='Fit', lw=3)
    ax.legend()

    if DO_TAB:
        tab = TabPlot()
        for k, i in names_indices.items():
            print(f"tab {k}")
            try:
                dx = {'slope': 10, 'bg': 5000}[k]
            except KeyError:
                if 'amplitude' in k:
                    dx = 10000
                elif 'max' in k:
                    dx = 4000
                elif 'center' in k:
                    dx = 50
                else:
                    # print("Fuck")
                    dx = 10
            tab_ax = tab.new_ax(k, )

            init_guess = init_params[i]
            try:
                true = true_params[k]
                tab_ax.axvline(true, label='True')
            except KeyError:
                true = init_guess

            tab_x = np.linspace(min(true - dx/2, init_guess - dx), max(init_guess + dx, true + dx/2), 3000)
            _f = isolate_variable(logProb, init_params, i, func_args=[names_indices, n_peaks, init_logProb])
            y = _f(tab_x)

            tab_ax.axvline(init_guess, label='Guess', c='red')
            tab_ax.axvline(final_params[names_indices[k]], label='Fit', c='Green')

            tab_ax.plot(tab_x, y, marker='.')
            tab_ax.legend()


if __name__ == '__main__':
    np.random.seed(0)
    # # ==============================
    JAX = True
    DO_TAB = True
    bins = np.linspace(0, 100, 500)
    x = 0.5 * (bins[1:] + bins[:-1])
    amps = [12000, 50000, 30000, 25600]  # total counts in peak
    bf_rate = 3000  # bg counts per bin
    slope_rate = 100
    centers = [10, 35.6, 55.7, 68]
    sigmas = 1, 2, 4, 7
    bg_guess_scale = 0.6
    #==================================

    if not hasattr(sigmas, '__iter__'):
        sigmas = [sigmas] * len(centers)

    print(f"True params:")
    print(f"\tbg: {bf_rate/(bins[1] - bins[0])}")

    true_bg = bf_rate/(bins[1] - bins[0])
    true_params = {'bg': true_bg}

    for i, amp in enumerate(amps):
        print(f"\t_{i}_amp: {amp}")
        _max = amp/(np.sqrt(2*np.pi) * sigmas[i])
        print(f"\t_{i}_max: {_max}")
        print()
        true_params[f'_{i}_amplitude'] = amp
        true_params[f'_{i}_max'] = _max
        true_params[f'_{i}_center'] = centers[i]
        true_params[f'_{i}_sigma'] = sigmas[i]

    data = np.empty(0, dtype=float)

    for c, amp, sigma in zip(centers, amps, sigmas):
        data = np.concatenate([data, np.random.normal(c, sigma, amp)])

    slope_bg = np.sqrt(np.random.uniform(0, bins[-1]**2, int(slope_rate * (len(bins) - 1))))
    bg = np.random.uniform(0, bins[-1], int(bf_rate * (len(bins) - 1)))
    data = np.concatenate([slope_bg, bg, data])

    y, bins = np.histogram(data, bins)

    nbins_filter = 100/np.median(y)
    bwidth = x[1] - x[0]

    gaus_fit(bins, y, center_guesses=centers, jit=JAX)

    plt.show()




    # def _logProb(args_dict):
    #     amplitude = args_dict['amplitude']
    #     center = args_dict['center']
    #     sigma = args_dict['sigma']
    #     bg = args_dict['bg']
    #     model_rates = bg + bin_widths * amplitude * norm.pdf(x, loc=center, scale=sigma)
    #     sel = jnp.where(model_rates > 0)
    #     model_rates = model_rates[sel]
    #     out = -jnp.sum(model_rates) + jnp.sum(counts[sel] * jnp.log(model_rates))
    #     return -out

    # center_is = np.searchsorted(bins, center_guesses, side='right') - 1
    # bin_widths = np.asarray(bins[1:] - bins[:-1])
    # bin_widths_at_centers = bin_widths[center_is]
    # mean_bin_width = np.mean(bin_widths_at_centers)

    # bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # assert len(counts) == len(bin_centers), "`bins` argument must be len(counts) + 1"

    # ax = mpl_hist(bins, counts)
    # ax.set_ylabel("Counts")
    # ax.set_xlabel("x")
    #
    # I = np.sum(bin_widths * counts)
    # center_is = np.searchsorted(bins, center_guesses, side='right') - 1
    #
    # bg_guess = np.median(counts)
    # amp_guess = I - np.sum(bg_guess * bin_widths)
    # # sigma_guess = 3*np.mean(bin_widths)
    # max_guess = np.mean(counts[center_is[0] - center_playN: center_is[0] + center_playN]) - bg_guess
    #
    # center_min, center_max = center_guesses[0] - mean_bwidth * center_playN, \
    #                          center_guesses[0] + mean_bwidth * center_playN
    # # sigma_min = 0.756 * mean_bwidth
    #
    # init_params = Parameters()
    # init_params.add('amplitude', value=amp_guess, min=np.sqrt(bg_guess), max=I,)
    #
    # init_params.add('center', value=center_guesses[0], min=center_min, max=center_max)
    # init_params.add('max', value=max_guess/mean_bwidth, min=0, max=max(counts))#, brute_step=mean_bwidth*0.5)
    # init_params.add('sigma', expr='amplitude*0.398942/max') #, value=sigma_guess, min=sigma_min, max=(x[-1] - x[0])*0.5) #, brute_step=mean_bwidth*0.5)
    # init_params.add('bg', value=bg_guess, min=np.sqrt(bg_guess)/10, max=max(counts),)  # brute_step=max(counts)/20)
    #
    # ax.plot(x, model(init_params), label='initial model')
    #
    # fitter = Minimizer(logProb, init_params)
    #
    # result = fitter.minimize(method='nelder', params=init_params)  # brute_step
    # # print(result)
    # #
    # # print(fit_report(result))
    #
    # fit_params = result.params
    #
    # H = hessian(_logProb)({k: v.value for k, v in fit_params.items() if k != 'max'})
    # m = np.array([[x for x in v.values()] for v in H.values()])
    # errs = {k: np.sqrt(np.abs(f)) for k, f in zip(H.keys(), np.diag(np.linalg.inv(m)))}
    #
    # true_params = {"sigma": sigma, 'center': center, 'amplitude': amp}
    #
    # for i, (k, v) in enumerate(fit_params.items()):
    #     if k in true_params:
    #         true = true_params[k]
    #     else:
    #         true = None
    #
    #     try:
    #         err = errs[k]
    #     except KeyError:
    #         err = 0
    #
    #     param = ufloat(v.value, err)
    #     v.stderr = err
    #     print(k, param, f'{true}')
    #     # print(k, v)
    #
    # other_params = fit_params.copy()
    # print(f"init log_prob: {logProb(init_params)}")
    # print(f"Final log_prob: {logProb(fit_params)}")
    # print(f"init - final: {logProb(init_params) - logProb(fit_params)}")
    #
    # ax.plot(x, model(fit_params), label='fit')
    # ax.legend()

