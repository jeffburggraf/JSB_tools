import matplotlib.pyplot as plt

from JSB_tools import calc_background, mpl_hist
import numpy as np
from lmfit.models import GaussianModel
from lmfit.model import CompositeModel
from scipy.signal import find_peaks


def multi_peak_fit(centers, bins, counts):
    """
    Todo: deal with situation when peaks are so close that a seconds peak isnt found.
    Args:
        centers:
        bins:
        counts:

    Returns:

    """
    assert len(bins) - 1 == len(counts), [len(bins), len(counts)]
    center_idx = np.searchsorted(bins, centers, side='right') - 1
    model = None
    params = None
    y_bg = calc_background(counts)
    y = counts - y_bg
    yerr = np.sqrt(counts + y_bg)
    x = (bins[1:] + bins[:-1])/2
    b_widths = bins[1:] - bins[:-1]
    y /= b_widths
    peaks, peak_infos = find_peaks(y, height=1.5*yerr, width=0, prominence=4*yerr, rel_height=0.25)
    _select_peaks = np.argmin(np.abs([i - peaks for i in center_idx]), axis=1)
    peaks = peaks[_select_peaks]
    peak_infos = {k: v[_select_peaks] for k, v in peak_infos.items()}
    sigma_guesses = peak_infos['widths']/1.51706*b_widths[peaks]
    amp_guesses = peak_infos['peak_heights']*sigma_guesses*np.sqrt(2*np.pi)
    # center_guesses = x[peaks]
    center_guesses = centers
    model = None
    params = None
    for index, (center_guess, sigma_guess, amp_guess) in enumerate(zip(center_guesses, sigma_guesses, amp_guesses)):
        m = GaussianModel(prefix=f'_{index}')
        if model is None:
            model = m
            params = model.make_params()
        else:
            model += m
            params.update(m.make_params())
        params[f'_{index}amplitude'].set(value=amp_guess)
        params[f'_{index}sigma'].set(value=sigma_guess)
        params[f'_{index}center'].set(value=center_guess)
    weights = np.where(yerr>0, yerr, 1)
    weights = 1.0/weights

    fit_result = model.fit(x=x, data=y, weights=weights, params=params)
    fit_result.plot()
    print(fit_result.fit_report())
    plt.figure()
    plt.plot(x, y)
    plt.plot(x[peaks], y[peaks]+y_bg[peaks], ls='None', marker='o')


    # x =
    # plt.figure()
    # y /= self.erg_bin_widths[_slice]  # make density
    # mpl_hist(self.erg_bins[_slice.start: _slice.stop + 1], y)
    #
    # # y /= np.mean(x[1:] - x[:-1])
    # peak_centers, peak_infos = find_peaks(unp.nominal_values(y), height=unp.std_devs(y), width=0)
    # # plt.plot(peak_centers, peak_infos['prominences'], ls='None', marker='o')
    # select_peak_ixs = np.argmin(np.array([np.abs(c - np.searchsorted(x, centers)) for c in peak_centers]).T, axis=1)
    # peak_widths = peak_infos['widths'][select_peak_ixs] * self.erg_bin_widths[_center]
    # amplitude_guesses = peak_infos['peak_heights'][select_peak_ixs] * peak_widths
    # sigma_guesses = peak_widths / 2.355
    #
    # for i, erg in enumerate(centers):
    #     m = GaussianModel(prefix=f'_{i}')
    #     if model is None:
    #         params = m.make_params()
    #         params[f'_{i}center'].set(value=erg)
    #         model = m
    #     else:
    #         model += m
    #         params.update(m.make_params())
    #     # bin_index = self.__erg_index__(erg)
    #     params[f'_{i}amplitude'].set(value=amplitude_guesses[i], min=0)
    #     params[f'_{i}center'].set(value=erg, min=erg - 0.1, max=erg + 0.1)
    #     params[f'_{i}sigma'].set(value=sigma_guesses[i])
    # weights = unp.std_devs(y)
    # weights = np.where(weights > 0, weights, 1)
    # weights = 1.0 / weights
    # plt.plot(x, model.eval(params=params, x=x))
    # fit_result = model.fit(data=unp.nominal_values(y), x=x, weights=weights, params=params)


def fake_data(n=1000, noise=1, peak_centers=None, amps=None, sigmas=None):
    rnd_scale = np.sum(amps)
    if amps is None:
        amps = 10000*np.array([1, 2, 3])
    if peak_centers is None:
        peak_centers = [100, 120, 400]
    if sigmas is None:
        sigmas = np.ones(len(peak_centers))*3
    assert len(amps) == len(peak_centers) == len(sigmas)

    amps = np.array(amps, dtype=float)
    peak_centers = np.array(peak_centers, dtype=float)

    bins = np.arange(0, n + 1, 0.5)
    out, _ = np.histogram(np.random.uniform(0, len(bins), int(rnd_scale*noise)), bins=bins)  # noise
    for a, c, s in zip(amps, peak_centers, sigmas):
        n_samples = int(a)
        out += np.histogram(np.random.randn(n_samples)*s + c, bins=bins)[0]
    return bins, out


centers = [100, 108, 180, 400, 600]
amps = 10000*np.array([1, 2, 2, 3, 0.05])
b, d = fake_data(peak_centers=centers, amps=amps
                 )

mpl_hist(b, d, np.sqrt(d))
multi_peak_fit([100, 107], b, d)

plt.show()
