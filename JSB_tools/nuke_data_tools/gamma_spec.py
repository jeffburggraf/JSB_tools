from JSB_tools.nuke_data_tools import Nuclide, GammaLine
from JSB_tools.TH1 import TH1F
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm
import math
from uncertainties import unumpy as unp
try:
    import ROOT
except ModuleNotFoundError as e:
    raise Exception('Must have ROOT installed to use this functionality!') from e


n_axs = 0
MAX_AXIS_CREATED = 10

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def get_similarity(nuclide: Nuclide, min_intensity, gamma_erg_bin_centers, peak_widths_i, peak_indicies,
                   peak_prominences):
    global n_axs

    def get_prob(gamma_line: GammaLine):
        try:
            g_index = find_nearest(gamma_erg_bin_centers, gamma_line.erg.n)
        except TypeError:
            print(gamma_line.erg, gamma_erg_bin_centers)
            raise
        width = peak_widths_i[g_index]
        find_peaks_index = find_nearest(peak_indicies, g_index)
        closest_index = peak_indicies[find_peaks_index]
        _prominence = peak_prominences[find_peaks_index]
        _prob = norm.pdf((g_index-closest_index), scale=width/np.sqrt(_prominence.n))

        return _prob, _prominence

    probs = []
    prominences = []
    line_intensities = []
    ergs = []
    for g in nuclide.decay_gamma_lines:
        if g.intensity >= min_intensity:
            prob, prominence = get_prob(g)
            if prob*prominence > 0:
                probs.append(prob)
                prominences.append(prominence)
                line_intensities.append(g.intensity.n)
                ergs.append(g.erg.n)
        else:
            break

    if len(probs) == 0:
        return None, None, None, None, None

    prob_powers = unp.std_devs(prominences)
    prob_power = 1.0/np.sum(prob_powers)
    weight = prob_power
    prob = np.product(np.array(probs)**prob_powers)**prob_power

    prominences = np.array(prominences)
    intensities_ = line_intensities[:]
    line_intensities = np.array(line_intensities)
    line_intensities /= np.linalg.norm(line_intensities)

    return np.dot(prominences, prob*line_intensities), ergs, intensities_, prob, prominences


def nuclide_search(spectrum_histo: TH1F, numberIterations=10, verbose=False, peak_width=1, sigma_prominence=2,
                   n_max_gamma_lines=3, min_intensity=0.05, a_z_hl_cut=None, exclude_511=True):
    """
    peak_width: either a number or a iterable  of the same length of `spectrum_histo`
    sigma_prominence: number of sigmas a peak's prominence must be above baseline to be considered for a peak
    """
    assert isinstance(spectrum_histo, TH1F)

    peak_widths_in_units_indicies = peak_width/spectrum_histo.bin_widths

    s = ROOT.TSpectrum()

    baseline = np.array(spectrum_histo.nominal_bin_values, dtype=np.float)

    s.Background(baseline, len(spectrum_histo), numberIterations, ROOT.TSpectrum.kBackDecreasingWindow,
                           ROOT.TSpectrum.kBackOrder8, ROOT.kTRUE,
                 ROOT.TSpectrum.kBackSmoothing7, ROOT.kTRUE)

    sig_plus_bg = spectrum_histo
    spectrum_histo = spectrum_histo - baseline

    if verbose:
        plt.figure()
        sig_plus_bg.plot(plt, logy=0, leg_label='sig + bk', alpha=0.5)
        plt.plot(spectrum_histo.bin_centers, baseline, label='bg')
        plt.legend()
        plt.errorbar(spectrum_histo.bin_centers, spectrum_histo.nominal_bin_values, yerr=spectrum_histo.std_errs,
                     ds="steps-mid")
        _, ax = spectrum_histo.plot()
        ax.set_title('Background subtracted result')

    _source = np.array(spectrum_histo.nominal_bin_values, dtype=np.float)

    peak_idxs, peak_infos = find_peaks(spectrum_histo.nominal_bin_values, prominence=sigma_prominence*hist.std_errs,
                                       width=peak_width)

    if exclude_511:
        index_511 = find_nearest(spectrum_histo.bin_centers, 511)
        peak_idxs_selector = [find_peaks_i for find_peaks_i, spectrum_index in enumerate(peak_idxs)
                     if (abs(spectrum_index-index_511)/peak_widths_in_units_indicies[spectrum_index]) > 1.5]
    else:
        peak_idxs_selector = np.range(len(peak_idxs))
    peak_idxs = peak_idxs[peak_idxs_selector]
    peak_prominences = peak_infos['prominences'][peak_idxs_selector]
    peak_prominences_errors = peak_prominences*hist.rel_errors[peak_idxs]
    peak_prominences = unp.uarray(peak_prominences, peak_prominences_errors)
    # peak_widths = peak_infos['widths']
    if verbose:
        _, ax = spectrum_histo.plot(zorder=0)
        ax.scatter(spectrum_histo.bin_centers[peak_idxs], spectrum_histo.nominal_bin_values[peak_idxs],
                   marker='x', label='gamma lines', s=5, c='red', zorder=2)

        for index, (peak_idx, prominence) in enumerate(zip(peak_idxs, peak_prominences)):
            x = spectrum_histo.bin_centers[peak_idx]
            y = spectrum_histo.nominal_bin_values[peak_idx]
            ax.plot([x, x], [y-prominence.n, y], c='green', linewidth=1, label='prominences' if index==0 else None)

        ax.legend()

    nuclides = Nuclide.get_all_nuclides(a_z_hl_cut=a_z_hl_cut)
    nuclides = [n for n in nuclides if not n.is_stable]

    best_nuclides = []
    best_dot_products = []
    infos = []

    for n in nuclides:
        s,  ergs, line_intensities, prob, prominences = get_similarity(n, min_intensity, spectrum_histo.bin_centers, peak_widths_in_units_indicies,
                           peak_idxs, peak_prominences)
        if s is None:
            continue
        index = np.searchsorted(best_dot_products, s)
        best_dot_products.insert(index, s)
        best_nuclides.insert(index, n)
        info = ['erg: {erg}, intensity: {i}; prob: {prob}, prom: {prom} '.format(erg=erg, i=i, prob=prob, prom=prom)
                for erg, i, prom in zip(ergs, line_intensities, prominences)]
        infos.insert(index, info)

    for n, d, info in (zip(best_nuclides, best_dot_products, infos)):
        print('{0}, FOM: {1}... {2}'.format(n, d, info))


if __name__ == '__main__':
    from GlobalValues import shot_groups
    from PHELIXDataTTree import get_global_energy_bins
    shot_group = shot_groups['He+Ar']
    hist = TH1F(bin_left_edges=get_global_energy_bins(100, 3000))
    hist.Project(shot_group.tree, 'erg', weight='1.0/eff')
    nuclide_search(hist, verbose=True, a_z_hl_cut='5<hl<1000')
    #
    # print(Nuclide.from_symbol('Xe139').decay_gamma_lines)

    plt.show()
