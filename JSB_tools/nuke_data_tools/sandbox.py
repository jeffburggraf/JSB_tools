import numpy as np
from JSB_tools.nuke_data_tools import Nuclide
from Shot_data_analysis.PHELIXDataTTree import ttree_and, ttree_cut_range, get_global_energy_bins, erg_efficiency
from GlobalValues import shot_groups
from JSB_tools.TH1 import TH1F
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
# shot_group = shot_groups['N2']
# hist = TH1F(bin_left_edges=get_global_energy_bins())
# hist.Project(shot_group.tree, 'erg', weight='1.0/eff')
# hist.plot(logy=False)



n = Nuclide.from_symbol('W184')
# n = Nuclide.from_symbol('U238')
n.proton_induced_fiss_xs.plot()
plt.show()


#
# x = hist.bin_centers
# widths = [np.zeros_like(x), np.array(20/hist.bin_widths, dtype=int)]
# y = hist.nominal_bin_values
# yerr = hist.bin_std_devs
# peak_ix, peak_infos = find_peaks(y, prominence=yerr, width=widths)
# for k,v in peak_infos.items():
#     print(k, v)
# prominences = peak_infos['prominences']
# widths = peak_infos['widths']
# right_bases = np.interp(peak_infos['right_ips'], range(len(x)), x)
# left_bases = np.interp(peak_infos['left_ips'], range(len(x)), x)
# width_heights =peak_infos['width_heights']
# plt.scatter(x[peak_ix], y[peak_ix])
# for i in range(len(prominences)):
#     data_i = peak_ix[i]
#     _x_ = x[data_i]
#     width = widths[i]
#     y_high = y[data_i]
#     y_low = y_high - prominences[i]
#     y_mid = 0.5*(y_high + y_low)
#     plt.plot([_x_, _x_], [y_low, y_high], c='black', label='prominences' if i ==0 else None)
#     plt.plot([left_bases[i], right_bases[i]], [y_mid, y_mid], c='green', label='bases' if i ==0 else None)
#     plt.plot([left_bases[i], right_bases[i]], [width_heights[i], width_heights[i]], c='red', label='width_h' if i ==0 else None)
#
#
# plt.scatter(x[peak_ix], y[peak_ix])
# for i in range(10):pass
#
# plt.legend()
# # print(np.polyfit(x, y, deg=3, w=w))
# # coeff = np.polyfit(x, y, deg=5, w=w)
# # baseline = np.sum([c*x**i for i,c in enumerate(reversed(coeff))], axis=0)
# # plt.plot(x, baseline)
# #
# plt.show()


# if __name__ == "__main__":
#     import time, ROOT

    # while True:
    #     ROOT.gSystem.ProcessEvents()
    #     time.sleep(0.05)
