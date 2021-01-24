from JSB_tools.nuke_data_tools.gamma_spec import exp_decay_maximum_likely_hood
import numpy as np
np.random.seed(0)
from JSB_tools.TH1 import TH1F
from matplotlib import pyplot as plt
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from PHELIXDataTTree import get_global_energy_bins, erg_efficiency
from GlobalValues import shot_groups

# shot_group = shot_groups['He+Ar']
# tree = shot_group.tree
# hist = TH1F(bin_left_edges=get_global_energy_bins(100, 2000))
# hist.Project(tree, 'erg', weight='1/eff')
#
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
# hist.plot(ax=ax1)

#  =====================================
sigma_range = 1, 3
s_g_ratio = 10
peaks = [0, 5, 10, 20]
N = 1000
sliding_window = 5
#  =====================================
amplitudes = np.random.uniform(0.1, 1, len(peaks))

x_range = min(peaks) - 5*sigma_range[-1], max(peaks) + 5*sigma_range[-1]

hist = TH1F(*x_range, bin_width=sigma_range[0]/2)

sliding_window = int(sliding_window/hist.bin_width)

n_sig = int(sum(amplitudes*N))
n_noise = int(n_sig/s_g_ratio)

for peak, amp in zip(peaks, amplitudes):
    sigma = np.random.uniform(*sigma_range)
    for x in np.random.normal(peak, sigma, N):
        hist.Fill(x)

for x in np.random.uniform(*x_range, n_noise):
    hist.Fill(x)

rolling_window = 30


counts = hist.nominal_bin_values
grouped_values = np.array([counts[i-rolling_window//2:i+rolling_window//2] for i in range(rolling_window//2, len(hist)-rolling_window//2)])
medians = np.median(grouped_values, axis=1)
mad = np.median([np.abs(medians[i] - grouped_values[i]) for i in range(len(grouped_values))], axis=1)
print(medians, grouped_values)
# deviations = [np.median(medians-grouped_values, axis=1)]
# for

print(grouped_values.shape,medians.shape, mad.shape)
foms = ([x[rolling_window//2] for x in grouped_values] - medians)/(1.7*mad)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
print(foms)
ax2.plot(hist.bin_centers[rolling_window//2: -rolling_window//2], foms)
hist.plot(ax1)
plt.show()