from pyhdf.SD import SD
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
import numpy as np
from pathlib import Path

# np.random.seed(101)
# zvals = np.random.rand(100, 100) * 10
#  ===========================================================
max_sigma = 6
#  ===========================================================


# h_pjx = SD("/Users/burggraf1/Downloads/HRS_PJX3-s27590_hrs_pjx3.hdf")
# ccd = SD("/Users/burggraf1/Downloads/HRS_CCD-s28091_hrs_ccd_lower.hdf")
# print("h_pjx.datasets()", h_pjx.datasets())
# print("ccd.datasets()", ccd.datasets())
#
#  shape = [2, 550, 1100]
# streak_arrays = h_pjx.select('Streak_array')
# streak_array_0 = streak_arrays[:][0]
# streak_array_1 = streak_arrays[:][1]

def plot_streak_arrays(path: str, n_sigma=4):
    f_name = Path(path).name
    def clip(x):
        if n_sigma is None:
            return x
        shape = x.shape
        x = x.flatten()
        sigma = np.std(x)
        mean = np.mean(x)
        print(f_name, mean)

        out = np.where(np.abs(x - mean) < n_sigma * sigma, x,
                       mean + np.sign(x - mean) * n_sigma * sigma).reshape(shape)
        return out
    sd = SD(path)
    for i, streak_array in enumerate(sd.select('Streak_array')):
        streak_array = clip(streak_array)

        plt.figure()
        plt.title(f"{f_name}\nstreak_array_{i}")
        im = plt.imshow(streak_array, norm=LogNorm(min(streak_array.flatten()), max(streak_array.flatten())),
                        origin='lower')
        # pcm = plt.pcolormesh(x, y, values, rasterized=True)  # you don't need rasterized=True
        plt.colorbar(im)


plot_streak_arrays("/Users/burggraf1/Downloads/HRS_PJX3-s27590_hrs_pjx3.hdf")
plot_streak_arrays("/Users/burggraf1/Downloads/HRS_CCD-s28091_hrs_ccd_lower.hdf")
plt.show()
