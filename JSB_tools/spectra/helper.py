import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5agg')
cwd = Path(__file__).parent


def find_nearest_local_maximum(ys, start_index, smooth_width=3):
    """
    Find index of nearest local maximum, after "smoothening" the array.

    Good for finding peak centers for fit initial params

    Args:
        ys:
        start_index:
        smooth_width:

    Returns:

    """
    if not len(ys) or start_index < 0 or start_index >= len(ys):
        raise ValueError

    if smooth_width == 0:
        smooth_width = None

    elif smooth_width % 2 == 0:
        smooth_width += 1  # Make odd so that window is "even centered"

    n = len(ys)
    saved_sums = {}

    def get_val(index):
        if smooth_width is None:
            return ys[index]

        try:
            smoothed_val = saved_sums[index]
        except KeyError:
            i0, i1 = max(0, index - smooth_width//2), min(n - 1, index + smooth_width//2 + 1)
            vals = ys[i0: i1]
            smoothed_val = saved_sums[index] = np.average(vals)

        return smoothed_val

    def is_local_maximum(index):
        if index == 0:
            return get_val(index) > get_val(index + 1)

        if index == n - 1:
            return get_val(index) > get_val(index - 1)

        return get_val(index) >= get_val(index - 1) and get_val(index) >= get_val(index + 1)

    left_index = start_index
    right_index = start_index

    while left_index >= 0 or right_index < n:
        if left_index >= 0 and is_local_maximum(left_index):
            return left_index

        if right_index < n and is_local_maximum(right_index):
            return right_index

        left_index -= 1
        right_index += 1

    return None


def count_peaks(bins, ys, centers, window_width, smoothening_width=3):
    """
    Returns the sum of array entries in the neighborhood of `centers` according to `bins`.

    Args:
        bins:
        centers:
        ys:
        window_width:
        smoothening_width: Used for snapping to the nearest local maximum

    Returns:
        {'sums': [...], 'lefts': [...], 'rights': [...]}
    """
    assert len(ys) == len(bins) - 1
    indices = np.searchsorted(bins, centers, side='right') - 1

    if not hasattr(centers, '__iter__'):
        indices = [indices]
        iter_flag = True
    else:
        iter_flag = True

    out = {'sums': [], 'lefts': [], 'rights': []}

    for index in indices:
        i0 = find_nearest_local_maximum(ys, index, smoothening_width)
        center = 0.5 * (bins[i0] + bins[i0 + 1])

        rng_indices = np.searchsorted(bins, [center - window_width/2, center + window_width/2], side='right') - 1

        out['lefts'].append(bins[rng_indices[0]])
        out['rights'].append(bins[rng_indices[1] + 1])

        out['sums'].append(sum(ys[rng_indices[0]: rng_indices[1]]))

    out = {k: np.array(v) for k, v in out.items()}
    return out



if __name__ == '__main__':
    from JSB_tools.hist import mpl_hist
    x = np.array([7081, 7082, 7083, 7084, 7085, 7086, 7087, 7088, 7089, 7090, 7091,
       7092, 7093, 7094, 7095, 7096, 7097, 7098, 7099, 7100, 7101, 7102,
       7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113,
       7114, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122, 7123, 7124,
       7125, 7126, 7127])

    y = np.array([  0.55555556,   0.77777778,   0.88888889,   0.88888889,
         0.77777778,   1.        ,   1.11111111,   1.22222222,
         1.44444444,   1.88888889,   2.88888889,   4.22222222,
         6.55555556,  10.55555556,  15.33333333,  21.11111111,
        28.55555556,  38.44444444,  50.22222222,  61.44444444,
        75.        ,  88.22222222,  97.44444444, 105.88888889,
       112.88888889, 112.88888889, 109.66666667, 103.44444444,
        94.55555556,  83.11111111,  69.66666667,  57.77777778,
        45.        ,  33.        ,  25.88888889,  19.22222222,
        13.11111111,   9.55555556,   6.11111111,   3.88888889,
         2.44444444,   1.77777778,   0.88888889,   0.44444444,
         0.22222222,   0.22222222,   0.22222222])
    start = 24

    find_nearest_local_maximum(y, start, smooth_width=0)
    # y = [0, 1, 2, 3, 4, 3, 2, 1, 0]
    bins = np.arange(0, len(y) + 1)
    mpl_hist(bins, y)

    print(count_peaks(bins, 2, y, 5, ))

    plt.show()

