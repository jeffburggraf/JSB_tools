import datetime
import warnings
from typing import List, Dict
import numpy as np
from typing import Union, Sequence

import pendulum
from uncertainties import unumpy as unp
from uncertainties import UFloat
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.integrate import trapezoid
import pendulum

__default_stats_kwargs = {'loc': (0.7, 0.8)}



def get_stats(bins, y, errors=True, percentiles=(0.25, 0.5, 0.75)):
    """
    Returns dict of stats. Inspired by ROOT's TH1F default behavior.
    Args:
        bins: Bin left edges
        y: Bin values
        errors: Bin errors
        percentiles: Percentiles to include in return value.

    Returns: dict
        {'count': count,
        'mean': mean,
        'std': std,
        'percentiles': [(percentile_1, x_value1), (percentile_2, x_value2), ...]
        }

    """
    b_centers = 0.5*(bins[1:] + bins[:-1])
    x = b_centers
    if errors:
        x = unp.uarray(x, (bins[1:] - bins[:-1])/(2*np.sqrt(3)))
    else:
        y = unp.nominal_values(y)

    mean = sum(y*x)/sum(y)
    std = usqrt(sum(y*(x - mean)**2)/sum(y))

    cumsum = np.cumsum(unp.nominal_values(y))

    percentiles_xs = []

    for p in percentiles:
        frac = unp.nominal_values(cumsum[-1]*p)
        i = np.searchsorted(cumsum, frac, side='right') - 1

        if i < 0:
            percentiles_xs.append(x[0])
            continue
        elif i == len(cumsum):
            percentiles_xs.append(x[-1])
            continue

        x0 = x[i]

        y0 = cumsum[i]
        try:
            y1 = cumsum[i + 1]
        except:
            print()

        x1 = x[i + 1]
        di = (frac - y0)/(y1 - y0)
        dx = di*(x1 - x0)

        percentiles_xs.append(x0 + dx)

        count = sum(unp.nominal_values(y))

        if int(count) == count:
            count = int(count)

    return {'count': count,
            'mean': mean,
            'std': std,
            'percentiles': list(zip(percentiles, percentiles_xs))}


def mpl_hist(bin_edges, y, yerr=None, ax=None, label=None, fig_kwargs=None, title=None, poisson_errors=False,
             return_handle=False, stats_box=False, stats_kwargs=None, make_density=False,
             errorevery=1, **mpl_kwargs):
    """

    Args:
        bin_edges: Left edges of bins (must be of length len(y) + 1)
        y:
        yerr:
        ax:
        label: For legend
        fig_kwargs: kwargs for mpl.figure
        title:
        poisson_errors: If True and yerr is not provided, assume Poissonian errors.

        return_handle: Return the handle for custom legend creation. Form is tuple([handle1, handle2]).
            To make legend with marker and all, do e.g. fig.legend(handles, labels), where each element in handles is
            that which is returned due to this argument being True.

        stats_box: If true write stats box akin to ROOT histograms.
        stats_kwargs: Default is {'loc': (0.7, 0.8)}
        make_density:
        elinewidth:
        **mpl_kwargs:

    Returns:
            ax                        if not return_handle
            ax, [handle1, handle2]    if return_handle

    """
    if not len(bin_edges) == len(y) + 1:
        raise ValueError(f'`bin_edges` must be of length: len(y) + 1, '
                         f'not {len(bin_edges)} for bins and {len(y)} for y ')

    if fig_kwargs is None:
        fig_kwargs = {}

    if ax is None:
        plt.figure(**fig_kwargs)
        ax = plt.gca()
    elif ax is plt:
        ax = plt.gca()
    else:
        assert isinstance(ax, Axes), f"`ax` argument must be an mpl.Axes instance, not {type(ax)}"

    def sep_errs():  # place errors in separate array (i.e. no UFloats)
        nonlocal yerr, y
        yerr = unp.std_devs(y)
        y = unp.nominal_values(y)

    if isinstance(y[0], UFloat):
        sep_errs()
    else:
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.dtype == 'O':
            sep_errs()

    if yerr is None and poisson_errors:
        yerr = np.sqrt(np.where(y < 0, 0, y))

    if title is not None:
        ax.set_title(title)

    if make_density:
        bwidths = (bin_edges[1:] - bin_edges[:-1])
        y = y / bwidths
        if yerr is not None:
            yerr = yerr / bwidths

    assert y.ndim == 1, f"`y` must be a one dimensional array, not shape of {y.shape}"

    if isinstance(bin_edges[0], datetime.datetime):
        bin_centers = [bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)]  #
    else:
        bin_centers = [(bin_edges[i + 1] + bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)]
    yp = np.concatenate([y, [y[-1]]])

    capsize = mpl_kwargs.pop('capsize', None)

    # handle1 = ax.errorbar(bin_edges, yp, yerr=np.zeros_like(yp), label=label, capsize=0, ds='steps-post', elinewidth=elinewidth, **mpl_kwargs)
    handle1, = ax.plot(bin_edges, yp, label=label,  ds='steps-post',  **mpl_kwargs)

    handle1.set_marker('None')

    if "c" in mpl_kwargs:
        pass
    elif 'color' in mpl_kwargs:
        pass
    else:  # color was from color cycle. Fetch from handle.
        mpl_kwargs['color'] = handle1.get_color()
    mpl_kwargs.pop('ls', None)
    mpl_kwargs.pop('linestyle', None)

    handle2 = None
    if yerr is not None:
        handle2 = ax.errorbar(bin_centers, y, yerr, ls="None", capsize=capsize, errorevery=errorevery, **mpl_kwargs)  # draw error_bars and markers.

    # if label is not None:
    #     ax.legend()

    if stats_box:
        if stats_kwargs is None:
            stats_kwargs = __default_stats_kwargs
        else:
            stats_kwargs = {k: (stats_kwargs[k] if k in stats_kwargs else __default_stats_kwargs[k])
                            for k in __default_stats_kwargs}

        stats = get_stats(bin_edges, y)
        s = ""
        for k, label in zip(['count', 'mean', 'std'], ['count', r'$\mu$       ', r'$\sigma$       ']):

            v = f"${stats[k]:.L}$" if isinstance(stats[k], UFloat) else f"{stats[k]:.2g}"

            s += f'{label}  {v}\n'

        for p, x in stats['percentiles']:
            s += f'{int(100*p)}       {x}\n'

        props = dict(boxstyle='round', facecolor='tab:grey', alpha=0.75)

        ax.text(*stats_kwargs['loc'], s, transform=ax.transAxes, bbox=props, color=handle1[0].get_color())

    out = [ax]

    if return_handle:
        if handle2 is None:
            out += [(handle1,)]
        else:
            out += [(handle1, handle2)]

    if len(out) == 1:
        return ax
    else:
        return tuple(out)


def mpl_hist_from_data(bin_edges: Union[list, np.ndarray, int], data, weights=None, ax=None, label=None, fig_kwargs=None, title=None,
                       nominal_values=False, log_space=False, stats_box=False, norm=None, **mpl_kwargs):
    """
    Plots a histogram from raw data.

    Args:
        bin_edges: List or int.
        data: 1D list of data points.
        weights: Weights to each value in data.
        ax:
        label:
        fig_kwargs:
        title:
        nominal_values:
        log_space: If True, bin_edges must be an int and bins will be constant width in log space
        stats_box:
        norm: Scale data such that integral equals norm
        **mpl_kwargs:

    Returns:
        bin_values, *from mpl_hist*

    """
    assert hasattr(data, '__iter__'), f'Bad argument for `data`: Not an iterator. "{data}"'
    assert len(data) > 0, f'Bad argument for `data`: Empty array'

    if isinstance(data[0], UFloat):
        data = unp.nominal_values(data)

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if not np.isfinite(data.max() + data.min()):
        data = data[np.where(np.isfinite(data))]

    if isinstance(bin_edges, int):
        bin_edges = np.linspace(min(data), max(data), bin_edges + 1)

    yerr = None

    if isinstance(weights, (int, float)):
        y, _ = np.histogram(data, bins=bin_edges, weights=None)
        y = np.asarray(y, dtype=float)

        if not nominal_values:
            yerr = np.sqrt(y) * weights  # do first!

        y *= float(weights)

    elif weights is not None:
        y, _ = np.histogram(data, bins=bin_edges, weights=weights)

        if not nominal_values:
            y_counts, _ = np.histogram(data, bins=bin_edges, weights=None)
            rel_err = 1 / np.sqrt(y_counts)
            yerr = y * rel_err
    else:
        y, _ = np.histogram(data, bins=bin_edges, weights=None)

    if norm is not None:
        integral = trapezoid(y, x=0.5*(bin_edges[1:] + bin_edges[:-1]))
        scale = norm/integral
        y = y * scale

        if yerr is not None:
            yerr = yerr * scale

    return y, mpl_hist(bin_edges, y, yerr, ax=ax, label=label, fig_kwargs=fig_kwargs, title=title,
                       stats_box=stats_box,  **mpl_kwargs)




if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt5agg')

    y = [1,2,34]

    mpl_hist([1,2,3,4], y)
