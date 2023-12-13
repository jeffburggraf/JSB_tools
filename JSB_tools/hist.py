import warnings
from typing import List, Dict
import numpy as np
from typing import Union, Sequence
from uncertainties import unumpy as unp
from uncertainties import UFloat
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.integrate import trapezoid


def mpl_hist(bin_edges, y, yerr=None, ax=None, label=None, fig_kwargs=None, title=None, poisson_errors=False,
             return_handle=False, stats_box=False, stats_kwargs=None, elinewidth=1.1, errorevery=1, **mpl_kwargs):
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

    assert y.ndim == 1, f"`y` must be a one dimensional array, not shape of {y.shape}"

    bin_centers = [(bin_edges[i + 1] + bin_edges[i]) / 2 for i in range(len(bin_edges) - 1)]
    yp = np.concatenate([y, [y[-1]]])

    capsize = mpl_kwargs.pop('capsize', None)

    handle1 = ax.errorbar(bin_edges, yp, yerr=np.zeros_like(yp), label=label, capsize=0, ds='steps-post',
                          elinewidth=elinewidth, **mpl_kwargs)

    handle1[0].set_marker('None')

    if "c" in mpl_kwargs:
        pass
    elif 'color' in mpl_kwargs:
        pass
    else:  # color was from color cycle. Fetch from handle.
        mpl_kwargs['color'] = handle1[0].get_color()
    mpl_kwargs.pop('ls', None)
    mpl_kwargs.pop('linestyle', None)
    handle2 = ax.errorbar(bin_centers, y, yerr, ls="None", capsize=capsize, errorevery=errorevery, **mpl_kwargs)  # draw error_bars and markers.

    if label is not None:
        ax.legend()

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
        out += [(handle1, handle2)]

    if len(out) == 1:
        return ax
    else:
        return tuple(out)


def mpl_hist_from_data(bin_edges: Union[list, np.ndarray, int], data, weights=None, ax=None, label=None, fig_kwargs=None, title=None,
                       return_line_color=False, log_space=False, stats_box=False, norm=None, **mpl_kwargs):
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
        return_line_color:
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

    if log_space:
        assert isinstance(bin_edges, int), "`bin_edges` must be an int to use log spaced bins."
        if data.min() <= 0:
            data = data[np.where(data > 0)]

        _min = np.log10(min(data))
        _max = np.log10(max(data))

        bin_edges = np.logspace(_min, _max, bin_edges + 1)
    else:
        if isinstance(bin_edges, int):
            bin_edges = np.linspace(min(data), max(data), bin_edges + 1)

    y, _ = np.histogram(data, bins=bin_edges, weights=weights)
    yerr = np.sqrt(y)

    if norm is not None:
        integral = trapezoid(y, x=0.5*(bin_edges[1:] + bin_edges[:-1]))
        scale = norm/integral
        y = y * scale
        yerr = yerr * scale

    return y, mpl_hist(bin_edges, y, yerr, ax=ax, label=label, fig_kwargs=fig_kwargs, title=title,
                       stats_box=stats_box,  **mpl_kwargs)
