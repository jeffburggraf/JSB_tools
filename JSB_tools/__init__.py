"""
A messy collection of stuff that nuclear physicists like like to use.

todo: move some of these imports into functions to speed up loading of this module
"""
from __future__ import annotations
import warnings
from matplotlib.colors import LogNorm, SymLogNorm
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    go = make_subplots = ModuleNotFoundError
from typing import List, Dict
import numpy as np
from itertools import islice
from scipy.interpolate import InterpolatedUnivariateSpline
try:
    from sortedcontainers import SortedDict
except ModuleNotFoundError:
    SortedDict = ModuleNotFoundError
from pathlib import Path
from typing import Union, Sequence
import pickle
from atexit import register
from scipy.interpolate import interp1d
from uncertainties import unumpy as unp
from uncertainties.core import UFloat, ufloat, AffineScalarFunc
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
import time
import sys
from scipy.stats import norm
from matplotlib import cm
import traceback
from uncertainties import UFloat
from scipy import ndimage
from scipy.stats import pearsonr
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from JSB_tools.hist import mpl_hist, mpl_hist_from_data
from JSB_tools.common import ProgressReport, MPLStyle, ROOT_loop
import matplotlib
from uncertainties.umath import sqrt as usqrt
import re
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Colormap
from scipy.integrate import trapezoid
from numbers import Number
from matplotlib.widgets import AxesWidget, RadioButtons, cbook
try:
    raise ModuleNotFoundError
    import ROOT as _ROOT
    root_exists = True
except ModuleNotFoundError:
    root_exists = False


markers = ['p', 'X', 'D', 'o', 's', 'P', '^', '*']


def latex_form(val, n_digits=2):
    """
    Get latex form of number, as in 3.02 \times 10^{12}
    Args:
        val:
        n_digits:

    Returns:

    """
    s = f'{val:.{n_digits}e}'
    if not isinstance(val, AffineScalarFunc):
        m = re.match('([0-9.]+)[eE]([-+0-9]+)', s)
    else:
        m = re.match(r'(\(?[(0-9./+-]+\)?)[eE]([-+0-9]+)', s)

    base = m.groups()[0]
    exp = int(m.groups()[1])

    if isinstance(val, AffineScalarFunc):
        base = base.replace('+/-', r'\pm')
        base = base.replace('-/+', r'\mp')

    return fr"${base}\times 10^{{{exp}}}$"


def hist2D(datax, datay, ax=None, bins=35, logz=False, n_labels_x=5, n_labels_y=5, xfmt='.2g', yfmt='.2g'):
    """
    2D heatmap, similar to ROOTs TH2D

    Args:
        datax: x data values,  to be binned
        datay: y data values,  to be binned
        ax:
        bins:
        logz:
        n_labels_x:
        n_labels_y:
        xfmt:
        yfmt:

    Returns:

    """
    def get_min_after_zero(a):
        return min(flatZ[np.where(a > 0)])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    Z, xbins, ybins = np.histogram2d(datax, datay, bins=bins)
    Z = Z.transpose()

    flatZ = Z.flatten()
    abs_flatZ = np.abs(flatZ)
    if logz:
        if min(flatZ) < 0:
            linthresh = np.percentile(abs_flatZ, 1)
            if linthresh == 0:
                linthresh = get_min_after_zero(abs_flatZ)

            norm = SymLogNorm(linthresh=linthresh, vmin=min(flatZ), vmax=max(flatZ), )
        else:
            if min(flatZ) == 0:
                z0 = get_min_after_zero(flatZ)
                Z[np.where(Z == 0)] = z0
                _min = z0
            else:
                _min = np.min(flatZ)
            norm = LogNorm(vmin=_min, vmax=max(flatZ))

    else:
        norm = None

    im = ax.imshow(Z, origin='lower', norm=norm)
    cbar = fig.colorbar(im, ax=ax)

    y_labels = list(map(lambda x: float(f'{x:{yfmt}}'), np.linspace(0, ybins[-1], n_labels_x)))
    x_labels = list(map(lambda x: float(f'{x: {xfmt}}'), np.linspace(0, xbins[-1], n_labels_y)))

    ax.set_yticks(np.linspace(0, len(ybins) - 1, len(y_labels)), labels=y_labels)
    ax.set_xticks(np.linspace(0, len(xbins) - 1, len(x_labels)), labels=x_labels)

    ax.format_coord = lambda x, y: f'x={x * x_labels[-1]/(len(xbins) -1):g}, y={y * y_labels[-1]/(len(ybins) -1):g}'

    return {'ax': ax, 'cbar': cbar, 'im': im, 'xbins': xbins, 'ybins': ybins}


def binned_down_sample(bins, y, yerr, n):
    """
    Removes every nth entry from bins and values.
    Args:
        bins:
        y:
        yerr:
        n:

    Returns:

    """
    def get(_y):
        return np.array([np.mean(_y[i1:i2]) for i1, i2 in zip(range_[:-1], range_[1:])])
    range_ = np.arange(0, len(bins), n)

    bins = np.array([bins[i] for i in range_])

    y = get(y)

    if yerr is not None:
        yerr = get(yerr)

    return bins, y, yerr


def find_nearest(array, value, return_value=False, side='left'):
    """
    Find the **INDEX** of element in array that is nearest to value.

    If multiple values, return first or last depending on `side` argument.

    If directly in middle of two entries, return left or right value depending on `side` argument

    NOTE: ASSUMES SORTED

    Args:
        array: sorted array of values
        value:
        return_value: If True, return value itself instead of index.
        side: If left, return first value, else return last (in cases of multiple repeat values).
            In the case of value exactly in middle of two array entries, return left value or right value, respectively

    Returns:

    """
    idx = np.searchsorted(array, value, side=side)

    if idx == 0:
        return array[0] if return_value else 0

    if idx == len(array):
        idx -= 1
    else:
        dleft = value - array[idx-1]
        dright = array[idx] - value
        if dright < dleft:
            idx = idx
        elif dright > dleft:
            idx -= 1
        elif dright == dleft:
            if side == 'left':
                idx = idx - 1

    if return_value:
        return array[idx]
    else:
        return idx


class RadioButtons(RadioButtons):
    @property
    def active_index(self):
        for index, val in enumerate(self.labels):
            if val._text == self.value_selected:
                return index

    def __init__(self, ax, labels, label_colors=None, active=0, activecolor='blue', size=49,
                 orientation="vertical", **kwargs):
        """
        Add radio buttons to an `~.axes.Axes`.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        Further parameters are passed on to `Legend`.
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor

            p = ax.scatter([], [], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)

        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")

        kwargs.setdefault("frameon", False)
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts

        self.circles = self.box.legendHandles
        for c in self.circles:
            c.set_picker(5)
        self.cnt = 0

        self.connect_event('pick_event', self._clicked)
        self._observers = cbook.CallbackRegistry(signals=["clicked"])

    def _clicked(self, event):
        if (self.ignore(event) or event.mouseevent.button != 1 or
                event.mouseevent.inaxes != self.ax):
            return
        if event.artist in self.circles:
            # if hasattr(self, '_observers'):
            self.set_active(self.circles.index(event.artist))

    def set_label_colors(self, colors):
        assert len(colors) == len(self.labels)
        for index, color in enumerate(colors):
            self.set_label_color(index, color)

    def set_label_color(self, index, color):
        self.labels[index].set_color(color)
        self.circles[index].set_edgecolor(color)


def get_linear_fit(y_data, y_model, offsetQ=False):
    """
    Find the scale, a, and offset, b, which transforms the y_model values to be as close to y_data as
        possible (according to least-square).
    Args:
        y_data:
        y_model:
        offsetQ: If False, then fit using only a scaling constant.

    Returns: a, b

    """
    y_data = np.asarray(y_data)
    y_model = np.asarray(y_model)

    XY = np.sum(y_data * y_model)
    X2 = np.sum(y_model**2)

    if not offsetQ:
        a = XY/X2
        return a, 0

    NN = len(y_data)
    X = np.sum(y_model)

    Y = np.sum(y_data)

    a = -((NN * XY - X * Y) / (X ** 2 - NN * X2))
    b = -((-(X*XY) + X2*Y)/(X**2 - X2))

    return a, b


class BijectiveMap:
    def __init__(self, _dict: Union[dict, list[tuple]] = None):
        if not isinstance(_dict, dict):
            if _dict is None:
                _dict = {}
            else:
                _dict = dict(_dict)

        self._dict = _dict
        self._reverse_dict = {v: k for k, v in self._dict.items()}

    def _test(self):
        for k, v in self._dict.items():
            assert self._reverse_dict[v] == k
        assert len(self._dict) == len(self._reverse_dict)

    def set_default(self, key, default_value=None):
        """
        Return key as is if it exists, else, set key to default and return new value.

        Args:
            key:
            default_value:

        Returns:

        """
        try:
            out = self[key]
        except KeyError:
            self[key] = out = default_value

        return out

    def pop_item(self):
        k, v = self._dict.popitem()
        self._reverse_dict.pop(v)
        return k, v

    def clear(self):
        self._dict = {}
        self._reverse_dict = {}

    def pop(self, item):
        val = self[item]
        del self[item]
        return val

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default

    def update(self, vals: Union[dict, List[tuple]]):
        if isinstance(vals, dict):
            vals = vals.items()
        else:
            if len(vals[0]) != 2:
                raise ValueError(f"Invalid argument, {vals[0]} ")
        for k, v in vals:
            self[k] = v

    def copy(self):
        return BijectiveMap(self._dict.copy())

    def __raise_key_err(self, item):
        s = f'{item.__repr__()} not in BijectiveMap'
        if len(self) < 25:
            s += f'\n\t{self}'
        raise KeyError(s)

    def __eq__(self, other: Union[dict, BijectiveMap]):
        if isinstance(other, BijectiveMap):
            other = other._dict

        return self._dict == other

    def __setitem__(self, key, value):
        for k in [key, value]:
            try:
                del self[k]
            except KeyError:
                pass

        self._dict[key] = value
        self._reverse_dict[value] = key
        self._test()

    def __getitem__(self, item):
        try:
            return self._dict[item]
        except KeyError:
            try:
                return self._reverse_dict[item]
            except KeyError:
                pass
        self.__raise_key_err(item)

    def __delitem__(self, key):
        try:
            val = self._dict[key]
        except KeyError:
            val = key
            key = self._reverse_dict[key]
        del self._dict[key]
        del self._reverse_dict[val]
        self._test()

    def __ior__(self, other: Union[dict, BijectiveMap]):
        for k, v in other.items():
            self[k] = v
        return self

    def __or__(self, other: Union[dict, BijectiveMap]):
        out = self.copy()
        out |= other
        return out

    def __len__(self):
        """Returns the number of connections"""
        return len(self._dict)

    def __iter__(self):
        return self._dict.__iter__()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def __contains__(self, item):
        return item in self._dict or item in self._reverse_dict

    def __repr__(self):
        return '{' + ', '.join([f'{k.__repr__()} <-> {v.__repr__()}' for k, v in self.items()]) + '}'

    def items(self):
        return self._dict.items()


def weighted_pearsonr(x, y, yerr=None, xerr=None, _wx=None, _wy=None):
    if _wy is not None:
        wy = _wy
    else:
        if isinstance(y[0], UFloat):
            yerr = unp.std_devs(y)
            y = unp.nominal_values(y)

        if yerr is None:
            wy = np.ones(len(y))
        else:
            wy = 1.0 / np.where(yerr > 0, 1, yerr)

    if _wx is not None:
        wx = _wx
    else:
        if isinstance(x[0], UFloat):
            xerr = unp.std_devs(x)
            x = unp.nominal_values(x)

        if xerr is None:
            wx = np.ones(len(x))
        else:
            wx = 1.0/np.where(xerr > 0, 1, xerr)

    xbar = np.average(x, weights=wx)
    ybar = np.average(y, weights=wy)
    varx = np.average((x - xbar)**2, weights=wx)
    vary = np.average((y - ybar)**2, weights=wy)
    covar = np.average((x - xbar) * (y - ybar), weights=np.sqrt(wx * wy))
    out = covar/np.sqrt(varx*vary)

    return out


def MC_pearsonr(x, y, yerr=None, xerr=None, n=350, weighted=False):
    """
    pearsonr with errors.
    Args:
        x:
        y:
        yerr:
        xerr:
        n: Number of MC samples.

    Returns:

    """
    if isinstance(x[0], UFloat):
        xerr = unp.std_devs(x)
        x = unp.nominal_values(x)

    if isinstance(y[0], UFloat):
        yerr = unp.std_devs(y)
        y = unp.nominal_values(y)

    x = np.asarray(x)
    y = np.asarray(y)

    if yerr is not None:
        yerr = np.asarray(yerr)
    if xerr is not None:
        xerr = np.asarray(xerr)

    def get_vals(v, verr):
        if verr is not None:
            out = np.random.normal(size=n*len(v))
            out = out.reshape(n, len(v), )
            out *= verr
            out += v
        else:
            out = np.array(list(v) * n)
        return out

    xerr = np.asarray(xerr)
    yerr = np.asarray(yerr)

    ys = get_vals(y, yerr)
    xs = get_vals(x, xerr)
    corrs = []

    if weighted:
        wx = 1.0 / np.where(xerr > 0, xerr, 1)
        wy = 1.0 / np.where(xerr > 0, xerr, 1)
    else:
        wx = wy = None

    for x, y in zip(xs, ys):
        if weighted:
            r = weighted_pearsonr(x, y, _wx=wx, _wy=wy)
        else:
            r = pearsonr(x, y).statistic
        corrs.append(r)

    return ufloat(np.mean(corrs), np.std(corrs))


def sub_plot_grid(n, favor_more_rows=False):
    if n == 1:
        out = 1, 1
    elif n == 2:
        out = 1, 2
    elif n == 3:
        out = 1, 3
    elif n == 4:
        out = 2, 2
    elif 5 <= n <= 6:
        out = 2, 3
    elif 7 <= n <= 8:
        out = 2, 4
    elif n == 9:
        out = 3, 3
    elif 10 <= n <= 12:
        out = 3, 4
    elif n <= 16:
        out = 4, 4
    else:
        raise OverflowError

    out = tuple(list(sorted(out)))
    if favor_more_rows:
        out = out[::-1]

    return out


def no_openmc_warn():
    warnings.warn("OpenMC not installed! Some functionality limited. ")


def nearest_i(vals, x):
    """
    Given sorted array of values, `vals`, find the indices of vals are nearest to each element in x.
    Args:
        vals:
        x:

    Returns:

    """
    is_ = np.searchsorted(vals, x, side='right') - 1
    midq = x < 0.5 * (vals[is_] + vals[is_ + 1])
    out = np.where(midq, is_, is_ + 1)
    if hasattr(midq, '__iter__'):
        return out[0]
    return out


def errorbar(x, y, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    yerr = unp.std_devs(y)
    y = unp.nominal_values(y)
    ax.errorbar(x, y, yerr, **kwargs)
    return ax


def rebin(old_bins, old_ys, new_bins, is_density=False, return_under_over_flow=False):
    """
    Rebin function (March 2024)

    Args:
        old_bins:
        old_ys:
        new_bins:
        is_density: If old_ys is a density (i.e. not counts), then set this to True
        return_under_over_flow: Return underflow/overflow (values lying outside new_bins)

    Returns:

    """
    if isinstance(old_ys[0], UFloat):
        old_ys = unp.nominal_values(old_ys)

    if is_density:
        new_bins = np.asarray(new_bins, float)
        old_bins = np.asarray(old_bins, float)
        old_ys = np.asarray(old_ys, float)

    def distrib(val, low0, high0, low1, high1):
        l = max(low0, low1)
        r = min(high0, high1)
        if r - l <= 0:
            return 0

        if is_density:
            val *= (high0 - low0)

        return val * (r-l)/(high0 - low0)

    yout = np.zeros(len(new_bins) - 1, dtype=float)

    G_inew = 0

    over_flow = under_flow = 0
    if new_bins[0] > old_bins[0]:
        under_flow = distrib(old_ys[0], old_bins[0], old_bins[1], old_bins[0], new_bins[0])

    if new_bins[-1] < old_bins[-1]:
        over_flow = distrib(old_ys[-1], old_bins[-2], old_bins[-1], new_bins[-1], old_bins[-1])

    for iold in range(len(old_ys)):  # loop through given bin values and distribute into new bins.
        old_left = old_bins[iold]
        old_right = old_bins[iold + 1]

        try:
            while (new_right := new_bins[G_inew + 1]) < old_left:
                G_inew += 1
                continue
        except IndexError:  # Have reached end of new_bins
            break

        inew = G_inew

        while (new_left := new_bins[inew]) < old_right:
            try:
                new_right = new_bins[inew + 1]
            except IndexError:  # Have reached end of new_bins
                break

            yout[inew] += distrib(old_ys[iold], old_left, old_right, new_left, new_right)
            inew += 1

    if is_density:
        yout /= (new_bins[1:] - new_bins[:-1])

    yout = np.asarray(yout)

    if return_under_over_flow:
        return yout, under_flow, over_flow
    else:
        return yout


def MCrebin(old_bin_edges, new_bin_edges, ys, N=1000):
    """
    Monte Carlo re-binning.

    Args:
        old_bin_edges: Bin edges corresponding to `ys`
        ys: Y values (len(old_bin_edges) - 1)
        new_bin_edges:
        N: Number of random samples per bin

    Returns:

    """
    samples = []
    weights = []
    for l0, l1, y in zip(old_bin_edges[:-1], old_bin_edges[1:], ys):
        samples.extend(np.random.uniform(l0, l1, N))
        weights.extend(np.ones(N)*(y/N))

    out, _ = np.histogram(samples, bins=new_bin_edges, weights=weights)

    return out


def mpl_2dhist_from_data(x_bin_edges, y_bin_edges, x_points, y_points, weights=None, ax: Axes3D = None, cmap: Colormap = None):
    """

    Args:
        x_bin_edges:
        y_bin_edges:
        x_points: X values
        y_points: Y values
        weights:
        ax:
        cmap: A Colormap instance.

    Returns:

    """
    if isinstance(x_bin_edges, int):
        x_bin_edges = np.linspace(min(x_points), max(x_points), x_bin_edges)
    if isinstance(y_bin_edges, int):
        y_bin_edges = np.linspace(min(y_points), max(y_points), y_bin_edges)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.gcf()

    H, xedges, yedges = np.histogram2d(x_points, y_points, bins=(x_bin_edges, y_bin_edges), weights=weights)
    H = H.T

    xcenters = 0.5 * (xedges[1:] + xedges[:-1])
    ycenters = 0.5 * (yedges[1:] + yedges[:-1])
    xbwidths = xedges[1:] - xedges[:-1]
    ybwidths = yedges[1:] - yedges[:-1]

    X, Y = np.meshgrid(xcenters, ycenters)
    XW, YW = np.meshgrid(xbwidths, ybwidths)

    x_data = X.flatten()
    y_data = Y.flatten()
    z_data = H.flatten()
    xw = XW.flatten()
    yw = YW.flatten()

    if cmap is None:
        cmap = cm.get_cmap('jet')

    max_height = np.max(z_data)
    min_height = np.min(z_data)

    rgba = [cmap((k - min_height) / (max_height - min_height)) for k in z_data]
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(min(z_data), max(z_data)))

    _ = ax.bar3d(x_data,
             y_data,
             np.zeros(len(z_data)),
             xw, yw, z_data, color=rgba)

    # cbar = plt.colorbar(sm)

    return ax, x_data, y_data, z_data


def flatten_dict_values(X):
    """
    Flatten out all dictionary keys
    Args:
        X:

    Returns:

    """
    out = []
    if isinstance(X, dict):
        for k in X.values():
            out.extend(flatten_dict_values(k))
    else:
        return [X]

    return out


def flatten(X):
    """
    Flatten iter object to be 1-dimension.
    Args:
        X:

    Returns:

    """
    out = []
    if hasattr(X, '__iter__'):
        for x in X:
            out.extend(flatten(x))
    else:
        return [X]

    return out


def rand_choice(bins, probs, size=1, interpolation='uniform'):
    """Random choice from a binned prob. dist."""
    assert len(bins) - 1 == len(probs)

    cumsum = np.cumsum(probs)
    cumsum = cumsum/cumsum[-1]

    idxs = np.searchsorted(cumsum, np.random.random(size))

    half_b_widths = bins[1:] - bins[:-1]

    if interpolation == 'uniform':
        interps = np.random.uniform(-0.5, 0.5, size) * half_b_widths[idxs]
    elif interpolation == 'linear':
        raise NotImplementedError("Todo")
    elif interpolation is None:
        interps = 0
    else:
        raise ValueError(f"Invalid `interpolation` argument, {interpolation}")

    return bins[idxs] + interps


def set_ylims(ax:Axes, lines: List[Line2D], *args, **kwargs):
    """
    Rescale y-axis to fit y-data in current view.
    Use either the current spectra or all spectra for y-data.
    Args:
        all_spectraQ: If True, only scale y to data of current spectra.
        *args:
        **kwargs:

    Returns:

    """
    xlim = ax.get_xlim()
    ylim = np.inf, -np.inf

    for line in lines:
        x, y = line.get_data()
        start, stop = np.searchsorted(x, xlim)
        yc = y[max(start - 1, 0):(stop + 1)]
        ylim = min(ylim[0], np.nanmin(yc)), max(ylim[1], np.nanmax(yc))

    ax.set_xlim(xlim, emit=False)

    dy = 0.07 * (ylim[1] - ylim[0])
    # y axis: set dataLim, make sure that autoscale in 'y' is on
    corners = (xlim[0], ylim[0] - dy), (xlim[1], ylim[1] + dy)
    # print(dy, (xlim[0], ylim[0]), (xlim[1], ylim[1]))
    ax.dataLim.update_from_data_xy(corners, ignore=True, updatex=False)
    ax.autoscale(enable=True, axis='y')

    ax.xlim = xlim
    ax.get_figure().canvas.draw()


def _float(x):
    if x is None:
        return None
    elif isinstance(x, UFloat):
        return float(x.nominal_value)
    else:
        return float(x)


class InteractivePlot:
    """
    Todo: make the add_plot internals function like add_persistant. (i.e. remove a lot of self.xxx: [List]
    """
    color_cycle = ['blue', 'red', 'green', 'black', 'gray']

    def __init__(self, frame_labels=None, slider_prefix='', n_subplots=2):
        assert n_subplots in [1, 2]  # todo

        self.ys = []
        self.yerrs = []
        self.ys_kwargs = []

        self.persistent = []

        self.xs = []

        self.step_titles = []
        self.frame_labels = frame_labels
        self.leg_labels = []

        self.fig = make_subplots(n_subplots, 1,  shared_xaxes=True)
        self.init_slider = None

        self.colors = []
        self.line_types = []

        self.slider_prefix = slider_prefix

    def add_persistent(self, x, y, yerr=None, n_frames=None, leg_label=None, color=None, line_type=None, **plot_kwargs):
        y = np.array(y, copy=False)
        assert y.ndim == 1

        if isinstance(y[0], UFloat):
            yerr = list(map(unp.std_devs, y))
            y = list(map(unp.nominal_values, y))

        if n_frames is None:
            n_frames = self.n_frames

        out = {'y': y, 'x': x,
               'leg_label': leg_label,
               'n_frames': n_frames,
               "yerr": yerr,
               'plot_kwargs': plot_kwargs}

        if color is None:
            l = (len(self.persistent))
            m = len(InteractivePlot.color_cycle)
            color = InteractivePlot.color_cycle[l % m]

        out['color'] = color
        out['line_type'] = line_type
        self.persistent.append(out)

    def add_ys(self, x, ys, yerr=None, leg_label=None, color=None, line_type=None, return_color=False, **plot_kwargs):

        if isinstance(line_type, str):
            line_type = line_type.lower()

        self.line_types.append(line_type)

        if isinstance(ys[0][0], UFloat):
            yerr = list(map(unp.std_devs, ys))
            ys = list(map(unp.nominal_values, ys))

        self.yerrs.append(yerr)
        self.ys.append(ys)
        self.ys_kwargs.append(plot_kwargs)

        self.xs.append(x)
        self.leg_labels.append(leg_label)
        if color is None:
            l = (len(self.ys) - 1)
            m = len(InteractivePlot.color_cycle)
            color = InteractivePlot.color_cycle[l % m]
        self.colors.append(color)
        if return_color:
            return color

    @property
    def n_frames(self):
        if not len(self.ys):
            return 0
        return max(map(len, self.ys))

    def __minmax(self, s):
        assert s in ['min', 'max']

        f = max if s == 'max' else min
        ys = [getattr(np, s)(self.ys)]  # bug here causeing the truth value of an array with more than one element is ambiguous.
        ys.extend([f(p['y']) for p in self.persistent])
        return f(ys)

    @property
    def max_y(self):
        return self.__minmax('max')

    @property
    def min_y(self):
        return self.__minmax('min')

    @property
    def steps(self):
        n_frames = len(self.ys[0])
        steps_visibility = []
        for frame in range(self.n_frames):
            vis = []
            for n in map(len, self.ys):
                vis.extend([True if i == frame else False for i in range(n)])
            for _ in self.persistent:
                vis.append(True)
            steps_visibility.append(vis)

        out = []
        if self.frame_labels is None:
            self.frame_labels = [str(i) for i in range(1, n_frames + 1)]

        for frame_label, visible in zip(self.frame_labels, steps_visibility):
            frame_label = str(frame_label)
            step = dict(
                method="update",
                args=[{"visible": visible},
                      {"title": frame_label}],  # layout attribute
            )
            out.append(step)

        return out
        # for l in self.ys:

    def plot(self):
        n_traces = 0

        def get_line_type(arg):
            return {None: None, 'hist': {'shape': 'hvh'}}[arg]

        for index_step, (ys, yerrs, plot_kwargs, x, color, leg_label, lt) in \
                enumerate(zip(self.ys, self.yerrs, self.ys_kwargs, self.xs, self.colors, self.leg_labels,
                              self.line_types)):

            line = get_line_type(lt)

            if yerrs is None:
                yerrs = [None]*len(ys)

            for index_plot, (y, yerr) in enumerate(zip(ys, yerrs)):
                n_traces += 1
                self.fig.add_trace(
                    go.Scatter(
                        visible=index_plot == 0,
                        x=x,
                        y=y,
                        error_y=dict(type='data', array=yerr),
                        marker_color=color,
                        line=line,
                        name=leg_label,
                        **plot_kwargs
                    ),
                    row=1, col=1
                )

        for persistent in self.persistent:
            n_traces += 1
            line = get_line_type(persistent['line_type'])
            self.fig.add_trace(
                go.Scatter(
                    visible=1,
                    x=persistent["x"],
                    y=persistent['y'],
                    error_y=dict(type='data', array=persistent['yerr']),
                    marker_color=persistent['color'],
                    line=line,
                    name=persistent['leg_label'],
                    **persistent['plot_kwargs']
                ),
                row=2, col=1
            )

        sliders = [dict(
            active=0,
            currentvalue={"prefix": self.slider_prefix},
            pad={"t": 50},
            steps=self.steps
        )]

        self.fig.update_layout(
            sliders=sliders, bargap=0, bargroupgap=0.0,
            yaxis={'title': 'Rate [Hz]', 'rangemode': 'tozero', 'autorange': True},
            xaxis={'title': 'Energy [keV]'}
        )
        # self.fig.update_yaxes(fixedrange=True)
        self.fig.show()


def human_friendly_time(time_in_seconds, unit_precision=2):
    """

    Args:
        time_in_seconds:
        unit_precision: Number of units to print, e.g. for 3 months and 2 days and 10 minutes
            If 1: 3 months
            If 2: 3 months 2 days
            If 3: 3 months 2 days  # num hours of 0 is omitted
            If 4: 3 months 2 days 10 minutes

    Returns:

    """
    rel_error = None
    time = time_in_seconds
    assert unit_precision >= 1
    if isinstance(time_in_seconds, UFloat):
        time = time_in_seconds.n
        rel_error = time_in_seconds.std_dev/time_in_seconds.n

    if time == np.inf or time == np.nan:
        return str(time)

    if time < 1:
        out = "{:.2e} seconds ".format(time)
        if rel_error is not None:
            out += f'+/- {100*rel_error:.1f}%'
        return out
    elif time < 60:
        out = "{:.1f} seconds ".format(time)
        if rel_error is not None:
            out += f'+/- {100 * rel_error:.1f}%'
        return out

    seconds_in_a_minute = 60
    seconds_in_a_hour = 60 * seconds_in_a_minute
    seconds_in_a_day = seconds_in_a_hour * 24
    seconds_in_a_month = seconds_in_a_day * 30
    seconds_in_a_year = 12 * seconds_in_a_month

    n_seconds = time % seconds_in_a_minute
    n_minutes = (time % seconds_in_a_hour) / seconds_in_a_minute
    n_hours = (time % seconds_in_a_day) / seconds_in_a_hour
    n_days = (time % seconds_in_a_month) / seconds_in_a_day
    n_months = (time % seconds_in_a_year) / seconds_in_a_month
    n_years = (time / seconds_in_a_year)
    units = np.array(['years', 'months', 'days', 'hours', 'minutes', 'seconds'])
    values = np.array([n_years, n_months, n_days, n_hours, n_minutes, n_seconds])
    outs = []

    printables = np.where(values >= 1)[0]
    printables = printables[np.where(printables - printables[0] < unit_precision)]
    value, unit = None, None
    for unit, value in zip(units[printables], values[printables]):
        outs.append(f"{int(value)} {unit}")
    if unit == 'seconds':
        outs[-1] = f'{value:.2g} {unit}'
    else:
        outs[-1] = f'{value:.2g} {unit}'
    out = ' '.join(outs)
    if rel_error is not None:
        out += f' (+/- {100*rel_error:.1f}$)'

    return out


def get_bin_index(bins, x_values):
    """
    Get bin index from an x value(s)
    Args:
        bins:
        x_values:

    Returns:

    """
    return np.searchsorted(bins, x_values, side='right') - 1


def multi_peak_fit(bins, y, peak_centers: List[float], baseline_method='ROOT', baseline_kwargs=None,
                   fit_window: float = None, debug_plot=False):
    """
    Fit one or more peaks in a close vicinity.
    Args:
        bins:
        y:
        peak_centers: List of peak_centers of peaks of interest. Doesn't have to be exact, but the closer the better.
        baseline_method: either 'ROOT' or 'median'
        baseline_kwargs: Arguments send to calc_background() or rolling_median() (See JSB_tools.__init__)
        fit_window: A window that should at least encompass the peaks (single number in KeV).
        debug_plot: Produce an informative plot.

    Returns:

    """
    from scipy.signal import find_peaks
    from lmfit.models import GaussianModel
    model = None
    params = None

    bin_widths = bins[1:] - bins[:-1]
    x = 0.5*(bins[1:] + bins[:-1])

    if baseline_kwargs is None:
        baseline_kwargs = {}
    baseline_method = baseline_method.lower()

    if baseline_method == 'root':
        baseline = calc_background(y, **baseline_kwargs)
    elif baseline_method == 'median':
        if 'window_width_kev' in baseline_kwargs:
            _window_width = baseline_kwargs['window_width_kev']
        else:
            _window_width = 30
        _window_width /= bin_widths[len(bin_widths)//2]
        baseline = rolling_median(values=y, window_width=_window_width)
    else:
        raise TypeError(f"Invalid `baseline_method`: '{baseline_method}'")

    y -= baseline

    centers_idx = list(sorted(get_bin_index(bins, peak_centers)))
    _center = int((centers_idx[0] + centers_idx[-1])/2)
    _bin_width = bin_widths[_center]

    if fit_window is None:
        if len(peak_centers) > 1:
            fit_window = 1.5*max([max(peak_centers) - min(peak_centers)])
            if fit_window*_bin_width < 10:
                fit_window = 10/_bin_width
        else:
            fit_window = 10/_bin_width

    _slice = slice(int(max([0, _center - fit_window//2])), int(min([len(y)-1, _center+fit_window//2])))
    y = y[_slice]
    x = x[_slice]

    density_sale = bin_widths[_slice]  # array to divide by bin widths.
    y /= density_sale  # make density

    peaks, peak_infos = find_peaks(unp.nominal_values(y), height=unp.std_devs(y), width=0)

    select_peak_ixs = np.argmin(np.array([np.abs(c - np.searchsorted(x, peak_centers)) for c in peaks]).T, axis=1)
    peak_widths = peak_infos['widths'][select_peak_ixs]*bin_widths[_center]
    amplitude_guesses = peak_infos['peak_heights'][select_peak_ixs]*peak_widths
    sigma_guesses = peak_widths/2.355

    for i, erg in enumerate(peak_centers):
        m = GaussianModel(prefix=f'_{i}')
        # erg = extrema_centers[np.argmin(np.abs(erg-extrema_centers))]
        if model is None:
            params = m.make_params()
            params[f'_{i}center'].set(value=erg)
            model = m
        else:
            model += m
            params.update(m.make_params())

        params[f'_{i}amplitude'].set(value=amplitude_guesses[i], min=0)
        params[f'_{i}center'].set(value=erg)
        params[f'_{i}sigma'].set(value=sigma_guesses[i])

    weights = unp.std_devs(y)
    weights = np.where(weights>0, weights, 1)
    weights = 1.0/weights

    fit_result = model.fit(data=unp.nominal_values(y), x=x, weights=weights, params=params)

    if debug_plot:
        ax = mpl_hist(bins[_slice.start: _slice.stop + 1], y*density_sale, label='Observed')
        _xs_upsampled = np.linspace(x[0], x[-1], 5*len(x))
        density_sale_upsampled = density_sale[np.searchsorted(x, _xs_upsampled)]
        model_ys = fit_result.eval(x=_xs_upsampled, params=fit_result.params)*density_sale_upsampled
        model_errors = fit_result.eval_uncertainty(x=_xs_upsampled, params=fit_result.params)*density_sale_upsampled
        ax.plot(_xs_upsampled, model_ys, label='Model')
        ax.fill_between(_xs_upsampled, model_ys-model_errors, model_ys+model_errors, alpha=0.5, label='Model error')
        ax.legend()
        ax.set_ylabel("Counts")
        ax.set_xlabel("Energy")
        for i in range(len(peak_centers)):
            amp = ufloat(fit_result.params[f'_{i}amplitude'].value, fit_result.params[f'_{i}amplitude'].stderr)
            _x = fit_result.params[f'_{i}center'].value
            _y = model_ys[np.searchsorted(_xs_upsampled, _x)]
            ax.text(_x, _y*1.05, f'N={amp:.2e}')

    return fit_result


def discrete_interpolated_median(list_, poisson_errors=False):
    """
    Median of a list of integers.
    Solves the problem of the traditional median being unaffected by values equal to the traditional median value.
    Args:
        list_: An iterable of integers
        poisson_errors: Return ufloat

    Returns:

    """

    values, freqs = np.unique(list_, return_counts=True)
    cumsum = np.cumsum(freqs)
    m_i = np.searchsorted(cumsum, cumsum[-1]/2)
    m = values[m_i]
    nl = np.sum(freqs[:m_i])
    ne = freqs[m_i]
    ng = np.sum(freqs[m_i + 1:])
    dx = values[m_i + 1] - values[m_i] if ng > nl else values[m_i] - values[m_i - 1]
    out = m + dx*(ng-nl)/(2*ne)
    if not poisson_errors:
        return out
    else:
        return ufloat(out, np.sqrt(cumsum[-1]))


def rolling_median(window_width, values):
    """
    Rolling median (in the y direction) over a uniform window. Window is clipped at the edges.
    Args:
        window_width: Size of independent arrays for median calculations.
        values: array of values

    Returns:

    """
    if isinstance(values[0], UFloat):
        _v = unp.nominal_values(values)
        rel_errors = unp.std_devs(values)/np.where(_v != 0, _v, 1)
        values = _v
    else:
        rel_errors = None

    window_width = int(window_width)
    n = min([window_width, len(values)])
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    window_indicies = (range(max([0, i - n // 2]), min([len(values) - 1, i + n // 2])) for i in range(len(values)))

    medians = np.array([np.median(values[idx]) for idx in window_indicies])
    if rel_errors is None:
        return medians
    else:
        return unp.uarray(medians, np.abs(rel_errors*medians))


def shade_plot(ax, window, color='blue', alpha=0.5, label=None):
    """
    Shades a region on the x-axis. Returns the handle for use in fig.legend([handle], ["shade"]) or similar.
    Args:
        ax:
        window: Tuple.List of len(2). i.e. [x1, x2]
        color: Color of shade.
        alpha: Transparency.
        label: Legend label.

    Returns:

    """
    if ax is plt:
        ax = plt.gca()
    _ylims = ax.get_ylim()
    y1, y2 = [ax.get_ylim()[0]] * 2, [ax.get_ylim()[1]] * 2
    handle = ax.fill_between(window, y1, y2, color=color, alpha=alpha, label=label)
    if 1 or ax.get_yscale() != 'log':
        ax.set_ylim(*_ylims)
    return handle


def calc_background(counts, num_iterations=20, clipping_window_order=2, smoothening_order=5, median_window=None):
    """

    Args:
        counts: Signal to operate on. Can be uarray
        num_iterations:
        clipping_window_order:
        smoothening_order:
        median_window: Only valid if ROOT isn't installed and thus background is calculated via a rolling median of
            width `median_window`

    Returns:

    """
    assert clipping_window_order in [2, 4, 6, 8]
    assert smoothening_order in [3, 5, 7, 9, 11, 13, 15]
    spec = _ROOT.TSpectrum()
    result = unp.nominal_values(counts)
    if isinstance(counts[0], UFloat):
        rel_errors = unp.std_devs(counts)/np.where(result != 0, result, 1)
    else:
        rel_errors = None

    if not root_exists:
        warnings.warn("No ROOT. Background estimation accomplished by rolling median")
        if median_window is None:
            median_window = int(len(counts)//10)
        result = rolling_median(median_window, result)
        if rel_errors is not None:
            return unp.uarray(result, result*rel_errors)
    else:
        assert median_window is None, '`median_window` is not needed when ROOT is installed. '

    clipping_window = getattr(_ROOT.TSpectrum, f'kBackOrder{clipping_window_order}')
    smoothening = getattr(_ROOT.TSpectrum, f'kBackSmoothing{smoothening_order}')
    spec.Background(result, len(result), num_iterations, _ROOT.TSpectrum.kBackDecreasingWindow,
                    clipping_window, _ROOT.kTRUE,
                    smoothening, _ROOT.kTRUE)
    if rel_errors is None:
        return result
    else:
        return unp.uarray(result, np.abs(rel_errors*result))


mpl_style = MPLStyle

g_A = 1/np.sqrt(2.0 * np.pi)


def gaussian(x, mu, sig):
    return g_A / sig * np.exp(-0.5 * np.power((x - mu) / sig, 2.0))


def norm2d_kernel(length_x, sigma_x, length_y=None, sigma_y=None):
    if sigma_y is None:
        sigma_y = sigma_x

    if length_y is None:
        length_y = length_x
    xs = norm(scale=sigma_x).pdf(np.linspace(-length_x//2, length_x//2, length_x))
    ys = norm(scale=sigma_y).pdf(np.linspace(-length_y//2, length_y//2, length_y))
    out = [[x*y for x in xs] for y in ys]
    out = out/np.sum(out)
    return out


def convolve_gauss2d(a, sigma_x, kernel_sigma_window: int = 8, sigma_y=None):
    if sigma_y is None:
        sigma_y = sigma_x

    kernel = norm2d_kernel(length_x=int(kernel_sigma_window*sigma_x), length_y=int(kernel_sigma_window*sigma_y),
                           sigma_x=sigma_x, sigma_y=sigma_y)
    plt.imshow(kernel)
    plt.figure()
    out = ndimage.convolve(a, kernel)
    # out = convolve2d(a, kernel, mode='same', boundary='symm')
    # out = np.fft.irfft2(np.fft.rfft2(a) * np.fft.rfft2(kernel, a.shape))
    return out


def fast_norm(x, sigma, mean, no_norm=False):  # faster than scipy.stat.norm
    return 1.0/np.sqrt(2 * np.pi)/sigma * np.e**(-0.5*((x - mean)/sigma)**2)


def convolve_gauss(a, sigma: Union[float, int], kernel_sigma_window: int = 6, mode='same', return_kernel=False,
                   reflect=False, est_errors=False):
    """
    Simple gaussian convolution.
    Args:
        a: The array to be convolved
        sigma: The width of the convolution (in units of array incicies)
        kernel_sigma_window: It's not efficient to make the window larger that a few sigma, so cut off at this value
        mode: See np.convolve
        reflect: Reflect array at boundaries to mitigate boundary effects.

    Returns:

    """
    if est_errors:
        out_unconvolved = a.copy()
    else:
        out_unconvolved = None

    if sigma == 0:
        if return_kernel:
            return a, np.ones(1)
        else:
            return a

    kernel_size = min([kernel_sigma_window * int(sigma), len(a)//2])
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel = np.e ** (-0.5 * (kernel_x / sigma) ** 2)
    kernel /= np.sum(kernel)

    if reflect:
        j = int(sigma * kernel_sigma_window)
        a = np.concatenate([a[j:0:-1], a, a[-1:-j-1:-1]])

    out = np.convolve(a, kernel, mode=mode)

    if reflect:
        out = out[j:-j]

    if est_errors:
        errs = np.sqrt(np.convolve((out - out_unconvolved) ** 2, kernel, mode='same'))
        out = (out, errs)

    if not return_kernel:
        return out
    else:
        return out, kernel


def convolve_unbinned(x, y, sigma, n_sigma_truncate=5, no_re_norm=False):
    """
    Convolve the signal (y) with a gauss kernel over a non-uniform grid (x).

    Args:
        x:
        y:
        sigma:
        n_sigma_truncate:
        no_re_norm:

    Returns:

    """
    if sigma == 0:
        return y

    r = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    f = -0.5/sigma**2

    if max(x[1:] - x[:-1]) < 2 * sigma or no_re_norm:
        re_norm = False
    else:
        re_norm = True

    def my_norm(centers, x):
        return r * np.e ** (f*(centers - x) ** 2)

    def new_val(index):
        i1 = np.searchsorted(x, x[index] - n_sigma_truncate * sigma)
        i2 = i1 + np.searchsorted(x[i1:], x[index] + n_sigma_truncate * sigma)

        i1 = max(0, i1)

        if i2 - i1 == 1:
            return y[index]

        x_truncated = x[i1: i2]
        y_truncated = y[i1: i2]

        norm = my_norm(x_truncated, x[index])

        # if x[-1] - x[0] < 4 * sigma:
        if re_norm:
            norm /= trapezoid(norm, x=x_truncated)  # Only needed in special cases where sigma < gap

        f = y_truncated * norm

        out: np.ndarray = trapezoid(f, x=x_truncated)
        return out

    return np.array([new_val(i) for i in range(len(x))])


def hist_gaus_convolve(bins, bin_values, sigma, is_density=False, check_norm=False):
    """
    Performs a gaussian convolution over hist (non-uniform bins allowed!).
    If bin_values refer to a density, then they are converted to counts and then back to density before returning.

    Args:
        bins: Histogram bins
        bin_values: Histogram bin values (len(bin_values) == len(bins) - 1)
        sigma: std in units the same as bins.
        is_density: Set to True if the histogram is a density (e.g. counts/MeV)
        check_norm: Verify that integral is unchanged, if not force equality (and suffer a performance penalty)

    Returns:

    """
    lefts = np.array(bins[:-1])
    rights = np.array(bins[1:])
    out = np.zeros_like(bin_values)

    if is_density:
        bwidths = bins[1:] - bins[:-1]
        bin_values = bin_values*bwidths  # convert to abs. bin values

    def kernal(index):
        xcenter = 0.5*(lefts[index] + rights[index])
        f = norm(loc=xcenter, scale=sigma).cdf
        gaus_areas = f(rights) - f(lefts)
        return bin_values[index] * gaus_areas

    for i in range(len(out)):

        out += kernal(i)

    if is_density:
        out /= bwidths  # convert back to density.

    if check_norm:
        if is_density:
            s = bwidths
        else:
            s = 1

        tot0 = sum(s * y)
        tot1 = sum(s * out)

        if not np.isclose(tot0, tot1):
            out *= tot0/tot1

    return out


def fill_between(x, y, yerr=None, ax=None, fig_kwargs=None, label=None, binsxQ=False, **mpl_kwargs):
    if fig_kwargs is None:
        fig_kwargs = {}

    if yerr is not None:
        assert not isinstance(y[0], UFloat)
        y = unp.uarray(y, yerr)

    if binsxQ:
        assert len(x) == len(y) + 1
        x = np.array(x)
        x = (x[1:] + x[:-1])/2

    if ax is None:
        plt.figure(**fig_kwargs)
        ax = plt.gca()
    y1 = unp.nominal_values(y) + unp.std_devs(y)
    y2 = unp.nominal_values(y) - unp.std_devs(y)
    alpha = mpl_kwargs.pop('alpha', None)
    ls = mpl_kwargs.pop('ls', None)
    fill_color = mpl_kwargs.pop('c', None)
    if fill_color is None:
        fill_color = mpl_kwargs.pop('color', None)
    if alpha is None:
        alpha = 0.4

    ax.fill_between(x, y1, y2, alpha=alpha, color=fill_color, **mpl_kwargs)
    ax.plot(x, unp.nominal_values(y), label=label, ls=ls, c=fill_color, **mpl_kwargs)

    if label is not None:
        ax.legend()
    return ax


class __TracePrints(object):

    def __init__(self):
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)

    def flush(self): pass


def trace_prints():
    """
    When there is a pesky print statement somewhere, use this to find it.
    Run this function at beginning of script
    """
    sys.stdout = __TracePrints()


def closest(sorted_dict: SortedDict, key):
    """Return closest key in `sorted_dict` to given `key`."""
    assert isinstance(sorted_dict, SortedDict)
    assert len(sorted_dict) > 0
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))


class TBrowser:
    def __init__(self):
        assert root_exists, 'Must install ROOT to use TBRowser'
        tb = _ROOT.TBrowser()
        while type(tb.GetBrowserImp()) is not _ROOT.TBrowserImp:
            _ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
        del tb


def cm_2_best_unit(list_or_number):
    """
    Find a good units for a number expressed in centimeters.
    e.g. 1.24E-4 cm -> 1.24 um
    Args:
        list_or_number:

    Returns: (Number in new units, new units (str))

    """
    if hasattr(list_or_number, '__iter__'):
        y = np.max(list_or_number)
        list_or_number = np.array(list_or_number)
    else:
        y = list_or_number
    unit_names = ["nm", "um", "mm", "cm", "m", "km"]
    orders = np.array([-7, -4, -1, 0, 2, 5])
    test_value = np.max(y)
    i = np.searchsorted(orders, np.log10(test_value), side='right') - 1
    i = max([0, i])
    units = unit_names[i]
    unit_conversion = 10. ** -orders[i]
    return list_or_number*unit_conversion, units


class FileManager:
    root_files: Dict[Path, _ROOT.TFile] = {}
    # todo: make gui for deleting files

    def _load_from_file(self):
        with open(self._save_path, 'rb') as f:
            for k, v in pickle.load(f).items():
                self.file_lookup_data[k] = v

    def __init__(self, path_to_root_dir: Union[str, Path] = None, recreate=False):
        """
        Creates a human friendly link between file and a dictionary of descriptive attributes that make it easy to
            access files created in a previous script.

        Note to self:
            When using this for complex workflow of lots of inter-related files, have all associations created in a
            single file! You will thank yourself later when de-bugging or modifying things because FileManager creation
            and management is all in one place as opposed to spread over several files.

        Args:
            path_to_root_dir: Path to the top directory. None for cwd.
            recreate: If you are loading an existing FileManager, then this must be False, else it will override the
                previous data.

        Examples:
            When a series of files (of any type) are created, they can be loaded later without the need to use a
             regex to lookup the file. e.g. many files are created from a series of MCNP simulations for which the
             energy and position of the particle source varies.

                cwd = Path(__file__).parent  # top directory of the simulations.
                f_man = FileManager(cwd, recreate=True)

                for pos in positions:
                    for energy in energies:

                        input_deck_name = f"{pos}_{energy}"

                         # The following will create a directory for the simulation where src energy=energy and source
                         #  position=pos. The function returns the path of the created input deck
                         #  (See JSB_tools.MCNP_helper.inputDeck)
                         f_path = i.write_inp_in_scope(input_deck_name) # this
                         outp_path = f_path.parent/'outp'  # this is the name of the outp file MCNP will create

                         # Even though `outp_path` doesn't exists yet, I can make a quick lookup with using FileManager
                         #  as follows:
                         f_man.add_path(outp_path, src_energy=energy, source_pos=position)
                # upon exiting the script, FileManager will save the association between the files and the key/values
                # in a pickle file named __file_lookup__.pickle in the top directory specified by `path_to_root_dir` in
                # the FileManager instantiation.

                In another script, say, that analyses the outp files, one could do the following (almost identical to
                the routine for initially creating the FIleManeger.
                cwd = Path(__file__).parent  # top directory of the simulations.
                f_man = FileManager(cwd, recreate=False)  # NOTE THAT IS False



        todo: Make it so files in the current/any sub dir are valid. The 'root_dir' is just the dir that containes the
            __file_info__.pickle.
            Make  "__file_info__.pickle" a hidden file
            This is a good place to use doctests



        """
        if path_to_root_dir is None:
            path_to_root_dir = Path.cwd()

        assert Path(path_to_root_dir).is_dir()
        self.root_directory = Path(path_to_root_dir)
        assert self.root_directory.parent.exists() and self.root_directory.parent.is_dir(),\
            f'Supplied root directory, "{self.root_directory}", is not a valid directory'
        if not self.root_directory.exists():
            print(f'Creating directory for FileContainer:\n{self.root_directory}')
            self.root_directory.mkdir()

        self._file_lookup_data: Dict[Path, dict] = {}

        # path to file that stores association information
        self._save_path = self.root_directory / "__file_lookup__.pickle"

        if not recreate:
            try:
                self._load_from_file()
                # with open(self._save_path, 'rb') as f:
                #     self.file_lookup_data: Dict[Path, Dict] = pickle.load(f)
            except (EOFError, FileNotFoundError) as e:
                recreate = True

        if recreate:
            self.file_lookup_data: Dict[Path, Dict] = {}
            self._save_path.unlink(missing_ok=True)

        register(self.__at_exit__)

    @property
    def file_lookup_data(self):
        return self._file_lookup_data

    @file_lookup_data.setter
    def file_lookup_data(self, val):
        self._file_lookup_data = val

    def remove_path(self, path):
        del self.file_lookup_data[path]

    def __save_lookup_data__(self):

        with open(self._save_path, 'wb') as f:
            pickle.dump(self.file_lookup_data, f)

        with open(self._save_path.parent/'__file_lookup__.txt', 'w') as f:
            for p, d in self.file_lookup_data.items():
                try:
                    _p = p.relative_to(self.root_directory)
                except ValueError:
                    _p = p
                f.write(f"{_p}: {d}\n")

    @staticmethod
    def auto_gen_path(attribs: Dict, root_path, extension='') -> Path:
        """
        Generate a simple (obscure) path, and save the attribs to a text file for reference.
        Args:
            attribs:
            root_path: Root path will be prepended to name. If None, then no path is prepended
            extension:

        Returns: Absolute path

        """
        existing_paths = list(Path(root_path).iterdir())
        root_path = Path(root_path)

        def get_new_path(i):
            out = (root_path/f"{i}").with_suffix(extension)
            return out

        i = 0
        while (new_path := get_new_path(i)) in existing_paths:
            i += 1

        return new_path

    @staticmethod
    def __verify_attribs__(attribs: Dict):  #why was this here again?
        pass
        # for kv in attribs.items():
        #     try:
        #         _ = {kv}
        #     except TypeError as e:
        #         assert False, f"Type error for the following value: {kv}\n" \
        #                       f"Make sure all attribs are hashable.\nThe error:\n" \
        #                       f"\t{e}"

    def add_path(self, rel_path_or_abs_path=None, missing_ok=False, overwrite_ok=False, **lookup_attributes) -> Path:
        """
        Add a path and lookup attributes to the list of saved files.
        Args:
            rel_path_or_abs_path:  Either a path relative to the self.root_directory, or an absolute path rel. to
                sys root
            missing_ok:  Raise error if missing?
            overwrite_ok: If True, can overwrite existing entries with either the same lookup_attributes or the same path.
            **lookup_attributes: kwargs used for easy lookup later.
        :return: Returns path to file

        Returns: Returns path to file.

        """
        FileManager.__verify_attribs__(lookup_attributes)
        assert len(lookup_attributes) != 0, \
            "If you're not going to provide any attributes then this tool is no for you."

        if rel_path_or_abs_path is None:
            rel_path_or_abs_path = self.auto_gen_path(lookup_attributes, self.root_directory)

        rel_path_or_abs_path = Path(rel_path_or_abs_path)

        if str(rel_path_or_abs_path.anchor) != '/':
            rel_path_or_abs_path = self.root_directory / Path(rel_path_or_abs_path)

        abs_path = rel_path_or_abs_path

        if not missing_ok:
            assert abs_path.exists(), f'The path, "{abs_path}", does not exist. Use missing_ok=True to bypass this error'

        remove_path = []
        for path, attribs in self.file_lookup_data.items():
            if lookup_attributes == attribs:
                if not overwrite_ok:
                    assert path == abs_path, \
                        f'FileManger requires a unique set of attributes for each file added.\n'\
                        f'"{lookup_attributes}" has already been used.\nPass arg overwrite=True to disable this error. '
                else:
                    remove_path.append(path)

        for path in remove_path:
            del self.file_lookup_data[path]

        self.file_lookup_data[abs_path] = lookup_attributes
        self.__save_lookup_data__()
        return rel_path_or_abs_path

    def find_path(self, missing_ok=False, **lookup_attributes) -> Union[None, Path]:
        """
        Return the path to a file who's keys/values **exactly** match `lookup_kwargs`. There can only be one.
        Args:
            missing_ok: whether to raise an error if file not found
            **lookup_attributes:

        Returns:

        """
        self._load_from_file()
        assert isinstance(missing_ok, int), f'Invalid `missing_ok` arg:\n\t"{missing_ok}"'
        for path, attribs in self.file_lookup_data.items():
            if lookup_attributes == attribs:
                return path
        available_files_string = '\n'.join(map(str, self.file_lookup_data.values()))
        if not missing_ok:
            raise FileNotFoundError(f"No file with the following matching keys/values:\n {lookup_attributes}\n"
                                    f"Currently linked files are:\n{available_files_string}")

    def find_paths(self, rtol=0.01, reject_attribs: List = None, **lookup_attributes) -> Dict[Path, dict]:
        """
        Find of all file paths for which the set of `lookup_attributes` is a subset of the files attributes.
        Return a dictionary who's keys are file paths, and values are the corresponding
            lookup attributes (all of them for the given file, not just the ones the user searched for)
        Args:
            rtol: If a config is found where a given attrib (a float) is close to within rtol, then consider values
                equal.
            reject_attribs: Any files that have one of these attribs are not included in search
            **lookup_attributes: key/values

        Examples:
            A FileManeger exists that links files containing the following attributes:
                f1 -> {"energy": 10, "position": 3, "particle": "neutron"}
                f2 -> {"energy": 12, "position": 3, "particle": "proton"}
                f2 -> {"energy": 19, "position": 3, "particle": "proton"}
                lookup_kwargs = (position=3) will return all file paths
                lookup_kwargs = (position=3, particle=proton) will return  file paths f2 and f3
                lookup_kwargs = (energy=10) will return  file path f1
            will match with

        Returns: Dictionary,  {Path1: file_attributes1, Path2: file_attributes2, ...}
        Todo: Find a way to make debugging not found easier.
        """
        self._load_from_file()
        lookup_kwargs = lookup_attributes  #.items()
        matches = {}

        if len(lookup_kwargs) == 0:
            return self.file_lookup_data

        def test_match(d_search, d_exists):
            for k in d_search.keys():
                if k not in d_exists:
                    return False
                exists_value = d_exists[k]
                search_value = d_search[k]
                if isinstance(exists_value, Number):
                    if isinstance(search_value, Number):
                        if not np.isclose(exists_value, search_value, rtol=rtol):
                            return False
                    else:
                        return False

                elif isinstance(exists_value, dict) and isinstance(search_value, dict):
                    if not test_match(exists_value, search_value):
                        return False
                else:
                    if exists_value != search_value:
                        return False
            return True

        for path, attribs in self.file_lookup_data.items():

            _continue_flag = False
            if reject_attribs is not None:

                for r in reject_attribs:
                    if r in attribs:
                        _continue_flag = True
                        break

            if _continue_flag:
                continue

            if test_match(lookup_kwargs, attribs):
                matches[path] = {k: v for k, v in attribs.items()}

        if len(matches) == 0:
            warnings.warn(f"No files found containing the following attribs: {lookup_attributes}")
        return matches

    def find_tree(self, tree_name="tree", **lookup_attributes) -> _ROOT.TTree:
        path = self.find_path(**lookup_attributes)
        if (path is None) or not path.exists():
            raise FileNotFoundError(f"Attempted to load ROOT tree on non-existent file. Attributes:{lookup_attributes}")
        return self.__load_tree_from_path__(path=path, tree_name=tree_name)

    @staticmethod
    def __load_tree_from_path__(path, tree_name='tree'):
        if not path.exists():
            raise FileNotFoundError(f"Attempted to load ROOT tree on non-existent file, '{path}'")
        f = _ROOT.TFile(str(path))
        FileManager.root_files[path] = f

        assert tree_name in map(lambda x:x.GetName(), f.GetListOfKeys()), \
            f'Invalid `tree_name`, "{tree_name}". ROOT file, "{path}", does not contain a key named "{tree_name}"'
        tree = f.Get(tree_name)
        return tree

    def find_trees(self, tree_name="tree", **lookup_attributes) -> Dict[_ROOT.TTree, dict]:
        """
        Same concept of find_paths, except the dictionary keys are ROOT trees.
        Args:
            tree_name:
            **lookup_attributes:

        Returns:

        """
        matches = {}
        for path, attribs in self.find_paths(**lookup_attributes).items():
            tree = self.__load_tree_from_path__(path=path, tree_name=tree_name)

            matches[tree] = attribs
        return matches

    def pickle_data(self, data, file_name=None, **lookup_attributes):
        """
        Save `data` to pickle file with the provided `lookup_attributes`
        Args:
            data: Data to be saved
            file_name: Name of pickle file. If not provided, then pick name automatically.
            **lookup_attributes:

        Returns:

        """
        if file_name is None:
            i = 0
            while file_name := (self.root_directory / f"file_{i}.pickle"):
                i += 1
                if file_name not in self.file_lookup_data:
                    break
        file_name = self.root_directory / file_name

        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
        self.add_path(file_name, **lookup_attributes)

    @property
    def all_files(self) -> Dict[Path, Dict[str, str]]:
        return {k: v for k, v in self.file_lookup_data.items()}

    def unpickle_data(self, **lookup_kwargs):
        """
        Unpickle and return the file who's keys/values match exactly
        Args:
            **lookup_kwargs:

        Returns:

        """
        path = self.find_path(**lookup_kwargs)

        with open(path, 'rb') as f:
            return pickle.load(f)

    def __at_exit__(self):
        pass
        # self.__save_lookup_data__()

    # def __del__(self):
    #     self.__at_exit__()

    @property
    def available_files(self):
        outs = []
        for path, keys_values in self.file_lookup_data.items():

            outs.append(f'{keys_values}   {path}  [{"exists" if path.exists() else "missing"}]')
        return '\n'.join(outs)

    def __repr__(self):
        outs = ['-'*80]
        for path, keys_values in self.file_lookup_data.items():
            outs.append(f"{keys_values}\n\t{path}\n")
        outs[-1] = outs[-1][:-1]
        outs.append(outs[0] + '\n')
        out = "\n".join(outs)
        out = f"Files in FileManeger at '{self._save_path}:'\n" + out
        return out
        # return "FileManager\nAvailable files:\nAttribs\tPaths\n{}".format(self.available_files)

    def clean(self):
        for path in self.file_lookup_data.keys():
            path = Path(path)
            path.unlink(missing_ok=True)
        self._save_path.unlink(missing_ok=True)

    def __iadd__(self, other: FileManager):
        for path, attribs in other.all_files.items():
            if path in self.file_lookup_data:
                assert attribs == self.file_lookup_data[path], f"Encountered two files with identical paths and " \
                                                                 "different attribs during merge. This is not allowed.\n" \
                                                                 f"{attribs}"
            self.file_lookup_data[path] = attribs
        return self


def interp1d_errors(x: Sequence[float], y: Sequence[UFloat], x_new: Sequence[float], order=2):
    """
    Extends interpolation to data with errors
    Args:
        x:
        y: uncertain array
        x_new: Values to interpolate.
        order:

    Returns: unp.uarray

    """
    orders = {0: 'zero', 1: 'linear', 2: 'quadratic', 3: 'cubic'}
    assert isinstance(order, int)
    assert order in orders, f'Invalid order, "{order}". Valid are:\n\t{list(orders.keys())}'
    order = orders[order]
    assert hasattr(y, '__iter__')
    assert hasattr(x, '__iter__')
    x = np.array(x)
    if all(x[np.argsort(x)] == x):
        assume_sorted = True
    else:
        assume_sorted = False

    assert hasattr(x_new, '__iter__')
    if not isinstance(y[0], UFloat):
        y = unp.uarray(y, np.zeros_like(y))
    if isinstance(x[0], UFloat,):
        raise NotImplementedError('Errors in x not implemented yet. Maybe someday')
    y_errors = unp.std_devs(y)
    y_nominal = unp.nominal_values(y)
    new_nominal_ys = interp1d(x, y_nominal, kind=order, copy=False, bounds_error=False, fill_value=(0, 0), assume_sorted=assume_sorted)(x_new)
    new_stddev_ys = interp1d(x, y_errors, kind=order, copy=False, bounds_error=False,  fill_value=(0, 0), assume_sorted=assume_sorted)(x_new)
    return unp.uarray(new_nominal_ys, new_stddev_ys)


if __name__ == '__main__':
    # import numpy as np
    # from matplotlib import pyplot as plt
    #
    # x = np.linspace(0, np.pi * 2, 1000)
    #
    # f = TabPlot()
    # axs = []
    # for i in range(1, 5):
    #     ax = f.new_ax(i)
    #     axs.append(ax)
    #
    #     ax.plot(x, np.sin(x * i), label='label 1')
    #     ax.plot(x, np.sin(x * i) ** 2, label='label 2')
    #     ax.plot(x, np.sin(x * i) ** 3, label=f'label 3')
    #     ax.legend()
    #     ax.set_title(str(i))
    #
    #     if i == 4:
    #         bins = np.concatenate([x, [x[-1] + x[1] - x[0]]])
    #         y = np.sin(x * i) ** 2
    #         mpl_hist(bins, y, yerr=y, ax=ax)
    plt.show()
    # d = {1: {1: {2: [3], 3:5}, 3: {1: [2,1,4]}}}
    # for h in flatten_dict_values(d):
    #     print(h)


