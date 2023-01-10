"""
A messy collection of stuff that nuclear physicists like like to use.

todo: move some of these imports into functions to speed up loading of this module
"""
from __future__ import annotations
import warnings
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    go = make_subplots = ModuleNotFoundError
from typing import List, Dict
import numpy as np
from itertools import islice
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
from uncertainties import UFloat, ufloat
from matplotlib.cm import ScalarMappable
import time
import sys
from scipy.stats import norm
from matplotlib import cm
import traceback
from uncertainties import UFloat
from scipy import ndimage
from matplotlib.widgets import Button
from uncertainties.umath import sqrt as usqrt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Colormap
from scipy.integrate import trapezoid
from numbers import Number
try:
    import ROOT
    root_exists = True
except ModuleNotFoundError:
    root_exists = False

cwd = Path(__file__).parent

style_path = cwd/'mpl_style.txt'

markers = ['p', 'X', 'D', 'o', 's', 'P', '^', '*']


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


def rebin(oldbins, oldy, newbins, force_equal_norm=False, kind='linear'):
    """
    Rebin a histogram.
    Do not use for density histograms, use interpolation instead.

    Args:
        oldbins:
        oldy:
        newbins:
        force_equal_norm: If True, force integrals to be the same.
        kind:

    Returns: new histogram bin values.

    Example:
        np.random.seed(0)
        xmax = 120
        x1 = np.arange(0, xmax, 1, dtype=float)
        x1 += np.linspace(0, 0.75, len(x1))

        x2 = np.linspace(0, xmax, 100)

        y1 = np.zeros(len(x1) - 1)

        points = np.concatenate(
            [8*np.random.randn(10000) + 55,
             4*np.random.randn(4000) + 20,
             np.random.randn(4500) + 80,
             0.3*np.random.randn(4500) + 5,
             ])

        for i in np.searchsorted(x1, points, side='right') - 1:
            if i < 0:
                continue
            y1[i] += 1

        ax = mpl_hist(x1, y1, label='Original bins')
        s, y2 = rebin(x1, y1, x2)

        mpl_hist(x2, y2, ax=ax, label='Rebinned')

        _x = np.linspace(0, xmax-3, 1000)

        ax.plot(_x, s.evaluate(_x), label='Interp')

        ax.legend()

        plt.show()



    """
    # xmids = 0.5*(oldbins[:-1] + oldbins[1:])
    # if kind != 'linear':
    #     x = np.hstack((oldbins[0], xmids, oldbins[-1]))
    #     y = np.hstack((oldy[0], oldy, oldy[-1]))
    # else:
    #     x = xmids
    #     y = oldy

    # todo: find a way to do higher order without occilations.
    # cumy = cumulative_trapezoid(y, x, initial=0)
    if isinstance(oldy[0], UFloat):
        yerr = unp.std_devs(oldy)
        oldy = unp.nominal_values(oldy)
    else:
        yerr = None
        oldy = np.array(oldy)

    oldbins = np.array(oldbins)
    newbins = np.array(newbins)

    cumy = np.cumsum(oldy)
    interp = interp1d(oldbins[1:], cumy, kind=kind, bounds_error=False, assume_sorted=True, fill_value=(0, cumy[-1]))

    newy = np.array([float(interp(newbins[i + 1])) - float(interp(newbins[i])) for i in range(len(newbins) - 1)])

    if yerr is not None:
        old_bwidths = oldbins[1:] - oldbins[:-1]
        new_bwidths = newbins[1:] - newbins[:-1]
        err_density = yerr**2/old_bwidths
        oldxmids = 0.5*(oldbins[:-1] + oldbins[1:])
        newxmids = 0.5*(newbins[:-1] + newbins[1:])

        err_interp = interp1d(oldxmids, err_density, bounds_error=False, assume_sorted=True, fill_value=(0, 0))
        newyerr = np.sqrt(new_bwidths*err_interp(newxmids))
        newy = unp.uarray(newy, newyerr)

    if force_equal_norm:
        newy *= sum(oldy)/sum(newy)

    return newy


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

    cbar = plt.colorbar(sm)

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


class TabPlot:
    """
    Make a mpl Figure with a button to switch between plots.

    Examples:
        import numpy as np
        from matplotlib import pyplot as plt

        x = np.linspace(0, np.pi * 2, 1000)

        f = TabPlot()

        for i in range(1, 20):
            ax = f.new_ax(i)
            ax.plot(x, np.sin(x * i), label='label 1')
            ax.plot(x, np.sin(x * i) ** 2, label='label 2')
            ax.plot(x, np.sin(x * i) ** 3, label=f'label 3')
            ax.legend()
            ax.set_title(str(i))

        plt.show()

    """
    _instnaces = []

    old_show = plt.show

    def new_show(*args, **wkargs):
        for self in TabPlot._instnaces:
            try:
                self.button_funcs[0]()
            except IndexError:
                pass
        return TabPlot.old_show(*args, **wkargs)

    plt.show = new_show  # link calls to plt.show to a function that initiates all TabPlots

    def __init__(self, figsize=(10, 8), *fig_args, **fig_kwargs):
        TabPlot._instnaces.append(self)

        self.fig = plt.figure(figsize=figsize, *fig_args, **fig_kwargs)

        self._vis_flag = True

        self.plt_axs = []

        self.suptitles = []

        self.button_funcs = []
        self.button_axs = [[]]  # list of lists for each row of buttons
        self.button_labels = []
        self.buttons = []

        self.index = 0

        self.max_buttons_reached = False

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def __len__(self):
        return len(self.button_labels)

    @property
    def button_len(self):
        return sum(map(len, self.button_labels))

    def on_press(self, event):
        sys.stdout.flush()
        if event.key == 'right':
            self.button_funcs[(self.index + 1)%len(self.button_funcs)](event)
        elif event.key == 'left':
            self.button_funcs[(self.index - 1)%len(self.button_funcs)](event)

        elif event.key == 'down':
            di = len(self.button_axs[0])
            self.button_funcs[(self.index - di) % len(self.button_funcs)](event)

        elif event.key == 'up':
            di = len(self.button_axs[0])
            self.button_funcs[(self.index + di) % len(self.button_funcs)](event)

    def get_button_func(self):
        """
        Callback function on button click. Generates and returns the func for the last call to self.new_ax(...)

        Returns:
                Callback func

        """
        index = len(self.plt_axs) - 1

        def set_vis(event=None):
            for axs_group in self.plt_axs:
                if axs_group is self.plt_axs[index]:
                    [ax.set_visible(1) for ax in axs_group]
                else:
                    [ax.set_visible(0) for ax in axs_group]

            if self.suptitles[index] is not None:
                title = self.suptitles[index]
            else:
                title = self.button_labels[index]
            self.fig.suptitle(title)

            self.index = index  # save current "place" in list of plots (e.g. self.button_funcs).
            # button = self.buttons[index]

            self.fig.canvas.draw_idle()

        return set_vis

    def add_aux_axis(self, ax):
        """
        Add an axis, `ax`, to the list of axis that will switch on/off with the last axis returned by self.
        Args:
            ax:

        Returns:

        """
        self.plt_axs[-1] = np.concatenate([self.plt_axs[-1], [ax]])

        if len(self.button_labels) == 1:
            ax.set_visible(1)
        else:
            ax.set_visible(0)
            # [ax.set_visible(0) for ax in axs_flat]

    def _set_new_twinx(self):
        # old_f = Axes.twinx
        axes_group_index = len(self.plt_axs) - 1

        def get_f(ax):
            def new_f(*args, **kwargs):
                twin_ax = Axes.twinx(ax, *args, **kwargs)
                axis_group = self.plt_axs[axes_group_index]
                self.plt_axs[axes_group_index] = np.concatenate([axis_group, [twin_ax]])

                return twin_ax

            return new_f

        for ax in self.plt_axs[axes_group_index]:
            ax.twinx = get_f(ax)

    def new_ax(self, button_label, nrows=1, ncols=1, sharex=False, sharey=False, suptitle=None, subplot_kw=None,
               *args, **kwargs) -> Union[np.ndarray, Axes]:
        """
        Raises OverflowError if too many axes have been created.
        Args:
            button_label:
            nrows:
            ncols:
            sharex:
            sharey:
            suptitle:
            subplot_kw: kwargs dict to be passed to subplots, e.g. like doing fig.add_subplot(**subplot_kw)
            *args:
            **kwargs:

        Returns: Single Axes instance if nrows == ncols == 1, else an array of Axes likewise to plt.subplots(...) .

        """
        if self.max_buttons_reached:
            raise OverflowError("Too many buttons! Create a new TapPlot.")

        if subplot_kw is None:
            subplot_kw = {}

        self.suptitles.append(suptitle)

        # subplot_kw=dict(projection='polar')
        button_label = f"{button_label: <4}"
        self.button_labels.append(button_label)

        axs = self.fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, subplot_kw=subplot_kw,
                                *args, **kwargs)

        if not hasattr(axs, '__iter__'):
            axs = np.array([axs])

        axs_flat = axs.flatten()

        self.plt_axs.append(axs_flat)

        b_unit = 0.1/5  # width of one character

        try:
            prev_x = self.button_axs[-1][-1].get_position().x1
        except IndexError:
            prev_x = 0

        button_x = prev_x + b_unit
        button_width = len(button_label) * b_unit

        if button_x + button_width > 0.95:
            if len(self.button_axs[-1]) == 0:  # first button of row is too long!
                assert False, f"Button label too long!, '{button_label}'"
            button_x = b_unit
            for row_index, b_axs in enumerate(self.button_axs):
                for b_ax in b_axs:
                    old_pos = list(b_ax.get_position().bounds)
                    new_y = 0.1*(len(self.button_axs) - row_index)
                    old_pos[1] = new_y
                    b_ax.set_position(old_pos)

            self.button_axs.append([])

        if not plt.gcf() is self.fig:
            plt.figure(self.fig.number)

        # [left, bottom, width, height]
        button_ax = plt.axes([button_x, 0, button_width, 0.075], projection='rectilinear')
        self.button_axs[-1].append(button_ax)

        button = Button(button_ax, button_label)
        self.buttons.append(button)

        self.button_funcs.append(self.get_button_func())
        button.on_clicked(self.button_funcs[-1])

        new_fig_bottom = self.button_axs[0][0].get_position().y1 + 0.1
        fig_top = self.fig.axes[0].get_position().y1

        self.fig.subplots_adjust(bottom=new_fig_bottom)

        if new_fig_bottom >= 0.75*fig_top:
            self.max_buttons_reached = True

        if self._vis_flag:
            self._vis_flag = False
            if suptitle is not None:
                self.fig.suptitle(suptitle)
            else:
                self.fig.suptitle(button_label)
        else:
            [ax.set_visible(0) for ax in axs_flat]

        self._set_new_twinx()
        return axs if len(axs_flat) > 1 else axs[0]

    def legend(self):
        for axs in self.plt_axs:
            for ax in axs:
                ax.legend()


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
    spec = ROOT.TSpectrum()
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

    clipping_window = getattr(ROOT.TSpectrum, f'kBackOrder{clipping_window_order}')
    smoothening = getattr(ROOT.TSpectrum, f'kBackSmoothing{smoothening_order}')
    spec.Background(result, len(result), num_iterations, ROOT.TSpectrum.kBackDecreasingWindow,
                    clipping_window, ROOT.kTRUE,
                    smoothening, ROOT.kTRUE)
    if rel_errors is None:
        return result
    else:
        return unp.uarray(result, np.abs(rel_errors*result))


class MPLStyle:
    fig_size = (15, 10)

    @staticmethod
    def set_bold_axes_labels():
        def new_func(axis):
            orig_func = getattr(Axes, f'set_{axis}label')

            def f(self, *args, **kwargs):
                args = list(args)
                args[0] = fr"\textbf{{{args[0]}}}"
                return orig_func(self, *args, **kwargs)

            return f

        for x in ['x', 'y']:
            setattr(Axes, f'set_{x}label', new_func(x))

    def __init__(self, minor_xticks=True, minor_yticks=True, bold_ticklabels=True, bold_axes_labels=True,
                 usetex=True, fontscale=None, fig_size=(15,8)):
        """

            Args:
                usetex:
                fontscale: 1.5 for half-width latex document.

            Returns:

            """
        plt.style.use(style_path)

        if bold_ticklabels:
            plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
        else:
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        plt.rcParams['xtick.minor.visible'] = minor_xticks
        plt.rcParams['ytick.minor.visible'] = minor_yticks

        plt.rcParams['figure.figsize'] = fig_size

        if bold_axes_labels and usetex:
            self.set_bold_axes_labels()

        if not usetex:
            plt.rcParams.update({
                "text.usetex": False, })
        else:
            pass

        if fontscale is not None:
            for k in ['font.size', 'ytick.labelsize', 'xtick.labelsize', 'axes.labelsize', 'legend.fontsize',
                      'legend.title_fontsize']:
                plt.rcParams.update({k: plt.rcParams[k] * fontscale})


mpl_style = MPLStyle


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


def convolve_gauss(a, sigma: Union[float, int], kernel_sigma_window: int = 6, mode='same'):
    """
    Simple gaussian convolution.
    Args:
        a: The array to be convolved
        sigma: The width of the convolution (in units of array incicies)
        kernel_sigma_window: It's not efficient to make the window larger that a few sigma, so cut off at this value
        mode: See np.convolve

    Returns:

    """
    sigma = int(sigma)
    if sigma == 0:
        return a
    kernel_size = min([kernel_sigma_window * sigma, len(a)//2])
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel = norm(loc=0, scale=sigma).pdf(kernel_x)
    kernel /= np.sum(kernel)
    return np.convolve(a, kernel, mode=mode)


def convolve_unbinned(x, y, sigma, n_sigma_truncate=5, thresh=0.0001):
    out = np.zeros_like(y)

    # def kernal(index):  # un-optimized
    #     weights = norm(loc=x[index], scale=sigma).pdf(x)
    #     weights /= sum(weights)
    #     return y[index] * weights
    # A = 1/(sigma*np.sqrt(np.pi))
    min_y = max(y)*thresh

    def my_norm(center, xs):
        return np.e**(-0.5*((center - xs)/sigma)**2)

    def kernal(index):  # optimized
        i1 = np.searchsorted(x, x[index] - n_sigma_truncate*sigma)
        i2 = i1 + np.searchsorted(x[i1:], x[index] + n_sigma_truncate*sigma)

        x_truncated = x[i1: i2]
        # weights = norm(loc=x[index], scale=sigma).pdf(x_truncated)
        weights = my_norm(x[index], x_truncated)
        weights /= sum(weights)
        return (i1, i2), y[index] * weights

    for i in range(len(out)):
        # if y[i] > min_y:
            (i1, i2), w = kernal(i)  # optimized
            out[i1: i2] += w   # optimized

        # w = kernal(i)  # un-optimized
        # out += w  # un-optimized
    return out


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


__default_stats_kwargs = {'loc': (0.7, 0.8)}


def mpl_hist(bin_edges, y, yerr=None, ax=None, label=None, fig_kwargs=None, title=None, poisson_errors=False,
             return_handle=False, stats_box=False, stats_kwargs=None, elinewidth=1.1, **mpl_kwargs):
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

    handle1[0].set_marker(None)

    if "c" in mpl_kwargs:
        pass
    elif 'color' in mpl_kwargs:
        pass
    else:  # color was from color cycle. Fetch from handle.
        mpl_kwargs['color'] = handle1[0].get_color()
    mpl_kwargs.pop('ls', None)
    mpl_kwargs.pop('linestyle', None)
    handle2 = ax.errorbar(bin_centers, y, yerr, ls="None", capsize=capsize, **mpl_kwargs)  # draw error_bars and markers.

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


class ProgressReport:
    def __init__(self, i_final, sec_per_print=2, i_init=0):
        self.__i_final__ = i_final
        self.__i_init__ = i_init
        self.__sec_per_print__ = sec_per_print
        # self.__i_current__ = i_init
        self.__next_print_time__ = time.time() + sec_per_print
        self.__init_time__ = time.time()

        self.events_log = [self.__i_init__]
        self.times_log = [self.__init_time__]
        # self.__rolling_average__ = []

    @property
    def elapsed_time(self):
        return time.time()-self.__init_time__

    def __report__(self, t_now, i, added_msg):

        self.events_log.append(i)
        self.times_log.append(t_now)

        evt_per_sec = (self.events_log[-1] - self.events_log[0])/(self.times_log[-1] - self.times_log[0])

        if len(self.times_log) >= max(2, int(6/self.__sec_per_print__)):
            self.dts_log = self.times_log[:5]
            self.events_log = self.events_log[:5]

        evt_remaining = self.__i_final__ - i
        sec_remaining = evt_remaining/evt_per_sec
        sec_per_day = 60**2*24
        days = sec_remaining//sec_per_day
        hours = (sec_remaining % sec_per_day)//60**2
        minutes = (sec_remaining % 60**2)//60
        sec = int(sec_remaining % 60)
        msg = " {0} seconds".format(sec)
        if minutes:
            msg = " {0} minute{1},".format(minutes, 's' if minutes > 1 else '') + msg
        if hours:
            msg = " {0} hour{1},".format(hours, 's' if hours > 1 else '') + msg
        if days:
            msg = "{0} day{1},".format(days, 's' if days > 1 else '') + msg
        print(f"{added_msg}... {msg} remaining {100*i/self.__i_final__:.2f}% complete")

    def log(self, i, msg=""):
        t_now = time.time()
        if t_now > self.__next_print_time__:
            self.__report__(t_now, i, msg)
            self.__next_print_time__ += self.__sec_per_print__
            return True
        return False


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
        tb = ROOT.TBrowser()
        while type(tb.GetBrowserImp()) is not ROOT.TBrowserImp:
            ROOT.gSystem.ProcessEvents()
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


def ROOT_loop():
    try:
        import time
        while True:
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
    except ModuleNotFoundError:
        warnings.warn('ROOT not installed. Cannot run ROOT_loop')


class FileManager:
    root_files: Dict[Path, ROOT.TFile] = {}
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

    def find_tree(self, tree_name="tree", **lookup_attributes) -> ROOT.TTree:
        path = self.find_path(**lookup_attributes)
        if (path is None) or not path.exists():
            raise FileNotFoundError(f"Attempted to load ROOT tree on non-existent file. Attributes:{lookup_attributes}")
        return self.__load_tree_from_path__(path=path, tree_name=tree_name)

    @staticmethod
    def __load_tree_from_path__(path, tree_name='tree'):
        if not path.exists():
            raise FileNotFoundError(f"Attempted to load ROOT tree on non-existent file, '{path}'")
        f = ROOT.TFile(str(path))
        FileManager.root_files[path] = f

        assert tree_name in map(lambda x:x.GetName(), f.GetListOfKeys()), \
            f'Invalid `tree_name`, "{tree_name}". ROOT file, "{path}", does not contain a key named "{tree_name}"'
        tree = f.Get(tree_name)
        return tree

    def find_trees(self, tree_name="tree", **lookup_attributes) -> Dict[ROOT.TTree, dict]:
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

    d = {1: {1: {2: [3], 3:5}, 3: {1: [2,1,4]}}}
    for h in flatten_dict_values(d):
        print(h)


