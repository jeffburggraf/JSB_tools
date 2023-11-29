from __future__ import annotations
import warnings
from typing import List, Dict
import numpy as np
from typing import Union, Sequence
import sys
from matplotlib.widgets import Button
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


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
    class TabOverflowError(OverflowError):
        pass

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

    def _key_press(self, event):
        if event.key == 'y':
            ax_group = self.plt_axs[self.index]
            for ax in ax_group:
                lines = [l for l in ax.lines]
                set_ylims(ax, lines)

    def __len__(self):
        return len(self.button_labels)

    @property
    def axes_flat(self) -> List[Axes]:
        return flatten(self.plt_axs)

    def __init__(self, figsize=(10, 8), universal_zoom=True, *fig_args, **fig_kwargs):
        """

        Args:
            figsize:
            universal_zoom: Zoom on one axis will be applied to all other axis (with the same subplots nrow/ncols).
            *fig_args:
            **fig_kwargs:
        """
        TabPlot._instnaces.append(self)
        plt.rcParams['keymap.forward'] = []
        plt.rcParams['keymap.back'] = []

        self.universal_zoom = universal_zoom

        self.fig = plt.figure(figsize=figsize, *fig_args, **fig_kwargs)

        self.fig.canvas.mpl_connect('key_press_event', self._key_press)

        self._vis_flag = True

        self.plt_axs = []
        self._axis_shapes = []

        self.global_axs = []

        self.suptitles = []

        self.button_funcs = []
        self.button_axs = [[]]  # list of lists for each row of buttons
        self.button_labels = []
        self.buttons = []

        self.index = 0

        self.max_buttons_reached = False

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    @property
    def button_len(self):
        return sum(map(len, self.button_labels))

    def on_press(self, event):
        sys.stdout.flush()
        if event.key in ['right', 'up']:
            index = min(len(self.button_funcs) - 1, self.index + 1)
            self.button_funcs[index](event)
        elif event.key in ['down', 'left']:
            index = max(0, self.index - 1)
            self.button_funcs[index](event)

    def add_button_callback(self, func, index=None, self_kwargs: List[str] = None):
        """
        Add a callback that will be called () when button corresponding to index is clicked.
            The callback function will be passed
        Args:
            func:
            index: If None, use last button added. This value is always passed to callback funciton
            self_kwargs: Attribute names of TabPlot (self), as they are at the time of THIS function call,
                that will be passed to callback as kwargs.
        Returns:

        """
        if index is None:
            index = len(self) - 1

        if self_kwargs is not None:
            kwargs = {name: getattr(self, name) for name in self_kwargs}
        else:
            kwargs = {}

        def new_func(event):
            return func(index, **kwargs)

        self.buttons[index].on_clicked(new_func)

    def get_button_func(self):
        """
        Callback function on button click. Generates and returns the func for the last call to self.new_ax(...)

        Returns:
                Callback func

        """
        index = len(self.plt_axs) - 1

        current_axes_group = self.plt_axs[index]

        def set_vis(event=None):
            for shape, axs_group in zip(self._axis_shapes, self.plt_axs):
                same_shape = False
                if shape == self._axis_shapes[index]:
                    same_shape = True

                if axs_group is self.plt_axs[index]:
                    # for ax in axs_group:
                    [ax.set_visible(1) for ax in axs_group]
                    [ax.set_navigate(True) for ax in axs_group]
                else:
                    [ax.set_visible(0) for ax in axs_group]

                    if self.universal_zoom and same_shape:
                        [ax.set_navigate(True) for ax in axs_group]
                    else:
                        [ax.set_navigate(False) for ax in axs_group]

            if self.suptitles[index] is not None:
                title = self.suptitles[index]
            else:
                title = self.button_labels[index]

            self.fig.suptitle(title)

            for ax in self.global_axs:
                if ax.data_coordsx:
                    ax.set_xlim(current_axes_group[0].get_xlim())
                else:
                    ax.set_xlim((0, 1))

                if ax.data_coordsy:
                    ax.set_ylim(current_axes_group[0].get_ylim())
                else:
                    ax.set_ylim((0, 1))

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

    @property
    def char_units(self):
        s = 'abcdefghijklmnopqrstuvwxyz'
        _t = self.fig.text(0, 0, s)
        points = self.fig.transFigure.inverted().transform(_t.get_tightbbox())
        width = (points[1, 0] - points[0, 0])/len(s)
        height = points[1][1] - points[0, 1]
        _t.remove()
        del _t
        return width, height

    def add_global_ax(self, data_coordsx=True, data_coordsy=True) -> Axes:
        """
        Return an axes that will be persistent over all tabs. Limits will adjust to each axis as necessary.

        Args:
            data_coordsx: If True, X-axis will be in same units as button axes.
                If False, fractional units are used,  i.e. values betweeen 0, and 1

            data_coordsy: Same as above, but for y-axis.

        Returns: Axes

        """
        if not all([len(g) == 1 for g in self.plt_axs]):
            raise NotImplementedError("Cannot add global axis when multiple subplots are present.")

        ax_new = self.fig.add_subplot()
        ax_new.data_coordsx = data_coordsx
        ax_new.data_coordsy = data_coordsy

        ax_new.patch.set_alpha(0)

        ax_new.axes.get_xaxis().set_ticks([])
        ax_new.axes.get_yaxis().set_ticks([])

        self.global_axs.append(ax_new)

        return ax_new

    def new_ax(self, button_label=None, nrows=1, ncols=1, sharex=False, sharey=False, suptitle=None, figsize=None,
               subplot_kw=None, gridspec_kw=None, height_ratios=None, width_ratios=None, *args, **kwargs) -> Union[np.ndarray, Axes]:
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
        if button_label is None:
            button_label = f'{len(self.button_labels)}'

        if self.max_buttons_reached:
            raise TabPlot.TabOverflowError("Too many buttons! Create a new TapPlot.")

        if subplot_kw is None:
            subplot_kw = {}

        if gridspec_kw is None:
            gridspec_kw = {}

        gridspec_kw.setdefault('height_ratios', height_ratios)
        gridspec_kw.setdefault('width_ratios', width_ratios)

        if figsize is not None:
            raise ValueError("figsize argument is applied in the TabPlot constructor")

        self.suptitles.append(suptitle)

        button_label = f"{button_label: <4}"
        self.button_labels.append(button_label)

        axs = self.fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, subplot_kw=subplot_kw,
                                gridspec_kw=gridspec_kw, *args, **kwargs)

        if not hasattr(axs, '__iter__'):
            axs = np.array([axs])

        axs_flat = axs.flatten()

        self.index = len(self) - 1
        self.plt_axs.append(axs_flat)
        self._axis_shapes.append(axs.shape)

        try:
            prev_x = self.button_axs[-1][-1].get_position().x1
        except IndexError:
            prev_x = 0

        char_w, char_h = self.char_units
        button_x = prev_x + char_w
        _text = self.fig.text(0, 0, button_label)
        _text_points = self.fig.transFigure.inverted().transform(_text.get_tightbbox())
        _text.remove()
        del _text
        button_width = _text_points[1, 0] + 2 * char_w
        button_height = 3 * char_h

        if button_x + button_width > 0.95:
            if len(self.button_axs[-1]) == 0:  # first button of row is too long!
                assert False, f"Button label too long!, '{button_label}'"
            button_x = char_w
            for row_index, b_axs in enumerate(self.button_axs):
                for b_ax in b_axs:
                    old_pos = list(b_ax.get_position().bounds)
                    new_y = button_height*1.5*(len(self.button_axs) - row_index) + char_h/2
                    old_pos[1] = new_y
                    b_ax.set_position(old_pos)

            self.button_axs.append([])

        if not plt.gcf() is self.fig:
            plt.figure(self.fig.number)

        # [left, bottom, width, height]
        button_ax = plt.axes([button_x, char_h/2, button_width, button_height], projection='rectilinear')
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
