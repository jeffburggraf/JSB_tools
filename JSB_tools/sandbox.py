from __future__ import annotations
import numpy as np
from JSB_tools.list_reader import MaestroListFile
from JSB_tools import Nuclide
import plotly.graph_objects as go
from uncertainties import UFloat
from uncertainties import unumpy as unp
from JSB_tools import rolling_median



l1 = MaestroListFile.from_pickle('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot134.Lis')
l2 = MaestroListFile.from_pickle('/Users/burggraf1/PycharmProjects/IACExperiment/exp_data/friday/shot132.Lis')


def plotly(self: MaestroListFile, erg_min=None, erg_max=None, eff_corr=True, time_bins=None, time_bin_width=15,
           time_step: int = 5, time_min=0, time_max=None, percent_of_max=False, remove_baseline=False, title=None,
           interactive_plot: InteractivePlot = None, nominal_values=True, leg_label=None, dont_plot=False):
    """

    Args:
        self:
        erg_min:
        erg_max:
        eff_corr:
        time_bins:
        time_bin_width:
        time_step:
        time_min:
        time_max:
        percent_of_max:
        remove_baseline:
        title:
        interactive_plot:
        nominal_values:
        leg_label:
        dont_plot:

    Returns:
        tuple(InteractivePlot, (args))
            where InteractivePlot is obvious and (args) are the arguments send to this function that can be used to plot
             an overlay with the same attributes.

    """
    if time_bins is not None:
        time_bins = np.array(time_bins, copy=False)
        assert time_bins.ndim == 2 and time_bins.shape[1] == 2
        # assert None is time_bin_width is time_step is time_min is time_max,
    else:
        if time_max is None:
            time_max = self.times[-1]
        assert None not in [time_min, time_bin_width, time_step]
        time_max = time_min + time_step * ((time_max - time_min) // time_step)
        time_centers = np.arange(0, time_max + time_step, time_step)
        lbins = time_centers - time_bin_width / 2
        rbins = time_centers + time_bin_width / 2
        time_bins = list(zip(np.where(lbins > 0, lbins, 0), np.where(rbins <= time_max, rbins, time_max)))

    time_bins = np.array(time_bins, copy=False)


    ys = []
    labels4frames = []
    for (b0, b1) in time_bins:
        _y, bin_edges = self.get_erg_spectrum(erg_min, erg_max, b0, b1, eff_corr=eff_corr,
                                              nominal_values=nominal_values,
                                              return_bin_edges=True)
        _y /= (b1 - b0)

        if remove_baseline:
            _y -= rolling_median(45, _y)
        ys.append(_y)
        labels4frames.append(f"{b0} <= t < {b1}")

    if interactive_plot is None:
        interactive_plot = InteractivePlot(labels4frames, "time ")

    x = (bin_edges[1:] + bin_edges[:-1])/2

    if leg_label is None:
        leg_label = self.file_name

    color = interactive_plot.add_ys(x, ys, leg_label=leg_label, line_type='hist', return_color=True)

    tot_time = (time_bins[-1][-1] - time_bins[0][0])
    if nominal_values:
        tot_y = np.mean(ys, axis=0)
        error_y = None
    else:
        tot = np.mean(ys, axis=0)
        tot_y = unp.nominal_values(tot)
        error_y = unp.std_devs(tot)

    interactive_plot.add_persistent(x, tot_y, yerr=error_y, leg_label='All time', opacity=0.5)

    if not dont_plot:
        interactive_plot.plot()

    return {"interactive_plot": interactive_plot, "remove_baseline": remove_baseline, "time_bins": time_bins,
            "erg_max": erg_max, "erg_min": erg_min,
            "nominal_values": nominal_values}


class InteractivePlot:
    color_cycle = ['blue', 'red', 'green', 'black', 'gray']

    def __init__(self, frame_labels=None, slider_prefix=''):
        self.ys = []
        self.yerrs = []

        self.persistent = []

        self.xs = []

        self.step_titles = []
        self.frame_labels = frame_labels
        self.leg_labels = []

        self.fig = go.Figure()
        self.init_slider = None

        self.colors = []
        self.line_types = []

        self.slider_prefix = slider_prefix

    def add_persistent(self, x, y, yerr=None, n_frames=None, leg_label=None, color=None, line_type=None, **plot_kwargs):
        y = np.array(y, copy=False)
        assert y.ndim == 1

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

    def add_ys(self, x, ys, yerr=None, leg_label=None, color=None, line_type=None, return_color=False):
        if isinstance(line_type, str):
            line_type = line_type.lower()

        self.line_types.append(line_type)

        if isinstance(ys[0][0], UFloat):
            yerr = list(map(unp.std_devs, ys))
            ys = list(map(unp.nominal_values, ys))

        self.yerrs.append(yerr)
        self.ys.append(ys)

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

        for step in steps_visibility:
            print(len(step))

        return out
        # for l in self.ys:

    def plot(self):
        n_traces = 0
        for index_step, (ys, yerrs, x, color, leg_label, lt) in \
                enumerate(zip(self.ys, self.yerrs, self.xs, self.colors, self.leg_labels, self.line_types)):

            line = {None: None, 'hist': {'shape': 'hvh'}}[lt]

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
                                name=leg_label
                            ),
                        )

        for persistent in self.persistent:
            n_traces += 1
            self.fig.add_trace(
                go.Scatter(
                    visible=1,
                    x=persistent["x"],
                    y=persistent['y'],
                    error_y=dict(type='data', array=persistent['yerr']),
                    marker_color=persistent['color'],
                    line=persistent['line_type'],
                    name=persistent['leg_label'],
                    **persistent['plot_kwargs']
                ),
            )
        print("n_traces", n_traces)
            # self.fig.add_trace(
            #     go.Scatter(
            #         visible=0,
            #         x=persistent['x'],
            #         y=persistent['y'],
            #         # error_y=dict(type='data', array=yerr),
            #         marker_color=persistent['color'],
            #         line=persistent['line_type'],
            #         name=persistent['leg_label']
            #     ),
            # )

        sliders = [dict(
            active=0,
            currentvalue={"prefix": self.slider_prefix},
            pad={"t": 50},
            steps=self.steps
        )]

        self.fig.update_layout(
            sliders=sliders, bargap=0, bargroupgap=0.0
        )

        self.fig.show()


args = plotly(l1, time_step=20, erg_min=60, erg_max=1500, time_max=350, nominal_values=True, dont_plot=True)
plotly(l2, **args)



# p = InteractivePlot()
#
# n= 10
# x = np.linspace(0, np.pi*2, 10)
# print(x)
# ys1 = np.array([np.cos(x-i*np.pi*2/n) for i in range(n)])
# ys2 = [np.sin(x-i*np.pi*2/n) for i in range(n//2)]
#
# p.add_ys(x, ys1, leg_label="Cos", line_type='hist')
# p.add_ys(x, ys2, yerr=np.sqrt(np.abs(ys2)), leg_label="Sin")
# p.add_persistent(x, ys1[0]+0.1, leg_label='omg')
# print(p.fig)
# p.plot()
# print(p.steps)