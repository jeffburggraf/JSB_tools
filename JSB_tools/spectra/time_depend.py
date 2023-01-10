import numpy as np
from matplotlib import pyplot as plt
from JSB_tools import mpl_hist
from pathlib import Path
from functools import cached_property
from typing import List
from mpl_interactions.widgets import RangeSlider


class InteractiveSpectra:
    def add_spectra(self, energy_binned_times, energy_bins):
        self.energy_binned_timess.append(energy_binned_times)
        self.erg_binss.append(energy_bins)

        if self.times_range[0] is None:
            self.times_range[0] = min([ts[0] for ts in energy_binned_times if len(ts)])
        else:
            self.times_range[0] = min(self.times_range[0], min([ts[0] for ts in energy_binned_times if len(ts)]))

        if self.times_range[-1] is None:
            self.times_range[-1] = max([ts[-1] for ts in energy_binned_times if len(ts)])
        else:
            self.times_range[-1] = max(self.times_range[-1], max([ts[-1] for ts in energy_binned_times if len(ts)]))

    def __init__(self, init_slider_width=10):
        self.times_range: List[float] = [None, None]

        self.erg_binss = []
        self.energy_binned_timess = []

        self.fig, self.ax = plt.subplots()

        plt.subplots_adjust(bottom=0.2)

        self.slider_ax_left = plt.axes([0.1, 0.1, 0.8, 0.05])
        self.slider_ax_right = plt.axes([0.1, 0.02, 0.8, 0.05])

        self.slider_ax_left.set_axis_off()
        self.slider_ax_right.set_axis_off()

        self.slider_width = init_slider_width
        self.slider_pos = init_slider_width/2

        self.slider_left: plt.Slider = None
        self.slider_right: plt.Slider = None

    def _slider_left(self):
        pass

    def _slider_right(self):
        pass

    def show(self):
        self.slider_left = plt.Slider(self.slider_ax_left, "l", valmin=self.times_range[0],
                                         valmax=self.times_range[1], valinit=self.times_range[0])
        self.slider_right = plt.Slider(self.slider_ax_right, '', valmin=self.times_range[0],
                                 valmax=self.times_range[1], valinit=self.times_range[0] + 0.5*self.slider_width)

        # self.slider.on_changed(self._slider_func)
        # self.ranged_slider.on_changed(self._ranged_func)



if __name__ == '__main__':
    # for dir_ in Path(r'C:\Users\jeffb\PycharmProjects\IACExperiment\exp_data').iterdir():
    #     if dir_.is_dir():
    #         for path in dir_.iterdir():
    #             if path.suffix in ['.pylist', '.list_meta', '.marshal', '.marshalSpe']:
    #                 path.unlink()
    from JSB_tools.spectra import ListSpectra
    from analysis import Shot

    l = Shot(134).list
    i = InteractiveSpectra()
    i.add_spectra(l.energy_binned_times, l.erg_bins)
    i.show()
    plt.show()
