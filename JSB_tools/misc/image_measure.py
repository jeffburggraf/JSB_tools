import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib import transforms
from matplotlib.widgets import TextBox
import matplotlib
# matplotlib.use('TkAgg')


class Image:
    def clear(self):
        for a in self.straight_edges + self.plot_points + self.dist_labels:
            a.remove()
        self.straight_edges = []
        self.plot_points = []
        self.dist_labels = []
        self.update()

    def on_keypress(self, evt):
        if evt.key == ' ':
            self.holding_space = True
            self.fig.canvas.set_cursor(3)
        elif evt.key == 'escape':
            self.clear()

    def on_keyrelease(self, evt):
        if evt.key == ' ':
            self.holding_space = False
            self.fig.canvas.set_cursor(1)
        elif evt.key == 'h':
            self.ax.set_xlim(self.lims[0])
            self.ax.set_ylim(self.lims[1])
            self.fig.canvas.draw_idle()

    def get_ref_input(self, xdata, ydata):
        t = self.ax.transData + self.fig.transFigure.inverted()
        xfig, yfig = t.transform([xdata, ydata])
        ax = self.fig.add_axes([xfig, yfig, 0.05, 0.03])
        self.txt_box = TextBox(ax, "Length", color='red')
        self.txt_box.on_submit(self.set_ref)

    def set_ref(self, val):
        self.ref_scale = float(val)/self.norm
        self.txt_box.ax.remove()
        del self.txt_box
        self.points = []
        self.update()

    @property
    def norm(self):
        assert len(self.points) == 2
        return np.linalg.norm(self.points[1] - self.points[0])

    @property
    def dx(self):
        return self.points[1][0] - self.points[0][0]

    @property
    def dy(self):
        return self.points[1][1] - self.points[0][1]

    def perform_snap(self):
        if abs(self.dy/self.dx) < 0.035:
            y = 0.5 * (self.points[0][1] + self.points[1][1])
            self.points[0][1] = self.points[1][1] = y
            return True

        elif abs(self.dx / self.dy) < 0.05:
            x = 0.5 * (self.points[0][0] + self.points[1][0])
            self.points[0][0] = self.points[1][0] = x
            return True
        return False

    @property
    def x_points(self):
        return [p[0] for p in self.points]

    @property
    def y_points(self):
        return [p[1] for p in self.points]

    def on_click(self, evt):
        if self.holding_space:
            # xlims = self.ax.get_xlim()
            # ylims = self.ax.get_ylim()
            if self.ref_scale is None and self.txt_box is not None:
                return
            x, y = np.array(evt.xdata), np.array(evt.ydata)
            self.points.append(np.array([x, y]))

            point, = self.ax.plot([self.points[-1][0]], [self.points[-1][1]],
                                  ls='None', marker='+', color='red',
                                  markersize=7)

            self.plot_points.append(point)

            if len(self.points) == 2:
                if self.perform_snap():
                    [p.set_data(self.x_points, self.y_points) for p in self.plot_points[-2:]]

                line, = self.ax.plot(*np.transpose(self.points), color='red', lw=1.5)

                self.straight_edges.append(line)

                mpx, mpy = np.sum(self.points, axis=0)/2

                if self.ref_scale is not None:
                    text = self.ax.text(mpx, mpy, f'{self.ref_scale * self.norm:.2f}', fontsize=14,
                                        color='green', fontweight='bold')
                    self.dist_labels.append(text)
                    self.points = []
                else:
                    self.get_ref_input(mpx, mpy)

            # self.ax.set_xlim(xlims)
            # self.ax.set_ylim(ylims)

            self.update()

    def update(self):
        self.fig.canvas.draw_idle()

    def __init__(self, path):
        self.ref_scale = None
        self.txt_box = None

        self.holding_space = False
        self.img = plt.imread(path)
        self.fig, ax = plt.subplots()
        self.ax: Axes = ax

        self.im = self.ax.imshow(self.img)

        self.straight_edges = []
        self.plot_points = []
        self.dist_labels = []

        self.points = []

        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.fig.canvas.mpl_connect('key_release_event', self.on_keyrelease)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.autoscale(False)

        self.lims = self.ax.get_xlim(), self.ax.get_ylim()






if __name__ == '__main__':
    p = '/Users/jeffreyburggraf/Pictures/Himalaya.png'
    i = Image(p)
    class Fake_press1:
        xdata =  0
        ydata = 0


    class Fake_press2:
        xdata =  100
        ydata = 300

    # i.holding_space = True
    # i.on_click(Fake_press1)
    # i.on_click(Fake_press2)
    plt.show()
