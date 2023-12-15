import pickle
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib import transforms
from matplotlib.widgets import TextBox, CheckButtons, Button
import matplotlib
from pathlib import Path
# matplotlib.use('TkAgg')


class Image:
    graphics_attrs = ['straight_edges', 'space_click_points', 'tab_click_points', 'dist_labels', 'linenum_labels',
                      'theta_ref_lines']
    def clear(self, only_last=False):
        for s in Image.graphics_attrs:
            list_ = getattr(self, s)
            for el in reversed(list_):
                try:
                    el.remove()
                except ValueError:
                    continue

                if only_last:
                    break

            if not only_last:
                setattr(self, s, [])

        self.update()

    def on_keypress(self, evt):
        if evt.key == 'z':
            raise NotImplementedError("TODO")

        elif evt.key == ' ':
            self.holding_space = True
            self.fig.canvas.set_cursor(3)
        elif evt.key == 'escape':
            self.clear()
            if not self._from_pickled:
                self.ref_scale = None
            if self.ref_txt_box is not None:
                self.ref_txt_box.ax.remove()
                self.ref_txt_box = None

        elif evt.key == 'tab':
            self.holding_tab = True

    def on_keyrelease(self, evt):
        if evt.key == ' ':
            self.holding_space = False
            self.fig.canvas.set_cursor(1)
        elif evt.key == 'h':
            self.ax.set_xlim(self.lims[0])
            self.ax.set_ylim(self.lims[1])
            self.fig.canvas.draw_idle()
        elif evt.key == 'tab':
            self.holding_tab = False

            l0 = self.tab_click_points[0]
            def get_v(l):
                dx = l.get_xdata()[0] - l0.get_xdata()[0]
                dy = l.get_ydata()[0] - l0.get_ydata()[0]
                if self.ref_scale is not None:
                    dx *= self.ref_scale
                    dy *= self.ref_scale

                return f'({dx:.3f}, {dy:.3f})'

            pts = [get_v(l) for l in self.tab_click_points]

            print(f"Points:\n[{','.join(pts)}]")

    def request_ref_input(self):
        ax = self.fig.add_axes([0.1, 0.9, 0.1, 0.03])

        self.ref_txt_box = TextBox(ax, "Ref. Length:", color='red')
        self.ref_txt_box.on_submit(self.set_ref)

    def set_ref(self, s):
        s = str(s)
        vals = s.split(' ')
        if len(vals) > 2:
            raise ValueError("Invalid ref. value. Examples:\n5\n 5 cm")
        elif len(vals) == 2:
            self.units = vals[1]

        val = vals[0]

        self.ref_input = s  # for pickling

        norm = np.linalg.norm(self.points[1] - self.points[0])
        self.ref_scale = float(val)/norm
        self.ref_txt_box.ax.remove()
        self.ref_txt_box = None

        # mpx, mpy = np.sum(self.points, axis=0) / 2

        self.dist_labels[-1].set_text(s)

        if self.units is not None:
            self.ax.set_xlabel(f"x [{self.units}]")
            self.ax.set_ylabel(f"y [{self.units}]")

        self.points = []

        self.ax.xaxis.set_major_formatter(self._xfmt)
        self.ax.yaxis.set_major_formatter(self._yfmt)

        self.update()

    def _xfmt(self, x, pos):
        s = 1 if self.ref_scale is None else self.ref_scale
        return f'{x * s: .4g}'

    def _yfmt(self, y, pos):
        s = 1 if self.ref_scale is None else self.ref_scale
        return f'{y * s: .4g}'

    # @property
    # def norm(self):
    #     assert len(self.points) == 2
    #     return np.linalg.norm(self.points[1] - self.points[0])

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
        x, y = np.array(evt.xdata), np.array(evt.ydata)

        print(self.holding_tab)
        if self.holding_tab:
            self.tab_click_points.append(self.ax.plot([x], [y],ls='None', marker='+', color='blue',
                                  markersize=7)[0])
            self.update()

        elif self.holding_space:
            if self.ref_scale is None and self.ref_txt_box is not None:
                return

            self.points.append(np.array([x, y]))

            point, = self.ax.plot([self.points[-1][0]], [self.points[-1][1]],
                                  ls='None', marker='+', color='red',
                                  markersize=7)

            self.space_click_points.append(point)

            if len(self.points) == 2:
                norm = np.linalg.norm(self.points[1] - self.points[0])

                dx, dy = np.array(self.img.shape) * 0.02

                if self.snapQ:
                    if self.perform_snap():
                        [p.set_data(self.x_points, self.y_points) for p in self.space_click_points[-2:]]

                line, = self.ax.plot(*np.transpose(self.points), color='red', lw=1.5)

                self.straight_edges.append(line)

                mpx, mpy = np.sum(self.points, axis=0)/2

                if self.ref_scale is not None:
                    line_label = f'{len(self.linenum_labels)}'
                else:
                    line_label = 'ref'

                xl, xy = self.points[0][0], self.points[0][1],
                line_label = self.ax.text(xl - dx, xy - 1.5 * dy, line_label, fontsize=9,
                                          color='black', fontweight='bold')

                self.linenum_labels.append(line_label)
                description = f'Line {len(self.linenum_labels)}: p1={self.points[0]}, p2={self.points[1]}'

                if not self.snapQ:
                    dy_line = self.points[1][1] - self.points[0][1]
                    dx_line = self.points[1][0] - self.points[0][0]
                    theta = 180 / np.pi * np.arctan(dy_line / dx_line)

                    if dx_line < 0:
                        theta *= -1

                    ref_line, = self.ax.plot([self.points[0][0], self.points[0][0] + np.sign(dx_line) * 4 * dx, ],
                                             [self.points[0][1], self.points[0][1]],
                                             ls='--', marker='None', color='red')
                    self.theta_ref_lines.append(ref_line)
                    description += f', angle={theta:.2f}°'

                print(description)

                if self.ref_scale is not None:
                    c = 'green'
                    s = self.ref_scale * norm

                else:
                    s = norm
                    c = 'black'

                    self.request_ref_input()  # clears self.points

                text = self.ax.text(mpx, mpy, f'{s:.2f}', fontsize=14,
                                    color=c , fontweight='bold')
                self.dist_labels.append(text)

            elif len(self.points) == 3:
                self.points = [self.points[-1]]

            self.update()

    @property
    def snapQ(self):
        return self.check_buttons.get_status()[0]

    def update(self):
        self.fig.canvas.draw_idle()

    def save(self, *args, name=None):
        if name is None:
            name = self.path.name

        p = (self.path.parent/name).with_suffix('.pickle')

        data = {'space_click_points': [], 'path': self.path, 'ref_input': self.ref_input, 'theta': self._theta}
        for edge in self.straight_edges:
            ps = []
            for x, y in zip(edge._x, edge._y):
                ps.append([x, y])
            data['space_click_points'].append(ps)

        with open(p, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved to {p}")

    @classmethod
    def from_pickle(cls, path):
        path = Path(path)
        path = path.with_suffix('.pickle')

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # self = cls.__new__(cls, img=data['img'])
        self = cls.__new__(cls)
        self.__init__(data['path'])
        self._from_pickled = True

        self._rot_im(data['theta'])

        self.holding_space = True
        for i, points in enumerate(data['space_click_points']):
            for p in points:
                self.on_click(Fake_press(*p))

            if i == 0:
                self.set_ref(data['ref_input'])
                # self.ref_scale = data['ref_scale']

        self.holding_space = False
        # self.straight_edges = data['straight_edges']
        #
        return self

    def rot_on_sumbit(self, *args):
        theta = float(self.rotate_txt_box.text)
        self._rot_im(theta)

    def _rot_im(self, theta):
        self._theta = theta
        self.im.set_data(rotate(self.img, theta))
        self.update()

    def __init__(self, path):
        self._theta = 0

        self._from_pickled = False
        self.ref_input = '1'  # for pickling. Will be changed when ref is set. Includes units

        self.path = Path(path)

        self.ref_scale = None
        self.ref_txt_box = None

        self.holding_space = False
        self.holding_tab = False

        self.units = None

        if not hasattr(self, 'img'):
            self.img = plt.imread(path)

        self.fig, axs = plt.subplots(1, 2, width_ratios=[8, 1], figsize=(12, 8))

        title = (self.path.relative_to(self.path.parents[1]))

        self.fig.suptitle(title)

        self.ax: Axes = axs[0]
        self.button_ax = axs[-1]
        self.button_ax.set_axis_off()

        bx_ax = self.fig.add_axes([self.button_ax._position.x0, 0.05, 0.12, 0.5])
        self.check_buttons = CheckButtons(bx_ax, ['V/H snap'], [False])

        ax_save_button = self.fig.add_axes([self.button_ax._position.x0, 0.9, 0.1, 0.05])
        self.save_button = Button(ax_save_button, 'Save',)
        self.save_button.on_clicked(self.save)

        rotate_inp_ax = self.fig.add_axes([self.button_ax._position.x0 + 0.03, 0.83, 0.06, 0.05])
        self.rotate_txt_box = TextBox(rotate_inp_ax, "Image rotation: ", '0')
        self.rotate_txt_box.on_submit(self.rot_on_sumbit)

        self.img = np.flipud(self.img)

        self.im = self.ax.imshow(self.img) #, origin = 'lower')

        self.ax.invert_yaxis()

        self.straight_edges = []
        self.space_click_points = []
        self.theta_ref_lines = []
        self.dist_labels = []
        self.linenum_labels = []
        self.tab_click_points = []

        self.points = []

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.fig.canvas.mpl_connect('key_release_event', self.on_keyrelease)

        plt.autoscale(False)

        self.lims = self.ax.get_xlim(), self.ax.get_ylim()


class Fake_press:
    def __init__(self, x=500, y=500):
        self.xdata = x
        self.ydata = y

instructions ="""
Hold space and click in two places to create a line.
Upon creating the first line, an input box will appear where the reference distance must be input before continuing. 
Units can (optionally) be included in reference input, e.g. "10 cm".
  
Holding tab and clicking repeatedly will print out the coordinate locations of each click. 

Hold "O" and click to set (0,0) origin. 


"""
if __name__ == '__main__':
    p = Path('/Users/burgjs/PycharmProjects/miscMCNP/detectorModels/RMET3/xray_center1.tif')
    # p = Path('/Users/burgjs/PycharmProjects/MiscMCNP/detectorModels/RMET3/XRay1.tif')
    i = Image(p)
    # i = Image.from_pickle(p)

    plt.show()
