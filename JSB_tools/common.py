import warnings
import numpy as np

from matplotlib.figure import Figure
import time

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


cwd = Path(__file__).parent

style_path = cwd/'mpl_style.txt'


class ProgressReport:
    def __init__(self, i_final, sec_per_print=2, i_init=0, final_msg=None, call_func=None):
        self.__i_final__ = i_final
        self.__i_init__ = i_init
        self.__sec_per_print__ = sec_per_print
        # self.__i_current__ = i_init
        self.__next_print_time__ = time.time() + sec_per_print
        self.__init_time__ = time.time()

        self.events_log = [self.__i_init__]
        self.times_log = [self.__init_time__]

        self.final_msg = final_msg

        if call_func is None:
            self.call_func = print
        else:
            self.call_func = call_func

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

        sec_per_day = 60**2*24

        if evt_per_sec > 0:
            sec_remaining = evt_remaining/evt_per_sec
            days = sec_remaining // sec_per_day
            hours = (sec_remaining % sec_per_day) // 60 ** 2
            minutes = (sec_remaining % 60 ** 2) // 60
            sec = int(sec_remaining % 60)
        else:
            sec = np.inf
            minutes = hours = days = 0

        msg = " {0} seconds".format(sec)
        if minutes:
            msg = " {0} minute{1},".format(minutes, 's' if minutes > 1 else '') + msg
        if hours:
            msg = " {0} hour{1},".format(hours, 's' if hours > 1 else '') + msg
        if days:
            msg = "{0} day{1},".format(days, 's' if days > 1 else '') + msg

        msg = f"{added_msg}... {msg} remaining {100*i/self.__i_final__:.2f}% complete"

        self.call_func(msg)

    def log(self, i, msg=""):
        t_now = time.time()
        if t_now > self.__next_print_time__:
            self.__report__(t_now, i, msg)
            self.__next_print_time__ += self.__sec_per_print__
            return True
        return False

    def __del__(self):
        if self.final_msg is None:
            print(f"Complete in {self.elapsed_time:.1f} total seconds")
        else:
            print(f"{self.final_msg} ({self.elapsed_time:.1f} seconds)")


class MPLStyle:
    fig_size = (15, 10)
    has_been_called = False

    @staticmethod
    def using_tex():
        return plt.rcParams['text.usetex']

    def get_new_show(self, ):
        old_show = plt.show

        def show(*args, **kwargs):
            for i in plt.get_fignums():
                fig: Figure = plt.figure(i)
                ax: Axes = fig.gca()
                for a in ['x', 'y']:
                    try:
                        ax.ticklabel_format(axis=a, scilimits=getattr(self, f'scilimits{a}'))
                    except Exception as e:
                        # raise
                        warnings.warn(f"Exception in MPLStyle.get_new_show, in Fig.{fig.get_label()}, Axes: {ax.get_title()}. Error message: {e}")

            return old_show(*args, **kwargs)

        return show

    @staticmethod
    def set_bold_axes_labels():
        def new_func(axis, class_):
            orig_func = getattr(class_, f'set_{axis}label')

            def f(self, *args, **kwargs):
                args = list(args)
                args[0] = fr"\textbf{{{args[0]}}}"
                return orig_func(self, *args, **kwargs)

            return f

        for x in ['x', 'y', 'z']:
            if x == 'z':
                class_ = Axes3D
            else:
                class_ = Axes
            setattr(class_, f'set_{x}label', new_func(x, class_))

    def __init__(self, minor_xticks=True, minor_yticks=True, bold_ticklabels=True, bold_axes_labels=True,
                 usetex=True, fontscale=None, fig_size=(15, 8), scilimitsx=(-2, 4), scilimitsy=(-3, 3)):

        """

            Args:
                usetex:
                fontscale: 1.5 for half-width latex document.

            Returns:

            """
        self.scilimitsx = scilimitsx
        self.scilimitsy = scilimitsy

        MPLStyle.has_been_called = True
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
            plt.rcParams.update({"text.usetex": False, })
        else:
            pass

        if fontscale is not None:
            for k in ['font.size', 'ytick.labelsize', 'xtick.labelsize', 'axes.labelsize', 'legend.fontsize',
                      'legend.title_fontsize']:
                plt.rcParams.update({k: plt.rcParams[k] * fontscale})

        plt.show = self.get_new_show()


def ROOT_loop():
    import ROOT
    try:
        import time
        while True:
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
    except ModuleNotFoundError:
        warnings.warn('ROOT not installed. Cannot run ROOT_loop')
