import numpy as np
print()

# # from matplotlib.widgets import Button
# # from matplotlib.widgets import CheckButtons, Slider
# # from matplotlib.lines import Line2D
# # from matplotlib.axes import Axes
# # from matplotlib import pyplot as plt
# # from matplotlib.figure import  Figure
# #
# # fig, ax = plt.subplots()
# #
# # n = 4
# #
# # w_ax: Axes = plt.axes([0.9, 0.2, 0.25, 0.6])
# # w_ax.set_axis_off()
# #
# #
# # def f(*args):
# #     print(f"args: {args}; get_active: {checks.get_active()}; get_status: {checks.get_status()}")
# #     # print(checks.get_active())
# #     # print(checks.get_status())
# #     # print(checks)
# #
# # checks = CheckButtons(w_ax, [str(i) for i in range(n)], actives=[True if i == 0 else False for i in range(n)])
# # checks.on_clicked(f)
# #
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import RadioButtons
#
# t = np.arange(0.0, 2.0, 0.01)
# s0 = np.sin(2*np.pi*t)
# s1 = np.sin(4*np.pi*t)
# s2 = np.sin(8*np.pi*t)
#
# fig, ax = plt.subplots()
# l, = ax.plot(t, s0, lw=2, color='red')
# fig.subplots_adjust(left=0.3)
#
# axcolor = 'lightgoldenrodyellow'
# rax = fig.add_axes([0.05, 0.7, 0.15, 0.15], facecolor=axcolor)
# radio = RadioButtons(rax, ('2 Hz', '4 Hz', '8 Hz'))
#
#
# def hzfunc(label):
#     hzdict = {'2 Hz': s0, '4 Hz': s1, '8 Hz': s2}
#     ydata = hzdict[label]
#     l.set_ydata(ydata)
#     plt.draw()
# radio.on_clicked(hzfunc)
#
# rax = fig.add_axes([0.05, 0.4, 0.15, 0.15], facecolor=axcolor)
# radio2 = RadioButtons(rax, ('red', 'blue', 'green'))
#
#
# def colorfunc(label):
#     l.set_color(label)
#     plt.draw()
# radio2.on_clicked(colorfunc)
#
# rax = fig.add_axes([0.05, 0.1, 0.15, 0.15], facecolor=axcolor)
# radio3 = RadioButtons(rax, ('-', '--', '-.', ':'))
#
#
# def stylefunc(label):
#     l.set_linestyle(label)
#     plt.draw()
#
# radio3.on_clicked(stylefunc)
#
# plt.show()