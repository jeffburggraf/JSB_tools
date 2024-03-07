import re
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from JSB_tools import mpl_hist, convolve_gauss
import matplotlib.colors as mpl_colors
from pathlib import Path
import pendulum
from matplotlib import cm


matplotlib.use('Qt5agg')
cwd = Path(__file__).parent

X = []
Y = []
colors = []

with open(cwd/'172_pwr') as f:
    f.readline()
    for line in f.readlines():
        vals = line.split('\t')

        _, date, _, _, _, pwr, _, _, _ = vals

        date = pendulum.from_format(date, 'MMM-DD-YYYY')
        try:
            Y.append(float(pwr))
            X.append(date)
        except ValueError:
            continue

        colors.append([20.4, 25, 36, 44.5, 52.8, 60, 68, 66.5, 57.7, 45, 32.4, 22][date.month - 1])


fig, ax = plt.subplots()


months = mdates.MonthLocator(interval=3)
months_fmt = mdates.DateFormatter('%b')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.xaxis.set_minor_locator(months)
ax.xaxis.set_minor_formatter(months_fmt, )

ax.tick_params(axis='x', labelrotation=90, which="both")

cmap = cm.coolwarm
cnorm = mpl_colors.Normalize(min(colors), max(colors))

ax.scatter(X, Y, marker='o', c=colors, cmap=cmap)

ax.plot(X, Y, color='black', ds='steps-mid')
smooth = convolve_gauss(Y, 6, reflect=True)
ax.plot(X, smooth)

plt.show()

