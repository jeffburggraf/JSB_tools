import re
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from JSB_tools import mpl_hist, convolve_gauss
import matplotlib.colors as mpl_colors
from pathlib import Path
import pendulum
from JSB_tools.tab_plot import TabPlot
from matplotlib import cm
import pickle

with open('monthly_avgs.pickle', 'rb') as f:
    data = pickle.load(f)

matplotlib.use('Qt5agg')
cwd = Path(__file__).parent

X = []
pwrs = []
mean_temps = []
median_temps = []

with open(cwd/'172_pwr') as f:
    f.readline()
    for line in f.readlines():
        vals = line.split('\t')

        _, date, _, _, _, pwr, _, _, _ = vals

        date = pendulum.from_format(date, 'MMM-DD-YYYY')
        try:
            pwrs.append(float(pwr))
            X.append(date)
        except ValueError:
            continue

        mean_temp = data['mean_temps'][(date.year, date.month)]
        median_temp = data['median_temps'][(date.year, date.month)]
        mean_temps.append(mean_temp)
        median_temps.append(median_temp)


mean_temps = np.array(mean_temps)
median_temps = np.array(median_temps)

# ==============================================
color = median_temps
setpoint = None  # None for color midpoint to be mean temp
# ==============================================
tab = TabPlot(figsize=(16, 8))
# fig, axs = plt.subplots(2, 1, figsize=(16, 8))


months = mdates.MonthLocator(interval=3)
months_fmt = mdates.DateFormatter('%b')

cmap = cm.seismic

if setpoint is not None:
    color = np.abs(color - setpoint)

ax = tab.new_ax('Over time')
ax.xaxis.set_minor_locator(months)
ax.xaxis.set_minor_formatter(months_fmt, )
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(axis='x', labelrotation=90, which="both")


sc = ax.scatter(X, pwrs, marker='o', c=color, cmap=cmap)
ax.plot(X, pwrs, color='black', ds='steps-mid')
ax.set_xlabel("Date")
ax.set_ylabel("Power consumption [kWh]")
cbar = plt.colorbar(sc)

if setpoint is not None:
    cbar.set_label(f'Abs([Monthly temp] - {setpoint})')
else:
    cbar.set_label('Mean monthly temp [F]')

n_month_avg = 3
smooth = convolve_gauss(pwrs, n_month_avg, reflect=True)

ax.plot(X, smooth, c='tab:blue', label=f'{n_month_avg} mo. avg.', lw=2)
ax.axhline(np.mean(pwrs), label="Mean (all time)", c='green', ls='--')
ax.legend()

ax = tab.new_ax('Correlation')
Y = np.abs(color - np.median(color))
ax.scatter(Y, pwrs)

ax.set_xlabel(f"Abs. deviation from median temp ({int(np.median(color))} F)")
ax.set_ylabel("Power consumption [kWh]")

a, b = np.polyfit(Y, pwrs, 1)
xs = np.linspace(min(Y), max(Y), 1000)
ax.plot(xs, b + xs*a, label=f'Linear fit\npwr=({a:.2e})$\Delta$T + {b:.2f}')
ax.legend()

plt.show()

