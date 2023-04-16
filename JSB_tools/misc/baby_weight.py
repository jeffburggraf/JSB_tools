import re
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
from JSB_tools import mpl_hist
from pathlib import Path
import matplotlib.dates
import unicodedata
from lmfit.models import LinearModel


# ==========================
fit_interval = 1.4  # in months
# ==========================
month_olds = []
bday = dt.datetime(2022, 5, 15)

weights = []
with open(Path(__file__).parent/'weight_data') as f:
    # t = ' '.join(f.readlines())
    year = 2022
    _prev_month = None
    _prev_date = None

    for line in f.readlines():

        if m := re.match(r' *\* *(?P<month>[0-9]+)\/(?P<day>[0-9]+) +(?P<lbs>[0-9.]+)', line):
            month = int(m['month'])

            if _prev_month is not None and (_prev_month != 1 and month == 1):
                year += 1

            date = dt.datetime(year, month, int(m['day']))
            age = date - bday
            month_olds.append(age.days/30)

            weights.append(float(m['lbs']))

            if _prev_date is not None and date < _prev_date:
                print(line)

            _prev_date = date
            _prev_month = month

        else:
            if len(line):
                print(f"No match: {line}, {line.__repr__()}")

fig, ax = plt.subplots(figsize=(10, 8))

weights = np.array(weights)
month_olds = np.array(month_olds)

ax.plot(month_olds, weights, ls='None', marker='o')
plt.subplots_adjust(bottom=0.15)

fit_intervals = np.linspace(month_olds[0], month_olds[-1], int((month_olds[-1] - month_olds[0])/fit_interval))
interval_idxs = np.searchsorted(month_olds, fit_intervals)

model = LinearModel()

for index, (i0, i1) in enumerate(zip(interval_idxs[:-1], interval_idxs[1:])):
    i_prev = max(i0-1, 0)
    i1 = min(len(month_olds), i1 + 1)

    x = month_olds[i_prev: i1]
    y = weights[i_prev: i1]
    fit = model.fit(data=y, x=x)

    ls = '--' if index % 2 == 0 else '-.'

    x_text = x[0] + (x[-1] - x[0]) / 4
    _t = ax.text(x_text, fit.eval(x=x_text) + 0.75, fr"{fit.params['slope'].value:.1f} $\frac{{lbs}}{{mo}}$",
                 transform=ax.transData, fontsize=13)
    _x = x[:]
    _dy = y[-1] - y[0]
    _x[0] = (month_olds[i_prev] + month_olds[i0])/2
    ax.plot(x, fit.eval(x=_x), c='black', ls=ls)

ax.set_xlabel("Months old")
ax.set_ylabel("Weight [lbs]")

plt.show()

print()

