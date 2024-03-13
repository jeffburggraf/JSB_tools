"""
#  from https://mesonet.agron.iastate.edu/request/download.phtml?network=ID_ASOS

"""
import pickle

import pendulum
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import re
matplotlib.use('Qt5agg')
import  pendulum

cwd = Path(__file__).parent


snow_amounts = []
mean_temps = []
dates = []
fix, axs = plt.subplots(1, 2)

month = None

temps = {}
key = None

with open(cwd / 'temp.csv') as f:
    header = f.readline().split(',')
    temp_i = header.index('tmpf')

    while line := f.readline():
        vals = line.split(',')

        if vals[temp_i] == 'M':
            continue

        date = pendulum.from_format(vals[1], 'YYYY-MM-DD HH:SS', tz='MST')

        if date.month != month:
            key = (date.year, date.month)
            temps[key] = []

        month = date.month

        temp = float(vals[temp_i])
        temps[key].append(temp)
        # re.match()

mean_temps = {k: np.mean(v) for k, v in temps.items()}
median_temps = {k: np.median(v) for k, v in temps.items()}

data = {'mean_temps': mean_temps, 'median_temps': median_temps}
with open(cwd / 'monthly_avgs.pickle', 'wb') as f:
    pickle.dump(data, f)


