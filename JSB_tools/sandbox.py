from __future__ import annotations
import numpy as np
from JSB_tools.maestro_reader import MaestroListFile
from JSB_tools import Nuclide
import plotly.graph_objects as go
from uncertainties import UFloat
from uncertainties import unumpy as unp
from JSB_tools import rolling_median
from uncertainties import ufloat
from lmfit.model import Model
import matplotlib.pyplot as plt
from JSB_tools import ROOT_loop
#
with open("/Users/burggraf1/Downloads/High.tsv") as f:
    lines = f.readlines()
    data = []
    for line in lines:
        data.append(list(map(float, list(line.split()))))
    data = np.array(data)


def f(x, a, b, c):
    return a + b*np.e**(-c*x)


def root_f(x, p):
    return p[0] + p[1]*np.e**(-p[2]*x[0])


yss = [data[:, i] for i in range(5)]
x = data[:, 5]

y = [[ufloat(_, np.sqrt(_)) for _ in l] for l in yss]
# y = [ufloat(_, np.sqrt(_)) for _ in y]
y = np.mean(y, axis=1)
print(f"y = {y}")

weights = 1.0/unp.std_devs(y)
y = unp.nominal_values(y)
#
# def lmfit():
#     model = Model(f, )
#     params = model.make_params()
#     params['a'].set(value=min(y))
#     params['b'].set(value=max(y) - min(y))
#     params['c'].set(value=np.log(2)/(x[-1] - x[0]))
#
#     fit_result = model.fit(y, x=x, weights=weights, params=params)
#     # fit_result.plot_fit()
#     params = fit_result.params
#
#     fit_x = np.linspace(x[0], x[-1], 100)
#     fit_err = fit_result.eval_uncertainty(params, x=fit_x)
#     fit_y = fit_result.eval(params, x=fit_x)
#
#     plt.errorbar(x, y, 1.0/weights, ls='None', marker='_')
#     plt.plot(fit_x, fit_y)
#     print(fit_result.fit_report())
#     return fit_result
#
def root():
    import ROOT

    # tf1 = ROOT.TF1("func", "[0] + [1]*2.71828**(-[2]*x)", x[0], x[1])
    tf1 = ROOT.TF1("func", root_f, x[0], x[-1], 3)

    # tf1.SetParameter('g', lambda x:x)
    tf1.SetParameter(0, min(y))
    tf1.SetParameter(1, max(y) - min(y))
    tf1.SetParameter(2, np.log(2)/(x[-1] - x[0]))

    graph = ROOT.TGraphErrors(len(x), np.array(x, dtype=float), y, 0.1*np.ones(len(x), dtype=float), 1.0/weights)

    graph.Fit(tf1)

    graph.Draw()
    ROOT_loop()

root()
#
# # lmfit()
#
# plt.show()
#
# print(y)


