# from JSB_tools.nuke_data_tools import Nuclide
from matplotlib import pyplot as plt
import matplotlib
from mplhep import histplot
import  numpy as np
# print(matplotlib.__version__)
# # s = Nuclide.from_symbol('U238')
# from uncertainties import unumpy as unp
# # from openmc.data import FissionProductYields
# bins = np.arange(0, 100)
# y, _ = np.histogram(20*np.random.rand(1000) + 50, bins=bins)
# yerr = np.sqrt(y)
# print(y)
#
# histplot(y, bins=bins, yerr = yerr, ds='steps-mid')
# # n = Nuclide.from_symbol('U238')
import matplotlib.pyplot as plt
import numpy as np
import inspect

def hist_errorbars( data, xerrs=True, *args, **kwargs) :
    """Plot a histogram with error bars. Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar"""
    # pop off normed kwarg, since we want to handle it specially
    norm = False
    if 'normed' in kwargs.keys() :
        norm = kwargs.pop('normed')

    # retrieve the kwargs for numpy.histogram
    histkwargs = {}
    for key, value in kwargs.iteritems() :
        if key in inspect.getargspec(np.histogram).args :
            histkwargs[key] = value

    histvals, binedges = np.histogram( data, **histkwargs )
    yerrs = np.sqrt(histvals)

    if norm :
        nevents = float(sum(histvals))
        binwidth = (binedges[1]-binedges[0])
        histvals = histvals/nevents/binwidth
        yerrs = yerrs/nevents/binwidth

    bincenters = (binedges[1:]+binedges[:-1])/2

    if xerrs :
        xerrs = (binedges[1]-binedges[0])/2
    else :
        xerrs = None

    # retrieve the kwargs for errorbar
    ebkwargs = {}
    for key, value in kwargs.iteritems() :
        if key in inspect.signature(plt.errorbar).k :
            histkwargs[key] = value
    out = plt.errorbar(bincenters, histvals, yerrs, xerrs, fmt=".", **ebkwargs)

    if 'log' in kwargs.keys() :
        if kwargs['log'] :
            plt.yscale('log')

    if 'range' in kwargs.keys() :
        plt.xlim(*kwargs['range'])

    return out