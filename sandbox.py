# from abc import ABC, ABCMeta
# import numpy as np
# from matplotlib import pyplot as plt
#
# from JSB_tools.SRIM import find_SRIM_run
# from scipy.integrate import quad
# from scipy.interpolate import interp1d
#
# def channel2_erg(ch):
#     return 1 + 0.5*ch + 0.01*ch*2
#
#
# def erg_2_channel(erg):
#     return 5.*(-5. + np.sqrt(21. + 4.*erg))
#
#
# ergs = 10*np.random.randn(100000)
#
# # channels =
import re
with open("/Users/burggraf1/PycharmProjects/JSB_tools/requirements.txt") as f:
    for line in f.readlines():
        if m := re.match(r"(.+?)[><~]?=(.+)", line):
            print(f"  - {m.groups()[0]}={m.groups()[1]}")