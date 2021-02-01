import os
import sys
# from .outp_reader import OutP
from warnings import warn
from openmc.data import atomic_weight
import re
from typing import List
import numpy as np
import time
from numbers import Number
from itertools import islice
from sortedcontainers import SortedDict


class ProgressReport:
    def __init__(self, i_final, sec_per_print=2, i_init=0):
        self.__i_final__ = i_final
        self.__i_init__ = i_init
        self.__sec_per_print__ = sec_per_print
        # self.__i_current__ = i_init
        self.__next_print_time__ = time.time() + sec_per_print
        self.__init_time__ = time.time()
        self.__rolling_average__ = []

    def __report__(self, t_now, i):
        evt_per_sec = (i-self.__i_init__)/(t_now - self.__init_time__)
        self.__rolling_average__.append(evt_per_sec)
        evt_per_sec = np.mean(self.__rolling_average__)
        if len(self.__rolling_average__) >= 5:
            self.__rolling_average__ = self.__rolling_average__[:5]
        evt_remaining = self.__i_final__ - i
        sec_remaining = evt_remaining/evt_per_sec
        sec_per_day = 60**2*24
        days = sec_remaining//sec_per_day
        hours = (sec_remaining % sec_per_day)//60**2
        minutes = (sec_remaining % 60**2)//60
        sec = (sec_remaining % 60)
        msg = " {0} seconds".format(int(sec))
        if minutes:
            msg = " {0} minutes,".format(minutes) + msg
        if hours:
            msg = " {0} hours,".format(hours) + msg
        if days:
            msg = "{0} days,".format(days) + msg
        print(msg + " remaining.", i/self.__i_final__)

    def log(self, i):
        t_now = time.time()
        if t_now > self.__next_print_time__:
            self.__report__(t_now, i)
            self.__next_print_time__ += self.__sec_per_print__


def closest(sorted_dict: SortedDict, key):
    """Return closest key in `sorted_dict` to given `key`."""
    assert isinstance(sorted_dict, SortedDict)
    assert len(sorted_dict) > 0
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))


def ROOT_loop():
    try:
        import ROOT
        import time
        while True:
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
    except ModuleNotFoundError:
            warn('ROOT not installed. Cannot run ROOT_loop')



