import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from mpant_reader import MPANTList
from itertools import zip_longest
import time

a = [np.arange(np.random.uniform(0, 1000)) for i in range(1000)]


def get():
    return np.array(list(zip_longest(*a, fillvalue=0)))


def get2():
    return np.array(a)


# t0 = time.time()
# for i in range(100):
#     get()
# print(time.time() - t0)

t0 = time.time()
for i in range(100):
    get()
print(time.time() - t0)

