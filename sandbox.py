import numpy as np
from pathlib import Path


lines = ["X, Y, Z"]
for i in range(100):
    th = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, 4/1000)
    lines.append(f'{r*np.cos(th)}, {r*np.sin(th)}, 0')