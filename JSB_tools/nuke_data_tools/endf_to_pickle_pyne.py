"""
Due to limited ability of openmc to read non-neutron induced reactions from end files, this module will fill in some of
the gaps. Must be run with python 3.7 due to pyne's requirements.
"""
from typing import Tuple
from pyne.endf import Evaluation, Library
from global_directories import tendl_2019_proton_dir
import re
from matplotlib import pyplot as plt
from pathlib import Path


def loop_tendl_paths(_dir=None) -> Tuple[str, Path]:
    """
    The tendl library is unfortunately structured by many nested directories. This loops until it finds a file and
    yields it
    """
    if _dir is None:
        _dir = tendl_2019_proton_dir

    if Path.is_dir(_dir):
        for path in _dir.iterdir():
            yield from loop_tendl_paths(path)
    else:
        m = re.match('p-([A-Za-z]+)([0-9]+)\.tendl', _dir.name)
        if m:
            symbol = m.groups()[0]
            A = m.groups()[1]
            nuclide_name = f'{symbol}{A}'
            yield nuclide_name, _dir
    # yield from out

for nuclide_symbol, path in loop_tendl_paths():
    if nuclide_symbol == 'U238':
        e = Evaluation(path)
        922380000
        try:
            print(e.read())
        except:
            pass
        # print(e.fission)
        r = (e.reactions[4])

        # y = (r.xs.y)
        # x = (r.xs.x)
        # plt.plot(x, y)
plt.show()
        #
        # print(dir(r))
        # print(r.production)

