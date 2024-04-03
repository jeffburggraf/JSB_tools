"""
Put the following into your ~/profile:

Clean(){
    python /path/to/Clean.py "$1";
}

Add -outp to not clean first run

"""
import re
import platform
import os
import warnings
from pathlib import Path
import sys

try:
    from send2trash import send2trash
except ModuleNotFoundError:
    warnings.warn("send2trash module is needed in order to use clean.py")
    raise

cwd = Path(os.getcwd())

# print(sys.argv)
# assert False

if '-outp' == sys.argv[1]:
    safe_files = ['runtpe.h5', 'runtpe', 'outp', 'mctal']
else:
    safe_files = []

m = re.compile(r"(ptra[a-z]$)|(runtp[a-z]$)|(runtp[a-z]\.h5$)|(mcta[a-z]$)|(out[a-z]$)|(comou[a-z]$)|(meshta[a-z]$)|(mdat[a-z]$)|"
               r"(plot[m-z]\.ps)")

paths = list(cwd.iterdir())

_d = []
for f_path in paths:
    if m.match(f_path.name):
        if f_path.name in safe_files:
            continue

        trash_name = f'{f_path.name}-{f_path.parent.relative_to(f_path.parents[2])}'

        if platform.system() == 'Windows':
            trash_name = trash_name.replace('\\', '.')
        else:
            trash_name = trash_name.replace('/', '.')

        new_path = f_path.parent / trash_name

        os.rename(f_path, new_path)
        send2trash(new_path)
