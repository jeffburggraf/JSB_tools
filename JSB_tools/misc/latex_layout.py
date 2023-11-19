import re
from pathlib import Path


tab_char = ' '  # set to \t to make indentation bigger

def print_outline(path):
    assert Path(path).exists()
    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        if m := re.match(r'\\((?:sub)*)section{(.+?)}', line):
            if m.groups()[0] is not None:
                n_nubs = len(m.groups()[0])//3
            else:
                n_nubs = 0

            s = tab_char * n_nubs + m.groups()[1]
            print(s)


print_outline('/Users/burggraf1/PycharmProjects/nim2021/nim/manuscript/JSB_NiM2021.tex')