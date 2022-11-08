import re
from pathlib import Path


def iter_evaluations(path, i=0):

    i += 1
    path = Path(path)
    if path.is_dir():
        for sub_path in path.iterdir():
            yield from iter_evaluations(sub_path, i)
    else:
        if i % 2 == 0:
            yield path
        else:
            print(i)


for i in iter_evaluations("/Users/burggraf1/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools"):
    print(i)
