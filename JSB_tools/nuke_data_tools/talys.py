import pickle
import re
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from os import system

talys_dir = Path(__file__).parent/'data'/'talys'

tally_executable = '/Users/burggraf1/Talys/talys'


def run(target, projectile, min_erg=0, max_erg=120, erg_step=None,
        fission_yields=False, outdiscrete=False, maxlevelstar=30, **kwargs):
    """
    Args:
        target:
        projectile: n, p, d, t, h, a, g representing neutron, proton, deuteron, triton, 3He, alpha and gamma, respectively.
        min_erg:
        max_erg:
        erg_step:
        fission_yields:
        outdiscrete:
        maxlevelstar: The number of excited levels (starting from gs) considered in non-elastic reactions.
    Returns:

    """
    m = re.match("([A-Z][a-z]{0,2})-?([0-9]{1,3})", target)
    assert m
    a_symbol = m.groups()[0]
    mass = int(m.groups()[1])
    lines = [f'projectile {projectile}', f'element {a_symbol}',
             f'mass {mass}']

    if erg_step is None:
        lines.append(f'energy {projectile}{min_erg}-{max_erg}.grid')
    else:
        lines.append(f'energy {min_erg} {max_erg} {erg_step}')


    kwargs['maxlevelstar'] = maxlevelstar

    for k, v in kwargs.items():
        lines.append(f"{k} {v}")

    if fission_yields:
        lines.append('outfy y')

    if outdiscrete:
        lines.append('outdiscrete y')

    f_name = f"{a_symbol}{mass}-{projectile}"
    assert projectile in ['p', 'n', 'g']
    data_path = talys_dir/f_name
    Path.mkdir(data_path, exist_ok=True)
    s = '\n'.join(lines)
    with open(data_path/f_name, 'w') as f:
        f.write(s)

    cmd = f'cd {data_path};{tally_executable} < {f_name} > output'

    out = system(cmd)

    try:
        with open(data_path/'output') as f:
            line = f.readline()
            if "TALYS-error" in line:
                print(line + "\n".join(f.readlines()))
    except FileNotFoundError:
        raise FileNotFoundError("Output file not found. Something went wrong!")

    return out


def pickle_result(target, projectile):
    data = {}
    path = talys_dir/f'{target}-{projectile}'

    for p in path.iterdir():
        m_ground = re.match('rp[0-9]+\.tot', p.name)
        m_excited = re.match('rp[0-9]+\.L([0-9]+)', p.name)

        if m_ground or m_excited:

            with open(p) as f:
                lines = f.readlines()
            m = re.match('.+Production of +([0-9]+)([A-Z][a-z]{0,2})', lines[0])
            product = f'{m.groups()[1]}{m.groups()[0]}'
            if m_excited:
                product += f'_m{int(m_excited.groups()[0])}'
            if product == target:
                continue

            def get_values():
                if m_excited:
                    return tuple(map(float, line.split()[:2]))
                else:
                    return tuple(map(float, line.split()))

            for i in range(len(lines)):
                if lines[i][0] != '#':
                    break
            lines = lines[i:]
            ergs = []
            xss = []
            for line in lines:
                erg, xs = get_values()
                ergs.append(erg)
                xss.append(xs*1E-3)
            data[product] = xss

    with open(path/'data.pickle', 'wb') as f:
        pickle.dump(data, f)
        pickle.dump(ergs, f)


if __name__ == '__main__':
    # args = ('N14', 'g')
    target = 'U235'
    projectile = 'n'

    run(target, projectile, outdiscrete=True, fission_yields=False, max_erg=25,  filediscrete=30, maxlevelstar=30, fission='n')
    # pickle_result(*args)
