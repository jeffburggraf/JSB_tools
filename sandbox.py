import re
from matplotlib import pyplot as plt
import numpy as np

# f = open('/Users/jeffreyburggraf/Desktop/omg.gcode')


#  =================================================================
min_z = 0.08
max_z = 0.25
#  =================================================================
# lines = f.readlines()




#
# def get_code(z):
#     return f"G28 Z\nG0 F6000 Z{0.2-z:.3f} ; offset: {z}\nG92 Z0.2\n"
#
# x_max, x_min = None, None
# begin_skin_index = None
# for index, line in enumerate(lines):
#     if m1 := re.match(';MINX:([0-9.]+)', line):
#         x_min = float(m1.groups()[0])
#     elif m2 := re.match(';MAXX:([0-9.]+)', line):
#         x_max = float(m2.groups()[0])
#     elif re.match(';TYPE:SKIN', line):
#         begin_skin_index = index
#         break
# else:
#     assert False

x_min = 0
x_max = 1
def get_z(x):
    return (x-x_min)*(max_z-min_z)/(x_max-x_min) + min_z

for x in np.linspace(0, 1, 15)[::-1]:
    dx = x
    offset = get_z(x)-0.2
    print(f'{dx/10:.2f}   {offset:.2f}   {abs(100*offset/0.2):.0f}%')

out = lines[:begin_skin_index]

g1_match = re.compile('G1.+X(?P<x>[0-9.]+) Y(?P<y>[0-9.]+) E(?P<e>[0-9.]+)')

stop_modifying = False
for line in lines[begin_skin_index:]:
    if re.match(';TIME_ELAPSED', line):
        stop_modifying = True
    if (m := g1_match.match(line)) and not stop_modifying:
        x = float(m.group('x'))
        z = get_z(x)
        if "Z" not in line:
            line = line.replace('\n', '')
            new_line = f"{line} Z{z:.3f}\n"
            out.append(new_line)
            continue
    out.append(line)

print('distance from left [cm]    Z-offset [cm]   % of layer height')


for x in np.linspace(x_min, x_max, 15)[::-1]:
    dx = x - x_min
    offset = get_z(x)-0.2
    print(f'{dx/10:.2f}   {offset:.2f}   {abs(100*offset/0.2):.0f}%')


with open('/Users/jeffreyburggraf/Desktop/ZOffsetTest.gcode', 'w') as f:
    for line in out:
        f.write(line)
