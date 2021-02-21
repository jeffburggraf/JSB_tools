from __future__ import annotations
import re
from typing import List, Dict
import numpy as np

class GCommand:
    def __init__(self, cmd: str, line_num, xyz_mode, extrusion_mode, goem_type=None, prev_cmd: GCommand = None):
        self.xyz_mode = xyz_mode
        self.extrusion_mode = extrusion_mode
        self.line_num = line_num
        self.prev_cmd = prev_cmd
        self.goem_type = goem_type
        self.cmd = cmd
        self.F = 0

        self.x1, self.x2, self.y1, self.y2, self.z1, self.z2, self.e1, self.e2 = [None]*8

        def get_r1_r0(value, key, mode):
            if mode == 'relative':
                if self.prev_cmd is not None:
                    r1 = getattr(prev_cmd, key + '2')
                    if r1 is not None:
                        r2 = r1 + value
                    else:
                        r2 = None
                else:
                    r1 = None
                    r2 = None
            elif mode == 'absolute':
                if self.prev_cmd is not None:
                    r1 = getattr(prev_cmd, key+'2')
                else:
                    r1 = None
                r2 = value
            else:
                assert False, 'Bad mode'
            return r1, r2

        if m := re.match('F([0-9.]+)', cmd):
            self.F = float(m.groups()[0])
        for key in ['X', 'Y', 'Z', 'E']:
            if m := re.match(f'.+{key}([0-9.]+).+', cmd):
                x1, x2 = get_r1_r0(float(m.groups()[0]), key.lower(), self.xyz_mode)
                setattr(self, key.lower() + '1', x1)
                setattr(self, key.lower() + '2', x2)
            else:
                if prev_cmd is not None:
                    _k = key.lower()
                    setattr(self, _k + '1', getattr(prev_cmd,_k + '2' ))
                    setattr(self, _k + '2', getattr(prev_cmd,_k + '2' ))

    @property
    def dxy(self):
        if None not in [self.x2, self.x1, self.y2, self.y1]:
            return np.array([self.x2-self.x1, self.y2-self.y1])
        else:
            return 0

    @property
    def de(self):
        if None not in [self.e1, self.e2]:
            return self.e2-self.e1
        else:
            return 0


class AnalyseGCode:
    def __init__(self, path):
        with open(path) as f:
            self.lines = f.readlines()
        self.x_range = [None, None]
        self.y_range = [None, None]
        self.z_range = [None, None]
        self.layer_height = None
        self.filament_used = None
        self.time_required = None
        self.bed_temp = None
        self.linear_commands: List[GCommand] = []
        xyz_mode = 'absolute'
        extrusion_mode = 'absolute'
        geom_type = None
        for index, line in enumerate(self.lines):
            line_num = index + 1
            if len(self.linear_commands):
                prev_linear_command = self.linear_commands[-1]
            else:
                prev_linear_command = None

            if m := re.match(';MINX:([0-9.]+)', line):
                self.x_range[0] = float(m.groups()[0])
            elif m := re.match(';MINY:([0-9.]+)', line):
                self.y_range[0] = float(m.groups()[0])
            elif m := re.match(';MINZ:([0-9.]+)', line):
                self.z_range[0] = float(m.groups()[0])
            elif m := re.match(';MAXX:([0-9.]+)', line):
                self.x_range[1] = float(m.groups()[0])
            elif m := re.match(';MAXY:([0-9.]+)', line):
                self.y_range[1] = float(m.groups()[0])
            elif m := re.match(';MAXZ:([0-9.]+)', line):
                self.z_range[1] = float(m.groups()[0])
            elif m := re.match(';Layer height: +([0-9.]+)', line):
                self.layer_height = m.groups()[0]
            elif m := re.match(';Filament used: +([0-9.]+)', line):
                self.filament_used = m.groups()[0]
            elif m := re.match(';TIME:([0-9.]+)', line):
                self.time_required = m.groups()[0]
            elif m := re.match('M140 S([0-9.]+)', line):
                self.bed_temp = m.groups()[0]
            elif re.match(' *G91 *', line):
                xyz_mode = 'relative'
            elif re.match(' *G90 *', line):
                xyz_mode = 'absolute'
            elif re.match(' *M82 *', line):
                extrusion_mode = 'absolute'
            elif re.match(' *M83 *', line):
                extrusion_mode = 'relative'
            elif re.match('G28', line):
                self.linear_commands.append(GCommand("G0 X0 Y0 Z0 E0", line_num, xyz_mode, extrusion_mode,geom_type,
                                                prev_linear_command))
            elif line[:2] in ['G1', 'G2']:
                self.linear_commands.append(GCommand(line, line_num, xyz_mode, extrusion_mode, geom_type,
                                                     prev_linear_command))
            elif m := re.match(';TYPE:([A-Z]+)', line):
                geom_type = m.groups()[0]




g = AnalyseGCode('/Users/jeffreyburggraf/Desktop/ZOffsetTest.gcode')

for c in g.linear_commands:
    print(c.goem_type, c.de/np.linalg.norm(c.dxy))