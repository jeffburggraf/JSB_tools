from __future__ import annotations
from pathlib import Path
from JSB_tools import ROOT_loop
import re
import ROOT
import numpy as np
import time
from typing import Tuple
from typing import List, Union,  Dict
import shutil
from JSB_tools import ProgressReport

cwd = Path(__file__).parent


class Branch:
    __padding = None  # Set in Branch.text_heading
    all_branches: List[Branch] = []

    @staticmethod
    def order_branches(*some_names_in_order):
        names = [b.name for b in Branch.all_branches]
        arg_sort = [names.index(n) for n in some_names_in_order]
        for name in names:
            i = names.index(name)
            if i not in arg_sort:
                arg_sort.append(i)
        Branch.all_branches = [Branch.all_branches[i] for i in arg_sort]

    @staticmethod
    def text_heading():
        outs = [b.name for b in Branch.all_branches]
        Branch.__padding = np.array(list(map(lambda x: max(8, len(x)), outs))) + 2
        outs = [o.ljust(p) for o, p in zip(outs, Branch.__padding)]
        out = ",".join(outs) + '\n'
        return out

    @staticmethod
    def current_state2text():
        outs = []
        for padding, b in zip(Branch.__padding, Branch.all_branches):
            # f"{b.value: .4f<4}"
            value = f"{b.value: .2E}" if b.value is not None else f"{'_'}"
            value = f"{value: <{padding}}"
            outs.append(value)
        return ",".join(outs) + '\n'

    @staticmethod
    def clear_all_except(_except=None):
        if _except is None:
            for b in Branch.all_branches:
                b.clear()
        else:
            for b in Branch.all_branches:
                if b.name not in _except:
                    b.clear()

    @staticmethod
    def fill_tree():
        Branch.all_branches[0].tree.Fill()

    def __init__(self, name, tree):
        self.name = name
        self.tree = tree
        self._value = ROOT.vector('float')()
        self.tree.Branch(name, self._value)  # , f"{name}/F")
        Branch.all_branches.append(self)

    def __eq__(self, other):
        if len(self._value):
            return self._value[0] == other
        else:
            return False

    @property
    def value(self):
        assert len(self._value) < 2
        if len(self._value):
            return self._value[0]
        return None

    @value.setter
    def value(self, other):
        self.clear()
        self.fill(other)

    def __repr__(self):
        return f"<Branch, {self.name}, value=: {self._value}>"

    def fill(self, value):
        assert len(self._value) < 2
        self._value.push_back(float(value))

    def clear(self):
        self._value.clear()


class _Header:
    def __init__(self, file):
        file.readline()
        file.readline()
        self.title = file.readline()
        while line := file.readline():
            if not re.match(r"( +[0-9]\.[0-9]+E[+-][0-9]+){3}", line):
                break

        n_ids = list(map(int, line.split()[:11]))
        self.ids = []

        ids = []
        #  (bug fix) Replaced line below with regex that terminates header at correct position more generally
        # ids = list(map(int, " ".join([file.readline() for i in range(3)]).split()))
        while True:  # (Bug fix Jan/Germany2022)
            pos = file.tell()
            line = file.readline()
            if re.match('^( {0,7}[0-9]{1,2})+$', line):
                ids.extend(map(int, line.split()))
            else:
                file.seek(pos)
                break

        indicies = [0] + list(np.cumsum(n_ids))
        for i1, i2 in zip(indicies[:-1], indicies[1:]):
            self.ids.append(ids[i1: i2])


root_files = []  # for ROOT file persistence


def ptrac2root(ptrac_path: Union[Path, str], root_file_name=None, max_events: Union[None, int] = None, write_2_text=False,
               write_lookup_file=True) -> Tuple[Path, ROOT.TTree]:
    """
    Save MCNP PTRAC data to a ROOT TTree.

    Args:
        ptrac_path: Abs. path to PTRAC text file.
        root_file_name: Name of root TFile file containing Tree. None for automatic, which just adds ".root" to original file.
        max_events: Stop after writing this many events.
        write_2_text: Probably doesnm't work... but writes text file.
        write_lookup_file: Write a text file in current directory with info on MT values, term values, etc.

    Returns:
        (path to TFile containing Tree, TTree object)

    """
    ptrac_path = Path(ptrac_path)
    assert ptrac_path.exists()
    file = open(ptrac_path)
    header = _Header(file)

    if root_file_name is None:
        root_file_name = ptrac_path.name

    root_file_path = (ptrac_path.parent / root_file_name).with_suffix('.root')
    root_file = ROOT.TFile(str(root_file_path), 'recreate')
    root_files.append(root_file)
    tree = ROOT.TTree("tree", "tree")

    # mat is just an integer starting from 1, ie not the material number

    #  Below  is a dict mapping variable IDs to branch containers. Uninteresting variables,  e.g. _next_event, are
    #   are stings  and thus are not stored in the TTree.
    var_id_to_branches = {1: Branch('nps', tree),
                          2: '_next_event',
                          4: Branch('next_sur', tree),
                          7: "_next_event",
                          10: Branch('zaid', tree),
                          11: Branch('ntyn', tree),
                          12: Branch('surf', tree),
                          13: Branch('surf_theta', tree),
                          14: Branch('term', tree),
                          16: Branch('par', tree),
                          17: Branch('cell', tree),
                          20: Branch('x', tree),
                          21: Branch('y', tree),
                          22: Branch('z', tree),
                          23: Branch('dirx', tree),
                          24: Branch('diry', tree),
                          25: Branch('dirz', tree),
                          26: Branch('erg', tree),
                          27: Branch('wgt', tree),
                          28: Branch('time', tree)}

    def dict_get(ids) -> List[int, Union[Branch, str, None]]:

        """
        Returns a list of Branches corresponding to quantities in the order they appear in the PTRAC file.
        A KeyError represents a PTRAC entry that is either bugged or not of interest
        (since I didn't include it in `var_id_to_branches`).
        """
        out = []
        for _id in ids:
            try:
                out.append(var_id_to_branches[_id])
            except KeyError:
                out.append(None)
        return out
    try:
        patterns = {9000: (dict_get(header.ids[0]),),
                    1000: (dict_get(header.ids[1]), dict_get(header.ids[2])),
                    2000: (dict_get(header.ids[3]), dict_get(header.ids[4])),
                    3000: (dict_get(header.ids[5]), dict_get(header.ids[6])),
                    4000: (dict_get(header.ids[7]), dict_get(header.ids[8])),
                    5000: (dict_get(header.ids[9]), dict_get(header.ids[10]))}
    except IndexError as e:
        raise ValueError(f"Not able to process file! Are you sure this is an MCNP PTRAC file? ({e})")

    n_events = 0
    expected_event = 9000
    bnk_number_branch = Branch('bnk', tree)
    current_event_branch = Branch('event', tree)

    # re-orders the global set of branches for printing to text file. Not really important
    Branch.order_branches('nps', 'event', 'cell', 'x', 'y', 'z', 'erg', 'par', 'time', 'bnk', 'term')

    text_file = None
    if write_2_text:
        text_file = open(ptrac_path.with_suffix('.csv'), 'w')
        text_file.write(Branch.text_heading())

    finish_point = file.tell() if max_events is None else max_events
    current_position = (lambda: file.tell()) if not max_events else (lambda: n_events)
    proj = ProgressReport(finish_point)  # for printing remaining time to stdout

    linenum_debug = 1

    while max_events is None or n_events < max_events:
        # For each event type (SRC, BNK, stc) `pattern` is a mapping between variable positions
        #  (of the first (and second) event line(s)) in the PTRAC file, and their corresponding TTree Branch instances.
        pattern = patterns[1000 * (expected_event // 1000)]

        lines = [file.readline() for _ in range(len(pattern))]  # for all evt types  except 1000, read two lines.

        linenum_debug += len(pattern)

        if lines[-1] == '':  # end of file
            break
        if expected_event == 9000:  # end of particle history
            Branch.clear_all_except()  # don't spare any branches, clear them all
        else:
            Branch.clear_all_except(['nps'])  # nps shouldn't be reset until end of particle history

        current_event_branch.value = expected_event

        for p, line in zip(pattern, lines):
            for branch_or_flag, value in zip(p, line.split()):
                if isinstance(branch_or_flag, Branch):
                    branch_or_flag.fill(value)
                elif branch_or_flag == '_next_event':  # this is not a branch, but must conform to the paradigm
                    if expected_event // 1000 == 2:
                        bnk_number_branch.fill(expected_event % 1000)
                    expected_event = int(value)
                else:    # Not a branch or otherwise interesting value. No processing needed.
                    assert branch_or_flag is None, f"BAD TROUBLE. type of branch_or_flag: {type(branch_or_flag)}"

        if current_event_branch.value != 9000.:
            if text_file:
                text_file.write(Branch.current_state2text())
            Branch.fill_tree()
            n_events += 1

        proj.log(current_position())
        if max_events is not None and n_events > max_events:
            break
    tree.Write()
    if text_file:
        text_file.close()
    file.close()

    copied_lookup_file_path = ptrac_path.parent/'lookup.txt'
    if write_lookup_file and not copied_lookup_file_path.exists():
        lookup_file_path = Path(cwd / 'lookup.txt')
        shutil.copy(lookup_file_path, copied_lookup_file_path)

    Branch.all_branches = []
    return root_file_path, tree


class TTreeHelper:
    root_files = []

    def __init__(self, path):
        path = Path(path)
        assert path.exists()
        self.root_file = ROOT.TFile(str(path))
        TTreeHelper.root_files.append(self.root_file)
        self.tree = self.root_file.Get('tree')

        self.__i__ = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.tree.GetEntry(self.__i__)

        self.__i__ += 1
        if self.__i__ >= self.tree.GetEntries():
            raise StopIteration

        return self

    @property
    def is_nucleus(self):
        """Helium or larger"""
        return 33 <= self.tree.par[0] <= 37

    @property
    def event(self):
        return int(self.tree.event[0])

    @property
    def is_term(self):
        return self.tree.event[0] == 5000

    @property
    def is_src(self):
        return self.tree.event[0] == 1000

    @property
    def dirx(self):
        return self.tree.dirx[0]

    @property
    def diry(self):
        return self.tree.diry[0]

    @property
    def dirz(self):
        return self.tree.dirz[0]

    @property
    def x(self):
        return self.tree.x[0]

    @property
    def y(self):
        return self.tree.y[0]

    @property
    def z(self):
        return self.tree.z[0]

    @property
    def energy(self):
        return self.tree.erg[0]

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    @property
    def cell(self):
        return self.tree.cell[0]

    @property
    def weight(self):
        return self.tree.wgt[0]

    @property
    def time(self):
        return self.tree.time[0]


if __name__ == '__main__':
    ptrac2root(cwd / 'ptrac', max_events=10, write_2_text=True)
    tb = ROOT.TBrowser()
    ROOT_loop()
