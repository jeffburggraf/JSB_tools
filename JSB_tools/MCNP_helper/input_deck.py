from __future__ import annotations
from pathlib import Path
import re
import warnings
import platform
import os
import stat
from atexit import register, unregister
from uncertainties import UFloat
from numbers import Number
import pickle
from JSB_tools.MCNP_helper.geometry.geom_core import get_comment, Cell, MCNPNumberMapping, PHITSOuterVoid
import numpy as np
from typing import Dict, List, Union, Tuple, Iterable

NDIGITS = 7  # Number of significant digits to round all numbers to when used in input decks.


def _split_line(line):
    # todo: do this for phits too. Max line length is 200, where \ (or /) is used as the continue line symbol.
    #  Also, continue line must begin with 5 spaces
    _line0 = line
    assert isinstance(line, str)
    l_striped_line = line.lstrip()

    if len(l_striped_line) == 0:
        return line  # blank line

    if len(l_striped_line) and l_striped_line[0].lower() == "c":
        return line  # comment line
    comment_split = re.split(r"\$", line)

    line = comment_split[0]
    if len(comment_split) > 1:
        comment = "$".join(comment_split[1:])
    else:
        comment = ""
    max_columns = 78

    # sections of line split at a white space character such that each is less than 'max_columns' chars long.
    valid_lines = []
    while len(line) != 0:
        if len(line) > max_columns:  # line must be split.
            for index in range(max_columns, 0, -1):  # Starting from max_columns, loop backwards until first white space
                char = line[index]  # current Sub-line is line[0:index]. Looking for white space char at which to split
                if char in [" ", "\t"]:  # is char white space? then add sub-line to valid_lines then remove sub-line
                    valid_lines.append(line[:index].lstrip())
                    assert len(valid_lines[-1]) <= max_columns  # check for validity
                    line = line[index+1:]  # remove sub-line
                    break
            else:
                e = AssertionError("Cannot breakup following line to series of lines <=80 columns:\n{0}".format(_line0))
                raise e
        else:  # line is fine. Don't split
            valid_lines.append(line)
            break

    fixed = "&\n ".join(valid_lines)
    if len(comment) != 0:
        fixed += "$" + comment

    return fixed


class MCNPSICard:
    used_numbers = set()

    @staticmethod
    def clear_all():
        MCNPSICard.used_numbers = set()

    @staticmethod
    def get_next_si_number():
        si_card_number = 1
        while si_card_number in MCNPSICard.used_numbers:
            si_card_number += 1
        MCNPSICard.used_numbers.add(si_card_number)

        return si_card_number

    @staticmethod
    def __check_arrays__(array, var_name):
        assert hasattr(array, '__iter__'), '`{}` must be an iterator'.format(var_name)
        if isinstance(array[0], UFloat):
            array = [f.n for f in array]
        assert all([isinstance(x, Number) for x in array]), 'All `{0}` mut be a number, not {1}'.format(var_name,
                                                                                                        [x for x in array]) #type(array[0]))

        return array

    def __init__(self, variable_values, variable_probs, si_card_number=None, discrete=False, cell_vol_dist=False):
        self.variable_values = MCNPSICard.__check_arrays__(variable_values, 'variable_values')
        self.variable_probs = MCNPSICard.__check_arrays__(variable_probs, 'variable_probs')

        assert len(self.variable_values) == len(self.variable_probs), 'Length of `variable_probs` and ' \
                                                                      '`variable_values` must be equal'
        assert isinstance(discrete, bool), '`discrete` arg must be True or False.'

        if si_card_number is None:
            si_card_number = MCNPSICard.get_next_si_number()
        else:
            assert isinstance(si_card_number, int), '`si_card_number` must be an integer.'
        self.si_card_number = int(si_card_number)

        if discrete:
            self.si_option = 'L'
        else:
            self.si_option = 'A'

        assert isinstance(cell_vol_dist, bool), '`cell_vol_dist` must be True of False'
        if cell_vol_dist is True:
            self.sp_option = 'V'
        else:
            self.sp_option = ' ' * len(self.si_option)

        self.card = 'SI{0} {1} {2}\n'.format(self.si_card_number, self.si_option,
                                             ' '.join(map(lambda x: f'{x:.4e}', self.variable_values)))
        self.card += f"SP{self.si_card_number} {self.sp_option} {' '.join(map(lambda x: f'{x:.4e}', self.variable_probs))}"


    @classmethod
    def from_function(cls, function, variable_values, si_card_number=None, discrete=False, *func_args, **func_kwargs):
        assert callable(function), '`function` must be a callable.'
        variable_probs = MCNPSICard.__check_arrays__([function(x, *func_args, **func_kwargs) for x in variable_values],
                                                     'Values returned from function')

        return cls(variable_values, variable_probs, si_card_number=si_card_number, discrete=discrete)

    @classmethod
    def function_of_si(cls, si_independent, function, si_card_number=None, discrete=False):
        assert isinstance(si_independent, MCNPSICard), '`si_independent` must be MCNPSICard instance'
        assert callable(function), '`function` must be a callable'
        variable_values = [function(x) for x in si_independent.variable_values[1:]]

        if discrete:
            option = ' L '
        else:
            option = ' '

        jis = MCNPSICard.__check_arrays__(variable_values, 'variable_values')
        out = cls([0], [0], si_card_number)
        card = 'DS{0}{1}{2}'.format(out.si_card_number, option, ' '.join(map(str, jis)))
        out.card = card
        return out

    def __repr__(self):
        return self.card


class TallyErgBinsMixin:
    def __init__(self):
        self.erg_bins_spec = None

    @property
    def erg_bins_array(self):
        if self.erg_bins_spec is None:
            return None
        if isinstance(self.erg_bins_spec, np.ndarray):
            return self.erg_bins_spec
        elif isinstance(self.erg_bins_spec, tuple):
            return np.linspace(self.erg_bins_spec[0], self.erg_bins_spec[2], self.erg_bins_spec[1] + 1)

    def set_erg_bins(self, erg_min=None, erg_max=None, n_erg_bins=None, erg_bins_array=None, _round=3):
        if erg_bins_array is not None:
            assert hasattr(erg_bins_array, '__iter__')
            assert len(erg_bins_array) >= 2
            assert erg_min == erg_max == n_erg_bins, 'Can either specify energy bins by an array of values, or by ' \
                                                     'erg_min, erg_max, and the number of bins, n_erg_bins.'
            self.erg_bins_spec = np.array([round(x, _round) for x in erg_bins_array])

        else:
            assert erg_bins_array is None, "Can't specify bins by using an array and a min, max, and number of bins." \
                                           "Set erg_bins_array to None."
            self.erg_bins_spec = (erg_min, n_erg_bins, erg_max)

    def get_erg_bins_card(self):
        if self.erg_bins_spec is None:
            return ''
        else:
            out = f'\nE{self.tally_number} '
            if isinstance(self.erg_bins_spec, np.ndarray):
                out += ' '.join(map(str, self.erg_bins_array))
            elif isinstance(self.erg_bins_spec, tuple):
                emin, nbins, emax = self.erg_bins_spec
                out += f"{emin} {nbins - 1}i {emax}"

            return out


class TallyBase:
    # all_f4_tallies = MCNPNumberMapping("F4Tally", 1)
    all_tallies = {}

    def __init__(self, tally_base_number, cell_or_number=None, tally_number=None, tally_name=None, tally_comment=None):
        self.__tally_base_number__ = tally_base_number
        if tally_number is not None:
            assert isinstance(tally_number, (int, str))
            assert str(tally_number)[-1] == str(tally_base_number), f"F{tally_base_number} tally number must end in " \
                                                                    f"{tally_base_number}, not like '{tally_number}'"
            tally_number = str(tally_number)
            assert len(tally_number)>1, "Only use tally numbers with more than one digit."
            tally_number = int(tally_number[:-1])

        if tally_base_number not in TallyBase.all_tallies:
            TallyBase.all_tallies[tally_base_number] = MCNPNumberMapping(f"F{tally_base_number}Tally", 1)

        self._tally_number = tally_number

        if isinstance(cell_or_number, Cell):
            self.cell = cell_or_number
            self.cell_number = self.cell.cell_number
        else:
            self.cell = None
            self.cell_number = cell_or_number

        self.__name__ = tally_name
        self.tally_comment = tally_comment

        TallyBase.all_tallies[tally_base_number][tally_number] = self

    @property
    def tally_number(self):
        return int(f"{self._tally_number}{self.__tally_base_number__}")

    @staticmethod
    def clear():
        TallyBase.all_tallies = {}

    @staticmethod
    def get_all_tally_cards():
        outs = []
        for _, mapping in TallyBase.all_tallies.items():
            for tally in mapping.values():
                outs.append(tally.tally_card)
        return '\n'.join(outs)


class F4Tally(TallyBase, TallyErgBinsMixin):
    def __init__(self, cell_or_number: Union[Cell, int, str], particle: str, tally_number=None,
                 tally_name=None, tally_comment=None):
        """
        Args:
            cell_or_number: Cell instance for which the tally_n will be applied (or cell number)
            particle: MCNP particle designator
            tally_number: Must end in a 4. Or, just leave as None and let the code pick for you
            tally_name:  Used in JSB_tools.outp_reader to fetch tallies by name.
            tally_comment:
        """

        super().__init__(4, cell_or_number, tally_number, tally_name, tally_comment)
        super(TallyBase, self).__init__()

        assert isinstance(particle, str), 'Particle argument must be a string, e.g. "n", or, "p", or, "h"'
        # self.erg_bins_array = None
        self.__modifiers__ = []

        assert isinstance(particle, str), '`particle` argument must be a string.'
        particle = particle.lower()
        for key, value in {'photons': 'p', 'protons': 'h', 'electrons': 'e', 'neutrons': 'n',
                           'alphas': 'a'}.items():
            if particle == key or particle == key[:-1]:
                particle = value
                break

        self.particle = particle

    @property
    def name(self):
        return self.__name__

    def add_fission_rate_multiplier(self, mat: int) -> None:
        """
        Makes this tally_n a fission xs rate tally_n [fissions/(src particle)/cm3] for neutrons and protons
        Args:
            mat: Material number

        Returns: None

        """
        mod = 'FM{} -1 {mat} -2'.format(self.tally_number, mat=mat)
        self.__modifiers__.append(mod)

    @property
    def tally_card(self):
        comment = get_comment(self.tally_comment, self.__name__)
        out = 'F{num}:{par} {cell} {comment}'.format(num=self.tally_number, par=self.particle,
                                                     cell=self.cell_number, comment=comment)
        out += self.get_erg_bins_card()

        # if self.erg_bins_array is not None:
        #     out += '\nE{num} {bins}'.format(num=self.tally_number, bins=' '.join(map(str, self.erg_bins_array)))

        if len(self.__modifiers__):
            out += '\n' + '\n'.join(self.__modifiers__)
        return out


class F8Tally:
    #     todo: Finish this
    @staticmethod
    def get_bins_card(emin, emax, nbins=None, binwidth=None, epsilon_bin=1E-5):
        """
        Return the string for creating uniform bins from emin to emax.
        Args:
            emin:
            emax:
            nbins:
            binwidth

        Returns:

        """
        assert emin > 0

        if binwidth is not None:
            assert nbins is None
            nbins = int((emax - emin)/binwidth)
        else:
            assert nbins is not None
            nbins = int(nbins)

        if emin == 0:
            return f'0 {epsilon_bin:.5g} {nbins - 1}i {emax:.5g}'
        else:
            return f'0 {emin:.5g} {nbins - 1}i {emax:.5g}'


class InputDeck:
    def __init__(self, cleanup_msg=True, *args, **kwargs):
        assert '__internal__' in kwargs, '\nTo create an input file, use one of the two factory methods:\n' \
                                         '\tInputDeck.mcnp_input_deck(*args, **kwargs)\n' \
                                         '\tInputDeck.phits_input_deck(*args, **kwargs)'
        assert "inp_file_path" in kwargs, "Must supply 'inp_file_path' keyword argument."
        self.inp_file_path = Path(kwargs["inp_file_path"])
        assert self.inp_file_path.exists(), "Cannot find input deck:\n{0}".format(self.inp_file_path)

        self.__has_called_write_inp_in_scope__ = False  # Used to control warnings

        self.warn_msg_in_cleanpy = kwargs.get('warn_msg_in_cleanpy', True)
        self.is_mcnp = kwargs.get("is_mcnp", True)
        self.sch_cmd = kwargs.get('sch_cmd')

        self.__num_writes__ = 0

        self.__auto_directory__ = False
        self.__auto_file_names__ = False

        self.platform = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'

        self.gen_run_script = kwargs.get("gen_run_script", True)
        assert isinstance(self.gen_run_script, bool)

        new_file_dir = kwargs.get('new_file_dir')
        if new_file_dir is None:
            new_file_dir = self.inp_file_path.parent
        else:
            new_file_dir = Path(new_file_dir)
            assert new_file_dir.parent.exists(), f'`new_file_dir` must be a directory that exists.\n' \
                                                                    f'"{new_file_dir}"'
            new_file_dir.mkdir(exist_ok=True)
        self.inp_root_directory = new_file_dir
        self.directories_created = []

        self.__files_written_so_far__ = []

        self.cycle_rnd_seed = kwargs.get("cycle_rnd_seed")
        if self.cycle_rnd_seed is not False:
            assert self.cycle_rnd_seed is True, "'cycle_rnd_seed' must be a bool."
            self.cycle_rnd_seed = 1

        self.MCNP_EOF = None

        self.inp_lines = open(self.inp_file_path).readlines()

        self.__new_inp_lines__ = []

        if self.is_mcnp:
            assert len(self.inp_lines) >= 2, "MCNP input deck has less than two lines. Something is wrong."

            # heck for double blank line EOF (as per MCNP input format). Set self.MCNP_EOF.
            for index, (line1, line2) in enumerate(zip(self.inp_lines[:-1], self.inp_lines[1:])):
                if len(line1.split()) == len(line2.split()) == 0:
                    self.MCNP_EOF = index
                    break
            else:
                self.MCNP_EOF = len(self.inp_lines)

        if cleanup_msg:
            register(self.__del)

    @staticmethod
    def PHITS2MCNP_plotter(directory, new_file_name, dict_of_globals):
        dict_of_globals['PHITSOuterVoid'] = PHITSOuterVoid
        i = InputDeck.mcnp_input_deck(Path(__file__).parent/'PHITS2MCNP_geometry.inp', directory, cleanup_msg=False)
        i.write_inp_in_scope(dict_of_globals, new_file_name, )

    def __split_new_lines__(self):
        title_line = self.__new_inp_lines__[0]  # Don't split the title line. MCNP sucks and won't allow it
        if len(title_line) > 79:
            warnings.warn('Title line cannot be greater than 79 chars long due to MCNP limitations. Truncating\n'
                          'with comment line'.format(title_line))
            titles = [title_line[i*78:(i+1)*78] for i in range(len(title_line)//78)]
            titles.append('\n')
            title_line = '\nc '.join(titles)

        new_inp_lines = self.__new_inp_lines__[1:]
        self.__new_inp_lines__ = []
        for line in new_inp_lines[:self.MCNP_EOF]:
            split_lines = line.split('\n')
            # In case evaluated code returns multiple lines, append on a line by line basis
            self.__new_inp_lines__.extend('\n'.join([_split_line(l) for l in split_lines]))

        self.__new_inp_lines__.extend(new_inp_lines[self.MCNP_EOF:])
        self.__new_inp_lines__.insert(0, title_line)

    def cycle_random_number_mcnp(self):
        assert isinstance(self.cycle_rnd_seed, int)
        for index, line in enumerate(self.__new_inp_lines__):
            line = line.lower().lstrip()
            _m = re.match("rand.+(seed *= *[0-9]+).*", line)
            if _m:
                s = "seed = {0}".format(self.cycle_rnd_seed)
                new_line = re.sub("seed *= *[0-9]+", s, line)

                self.__new_inp_lines__[index] = new_line
                break
        else:
            for index in range(len(self.__new_inp_lines__)-1, 0, -1):
                line = self.__new_inp_lines__[index]
                if len(line.split()) > 0:
                    new = ""
                    if line[-1] != "\n":
                        new += "\n"
                    new += "rand seed = {0}\n".format(self.cycle_rnd_seed)
                    self.__new_inp_lines__.insert(index+1, new)
                    break
        self.cycle_rnd_seed += 2

    def __process_line__(self, old_line, line_number, dict_of_globals):
        l_stripped_line = old_line.lstrip()
        exception_msg = ""

        if len(old_line.lstrip()) == 0:
            return old_line, exception_msg

        if re.match('c .*', l_stripped_line):  # is comment line. No processing
            return old_line, exception_msg

        if "$" in old_line:
            dollar_split = old_line.split('$')
            comment = "".join(dollar_split[1:])
            old_line = dollar_split[0]
        else:
            comment = None

        f_name = Path(self.inp_file_path.parent.name) / self.inp_file_path.name

        def _exception(msg):
            return "\n\t{0}\nIn line {1} in file {2}:\n{3}\n\n".format(msg, line_number, f_name, old_line)

        new_line = ""
        exp_to_process = None

        for s in old_line:
            if exp_to_process is None:
                if s == "@":
                    exp_to_process = ""
                else:
                    new_line += s
            else:
                if s == "@":
                    if len(exp_to_process) == 0:
                        warnings.warn("Empty embedded code in line {0} in file {1} ".format(line_number, f_name))
                    else:
                        try:
                            evaluated = eval(exp_to_process, dict_of_globals)
                        except Exception as e:
                            err = _exception(str(e))
                            exception_msg += err
                            unregister(InputDeck.__del)  # Don't print closing messages if exception raised.
                        else:
                            if isinstance(evaluated, float):
                                _fmt = f'.{NDIGITS}g'
                                evaluated = f"{evaluated:{_fmt}}"

                            new_line += "{0}".format(evaluated)

                    exp_to_process = None
                else:
                    assert isinstance(exp_to_process, str)
                    exp_to_process += s
        if exp_to_process is not None:
            exception_msg += (_exception("Missing @"))

        if comment is not None:
            new_line += "${0}".format(comment)

        return new_line, exception_msg

    def __create_directory_if_needed__(self, new_file_name):
        new_inp_directory = self.inp_root_directory / new_file_name
        self.directories_created.append(new_inp_directory)

        if not new_inp_directory.exists():
            print("Creating new directory: {0}".format(new_inp_directory))
        new_inp_directory.mkdir(exist_ok=True)

        assert new_inp_directory.exists(), "Directory '{0}' doesn't exist.".format(new_inp_directory)
        return new_inp_directory

    @staticmethod
    def __pickle_globals__(new_file_full_path: Path, dict_of_globals: dict):
        def check_type(x):
            if isinstance(x, (Number, str, bool)):
                return True
            else:
                return False
        new_file_full_path = Path(new_file_full_path)
        new_f_path = new_file_full_path.with_suffix('.pickle')
        print('Saved globals to {}'.format(new_f_path))
        with open(new_f_path, 'wb') as f:
            for k, var in dict_of_globals.items():
                if check_type(var):
                    try:
                        pickle.dump((k, var), f)
                    except TypeError:
                        pass

    def get_inp_path(self, new_file_name):
        if new_file_name is None:
            new_file_name = self.inp_file_path.name
            _m = re.match(r"(.+)\..+", new_file_name)  # remove extension
            if _m:
                new_file_name = _m.groups()[0]

            new_file_name = "{0}_{1}".format(self.__num_writes__, new_file_name)
        new_inp_directory = self.__create_directory_if_needed__(new_file_name)
        new_file_full_path = new_inp_directory / new_file_name
        return new_file_full_path

    def write_inp_in_scope(self, dict_of_globals: Union[dict, List[dict]], new_file_name=None,
                           script_name="cmd", overwrite_globals=None, **mcnp_or_phits_kwargs) -> Path:
        """
        Creates and fills an input deck according to a dictionary of values. Usually, just use globals().

        If the return path is needed but you don't want to run write_inp_in_scope, call self.get_inp_path().

        Args:
            dict_of_globals: Dict of scope. Usually just use globals(). Can also be a list of dicts.
            new_file_name: name of generated input file
            script_name: Name of script to obe created that runs (all) simulation(s) automatically.
            overwrite_globals: dictionary of variable names and values that will take precedence over globals()
            **mcnp_or_phits_kwargs: Command line args for MCNP or PHITS

        Returns: new_file_name

        """
        self.__has_called_write_inp_in_scope__ = True

        if isinstance(dict_of_globals, list):
            for d in dict_of_globals[1:]:
                for k, v in d.items():
                    dict_of_globals[0][k] = v

            dict_of_globals = dict_of_globals[0]

        assert len(self.__new_inp_lines__) == 0

        new_file_full_path = self.get_inp_path(new_file_name)

        exception_msgs = ""
        if self.is_mcnp:
            lines = self.inp_lines[:self.MCNP_EOF]
        else:
            lines = self.inp_lines

        if overwrite_globals is not None:
            dict_of_globals = {k: (v if k not in overwrite_globals else overwrite_globals[k])
                               for k, v in dict_of_globals.items()}

        for line_num, line in enumerate(lines):
            line_num += 1
            new_line, ex_msg = self.__process_line__(line, line_num, dict_of_globals)
            exception_msgs += ex_msg
            self.__new_inp_lines__.append(new_line)

        if len(exception_msgs) > 0:
            raise Exception(exception_msgs)

        if self.is_mcnp:
            self.__split_new_lines__()
            self.__new_inp_lines__.extend(self.inp_lines[self.MCNP_EOF:])

        if self.gen_run_script is True:
            self.__append_cmd_to_run_script__(script_name, new_file_full_path, mcnp_or_phits_kwargs)
            if self.sch_cmd:
                self.__append_cmd_to_run_script__(script_name, new_file_full_path, mcnp_or_phits_kwargs,
                                                  csh_cmd=True)
        self.__write_file__(new_file_full_path)

        self.__pickle_globals__(new_file_full_path, dict_of_globals)

        self.__new_inp_lines__ = []
        return new_file_full_path

    def __write_file__(self, new_file_full_path):
        assert len(self.__new_inp_lines__) > 0, "Must call 'write_inp_in_scope' before '__write_file__'"
        # new_inp_directory = new_file_full_path.parent
        # new_inp_name = new_file_full_path.name
        if self.cycle_rnd_seed is not False:
            if self.is_mcnp:
                self.cycle_random_number_mcnp()
            else:
                raise NotImplementedError

        with open(new_file_full_path, "w") as f:
            for line in self.__new_inp_lines__:
                f.write(line)

        self.__num_writes__ += 1
        self.__files_written_so_far__.append(new_file_full_path)

    def __append_cmd_to_run_script__(self, script_name, new_file_full_path, mcnp_or_phits_kwargs, csh_cmd=False):
        if csh_cmd:
            f_path = (self.inp_root_directory / script_name).with_suffix('.csh')
        else:
            f_path = (self.inp_root_directory / script_name).with_suffix('.sh')

        # new_file_full_path = new_file_full_path.relative_to(self.inp_root_directory)
        if mcnp_or_phits_kwargs is None:
            script_kwargs = ""
        else:
            assert isinstance(mcnp_or_phits_kwargs, dict)
            script_kwargs = " ".join(["{0}={1}".format(key, value) for key, value in mcnp_or_phits_kwargs.items()])
        new_file_name = new_file_full_path.name
        if csh_cmd:
            cd_path = new_file_full_path.parent.relative_to(self.inp_root_directory)
        else:
            cd_path = new_file_full_path.parent

        cd_cmd = "cd {0}".format(cd_path)

        run_cmd = "{0};mcnp6 i={1} {2}".format(cd_cmd, new_file_name, script_kwargs) if self.is_mcnp else \
            "{0};phits.sh {1} {2}".format(cd_cmd, new_file_name, script_kwargs)
        if csh_cmd:
            new_cmd = "{0} < /dev/null > ! log_{1}.txt &\n"\
                .format(run_cmd, new_file_full_path.name)
        else:
            if self.platform == "Darwin":
                new_cmd = "osascript -e 'tell app \"Terminal\"\ndo script \"{0} 2>&1 | tee -i log_{1}.txt;exit\"\nend " \
                          "tell'\n".format(run_cmd, new_file_full_path.name)
            elif self.platform == "Linux":
                new_cmd = "gnome-terminal -x sh -c '{0} 2>&1 | tee -a -i log_{1}.txt;'\n".format(run_cmd,
                                                                                                 new_file_full_path.name)
            elif self.platform == "Windows":
                warnings.warn("Currently no implementation of the creation of a .bat file to automatically run the "
                              "simulations on Windows. ")
                return
            else:
                warnings.warn("Run script not generated. Platform not supported.")
                return

        with open(f_path, "w" if self.__num_writes__ == 0 else "a") as f:
            if self.__num_writes__ == 0:
                if csh_cmd:
                    f.write('#!/bin/csh -f\n')
            f.write(new_cmd)

        if self.__num_writes__ == 0:
            if self.platform in ["Linux", "Darwin"]:
                st = os.stat(f_path)
                os.chmod(f_path, st.st_mode | stat.S_IEXEC)

    def __del(self):
        if self.__has_called_write_inp_in_scope__:
            if self.is_mcnp:
                clean_fname = f"Clean-{self.inp_root_directory.name}.py"

                with open(Path(self.inp_root_directory)/clean_fname, 'w') as clean_file:
                    print(f"Created '{clean_fname}'. Run this script to remove PTRAC, OUTP, etc.")
                    import inspect
                    cmds = inspect.getsource(__clean__)
                    rel_directories_created = [d.relative_to(self.inp_root_directory) for d in self.directories_created]
                    cmds += '\n\n' + "paths = {}\n".format(list(map(str, rel_directories_created)))
                    cmds += '__clean__(paths, {})\n'.format(self.warn_msg_in_cleanpy)
                    clean_file.write(cmds)

            print('Run the following commands in terminal to automatically run the simulation(s) just prepared:\n')
            print('cd {0}\n./cmd.sh'.format(self.inp_root_directory))
            print('*or if using csh: ./cmd.csh   \n')

            if self.is_mcnp:
                print('Created "Clean.py". Running this script will remove all outp, mctal, ptrac, ect.')

        else:
            warnings.warn('\nTo evaluate and write input file use\n\ti.write_inp_in_scope(globals(), [optional args])\n'
                          'where `i` is an InputDeck instance.')

    @classmethod
    def mcnp_input_deck(cls, inp_file_path, new_file_dir=None, cycle_rnd_seed=False, gen_run_script=True,
                        warn_msg_in_cleanpy=True, sch_cmd=False, cleanup_msg=True):
        """

        Args:
            inp_file_path:
            new_file_dir:
            cycle_rnd_seed:
            gen_run_script:
            warn_msg_in_cleanpy:
            sch_cmd:
            cleanup_msg:

        Returns:

        """
        return InputDeck(inp_file_path=inp_file_path, new_file_dir=new_file_dir, cycle_rnd_seed=cycle_rnd_seed,
                         gen_run_script=gen_run_script, is_mcnp=True, __internal__=True,
                         warn_msg_in_cleanpy=warn_msg_in_cleanpy, sch_cmd=sch_cmd, cleanup_msg=cleanup_msg)

    @classmethod
    def phits_input_deck(cls, inp_file_path, new_file_dir=None, cycle_rnd_seed=False, gen_run_script=True,
                         sch_cmd=False):
        return InputDeck(inp_file_path=inp_file_path, new_file_dir=new_file_dir, cycle_rnd_seed=cycle_rnd_seed, gen_run_script=gen_run_script,
                         is_mcnp=False, __internal__=True, sch_cmd=sch_cmd)


def __clean__(paths, warn_message):
    import re
    import platform
    import os
    from pathlib import Path
    from tkinter import messagebox, Tk
    from send2trash import send2trash
    cwd = Path(__file__).parent

    try:
        root = Tk()
        root.withdraw()
    except Exception:
        user = input(f'Remove outp, ptrac, etc from "{cwd}" (y or n):')
        if user.lstrip().rstrip().lower() == 'y':
            return
    else:
        if warn_message:
            yes_or_no = messagebox.askquestion('Deleting files!',
                                               'Are you sure you want to clean files (outp, ptrac, runtpe, etc.) from '
                                               f'simulations performed in:\n "{cwd}" ?')
            if not (yes_or_no == "yes"):
                return

    m = re.compile(r"(ptra[a-z]$)|(runtp[a-z]$)|(runtp[a-z]\.h5$)|(mcta[a-z]$)|(out[a-z]$)|(comou[a-z]$)|(meshta[a-z]$)|(mdat[a-z]$)|"
                   r"(plot[m-z]\.ps)")

    for p in paths:
        sim_directory = cwd / p
        for f_path in Path(sim_directory).iterdir():
            if m.match(f_path.name):
                trash_name = str(f_path.relative_to(f_path.parents[2]))

                if platform.system() == 'Windows':
                    trash_name = trash_name.replace('\\', '..')
                else:
                    trash_name = trash_name.replace('/', '..')

                new_path = f_path.parent / trash_name
                os.rename(f_path, new_path)
                send2trash(new_path)


if __name__ == "__main__":
    f = F4Tally(10, 'p')
    f.set_erg_bins(erg_bins_array=[1,2,3])
    print(f.tally_card)
    p = Path('/Users/burgjs/PycharmProjects/miscMCNP/detectorModels/GRETA0/sims/Co60-10cm/outp')
    p.relative_to(p.parent)
