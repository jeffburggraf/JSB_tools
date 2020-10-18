import numpy as np
from myTools.outp_reader import OutP
from pathlib import Path
from matplotlib import pyplot as plt
import re
import warnings
from functools import cached_property


def _split_line(line):
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


class InputFile:
    def __init__(self, **kwargs):
        assert "inp_file_path" in kwargs, "Must supply 'inp_file_path' keyword argument."
        self.inp_file_path = Path(kwargs["inp_file_path"])
        assert self.inp_file_path.exists(), "Cannot find input deck:\n{0}".format(self.inp_file_path)

        self.is_mcnp = kwargs.get("is_mcnp", True)

        self.__num_writes__ = 0  # Todo: multiple inp files.

        self.write_directory = kwargs.get("write_directory", None)
        if self.write_directory is None:
            self.write_directory = self.inp_file_path.parent
        else:
            self.write_directory = Path(self.write_directory)
            assert self.write_directory.exists(), "Provided 'write_directory' does not exist:\n{0}".format(self.write_directory)

        self.__names__written_so_far__ = []

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

    def __split_new_lines__(self):
        new_inp_lines = self.__new_inp_lines__[:]
        self.__new_inp_lines__ = []
        for line in new_inp_lines[:self.MCNP_EOF]:
            self.__new_inp_lines__.append(_split_line(line))
        self.__new_inp_lines__.extend(new_inp_lines[self.MCNP_EOF:])

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

        if len(l_stripped_line) > 0 and l_stripped_line[0].lower() == "c":
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
                            new_line += "{0}".format(eval(exp_to_process, dict_of_globals))
                        except Exception as e:
                            err = _exception(str(e))
                            exception_msg += err

                    exp_to_process = None
                else:
                    assert isinstance(exp_to_process, str)
                    exp_to_process += s
        if exp_to_process is not None:
            exception_msg += (_exception("Missing @"))

        if comment is not None:
            new_line += "${0}".format(comment)

        return new_line, exception_msg

    def write_inp_in_scope(self, dict_of_globals, new_file_name=None, new_file_dir=None):
        assert len(self.__new_inp_lines__) == 0
        exception_msgs = ""
        if self.is_mcnp:
            lines = self.inp_lines[:self.MCNP_EOF]
        else:
            lines = self.inp_lines

        for line_num, line in enumerate(lines):
            line_num += 1
            new_line, ex_msg = self.__process_line__(line, line_num, dict_of_globals)
            exception_msgs += ex_msg
            self.__new_inp_lines__.append(new_line)

        if len(exception_msgs) > 0:
            raise Exception(exception_msgs)

        self.__split_new_lines__()
        self.__new_inp_lines__.extend(self.inp_lines[self.MCNP_EOF:])
        self.__write_file__(new_file_name, new_file_dir)
        self.__new_inp_lines__ = []

    def __write_file__(self, new_file_name, new_file_dir):
        assert len(self.__new_inp_lines__) > 0, "Must call 'write_inp_in_scope' before '__write_file__'"

        if new_file_dir is None:
            new_file_dir = self.write_directory
        else:
            new_file_dir = Path(new_file_dir)
            assert new_file_dir.exists(), "Specified directory doesn't exist."

        if new_file_name is None:
            new_file_name = "{0}_{1}".format(self.__num_writes__, self.inp_file_path.name)
        assert new_file_name not in self.__names__written_so_far__,\
            "Attempted to create file '{0}' twice".format(new_file_name)

        if new_file_dir == self.inp_file_path.parent:
            assert new_file_name != self.inp_file_path.name, \
            "Cannot name new input file the same as source input file."

        if self.cycle_rnd_seed is not False:
            self.cycle_random_number_mcnp()

        with open(new_file_dir/new_file_name, "w") as f:
            for line in self.__new_inp_lines__:
                f.write(line)

        self.__num_writes__ += 1
        self.__names__written_so_far__.append(new_file_name)

    @classmethod
    def mcnp_input_deck(cls, inp_file_path, write_directory=None, cycle_rnd_seed=False):
        return InputFile(inp_file_path=inp_file_path, is_mcnp=True, write_directory=write_directory,
                         cycle_rnd_seed=cycle_rnd_seed)

    @classmethod
    def phits_input_deck(cls, inp_file_path, write_directory=None, cycle_rnd_seed=False):
        return InputFile(inp_file_path=inp_file_path, is_mcnp=False, write_directory=write_directory,
                         cycle_rnd_seed=cycle_rnd_seed)

if __name__ == "__main__":
    p = Path(__file__).parent/"test.inp"
    i = InputFile.mcnp_input_deck(p, cycle_rnd_seed=True)
    from FFandProtonSims.GlobalValues import *
    imp = "dokdkof"
    gas_density = 100
    i.write_inp_in_scope(globals())
    i.write_inp_in_scope(globals())


