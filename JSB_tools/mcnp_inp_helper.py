from pathlib import Path
import re
import warnings
import platform
import os
import stat


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


class InputFile:
    def __init__(self, **kwargs):
        assert "inp_file_path" in kwargs, "Must supply 'inp_file_path' keyword argument."
        self.inp_file_path = Path(kwargs["inp_file_path"])
        assert self.inp_file_path.exists(), "Cannot find input deck:\n{0}".format(self.inp_file_path)

        self.is_mcnp = kwargs.get("is_mcnp", True)

        self.__num_writes__ = 0

        self.__auto_directory__ = False
        self.__auto_file_names__ = False

        self.platform = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'

        self.gen_run_script = kwargs.get("gen_run_script", True)
        assert isinstance(self.gen_run_script, bool)

        self.new_name = None
        self.new_directory = None

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

    def __set_directory_etc__(self,  new_file_name, new_file_dir):
        if new_file_name is None:
            new_name = self.inp_file_path.name
            _m = re.match("(.+)\..+", new_name)
            if _m:
                new_name = _m.groups()[0]

            self.new_name = "{0}_{1}".format(self.__num_writes__, new_name)
        else:
            self.new_name = new_file_name

        if new_file_dir is None:
            self.new_directory = self.inp_file_path.parent/self.new_name
        else:
            self.new_directory = Path(new_file_dir)/self.new_name

        if not self.new_directory.exists():
            print("Creating new directory: {0}".format(self.new_directory))
        self.new_directory.mkdir(exist_ok=True)

        assert self.new_directory.exists(), "Directory '{0}' doesn't exist.".format(self.new_directory)

    @property
    def new_file_full_path(self):
        return self.new_directory / self.new_name

    def write_inp_in_scope(self, dict_of_globals, new_file_name=None, new_file_dir=None, script_name="cmd",
                           **mcnp_or_phits_kwargs):
        assert len(self.__new_inp_lines__) == 0
        self.__set_directory_etc__(new_file_name, new_file_dir)

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

        if self.is_mcnp:
            self.__split_new_lines__()
        if self.is_mcnp:
            self.__new_inp_lines__.extend(self.inp_lines[self.MCNP_EOF:])
        if self.gen_run_script is True:
            self.__gen_run_script__(script_name, mcnp_or_phits_kwargs)
        self.__write_file__()

        self.__new_inp_lines__ = []

    def __write_file__(self):
        assert len(self.__new_inp_lines__) > 0, "Must call 'write_inp_in_scope' before '__write_file__'"

        if self.cycle_rnd_seed is not False:
            if self.is_mcnp:
                self.cycle_random_number_mcnp()
            else:
                raise NotImplementedError

        with open(self.new_directory / self.new_name, "w") as f:
            for line in self.__new_inp_lines__:
                f.write(line)

        self.__num_writes__ += 1
        self.__files_written_so_far__.append(self.new_directory / self.new_name)

    def __gen_run_script__(self, script_name, mcnp_or_phits_kwargs):
        f_path = self.new_directory.parent / script_name
        if mcnp_or_phits_kwargs is None:
            script_kwargs = ""
        else:
            assert isinstance(mcnp_or_phits_kwargs, dict)
            script_kwargs = " ".join(["{0}={1}".format(key, value) for key, value in mcnp_or_phits_kwargs.items()])

        cd_cmd = "cd {0}".format(self.new_directory)
        run_cmd = "{0};mcnp6 i={1} {2}".format(cd_cmd, self.new_name, script_kwargs) if self.is_mcnp else \
            "{0};phits.sh {1} {2}".format(cd_cmd, self.new_name, script_kwargs)

        if self.platform == "Darwin":
            new_cmd = "osascript -e 'tell app \"Terminal\"\ndo script \"{0} 2>&1 | tee -i log_{1}.txt;exit\"\nend " \
                      "tell'\n".format(run_cmd, self.new_name)
        elif self.platform == "Linux":
            new_cmd = "gnome-terminal -x sh -c '{0} 2>&1 | tee -a -i log_{1}.txt;'\n".format(run_cmd, self.new_name)
        elif self.platform == "Windows":
            warnings.warn("Currently no implementation of the creation of a .bat file to automatically run the "
                          "simulations on Windows. ")
            return
        else:
            warnings.warn("Run script not generated. Platform not supported.")
            return

        with open(f_path, "w" if self.__num_writes__ == 0 else "a") as f:
            f.write(new_cmd)

        if self.__num_writes__ == 0:
            print("Directory:\ncd {0}".format(self.new_directory.parent))
            if self.platform in ["Linux", "Darwin"]:
                st = os.stat(f_path)
                os.chmod(f_path, st.st_mode | stat.S_IEXEC)

        if self.__num_writes__ == 1:
            print("./cmd")

    @classmethod
    def mcnp_input_deck(cls, inp_file_path, cycle_rnd_seed=False, gen_run_script=True):
        return InputFile(inp_file_path=inp_file_path,cycle_rnd_seed=cycle_rnd_seed, gen_run_script=gen_run_script,
                         is_mcnp=True)

    @classmethod
    def phits_input_deck(cls, inp_file_path, cycle_rnd_seed=False, gen_run_script=True):
        return InputFile(inp_file_path=inp_file_path, cycle_rnd_seed=cycle_rnd_seed, gen_run_script=gen_run_script,
                         is_mcnp=False)

if __name__ == "__main__":
    p = Path(__file__).parent/"test.inp"
    i = InputFile.mcnp_input_deck(p, cycle_rnd_seed=True)
    from FFandProtonSims.GlobalValues import *
    imp = "dokdkof"
    gas_density = 100
    i.write_inp_in_scope(globals())
    i.write_inp_in_scope(globals())


