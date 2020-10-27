import openmc
import pickle
import re
from pathlib import Path
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER

pwd = Path(__file__).parent
pickle_data_directory = pwd / "data" / "decay"
NUCLIDE_SYMBOL_MATCH = re.compile("([A-Za-z]+)-([0-9]*)m*([1-9]?)")


def prettier_symbol(s):
    _m = NUCLIDE_SYMBOL_MATCH.match(s)
    if _m:
        return s
    if s == "neutron":
        return "n-1"
    if s == "photon":
        return s
    assert isinstance(s, str)
    s = s.replace("_", "")
    _m = re.match("([a-zA-Z]*)([0-9]+.*)", s)
    return _m.groups()[0] + "-" + _m.groups()[1]


class Nuclide:
    __instances__ = {}

    no_pickle_data_error_msg = "No decay data available! " \
                               "Download decay data from  https://www.nndc.bnl.gov/endf/b7.1/download.html\n" \
                               "Then run Nuclide.__write_classes__(<path/to/decay/data>) ".format(Path(__file__))

    def __init__(self, openmc_decay, **kwargs):
        self.daughters = []
        self.__daughter__info__ = []
        self.parents = []
        self.__parent__info__ = []
        self.gamma_lines = []

        if openmc_decay is not None:
            self.z = openmc_decay.nuclide["atomic_number"]
            self.a = openmc_decay.nuclide["mass_number"]
            self.half_life = openmc_decay.half_life
            self.isomeric_state = openmc_decay.nuclide["isomeric_state"]
            self.spin = openmc_decay.nuclide["spin"]
            self.symbol = "{0}-{1}{2}".format(ATOMIC_SYMBOL[int(self.z)], self.a,
                                              "m{0}".format(self.isomeric_state) if self.isomeric_state else "")

            if "gamma" in openmc_decay.spectra and openmc_decay.spectra["gamma"]["continuous_flag"] == "discrete" \
                    and openmc_decay.spectra["gamma"]["discrete_normalization"].n == 1:
                self.gamma_lines = [{k: v for k, v in gamma.items() if k in ["energy", "from_mode", "intensity"]} for gamma
                                    in openmc_decay.spectra["gamma"]["discrete"]]

            self.__openmc_decay__ = openmc_decay
            for m in self.__openmc_decay__.modes:
                info = {"daughter_symbol": prettier_symbol(m.daughter), "branching_ratio": m.branching_ratio,
                        "modes": m.modes}
                self.__daughter__info__.append(info)
            self.__daughter__info__ = list(sorted(self.__daughter__info__, key=lambda x: -x["branching_ratio"]))

        else:
            self.z = kwargs.get("z")
            self.a = kwargs.get("a")
            self.isomeric_state = kwargs.get("isomeric_state", None)
            self.spin = kwargs.get("spin", None)
            self.symbol = kwargs.get("symbol")
            self.half_life = kwargs.get("half_life", None)

        if self.symbol not in Nuclide.__instances__:
            Nuclide.__instances__[self.symbol] = self

    def __write_class__(self):
        file_name = self.symbol + ".pickle"
        print("writing pickle data file {0} in directory {1}".format(file_name, pickle_data_directory))
        with open(pickle_data_directory/file_name, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def __write_classes__(endf_decay_data_directory):
        endf_decay_data_directory = Path(endf_decay_data_directory)
        assert endf_decay_data_directory.exists(), Nuclide.no_pickle_data_error_msg
        for path in endf_decay_data_directory.iterdir():
            f = Path(path)
            if f.name[-5:] != ".endf":
                continue
            openmc_decay = openmc.data.Decay(openmc.data.endf.Evaluation(f))
            Nuclide(openmc_decay)

        for key in list(Nuclide.__instances__.keys()):
            parent = Nuclide.__instances__[key]
            for d_info in parent.__daughter__info__:
                daughter_symbol = d_info["daughter_symbol"]
                try:
                    daughter_class = Nuclide.__instances__[daughter_symbol]
                except KeyError:
                    daughter_class = Nuclide.from_symbol(daughter_symbol, False)
                p_info = {"parent_symbol": parent.symbol, "branching_ratio": d_info["branching_ratio"],
                          "modes": d_info["modes"]}
                daughter_class.__parent__info__.append(p_info)

            parent.__write_class__()

    def __set_daughters__(self):
        if len(self.daughters) != 0:
            return
        for daughter_info in self.__daughter__info__:
            daughter_symbol = daughter_info["daughter_symbol"]
            del daughter_info["daughter_symbol"]

            if self.symbol == daughter_symbol:
                daughter_info["nuclide"] = None  # Todo
            else:
                daughter_info["nuclide"] = Nuclide.from_symbol(daughter_symbol, __set_related__=False)

            self.daughters.append(daughter_info)

    def __set_parents__(self):
        if len(self.parents) != 0:
            return
        for parent_info in self.__parent__info__:
            parent_symbol = parent_info["parent_symbol"]
            del parent_info["parent_symbol"]

            if self.symbol == parent_symbol:
                parent_info["nuclide"] = None  # Todo
            else:
                parent_info["nuclide"] = Nuclide.from_symbol(parent_symbol, False)

            self.parents.append(parent_info)

    def __repr__(self):
        return "<Nuclide: {}; t_1/2 = {}>".format(self.symbol, self.half_life)

    def __getstate__(self):
        state = self.__dict__
        del state["__openmc_decay__"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    @classmethod
    def from_symbol(cls, symbol, __set_related__=True):
        _m = re.match("([A-Za-z]+)-([0-9]*)m*([1-9]?)", symbol)
        assert _m, "Wrong isotope symbol format '{0}'. Correct examples: Xe-139; Cl-38m1".format(symbol)
        if symbol not in cls.__instances__:
            f_name = symbol + ".pickle"
            f_path = pickle_data_directory/f_name
            if not f_path.exists():
                "Nuclide data for {0} not found".format(symbol)

                z = ATOMIC_NUMBER[_m.groups()[0]]
                a = int(_m.groups()[1])
                if len(_m.groups()[2]):
                    isomeric_state = int(_m.groups()[2])
                else:
                    isomeric_state = 0
                out = Nuclide(None, a=a, z=z, isomeric_state=isomeric_state)
            else:
                out = pickle.load(open(f_path, "rb"))
            cls.__instances__[symbol] = out
        else:
            out = cls.__instances__[symbol]
        assert isinstance(out, Nuclide)
        if __set_related__:
            out.__set_daughters__()
            out.__set_parents__()

        return out

    @staticmethod
    def __all_available_nuclides__():
        out = []
        for path in pickle_data_directory.iterdir():
            _m = re.match(r"([A-Za-z]+-[0-9]*m*[1-9]?)\.pickle", path.name)
            if _m:
                out.append(_m.groups()[0])

        return out

    @staticmethod
    def all_nuclides():
        out = []
        for symbol in Nuclide.__all_available_nuclides__():
            out.append(Nuclide.from_symbol(symbol))
        return out

    def get_nuclides_with_cut(self, exp):
        pass
    # todo: e.g. get_nuclides_with_cut("z==3 qnd n>0) -> all valid nuclides


if __name__ == "__main__":
    #  Download decay data from https://www.nndc.bnl.gov/endf/b7.1/download.html

    # Nuclide.__write_classes__("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay")

    pass
