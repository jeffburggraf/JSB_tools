from __future__ import annotations
import pickle
import openmc
import re
from pathlib import Path
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
from openmc.data.endf import Evaluation
from openmc.data import Reaction
from typing import Dict
from matplotlib import pyplot as plt
from warnings import warn

pwd = Path(__file__).parent
NUCLIDE_SYMBOL_MATCH = re.compile("([A-Za-z]+)-([0-9]*)m*([1-9]?)")

#  Todo: better next time:
#   Find a way to spread classes across modules without circular imports. Make IncidentProton a subclass of
#   IncidentParticle, which should be a subclass of Nuclide. This will greatly enhance scalability.
#   Pickle Decay class objects directly.
#   Define a few model level helper functions for things like converting between file names and z/a/mx.
#   Address FileNotFoundError when accessing obscure proton induced parents/daughters by creating a default class.
#   Make all data pulls as robust as possible to returning instances with attributes such as <symbol> equal to None
#   Investigate why some half-lives such as Te-123 are evaluating to 0 when they are in actuality very very long.
#   Is this aan artifact from openmc?


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
    pickle_data_directory = pwd / "data" / "decay"

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
            self.z = openmc_decay.daughter_nuclide["atomic_number"]
            self.a = openmc_decay.daughter_nuclide["mass_number"]
            self.half_life = openmc_decay.half_life
            self.isomeric_state = openmc_decay.daughter_nuclide["isomeric_state"]
            self.spin = openmc_decay.daughter_nuclide["spin"]
            self.symbol = "{0}-{1}{2}".format(ATOMIC_SYMBOL[int(self.z)], self.a,
                                              "m{0}".format(self.isomeric_state) if self.isomeric_state else "")

            if "gamma" in openmc_decay.spectra and openmc_decay.spectra["gamma"]["continuous_flag"] == "discrete" \
                    and openmc_decay.spectra["gamma"]["discrete_normalization"].n == 1:
                self.gamma_lines = [{k: v for k, v in gamma.items() if k in ["energy", "from_mode", "intensity"]} for gamma
                                    in openmc_decay.spectra["gamma"]["discrete"]]
                self.gamma_lines = sorted(self.gamma_lines, key=lambda x: -x["intensity"])

            self.__openmc_decay__ = openmc_decay
            for m in self.__openmc_decay__.modes:
                info = {"daughter_symbol": prettier_symbol(m.daughter_name), "branching_ratio": m.branching_ratio,
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

        if self.half_life is None or self.half_life>24^60*60*365*5:
            self.stable = True
        else:
            self.stable = False

        if self.symbol not in Nuclide.__instances__:
            Nuclide.__instances__[self.symbol] = self

    def __write_class__(self):
        file_name = self.symbol + ".pickle"
        print("writing pickle data file {0} in directory {1}".format(file_name, Nuclide.pickle_data_directory))
        with open(Nuclide.pickle_data_directory/file_name, "wb") as f:
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
            f_path = cls.pickle_data_directory/f_name
            if not f_path.exists():
                "Nuclide data for {0} not found".format(symbol)

                z = ATOMIC_NUMBER[_m.groups()[0]]
                a = int(_m.groups()[1])
                if len(_m.groups()[2]):
                    isomeric_state = int(_m.groups()[2])
                else:
                    isomeric_state = 0
                out = cls(None, a=a, z=z, isomeric_state=isomeric_state)
            else:
                # out = pickle.load(open(f_path, "rb"))
                out = CustomUnpickler(open(f_path, "rb")).load()
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
        for path in Nuclide.pickle_data_directory.iterdir():
            _m = re.match(r"([A-Za-z]+)-([0-9]+)m*([1-9]?)\.pickle", path.name)
            if _m:
                symbol = _m.groups()[0]
                z = ATOMIC_NUMBER[symbol]
                a = int(_m.groups()[1])
                n = a-z
                out.append({"a": a, "z": z, "n": n, "symbol": symbol})

        return out

    @staticmethod
    def all_nuclides():
        out = []
        for symbol in Nuclide.__all_available_nuclides__():
            out.append(Nuclide.from_symbol(symbol))
        return out

    @staticmethod
    def get_nuclides_with_cut(z_a_n_logical_exp, hl_cut=None):
        out = []
        for vars in Nuclide.__all_available_nuclides__():
            if eval(z_a_n_logical_exp, vars):
                nuclide = Nuclide.from_symbol(vars["symbol"] + "-{0}".format(vars["a"]))
                hl = nuclide.half_life
                if hl_cut is not None:
                    if eval(hl_cut, {"hl": hl}):
                        out.append(nuclide)
                else:
                    out.append(nuclide)

        return out

    def incident_proton_data(self):
        return IncidentProton.from_symbol(self.symbol)


class CrossSection1D:
    def __init__(self, ergs, xss):
        self.ergs = ergs
        self.xss = xss

    def plot(self, ax=None, fig_title=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            if fig_title is not None:
                fig.set_title(fig_title)

        ax.plot(self.ergs, self.xss)

        return ax


class IncidentProton:
    PROTON_PICKLE_FILE_PATH = Path(__file__).parent / "data" / "incident_proton"

    def __init__(self, symbol):
        self.symbol = symbol
        self.products = {}  # {'<symbol>':{'nuclide': None (for now), 'xs_data': CrossSection1D},...}
        self.parents: Dict[str, Nuclide] = {}  # ['<symbol1'>, '<symbol2>, ...]

        self.nuclide = Nuclide.from_symbol(self.symbol)

    @staticmethod
    def __get_symbol_and_filepath__(data_dir):
        symbols_paths = {}

        for f_path in data_dir.iterdir():
            f_name = f_path.name

            _m = re.match("([0-9]{4,6})M*([12345])*", f_name)

            if _m:
                zaid = int(_m.groups()[0])
                z = zaid // 1000
                a = zaid % 1000
                isometric_state = _m.groups()[1]
                if isometric_state is None:
                    isometric_state = 0
                else:
                    isometric_state = int(isometric_state)
                pretty_symbol = "{0}-{1}{2}".format(ATOMIC_SYMBOL[z], a,
                                                    "" if isometric_state == 0 else "m{0}".format(isometric_state))
                symbols_paths[pretty_symbol] = f_path

        return symbols_paths

    def __set_data__(self, open_mc_reaction):
        self.__open_mc_reaction__ = open_mc_reaction
        for prod in self.__open_mc_reaction__.products:
            prod_symbol = prettier_symbol(prod.particle)
            if prod_symbol == self.symbol:
                continue
            prod_data = {}
            self.products[prod_symbol] = prod_data
            prod_data["xs_data"] = CrossSection1D(prod.yield_.x / 1E6, prod.yield_.y * 1E3)
            prod_data["nuclide"] = prod_symbol

    @classmethod
    def from_symbol(cls, symbol) -> IncidentProton:
        f_path = cls.__get_pickle_file_path__(symbol)
        assert f_path.exists, "Cannot find data for symbol {}. The correct format is I-134m1, or Ar-40".format(symbol)
        try:
            with open(f_path, "rb") as f:
                self = CustomUnpickler(f).load()
        except FileNotFoundError:
            warn("could not find proton in=duced data for {}".format(symbol))
            return IncidentProton(symbol)

        for product_symbol, prod_data in self.products.items():
            if NUCLIDE_SYMBOL_MATCH.match(product_symbol):
                prod_data["nuclide"] = Nuclide.from_symbol(product_symbol)

        for parent_symbol in list(self.parents.keys()):
            self.parents[parent_symbol] = Nuclide.from_symbol(parent_symbol)

        return self

    @staticmethod
    def __get_pickle_file_path__(symbol):
        return IncidentProton.PROTON_PICKLE_FILE_PATH / (str(symbol) + ".pickle")

    def __write__(self):
        f_path = self.__get_pickle_file_path__(self.symbol)
        with open(f_path, "wb") as f:
            pickle.dump(self, f)


def pickle_proton_data(data_dir):
    data_dir = Path(data_dir)/"Files"
    assert IncidentProton.PROTON_PICKLE_FILE_PATH.exists()

    symbols_paths = IncidentProton.__get_symbol_and_filepath__(data_dir)  # {"<symbol_i>": "<endf data path>", ...}}

    # initialize empty IncidentProton instances
    all_data = {s: IncidentProton(s) for s in symbols_paths}

    for symbol, endf_path in symbols_paths.items():
        e = Evaluation(endf_path)
        reaction = Reaction.from_endf(e, 5)
        incident_proton = all_data[symbol]
        assert isinstance(incident_proton, IncidentProton), type(incident_proton)
        print("Loading proton activation data for {}".format(symbol))
        incident_proton.__set_data__(reaction)

    for parent_incident_proton in list(all_data.values()):  # loop through all Nuclides
        assert isinstance(parent_incident_proton, IncidentProton)
        print("Setting parent proton activation data for {}".format(parent_incident_proton.symbol))

        for daughter_symbol in parent_incident_proton.products.keys():  # loop through all each nulides daughters
            try:
                daughter_incident_proton = all_data[daughter_symbol]
            except KeyError:
                daughter_incident_proton = IncidentProton(daughter_symbol)
                all_data[daughter_symbol] = daughter_incident_proton
            daughter_incident_proton.parents[parent_incident_proton.symbol] = None

    for incident_proton in all_data.values():
        incident_proton.__write__()
        print("Finished {0}".format(incident_proton.symbol))


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Nuclide':
            return Nuclide
        elif name == "IncidentProton":
            from .padf2007 import IncidentProton
            return IncidentProton
        elif name == "CrossSection1D":
            return CrossSection1D
        return super().find_class(module, name)


#  Download PADF proton induced reaction files from https://www-nds.iaea.org/padf/
proton_data_dir = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/PADF_2007"
decay_data_dir = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay"

if __name__ == "__main__":
    #  Download decay data from https://www.nndc.bnl.gov/endf/b7.1/download.html
    assert Path(decay_data_dir).exists, "Cannot find decay data files. " \
                                         "Download decay files from https://www.nndc.bnl.gov/endf/b7.1/download.html" \
                                         " and set the <decay_data_dir> " \
                                         "variable to the location of the unzipped directory"
    Nuclide.__write_classes__(decay_data_dir)

    assert Path(proton_data_dir).exists, "Cannot find proton data files. " \
                                         "Download proton files from https://www-nds.iaea.org/padf/ and set the " \
                                         "<proton_dir> variable to the location of the unzipped directory"
    # pickle_proton_data(proton_data_dir)

    pass
