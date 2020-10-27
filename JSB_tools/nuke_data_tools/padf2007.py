from matplotlib import pyplot as plt
import openmc
from pathlib import Path
import re
from openmc.data.endf import Evaluation
from openmc.data import Reaction
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
from JSB_tools.nuke_data_tools import prettier_symbol, Nuclide, NUCLIDE_SYMBOL_MATCH
import pickle
from typing import List

#  Download PADF proton induced reaction files from https://www-nds.iaea.org/padf/

PICKLE_FILE_PATH = Path().parent / "data" / "incident_proton"


def __get_symbol_and_filepath__(data_dir):
    symbols_paths = {}

    for f_path in data_dir.iterdir():
        f_name = f_path.name

        _m = re.match("([0-9]{4,6})M*([12345])*", f_name)

        if _m:
            zaid = int(_m.groups()[0])
            z = zaid//1000
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
    def __init__(self, symbol):
        self.symbol = symbol
        self.products = {}  # {'<symbol>':{'nuclide': None (for now), 'xs_data': CrossSection1D},...}
        self.parents: List[Nuclide] = []  # ['<symbol1'>, '<symbol2>, ...]

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
            # print(self.products)

    @classmethod
    def from_symbol(cls, symbol):
        f_path = cls.__get_pickle_file_path__(symbol)
        assert f_path.exists, "Cannot find data for symbol {}. The correct format is I-134m1, or Ar-40".format(symbol)
        with open(f_path, "rb") as f:
            self = pickle.load(f)

        for product_symbol, prod_data in self.products.items():
            if NUCLIDE_SYMBOL_MATCH.match(product_symbol):
                prod_data["nuclide"] = Nuclide.from_symbol(product_symbol)

        for index, parent_symbol in enumerate(self.parents):
            self.parents[index] = Nuclide.from_symbol(parent_symbol)

        return self

    @staticmethod
    def __get_pickle_file_path__(symbol):
        return PICKLE_FILE_PATH/(str(symbol) + ".pickle")

    def __write__(self):
        f_path = self.__get_pickle_file_path__(self.symbol)
        with open(f_path, "wb") as f:
            pickle.dump(self, f)


def pickle_proton_data(data_dir):
    data_dir = Path(data_dir/"Files")
    assert PICKLE_FILE_PATH.exists()

    symbols_paths = __get_symbol_and_filepath__(data_dir)  # {"<symbol_i>": "<endf data path>", ...}}

    # initialize empty IncidentProton instances
    all_data = {s: IncidentProton(s) for s in symbols_paths}

    for symbol, endf_path in symbols_paths.items():
        e = Evaluation(endf_path)
        reaction = Reaction.from_endf(e, 5)
        incident_proton = all_data[symbol]
        assert isinstance(incident_proton, IncidentProton), type(incident_proton)
        incident_proton.__set_data__(reaction)

    for parent_incident_proton in all_data.values():
        assert isinstance(parent_incident_proton, IncidentProton)
        for daughter_symbol in parent_incident_proton.products.keys():
            try:
                daughter_incident_proton = all_data[daughter_symbol]
            except KeyError:
                daughter_incident_proton = IncidentProton(daughter_symbol)
            daughter_incident_proton.parents.append(parent_incident_proton.symbol)

    for incident_proton in all_data.values():
        incident_proton.__write__()

proton_dir = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/PADF_2007"

if __name__ == "__main__":
    assert Path(proton_dir).exists, "Cannot find proton dir files. " \
                                    "Download proton files from https://www-nds.iaea.org/padf/ and set <proton_dir> " \
                                    "variable to the location of the unzipped directory"
    pickle_proton_data(proton_dir)
