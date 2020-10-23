import openmc
import pickle
import re
from pathlib import Path
from matplotlib import pyplot as plt

#  Download decay data from https://www.nndc.bnl.gov/endf/b7.1/download.html
class DecayData:
    def __init__(self, symbol):
        if "_" in symbol:
            _m = re.match("([A-Z][a-z]*)([0-9]+)_m*([0-9]+)", symbol)
            excited = "" if _m.groups()[-1] is None else "m{0}".format(_m.groups()[-1])
            self.symbol = "{0}-{1}{2}".format(_m.groups()[0], _m.groups()[1], excited)
        else:
            _m = re.match("([A-Z][a-z]*)([0-9]+)", symbol)
            if _m:
                self.symbol = "{0}-{1}".format(_m.groups()[0], _m.groups()[1])
            else:
                self.symbol = symbol
        self._symbol = symbol

        self.z = 0
        self.a = 0
        self.half_life = None
        self.isomeric_state = None


def process_file(path):
    f = Path(path)
    decay = openmc.data.Decay(openmc.data.endf.Evaluation(f))
    data = DecayData(decay.nuclide["name"])
    data.half_life = decay.half_life
    data.z = decay.nuclide["atomic_number"]

    data.a = decay.nuclide["mass_number"]
    data.isomeric_state = decay.nuclide["isomeric_state"]
    print(data.symbol, data._symbol)
    if "gamma" in decay.spectra and decay.spectra["gamma"]["continuous_flag"] == "discrete"\
            and decay.spectra["gamma"]["discrete_normalization"].n == 1:
        # print(decay.spectra["gamma"]["discrete"])
        print(decay.spectra["gamma"])


if __name__ == "__main__":
    # path = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay"
    # for i, p in enumerate(Path(path).iterdir()):
    #     data = process_file(p)
    #     if i>1000:
    #         break
    #
    e = openmc.data.endf.Evaluation("/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/18040")
    # e = openmc.data.endf.get_evaluations("/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/18040")
    # for a in e:
    #     print(a)

    print(e.reaction_list)
    print(e.info )
    r = openmc.data.Reaction.from_endf(e, 5)
    print(r)
    print(r.xs)
    for prod in r.products:
        print(type(prod.particle))
        if str(prod.particle) == "Cl39":
            plt.plot(prod.yield_.x/1E6, prod.yield_.y*1E3)
            break
        print(prod.particle, prod.yield_.y)
        print(dir(prod))

        #

    plt.show()
    print(r.products)
    print(r)

