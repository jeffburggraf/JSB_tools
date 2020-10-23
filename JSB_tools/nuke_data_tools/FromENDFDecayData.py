import openmc
import pickle
import re
from pathlib import Path
from openmc.data import ATOMIC_SYMBOL, ATOMIC_NUMBER
#  Download decay data from https://www.nndc.bnl.gov/endf/b7.1/download.html

data_directory = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay/decay_data.pickle"


def prettier_symbol(s):
    assert isinstance(s, str)
    s = s.replace("_", "")
    _m = re.match("([a-zA-Z]*)([0-9]+.*)", s)
    return _m.groups()[0] + "-" + _m.groups()[1]


class Nuclide:
    def __init__(self, z, a, **kwargs):
        self.z = int(z)
        self.a = int(a)
        self.isomeric_state = kwargs.get("isomeric_state", 0)
        self.symbol = "{0}-{1}{2}".format(ATOMIC_SYMBOL[int(z)], a,
                                          "m{0}".format(self.isomeric_state) if self.isomeric_state else "")

    def __repr__(self):
        return "<Nuclide: {0}>".format(self.symbol)


class RadioactiveNuclide(Nuclide):
    __instances__ = {}

    def __init__(self, openmc_decay, **kwargs):
        super().__init__(openmc_decay.nuclide["atomic_number"], openmc_decay.nuclide["mass_number"],
                         **kwargs)
        self.half_life = openmc_decay.half_life
        self.isomeric_state = openmc_decay.nuclide["isomeric_state"]

        self.gamma_lines = []
        if "gamma" in openmc_decay.spectra and openmc_decay.spectra["gamma"]["continuous_flag"] == "discrete" \
                and openmc_decay.spectra["gamma"]["discrete_normalization"].n == 1:
            self.gamma_lines = [{k: v for k, v in gamma.items() if k in ["energy", "from_mode", "intensity"]} for gamma
                                in openmc_decay.spectra["gamma"]["discrete"]]

        self.__openmc_decay__ = openmc_decay
        # for m in openmc_decay.modes:
        # decay_data._daughters_and_b_ratios[prettier_symbol(m.daughter)] = m.branching_ratio
        self.daughters = []
        self.parents = []
        RadioactiveNuclide.__instances__[self.symbol] = self

    @staticmethod
    def __set_daughters__(instances=None):

        if instances is None:  # instances: set of all RadioactiveNuclide instances.
            instances = RadioactiveNuclide.__instances__
        for parent in instances.values():  # loop through every instance and set daughters
            for m in parent.__openmc_decay__.modes:  # __openmc_decay__ is set during instantiation
                daughter_symbol = prettier_symbol(m.daughter)  # ket the key of the daughter
                # If daughter in instances then set nuclide to instance. Else, must use Nuclide cls
                if daughter_symbol in instances:
                    daughter_nuclide = instances[daughter_symbol]
                    # if daughter_symbol != parent.symbol:
                    #     daughter_nuclide.parents.append(parent.symbol)
                else:
                    _m = re.match("([A-Za-z]+)-([0-9]*)m*([1-9]{0,1})", parent.symbol)
                    z = ATOMIC_NUMBER[_m.groups()[0]]
                    a = _m.groups()[1]
                    isomeric_state = _m.groups()[2]
                    daughter_nuclide = Nuclide(z, a, isomeric_state=isomeric_state)
                if 'sf' in m.modes:
                    daughter_nuclide = None  # Todo
                daughter_info = {"nuclide": daughter_nuclide, "branching_ratio": m.branching_ratio,
                                 "modes": m.modes}
                # Todo: 'sf' decay mode doesnt give list of nuclides. This could ben fixed.
                parent.daughters.append(daughter_info)
            parent.daughters = sorted(parent.daughters, key=lambda x: -x["branching_ratio"])

    @staticmethod
    def __set_parents__(instances=None):
        if instances is None:  # instances: set of all RadioactiveNuclide instances.
            instances = RadioactiveNuclide.__instances__
        for obj in instances.values():
            assert isinstance(obj,  RadioactiveNuclide), type(obj)
            for daughter_info in obj.daughters:
                daughter = daughter_info["nuclide"]

                if isinstance(daughter, RadioactiveNuclide):
                    if not hasattr(daughter, "parents"):
                        daughter.parents = []
                        print("adding pparent att", daughter)
                    daughter.parents.append(obj)

    def __repr__(self):
        return "<RadioactiveNuclide: {}; t_1/2 = {}>".format(self.symbol, self.half_life)

    def __getstate__(self):
        state = self.__dict__
        del state["__openmc_decay__"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state


def process_files(paths, write_directory):
    write_directory = Path(write_directory)
    assert write_directory.exists()
    pickle_file = open(write_directory/"decay_data.pickle", "wb")
    for path in paths:
        f = Path(path)
        assert f.exists()
        if not f.name[-4:] == "endf":
            continue
        openmc_decay = openmc.data.Decay(openmc.data.endf.Evaluation(f))
        RadioactiveNuclide(openmc_decay)
    # RadioactiveNuclide.__set_daughters__()
    # RadioactiveNuclide.__set_parents__()

    pickle.dump(RadioactiveNuclide.__instances__, pickle_file)
#
# data = pickle.load(open("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay/decay_data.pickle", "rb"))
# d = (data["U-238"])
#
# assert isinstance(d, RadioactiveNuclide)
# print(d.parents)
# print(d.daughters)
# print(data["Xe-139"].parents)

if __name__ == "__main__":
    path = "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay"
    data = process_files(Path(path).iterdir(), "/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/decay")

    #
    # e = openmc.data.endf.Evaluation("/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/18040")
    # # e = openmc.data.endf.get_evaluations("/Users/jeffreyburggraf/PycharmProjects/JSB_tools/JSB_tools/nuke_data_tools/18040")
    # # for a in e:
    # #     print(a)
    #
    # print(e.reaction_list)
    # print(e.info )
    # r = openmc.data.Reaction.from_endf(e, 5)
    # print(r)
    # print(r.xs)
    # for prod in r.products:
    #     print(type(prod.particle))
    #     if str(prod.particle) == "Cl39":
    #         plt.plot(prod.yield_.x/1E6, prod.yield_.y*1E3)
    #         break
    #     print(prod.particle, prod.yield_.y)
    #     print(dir(prod))
    #
    #     #
    #
    # plt.show()
    # print(r.products)
    # print(r)

