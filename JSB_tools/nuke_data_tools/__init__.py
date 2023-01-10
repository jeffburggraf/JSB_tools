from JSB_tools.nuke_data_tools.nuclide import Nuclide, DecayNuclide
import warnings

__OPENMC_WARN = False
def openmc_not_installed_warning():
    global __OPENMC_WARN
    if __OPENMC_WARN:
        pass
    else:
        warnings.warn("OpenMC not installed! Some functionality limited. ")
        __OPENMC_WARN = True


if __name__ == "__main__":pass

