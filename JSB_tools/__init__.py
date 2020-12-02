from . import core
import os
import sys
from .outp_reader import OutP
from warnings import warn

def ROOT_loop():
    try:
        import ROOT
        import time
        while True:
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
    except ModuleNotFoundError:
            warn('ROOT not installed. Cannot run ROOT_loop')
# __all__ = [core.UncertainValue, OutP]
