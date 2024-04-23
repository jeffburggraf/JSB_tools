import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5agg')
cwd = Path(__file__).parent
import h5py
import matplotlib as mpl
#mpl.use('Agg') # For certain HPC backends
import matplotlib.pyplot as plt


class PTRAC:
    def __init__(self, fpath):
        self.file = h5py.File(fpath, 'r')

ptrack_file = h5py.File('ptrac.h5', 'r')
ptrack_grp = ptrack_file['ptrack']
sources = [data for data in ptrack_grp["Source"]]
terms = [data for data in ptrack_grp["Termination"]]
print()
#      # In older versions of h5py, it is a little tricky to access the datatype:
#  print("What fields are available?: ", ptrack_grp["Collision"].dtype.fields.keys())
# [data['energy'] for data in ptrack_grp["Source"]]
# print()

# -------------------------------------------------------
# NOTE: rename this file to plot_ptrac.py. It could not
# be attached to the PDF unless it had extension .txt
#
# This file assumes you have a valid installation of
# MCNPTools available, and have ran a corresponding input
# file `pincell.txt` to produce a ptrac.h5 output file. It
# is also necessary to have h5py and matplotlib installed
# with the python distribution.
#
# To run: python3 plot_ptrac.py
#
# It is straightforward to extend the parsing in this file
# to other uses. If you do not have MCNPTools, the first
# part of the script will successfully complete, and the
# mcnptools code below it can be removed.
# -------------------------------------------------------
# def plot_points(ax, points, color, label=None):
#  """Plot several points over geometry"""
#      x = [pair[0] for pair in points]
#      y = [pair[1] for pair in points]
#      ax.scatter(x,y,c=color,alpha=0.6,label=label)
#      return
# def plot_results_with_h5py():
#  """Directly plot results in h5py"""
#      ptrack_file = h5py.File('ptrac.h5', 'r')
#      ptrack_grp = ptrack_file['ptrack']
#      # In older versions of h5py, it is a little tricky to access the datatype:
#      print("What fields are available?: ", ptrack_grp["Collision"].dtype.fields.keys())
#      # Plot the distribution of the source energies
#      src_energies = [ data['energy'] for data in ptrack_grp["Source"] ]
#      fig = plt.figure()
#      ax = fig.add_subplot(111)
#      ax.hist(src_energies, bins = 25)
#      ax.set_xlabel("Energy (MeV)")
#      ax.set_ylabel("Number Samples")
#      # fig.savefig("energy_spectra.pdf", bbox_inches='tight')
#      # Bin fission sites by incident neutron enery, and plot H scatters
#      slow_fission_sites = []
#      fast_fission_sites = []
#      hydrogen_scatters = []
#      for entry in ptrack_grp["Collision"][1:1000000]: #load subset to reduce plotting strain
#          xy = (entry['x'],entry['y'])
#          if entry['reaction_type'] == 18: #MT number from ENDF format
#              if entry['energy'] > 1: #MeV
#                 fast_fission_sites.append( xy )
#              else:
#                 slow_fission_sites.append( xy )
#          elif entry['reaction_type'] == 2 and entry['zaid'] == 1001:
#             hydrogen_scatters.append( xy )
#      # Plot the points of fast and slow fissions, as well as hydrogen scatters
#      # simply to outline the geometry
#      fig = plt.figure()
#      ax = fig.add_subplot(111)
#      plot_points(ax, fast_fission_sites,'#1b9e77',label="Fast Fissions")
#      plot_points(ax, slow_fission_sites,'#d95f02',label="Slow Fissions")
#      plot_points(ax, hydrogen_scatters, '#7570b3', label="H Scatters")
#      ax.set_xlabel("x (cm)")
#      ax.set_ylabel("y (cm)")
#      ax.legend(loc='upper right')
#      ax.set_aspect(1)
#      fig.savefig("h5py_plot.pdf", bbox_inches='tight')
#      print("Number of fast fissions:{}".format(len(fast_fission_sites)))
#      print("Number of slower fissions:{}".format(len(slow_fission_sites)))
#      print("Number of scatters: {}".format(len(hydrogen_scatters)))
#
#
# # pdata = Ptrac("ptrac.h5", Ptrac.HDF5_PTRAC)
# # def plot_results_with_mcnptools():
# #  """Plot same results as before but with MCNPTools"""
# #  slow_fission_sites = []
# #  fast_fission_sites = []
# #  hydrogen_scatters = []
# #  from mcnptools import Ptrac
# #  # Open in mcnptools
# #  pdata = Ptrac("ptrac.h5", Ptrac.HDF5_PTRAC)
# #  # Read in batches
# #  num_collisions = 0
# #  while True:
# #  # Read histories in iterations, until 3M collisions have been
# # processed
# #  hists = pdata.ReadHistories(1000)
# #  if len(hists) == 0 or num_collisions > 3000000:
# #  break
# #  for h in hists: # history loop
# #  for e in range(h.GetNumEvents()): # event loop, per
# # history
# #  event = h.GetEvent(e)
# #  xy = (event.Get(Ptrac.X), event.Get(Ptrac.Y))
# #  if event.Type() == Ptrac.COL:
# #  num_collisions += 1
# #  mt_number = event.Get(Ptrac.RXN) # See manual
# #  if mt_number == 18:
# #  if event.Get(Ptrac.ENERGY) > 1.0:
# #  fast_fission_sites.append(xy)
# #  else:
# #  slow_fission_sites.append(xy)
# #  fig = plt.figure()
# #  ax = fig.add_subplot(111)
# #  plot_points(ax, fast_fission