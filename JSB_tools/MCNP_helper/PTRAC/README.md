#PTRAC

Tallies are convenient, but they can be limited.This is where PTRAC comes in. The format of a PTRAC file is uniquely obscure and poorly documented. 
The decoding of PTRAC files is an undertaking that you don't want to undertake. You should just use Geant instead. 
Or use this package to convert your PTRAC files into a ROOT tree. There's also an option to write the simulation
to a human-readable text file (use option `write_2_text=True`).

Example usage:
```
from JSB_tools.MCNP_helper.PTRAC import ptrac2root
from JSB_tools import ROOT_loop

path_to_ptrac_file = "path/to/file" 
ptrac2root(cwd / 'ptrac', max_events=None)

tb = ROOT.TBrowser()
ROOT_loop()
```

Done! A ROOT tree should appear in the working directory. Open up a TBrowser and check it out 
(ROOT_loop is just there to make this work).
A file names "lookup.txt" will be saved to the working directory. 
This file contains information of, for example, the IDs for all the particles, the meaning of 
banked event numbers (i.e. the creation of secondary particles, that are "banked" during transport to be handled after 
the current particle history finishes).
