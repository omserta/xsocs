import os
from kmap.process import qspace

import numpy as np

# output directory (some temporary files will also be written there)
workdir = '/path/to/output/'

# output HDF5 file name, default is qspace.h5 if not set
output_f = 'my_qspace.h5'

# path to the hdf5 "master" file (see id01_spec function)
master_f ='/path/to/master.h5'

# the beam energy (will take precedence over the one stored in the files, if any)
beam_energy = 8000.

# channels (pix.) per degree, used by xrayutilities when converting to
# qspace coordinates
# (will take precedence over the one stored in the files, if any)
ch_per_deg = [318., 318.]

# direct beam position in the detector coordinates
#  (will take precedence over the one stored in the files, if any)
center_chan = [140, 322]

# number of "bins" for the qspace cube
n_bins = (28, 154, 60)

# set disp_times to True if you want to output some info in stdout
qspace.disp_times = True

# size of the averaging window to use when downsampling
nav = [4, 4]

qspace.img_2_qspace(master_f,
                    workdir,
                    n_bins,
                    output_f=output_f,
                    # beam_energy=beam_energy,
                    # center_chan=center_chan,
                    # chan_per_deg=chan_per_deg,
                    nav=nav,
                    img_indices=None,
                    n_proc=None)
