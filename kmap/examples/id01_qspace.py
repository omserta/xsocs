import os
#import qspace
from kmap.process import qspace

import numpy as np

# output directory (some temporary files will also be written there)
workdir = '/path/to/output/'

# result file
result_file = '/path/to/results.txt'

# path to the hdf5 "master" file (see id01_spec function)
master_fn ='/path/to/master.h5'

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

res = qspace.img_2_qpeak(master_fn,
                         workdir,
                         n_bins,
                         # beam_energy=beam_energy,
                         # center_chan=center_chan,
                         # chan_per_deg=chan_per_deg,
                         nav=nav,
                         img_indices=None,
                         n_threads=None)

with open(result_file, 'w+') as res_f:
    for i, r in enumerate(res):
        res_str = 'Img {0} : {1} {2} {3} {4} {5} {6} {7}\n'.format(i, *r)
        res_f.write(res_str)
