from kmap.process import qspace

# ========================
# Files
# ========================
# output HDF5 file name, default is qspace.h5 if not set
output_f = 'my_qspace.h5'

# path to the hdf5 "master" file (see id01_spec function)
xsocs_h5 = '/path/to/xsocs.h5'

# ========================
# Acquisition parameters.
# ========================

# The following parameters are supposed to be already stored in the Xsocs input
# file. But you can force them to another value, if needed.

# the beam energy (will take precedence over the one stored in
# the files, if any)
beam_energy = 8000.

# channels (pix.) per degree, used by xrayutilities when converting to
# qspace coordinates
# (will take precedence over the one stored in the files, if any)
ch_per_deg = [318., 318.]

# direct beam position in the detector coordinates
#  (will take precedence over the one stored in the files, if any)
center_chan = [140, 322]

#========================
# Conversion parameters
#========================

# number of "bins" for the qspace cubes
n_bins = (28, 154, 60)

# set disp_times to True if you want to output some info in stdout
qspace.disp_times = True

# size of the averaging window to use when downsampling
image_binning = [4, 4]

# positions (on the sample) to convert to qspace
# if pos_indices will be ignored if rect_roi is provided
# rect_roi = [x_min, x_max, y_min, y_max] (sample positions)
# pos_indices = array with indices (of the sample positions array)
#   to convert to qspace
rect_roi = None
pos_indices = None

# set to true if you want to overwrite the output file
# otherwise an exception will be raised if the file already exists
overwrite = False

# number of processes to use
# If None, will use the number of availble core (see multiprocessing.cpu_count)
n_proc = None

qspace.img_2_qspace(xsocs_h5,
                    output_f,
                    n_bins,
                    # beam_energy=beam_energy,
                    # center_chan=center_chan,
                    # chan_per_deg=chan_per_deg,
                    image_binning=image_binning,
                    rect_roi=rect_roi,
                    pos_indices=pos_indices,
                    n_proc=n_proc,
                    overwrite=overwrite)
