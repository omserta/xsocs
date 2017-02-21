from xsocs.process.qspace import kmap_2_qspace, QSpaceConverter

# ========================
# Files
# ========================
# output HDF5 file name, default is qspace.h5 if not set
output_f = 'my_qspace.h5'

# path to the hdf5 "master" file (see id01_spec function)
xsocs_h5 = '/path/to/xsocs.h5'

#========================
# Conversion parameters
#========================

# number of "bins" for the qspace cubes
qspace_dims = (28, 154, 60)

# set disp_times to True if you want to output some info in stdout
QSpaceConverter.disp_times = True

# size of the averaging window to use when downsampling
image_binning = [4, 4]

# positions (on the sample) to convert to qspace
# if pos_indices will be ignored if rect_roi is provided
# rect_roi = [x_min, x_max, y_min, y_max] (sample positions)
roi = None

# set to true if you want to overwrite the output file
# otherwise an exception will be raised if the file already exists
overwrite = False

# number of processes to use
# If None, will use the number of availble core (see multiprocessing.cpu_count)
n_proc = None

kmap_2_qspace(xsocs_h5,
              output_f,
              qspace_dims,
              image_binning=image_binning,
              roi=roi,
              n_proc=n_proc,
              overwrite=overwrite)
