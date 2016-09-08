import os
import time

from kmap.util.id01_spec import merge_scan_data

# output directory (some temporary files will also be written there)
output_dir = '/path/to/output/'

# path to the spec file
spec_f ='/path/to/spec/scan.spec'

# path to the image files, if stored in a different place than the path written
# in the spec file (else, set to None)
img_base = '/path/to/img/dir/'

# the beam energy (note that this value can be changed later when calling the
# img_2_qpeak function)
beam_energy = 8000.

# merge_scan_data will take all scans from the spec file that have a matching
# image file. If you only want a subset of those scans you can provide a list
# of scan numbers (i.e : the #S xxx lines in the scan headers)
# for example, if we only want scans 48, 50, 52 :
scan_ids = xrange(48, 53, 2)

# this (temporary?) keyword is used to tell the function about the format
# of the spec file. So far there is only two supported values :
# 0 : files created before March 2016 (To Be Confirmed)
# 1 : files creqted after March 2016 (To Be Confirmed)
# At the moment it changed the way the nextNr value in the scan header is
# used to generate the image file name
# (in version 0 we actualy have to take nextNr-1 and pad it to get a 4 digit
# number, while in version 1 we pad the value to to 5 digits)
version = 1

# channels (pix.) per degree (used by xrayutilities when converting to
# qspace coordinates)
# (not that this value can also be changed later when calling the
# img_2_qpeak function)
chan_per_deg = [318., 318.]

# direct beam position in the detector coordinates
center_chan = [140, 322]

# the pixel size of the detector
pixelsize = [-1, -1]

# detector orientation
# (not used at the moment though, still a work in progres)
detector_orient = 'phi'

# the merge will actually create one file per scan, then a "master" file
# (in the output directory) that will contain links to those files. You can
# give the master file the name you want (if None, a name will be generated
# with the prefix found in the spec file)
master_f = None

# checks if some of the output files already exist
# set it to True if you dont care about overwriting files
overwrite = False

t_merge = time.time()

merge_scan_data(output_dir,
                    spec_f,
                    beam_energy=beam_energy,
                    chan_per_deg=chan_per_deg,
                    pixelsize=pixelsize,
                    center_chan=center_chan,
                    scan_ids=scan_ids,
                    master_f=master_f,
                    img_dir=img_base,
                    version=version,
                    overwrite=overwrite)

t_merge = time.time() - t_merge
print('Total time spent : {0}'.format(t_merge))
