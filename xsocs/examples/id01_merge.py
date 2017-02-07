#!/usr/bin/python
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2015-2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

__authors__ = ["D. Naudet"]
__date__ = "20/04/2016"
__license__ = "MIT"

import os
import time

from xsocs.util.id01_spec import merge_scan_data

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
# for example, if we only want scans 48.1, 54.1 and 68.1 we would write :
# scan_ids = ['48.1', '54.1', '68.1']
scan_ids = None

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
                    center_chan=center_chan,
                    scan_ids=scan_ids,
                    master_f=master_f,
                    img_dir=img_base,
                    version=version,
                    overwrite=overwrite)

t_merge = time.time() - t_merge
print('Total time spent : {0}'.format(t_merge))
