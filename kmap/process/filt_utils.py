# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
# ############################################################################*/

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "20/04/2016"

import time
cimport numpy as np
cimport cython
import numpy as np
from cython.parallel import prange, threadid

ctypedef fused data_t:
    numpy.float64_t
#    numpy.float32_t
#    numpy.int32_t
#    numpy.int64_t

def medfilt2D(image,
              kernel=(3, 3)):
    image_c = np.ascontiguousarray(image.reshape(-1))
    
    kernel_c = np.ascontiguousarray(kernel.reshape(-1),
                                    dtype=np.int32)



def _histogramnd_get_lut_fused(sample_t[:] i_sample,
                               int i_n_dims,
                               int i_n_elems,
                               sample_t[:] i_bins_rng,
                               int[:] i_n_bins,
                               lut_t[:] o_lut,
                               numpy.uint32_t[:] o_histo,
                               bint last_bin_closed):
