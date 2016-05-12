# /*##########################################################################
# coding: utf-8
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
__date__ = "01/03/2016"

import os
import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('util', parent_package, top_path)
    config.add_subpackage('test')

    # =====================================
    # filt_utils
    # =====================================
    filt_utils_dir = 'filt_utils'
    filt_utils_src = [os.path.join(filt_utils_dir, srcf)
                      for srcf in ['filt_utils.pyx']]
    filt_utils_inc = [numpy.get_include()]

    config.add_extension('filt_utils',
                         sources=filt_utils_src,
                         include_dirs=filt_utils_inc,
                         language='c')

    # =====================================
    # histogramnd_lut
    # =====================================
    filt_utils_dir = 'filt_utils'
    histogramnd_src = [os.path.join(filt_utils_dir, srcf)
                       for srcf in ['histogramnd_lut.pyx']]
    histogramnd_inc = [numpy.get_include()]

    config.add_extension('histogramnd_lut',
                         sources=histogramnd_src,
                         include_dirs=histogramnd_inc,
                         language='c')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
