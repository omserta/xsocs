# coding: utf-8
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
__license__ = "MIT"
__date__ = "01/03/2016"


from .QSpaceConverter import QSpaceConverter


def kmap_2_qspace(xsocsH5_f,
                  output_f,
                  qspace_dims,
                  image_binning=(4, 4),
                  roi=None,
                  sample_indices=None,
                  n_proc=None,
                  overwrite=False):
    """
    :param xsocsH5_f: path to the HDF5 file containing the scan counters
        and images
    :type xsocsH5_f: `str`

    :param output_f: name of the file that will contain the conversion results.
    :type output_f: `str`

    :param qspace_dims: qspace dimensions along the qx, qy and qz axis.
    :type qspace_dims: `array_like`

    :param output_f: Name of the output file the results will written to. This
        file will be created in *output_dir*. If not set, the file will be
        named 'qspace.h5'. This file will be overwritten if it already exists.
    :type output_f: *optional* str

    :param image_binning: size of the averaging window to use when downsampling
        the images (TODO : rephrase)
    :type image_binning: *optional* `array_like`

    :param roi: rectangular region which will be converted to qspace.
        This must be a four elements array containing x_min, x_max, y_min,
        y_max.
    :type roi: *optional* `array_like` (x_min, x_max, y_min, y_max)

    :param sample_indices: indices of the positions (on the sample) that have
        to be converted to qspace. **Ignored** if *roi* is provided.
        E.g : if the array [0, 1, 2] is provided, only the first 3 sample
        scans positions will be converted to qspace.
    :type sample_indices: *optional* `array_like`

    :param n_proc: number of process to use. If None, the number of process
        used will be the one returned by multiprocessing.cpu_count().
    :type n_proc: `int`

    :param overwrite: if set to False, an exception will be raise if the output
        file already exists.
    :type overwrite: bool
    """
    converter = QSpaceConverter(xsocsH5_f,
                                qspace_dims,
                                img_binning=image_binning,
                                output_f=output_f)

    if roi is not None:
        converter.roi = roi
    elif sample_indices is not None:
        converter.sample_indices = sample_indices

    converter.image_binning = image_binning

    converter.n_proc = n_proc

    converter.convert(overwrite=overwrite)

    rc = converter.status

    if rc != QSpaceConverter.DONE:
        raise ValueError('Conversion failed with CODE={0} :\n'
                         '{1}'''
                         ''.format(converter.status, converter.status_msg))


if __name__ == '__main__':
    pass
