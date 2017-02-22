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

from __future__ import absolute_import

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "01/03/2016"

import os

from . import KmapMerger
from . import KmapSpecParser


def merge_scan_data(output_dir,
                    spec_fname,
                    beam_energy=None,
                    chan_per_deg=None,
                    center_chan=None,
                    scan_ids=None,
                    img_dir=None,
                    n_proc=None,
                    version=1,
                    nr_padding=None,
                    nr_offset=None,
                    compression='lzf',
                    overwrite=False):
    """
    Creates a "master" HDF5 file and one HDF5 per scan. Those scan HDF5 files
    contain spec data (from *spec_fname*) as well as the associated
    image data. This file will either contain all valid scans or the one
    selected using the scan_ids parameter. A valid scan is a scan associated
    with an (existing) image file. Existing output files will be
    overwritten.

    :param output_dir: folder name into which output data (as well as
        temporary files) will be written.
    :type output_dir: str

    :param spec_fname: path to the spec file.
    :type output_dir: str

    :param beam_energy: beam energy in ....
    :type beam_energy: numeric

    :param chan_per_deg: 2 elements array containing the number of channels
        per degree (v, h) (as defined by xrayutilitied, used when converting to
        reciprocal space coordinates).
    :type chan_per_deg: array_like

    :param center_chan: 2 elements array containing the coordinates (v, h) of
        the direct beam position in the detector coordinates.
    :type center_chan: *optional* array_like

    :param scan_ids: array of scan numbers to add to the merged file. If
        None, all valid scans will be merged.
    :type scan_ids: *optional* array of int

    :param master_f: name of the "master" (top level) HDF5 file.
        If None, the file will be named master.h5. This file is created in the
        folder pointed to by output_dir.
    :type master_f: *optional* str

    :param img_dir: directory path. If provided the image files will be
        looked for into that folder instead of the one found in the scan
        headers.
    :type img_dir: *optional* str

    :param n_proc: Number of threads to use when merging files. If None, the
        number of threads used will be the value returned by the function
        `multiprocessing.cpu_count()`
    :type n_proc: *optional* str

    :param version: version of the spec file. It is currently used to get
    the offset and padding to apply to the nextNr value found in the spec scan
    headers. This nextNr is then used to generate the image file name. Set it
    to 0 if you are merging data generated before April 2016 (TBC).
    :type img_dir: *optional* int

    :param nr_padding: zero padding to apply to the nextNr number found
            in the SPEC file.
    :type nr_padding: int

    :param nr_offset: offset to apply to the nextNr number found
        in the SPEC file.
    :type nr_offset: int

    :returns: a list of scan IDs that were merged
    :rtype: *list*
    """

    base_spec = os.path.basename(spec_fname)

    spec_h5 = os.path.join(output_dir, '{}.h5'.format(base_spec))

    if os.path.exists(spec_h5) and not overwrite:
        raise ValueError('The temporary file {0} already exists.'
                         ''.format(spec_h5))

    parser = KmapSpecParser(spec_fname,
                            spec_h5,
                            img_dir=img_dir,
                            version=version,
                            nr_padding=nr_padding,
                            nr_offset=nr_offset)

    parser.parse()

    if parser.status != KmapSpecParser.DONE:
        raise ValueError('Parsing failed with error code {0}'
                         ''.format(parser.status))

    p_results = parser.results

    merger = KmapMerger(p_results.spec_h5,
                        p_results,
                        output_dir)

    merger.beam_energy = beam_energy
    merger.center_chan = center_chan
    merger.chan_per_deg = chan_per_deg
    merger.n_proc = n_proc
    merger.compression = compression

    merger.select(scan_ids, clear=True)

    merger.output_dir = output_dir

    merger.merge(overwrite=overwrite)

    if merger.status != KmapMerger.DONE:
        raise ValueError('Merging failed with error code {0}'
                         ''.format(parser.status))

    m_results = merger.results

    return m_results


if __name__ == '__main__':
    pass
