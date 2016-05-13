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

import re
import glob
import os.path

from multiprocessing import Pool, Event, Lock, cpu_count
from functools import partial

import h5py

from PyMca5 import EdfFile
from silx.io import spectoh5


# regular expression matching the imageFile comment line
_IMAGEFILE_LINE_PATTERN = ('^#C imageFile '
                           'dir\[(?P<dir>[^\]]*)\] '
                           'prefix\[(?P<prefix>[^\]]*)\] '
                           'nextNr\[(?P<nextNr>[^\]]*)\] '
                           'suffix\[(?P<suffix>[^\]]*)\]$')


# #######################################################################
# #######################################################################
# #######################################################################


def merge_scan_data(output_dir,
                    spec_fname,
                    beam_energy,
                    chan_per_deg,
                    pixelsize=(-1., -1.),
                    center_chan=(-1., -1.),
                    scan_ids=None,
                    master_f=None,
                    img_dir_base=None,
                    n_proc=None,
                    version=1,
                    compression='lzf'):

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
        per degree (as defined by xrayutilitied, used when converting to
        reciprocal space coordinates).
    :type chan_per_deg: array_like

    :param pixelsize: 2 elements array containing the pixel size of the,
        detector, in TBD.
    :type pixelsize: *optional* array_like

    :param center_chan: 2 elements array containing the coordinates of the
        direct beam position in the detector coordinates.
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

    :returns: a list of scan IDs that were merged
    :rtype: *list*
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    temporary_dir = os.path.join(output_dir, 'xsocs_tmp')
    if not os.path.exists(temporary_dir):
        os.makedirs(temporary_dir)

    temp_h5 = os.path.join(temporary_dir, 'temp_spec.h5')

    _spec_to_h5(spec_fname, temp_h5)

    scans_results = _find_scan_img_files(temp_h5,
                                         img_dir=img_dir_base,
                                         version=version)

    complete_scans = scans_results[0]

    if len(complete_scans) == 0:
        print('No complete scans found (scan + image file).')
        return None

    print('Complete scans (scan + image file) : {0}'
          ''.format(', '.join(complete_scans.keys())))

    # a declaration to quiet down flake8 complaining about the scan_id in
    # the except clause
    scan_id = 0

    if scan_ids is None:
        scans = complete_scans
    else:
        try:
            scans = {'{0}.1'.format(scan_id):
                     complete_scans['{0}.1'.format(scan_id)]
                     for scan_id in scan_ids}
        except KeyError:
            msg = 'Scan ID {0} not found.'.format(scan_id)
            raise ValueError(msg)

    print('Merging scan IDs : {}.'
          ''.format(', '.join(scans.keys())))

    _merge_data(output_dir,
                temp_h5,
                scans,
                beam_energy,
                chan_per_deg,
                pixelsize,
                center_chan,
                master_f='master.h5',
                overwrite=True,
                n_proc=n_proc,
                compression=compression)

    return scans.keys()


# #######################################################################
# #######################################################################
# #######################################################################


def _spec_to_h5(spec_filename, h5_filename):
    """
    Converts a spec file into a HDF5 file.

    :param spec_filename: name of the spec file to convert to HDF5.
    :type spec_filename: str

    :param h5_filename: name of the HDF5 file to create.
    :type h5_filename: str

    .. seealso : silx.io.convert_spec_h5.convert
    """
    spectoh5.convert(spec_filename,
                     h5_filename,
                     mode='w')


# ########################################################################
# ########################################################################


def _spec_get_img_filenames(spec_h5_filename):
    """
    Parsed spec scans headers to retrieve the associated image files.

    :param spec_h5_filename: name of the HDF5 file containing spec data.
    :type spec_h5_filename: str

    .. todo : can we suppose that there's
        only one scan per scan number?
        i.e : no 0.2, 1.2, ...
    .. todo : expecting only one imageFile comment line per scan. Is this
        always true?

    :return: 3 elements tuple : a dict containing the scans that have valid
        image file info, a list with the scans that dont have any image files,
        and a list of the scans that have more that one image files.
    :rtype: *list* (*dict*, *list*, *list*)
    """
    with h5py.File(spec_h5_filename, 'r') as h5_f:

        # scans for which a file name was found
        with_file = {}

        # scans for which no file name was found
        without_file = []

        # scans for which more than one file name was found
        # -> this case is not expected/handled atm
        error_scans = []

        # regular expression to find the imagefile line
        regx = re.compile(_IMAGEFILE_LINE_PATTERN)

        for k_scan, v_scan in h5_f.items():
            header = v_scan['instrument/specfile/scan_header']
            imgfile_match = [m for line in header
                             if line.startswith('#C')
                             for m in [regx.match(line.strip())] if m]

            # expecting only one imagefile line per scan
            if len(imgfile_match) > 1:
                error_scans.append(k_scan)
                continue

            # if no imagefile line
            if len(imgfile_match) == 0:
                without_file.append(k_scan)
                continue

            # extracting the named subgroups
            imgfile_grpdict = imgfile_match[0].groupdict()

            with_file[k_scan] = imgfile_grpdict

        return with_file, without_file, error_scans


# ########################################################################
# ########################################################################


def _find_scan_img_files(spec_h5_filename,
                         img_dir=None,
                         version=1):
    """
    Parses the provided "*spec*" HDF5 file and tries to find the edf file
    associated  with each scans. will look for the files in img_dir if
    provided (instead of looking for the files in the path written
    in the spec file).

    :param spec_h5_filename: name of the HDF5 file containing spec data.
    :type spec_h5_filename: str

    :param img_dir: directory path. If provided the image files will be
        looked for into that folder instead of the one found in the scan
        headers.
    :type img_dir: *optional* str

    :param version: version of the spec file. It is currently used to get
    the offset and padding to apply to the nextNr value found in the spec scan
    headers. This nextNr is then used to generate the image file name. Set it
    to 0 if you are merging data generated before April 2016 (TBC).
    :type img_dir: *optional* int

    :returns: 4 elements tuple : a dict containing the scans whose image file
        has been found, a dict containing the scans that have that have
        valid image file info in the scan header but whose image file has not
        been found, a list with the scans that dont have any image file info,
        and a list of the scans that have more that one image file info line.
    :rtype: *list* (*dict*, *dict*, *list*, *list*)
    """

    if not os.path.exists(img_dir):
        raise ValueError('Image folder not found : {0}'
                         ''.format(img_dir))

    imgfile_info = _spec_get_img_filenames(spec_h5_filename)
    with_files = imgfile_info[0]
    complete_scans = {}
    incomplete_scans = {}

    if version == 0:
        nextnr_ofst = -1
        nextnr_pattern = '{0:0>4}'
    else:
        nextnr_ofst = 0
        nextnr_pattern = '{0:0>5}'

    if img_dir:
        img_dir = os.path.expanduser(os.path.expandvars(img_dir))

    for scan_id, infos in with_files.items():
        parsed_fname = (infos['prefix'] +
                        nextnr_pattern.format(int(infos['nextNr']) +
                                              nextnr_ofst) +
                        infos['suffix'])
        img_file = None

        if not img_dir:
            parsed_fname = os.path.join(infos['dir'], parsed_fname)
            if os.path.isfile(parsed_fname):
                img_file = parsed_fname
        else:
            edf_fullpath = glob.glob(os.path.join(img_dir, parsed_fname))
            if edf_fullpath:
                img_file = edf_fullpath[0]

        if img_file:
            complete_scans[scan_id] = img_file
        else:
            incomplete_scans[scan_id] = infos

    result = [complete_scans, incomplete_scans]
    result.extend(elem for elem in imgfile_info[1:])

    return tuple(result)


# #######################################################################
# #######################################################################
# #######################################################################


def _merge_data(output_dir,
                spec_h5_fname,
                scans,
                beam_energy,
                chan_per_deg,
                pixelsize=[-1., -1.],
                center_chan=[-1, -1],
                master_f=None,
                overwrite=False,
                n_proc=None,
                compression='lzf'):

    """
    Creates a "master" HDF5 file and one HDF5 per scan. Those scan HDF5 files
    contain spec data as well as the associated image data.
    """

    output_dir = os.path.realpath(output_dir)

    # TODO : handle this better
    if master_f is None:
        master_f = os.path.split(output_dir)[1]
        if len(master_f) == 0:
            master_f = os.path.split(output_dir[:-1])[1] + '.h5'
    else:
        master_f = os.path.split(master_f)[1]

    master_f = os.path.join(output_dir, master_f)

    if not overwrite:
        mode = 'w-'
    else:
        mode = 'w'

    if n_proc is None:
        n_proc = cpu_count()

    def init(lock, evt):
        global g_spec_lock
        global g_term_evt
        g_spec_lock = lock
        g_term_evt = evt

    spec_lock = Lock()
    term_evt = Event()

    pool = Pool(n_proc,
                initializer=init,
                initargs=(spec_lock, term_evt,),
                maxtasksperchild=2)

    def callback(scan_id, result):
        if isinstance(result, Exception):
            term_evt.set()

    with h5py.File(master_f, mode) as m_h5f:

        results = {}
        for scan_id in sorted(scans.keys()):
            img_f = scans[scan_id]
            args = (scan_id, spec_h5_fname, output_dir, img_f,
                    beam_energy, chan_per_deg, pixelsize,
                    center_chan, compression)
            results[scan_id] = pool.apply_async(_add_edf_data,
                                                args,
                                                callback=partial(callback,
                                                                 scan_id))

        pool.close()
        pool.join()

        # checking if there was an error
        # TODO
        if term_evt.is_set():
            raise Exception('TODO : Error while merging spec/edf -> hdf5')

        for scan_id, async_res in results.items():
            entry, entry_fn, finished = async_res.get()
            if not finished:
                raise Exception('TODO : there was an error while merging'
                                'scan ID {0}'.format(scan_id))
            m_h5f[entry] = h5py.ExternalLink(entry_fn, entry)


# #######################################################################
# #######################################################################
# #######################################################################


def _add_edf_data(scan_id,
                  spec_h5_fn,
                  output_dir,
                  img_f,
                  beam_energy,
                  chan_per_deg,
                  pixelsize,
                  center_chan,
                  compression):

    """
    Creates an entry_*.h5 file with scan data from the provided
    "*spec*" HDF5 files, and adds the image data from the associated
    image file. This function is meant to be called in from _merge_data.
    """

    global g_spec_lock
    global g_term_evt

    entry = 'entry_{0:0>5}'.format(scan_id.split('.')[0])
    entry_fn = os.path.join(output_dir, entry + '.h5')

    if pixelsize is None:
        pixelsize = [-1., -1.]

    try:
        print('Merging scan ID {0}'.format(scan_id))

        if g_term_evt.is_set():
            return (entry, entry_fn, False)

        with h5py.File(entry_fn, 'w') as entry_h5f:
            entry_grp = entry_h5f.create_group(entry)
            g_spec_lock.acquire()
            with h5py.File(spec_h5_fn) as s_h5f:

                scan_grp = s_h5f[scan_id]

                for grp_name in scan_grp:
                    scan_grp.copy(grp_name,
                                  entry_grp,
                                  shallow=False,
                                  expand_soft=True,
                                  expand_external=True,
                                  expand_refs=True,
                                  without_attrs=False)
            g_spec_lock.release()
            img_det_grp = entry_grp.require_group('instrument/image_detector')
            img_data_grp = entry_grp.require_group('measurement/image_data')

            img_det_grp.create_dataset('pixelsize_dim0',
                                       data=float(pixelsize[0]))
            img_det_grp.create_dataset('pixelsize_dim1',
                                       data=float(pixelsize[1]))
            if beam_energy is not None:
                img_det_grp.create_dataset('beam_energy',
                                           data=float(beam_energy))
            if chan_per_deg is not None:
                img_det_grp.create_dataset('chan_per_deg_dim0',
                                           data=float(chan_per_deg[0]))
                img_det_grp.create_dataset('chan_per_deg_dim1',
                                           data=float(chan_per_deg[1]))
            if center_chan is not None:
                img_det_grp.create_dataset('center_chan_dim0',
                                           data=float(center_chan[0]))
                img_det_grp.create_dataset('center_chan_dim1',
                                           data=float(center_chan[1]))

            edf_file = EdfFile.EdfFile(img_f, access='r', fastedf=True)

            n_images = edf_file.GetNumImages()

            image = edf_file.GetData(0)
            dtype = image.dtype
            img_shape = image.shape
            dset_shape = (n_images, img_shape[0], img_shape[1])
            chunks = (1, dset_shape[1]//4, dset_shape[2]//4)

            image_dset = img_data_grp.create_dataset('data',
                                                     shape=dset_shape,
                                                     dtype=dtype,
                                                     chunks=chunks,
                                                     compression=compression,
                                                     shuffle=True)

            for i in range(n_images):
                if i % 500 == 0:
                    if g_term_evt.is_set():
                        return (entry, entry_fn, False)

                data = edf_file.GetData(i)
                image_dset[i, :, :] = data

            # creating some links
            img_data_grp['info'] = img_det_grp
            img_det_grp['data'] = image_dset

            # attributes
            grp = entry_grp.require_group('measurement/image_data')
            grp.attrs['interpretation'] = 'image'

            # setting the nexus classes
            grp = entry_grp.require_group('instrument')
            grp.attrs['NX_class'] = 'NXinstrument'

            grp = entry_grp.require_group('instrument/image_detector')
            grp.attrs['NX_class'] = 'NXdetector'

            grp = entry_grp.require_group('instrument/positioners')
            grp.attrs['NX_class'] = 'NXcollection'

            grp = entry_grp.require_group('measurement')
            grp.attrs['NX_class'] = 'NXcollection'

            grp = entry_grp.require_group('measurement/image_data')
            grp.attrs['NX_class'] = 'NXcollection'

    except Exception as ex:
        print(ex)
        return ex

    return (entry, entry_fn, True)


if __name__ == '__main__':
    # just adding those lines to make flake8 happy
    g_term_evt = 0
    g_spec_lock = 0
    pass
