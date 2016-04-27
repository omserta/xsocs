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
                    spec_h5_fname,
                    scans,
                    pixelsize_dim0=-1.,
                    pixelsize_dim1=-1.,
                    beam_energy,
                    master_f=None,
                    overwrite=False,
                    n_proc=None):
    pass


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
                         img_dir=None):
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


    :returns: 4 elements tuple : a dict containing the scans whose image file
        has been found, a dict containing the scans that have that have
        valid image file info in the scan header but whose image file has not
        been found, a list with the scans that dont have any image file info,
        and a list of the scans that have more that one image file info line.
    :rtype: *list* (*dict*, *dict*, *list*, *list*)
    """

    imgfile_info = _spec_get_img_filenames(spec_h5_filename)
    with_files = imgfile_info[0]
    complete_scans = {}
    incomplete_scans = {}

    if img_dir:
        img_dir = os.path.expanduser(os.path.expandvars(img_dir))

    for scan_id, infos in with_files.items():
        parsed_fname = (infos['prefix'] +
                        '{0:0>4}'.format(int(infos['nextNr'])-1) +
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
                pixelsize_dim0=-1.,
                pixelsize_dim1=-1.,
                beam_energy=-1.,
                master_f=None,
                overwrite=False,
                n_proc=None):

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
                    beam_energy, pixelsize_dim0, pixelsize_dim1,)
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
                  pixelsize_dim0,
                  pixelsize_dim1):

    global g_spec_lock
    global g_term_evt

    entry = 'entry_{0:0>5}'.format(scan_id.split('.')[0])
    entry_fn = os.path.join(output_dir, entry + '.h5')

    try:

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
                                       data=float(pixelsize_dim0))
            img_det_grp.create_dataset('pixelsize_dim1',
                                       data=float(pixelsize_dim1))
            img_det_grp.create_dataset('beam_energy',
                                       data=float(beam_energy))

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
                                                     compression='lzf',
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
        return ex

    return (entry, entry_fn, True)


if __name__ == '__main__':
    # just adding those lines to make flake8 happy
    g_term_evt = 0
    g_spec_lock = 0
    pass
