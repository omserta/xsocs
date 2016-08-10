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
import ctypes
from threading import Thread
import multiprocessing as mp
import multiprocessing.sharedctypes as mp_sharedctypes

from functools import partial

import h5py
import numpy as np
import xrayutilities as xu

# from scipy.signal import medfilt2d
from scipy.optimize import leastsq

from ..util.filt_utils import medfilt2D
from ..util.histogramnd_lut import histogramnd_get_lut, histogramnd_from_lut
# from silx.math import histogramnd

disp_times = False

positioners_tpl = '/{0}/instrument/positioners'
img_data_tpl = '/{0}/measurement/image/data'
measurement_tpl = '/{0}/measurement'
detector_tpl = '/{0}/instrument/image_detector'


class RecipSpaceConverter(object):
    def __init__(self,
                 data_h5f,
                 output_f=None):
        super(RecipSpaceConverter, self).__init__()

        self.__thread = None

        self.reset(data_h5f, output_f)

    def reset(self,
              data_h5f,
              output_f=None):

        self.__running_exception()

        self.__data_h5f = data_h5f
        self.__output_f = output_f
        self.__thread = None

        self.__qspace_size = None
        self.__image_binning = [1, 1]

        self.__results = None

    def __running_exception(self):
        if self.is_running():
            raise RuntimeError('Operation not permitted while '
                               'a conversion is already in progress.')

    def __get_output_f(self):
        if self.__output_f.endswith('.h5'):
            return self.__output_f
        # TODO : get the prefix from input file
        prefix = os.path.basename(self.__data_h5f).rsplit('.')[0]
        return os.path.join(self.__output_f, '{0}_qspace.h5'.format(prefix))

    output_f = property(__get_output_f)
    @output_f.setter
    def output_f(self, output_f):
        self.__output_f = output_f

    def is_running(self):
        return self.__thread is not None and self.__thread.is_alive()

    def abort(self, wait=True):
     if self.__thread is not None:
        self.__thread.abort(wait=wait)

    def convert(self,
                blocking=True,
                overwrite=False,
                callback=None,
                pos_indices=None,
                **kwargs):

        thread = self.__thread
        if thread is not None and thread.is_alive():
            raise RuntimeError('A conversion is already in progress.')

        if not overwrite and os.path.exists(self.__output_f):
            raise RuntimeError('The file {0} already exists. Use the '
                               'overwrite keyword to ignore this.'
                               ''.format(self.__output_f))

        callback = partial(self.__on_conversion_done,
                           callback=callback)

        thread = _ConvertThread(self.__data_h5f,
                                self.output_f,
                                self.__qspace_size,
                                self.__image_binning,
                                callback=callback,
                                overwrite=overwrite,
                                pos_indices=pos_indices,
                                **kwargs)

        self.__thread = thread

        thread.start()

        if blocking:
            self.wait()

    def wait(self):
        if self.__thread is not None:
            self.__thread.wait()

    def progress(self):
        if self.__thread is not None:
            return self.__thread.progress()
        return 0

    def __on_conversion_done(self, callback):
        self.__results = self.__thread.results()
        if callback:
            callback()

    def check_overwrite(self):
        """
        Checks if the output file(s) already exist(s).
        """
        output_f = self.__output_f
        if output_f is not None and os.path.exists(output_f):
            return [self.__output_f]
        return []

    def summary(self):
        """
        Gives an summary of what will be done.
        """
        # TODO : finish
        files = [self.output_f]
        return files

    def check_parameters(self):
        """
        Checks if the RecipSpaceConverter parameters are valid.
        Returns a list of strings describing those errors, if any,
        or an empty list.
        """
        errors = []
        if self.__image_binning is None:
            errors.append('invalid "image binning"')
        if self.__qspace_size is None:
            errors.append('invalid "qspace size"')
        return errors

    image_binning = property(lambda self: self.__image_binning)

    def check_consistency(self):
        """
        Check if all entries have the same values plus some other
        MINIMAL checks.
        This does not check if the parameter values are valid.
        Returns a list of strings describing those errors, if any,
        or an empty list.
        """
        errors = []

        params = _get_all_params(self.__data_h5f)

        def check_values(dic, key, description, error):
            values = [dic[scan][key] for scan in sorted(dic.keys())]
            if isinstance(values[0], (list, tuple)):
                values = [tuple(val) for val in values]
            values_set = set(values)
            if len(values_set) != 1:
                errors.append('Parameter inconsistency : '
                              '"{0}" : {1}.'
                              ''.format(description, '; '.join(str(m)
                                        for m in values_set)))

        check_values(params, 'n_images', 'Number of images', errors)
        check_values(params, 'n_positions', 'Number of X/Y positions', errors)
        check_values(params, 'img_size', 'Images size', errors)
        check_values(params, 'beam_energy', 'Beam energy', errors)
        check_values(params, 'chan_per_deg', 'Chan. per deg.', errors)
        check_values(params, 'center_chan', 'Center channel', errors)
        check_values(params, 'pixelsize', 'Pixel size', errors)
        check_values(params, 'detector_orient', 'Detector orientation', errors)

        n_images = params[params.keys()[0]]['n_images']
        n_positions = params[params.keys()[0]]['n_positions']
        if n_images != n_positions:
            errors.append('number of images != number of X/Y coordinates'
                          'on sample :'
                          '{0} != {1}'.format(n_images, n_positions))

        return errors

    def scan_params(self, scan):
        params = _get_all_params(self.__data_h5f)
        return params[scan]

    def __get_scans(self):
        params = _get_all_params(self.__data_h5f)
        return sorted(params.keys())

    @image_binning.setter
    def image_binning(self, image_binning):
        """
        Binning applied to the image before converting to qspace
        """
        err = False
        if len(image_binning) != 2:
            raise ValueError('image_binning must be a two elements array.')
        if None in image_binning:
            err = True
        else:
            image_binning_int = [int(image_binning[0]), int(image_binning[1])]
            if min(image_binning_int) <= 0:
                err = True
        if err:
            raise ValueError('<image_binning> values must be strictly'
                             ' positive integers.')
        self.__image_binning = image_binning_int

    qspace_size = property(lambda self: self.__qspace_size)

    @qspace_size.setter
    def qspace_size(self, qspace_size):
        """
        Size of the qspace volume.
        """
        err = False
        if len(qspace_size) != 3:
            raise ValueError('qspace_size must be a three elements array.')
        if None in qspace_size:
            err = True
        else:
            qspace_size_int = [int(qspace_size[0]),
                               int(qspace_size[1]),
                               int(qspace_size[2])]
            if min(qspace_size_int) <= 0:
                err = True
        
        if err:
            raise ValueError('<qspace_size> values must be strictly'
                             ' positive integers.')
        self.__qspace_size = qspace_size_int

    n_proc = property(lambda self: self.__n_proc)

    @n_proc.setter
    def n_proc(self, n_proc):
        if n_proc is None:
            self.__n_proc = None
            return

        n_proc = int(n_proc)
        if n_proc <= 0:
            self.__n_proc = None
        else:
            self.__n_proc = n_proc

    scans = property(__get_scans)


class _ConvertThread(Thread):
    def __init__(self,
                 data_h5f,
                 output_f,
                 qspace_size,
                 image_binning,
                 n_proc=None,
                 callback=None,
                 overwrite=False,
                 pos_indices=None,
                 **kwargs):
        super(_ConvertThread, self).__init__()

        self.__data_h5f = data_h5f
        self.__output_f = output_f
        self.__qspace_size = qspace_size
        self.__image_binning = image_binning
        self.__callback = callback

        if n_proc is None or n_proc <= 0:
            n_proc = mp.cpu_count()

        self.__n_proc = n_proc
        self.__overwrite = overwrite
        self.__pos_indices = pos_indices
        self.__kwargs = kwargs

        self.__results = None
        
        self.__shared_progress = mp_sharedctypes.RawArray(ctypes.c_int32,
                                                          n_proc)
        manager = mp.Manager()
        self.__term_evt = manager.Event()
        self.__manager = manager

    def results(self):
        return self.__results

    def progress(self):
        progress = np.frombuffer(self.__shared_progress, dtype='int32')
        return progress.max()

    def wait(self):
        self.join()

    def abort(self, wait=True):
        if self.is_alive():
            self.__term_evt.set()
            if wait:
                self.wait()

    def run(self):
        _img_2_qspace(self.__data_h5f,
                      self.__output_f,
                      self.__qspace_size,
                      overwrite=self.__overwrite,
                      pos_indices=self.__pos_indices,
                      image_binning=self.__image_binning,
                      n_proc=self.__n_proc,
                      shared_progress=self.__shared_progress,
                      manager=self.__manager,
                      term_evt=self.__term_evt,
                      **self.__kwargs)

        if self.__callback:
            self.__callback()


def _get_all_params(data_h5f):
    """
    Read the whole data and returns the parameters for each entry.
    Returns a dictionary will the scans as keys and the following fields :
    n_images, n_positions, img_size, beam_energy, chan_per_deg,
    center_chan, pixelsize, det_orient.
    Each of those fields are N elements arrays, where N is the number of
    scans found in the file.
    """
    n_images = []
    n_positions = []
    img_sizes = []
    beam_energies = []
    center_chans = []
    pixel_sizes = []
    chan_per_degs = []
    det_orients = []
    angles = []

    with h5py.File(data_h5f, 'r') as master_h5:
        entries = sorted(master_h5.keys())
        entry_files = []

        n_entries = len(entries)

        for entry_idx, entry in enumerate(entries):
            imgnr_tpl = measurement_tpl.format(entry) + '/imgnr'
            param_tpl = detector_tpl.format(entry) + '/{0}'
            angle_path = positioners_tpl.format(entry) + '/eta'

            img_data = master_h5.get(img_data_tpl.format(entry), None)

            if img_data is not None:
                if len(img_data.shape) == 2:
                    n_image = 1
                    img_x, img_y = img_data.shape
                else:
                    n_image = img_data.shape[0]
                    img_x, img_y = img_data.shape[1:3]
            else:
                n_image = None
                img_x = img_y = None

            img_size = [img_x, img_y]

            del img_data

            imgnr = master_h5.get(imgnr_tpl.format(entry), None)
            n_position = len(imgnr) if imgnr is not None else None

            del imgnr

            path = param_tpl.format('beam_energy')
            beam_energy = master_h5.get(path, np.array(None))[()]

            path = param_tpl.format('chan_per_deg_dim0')
            chan_per_deg_dim0 = master_h5.get(path, np.array(None))[()]
            path = param_tpl.format('chan_per_deg_dim1')
            chan_per_deg_dim1 = master_h5.get(path, np.array(None))[()]
            chan_per_deg = [chan_per_deg_dim0, chan_per_deg_dim1]

            path = param_tpl.format('center_chan_dim0')
            center_chan_dim0 = master_h5.get(path, np.array(None))[()]
            path = param_tpl.format('center_chan_dim1')
            center_chan_dim1 = master_h5.get(path, np.array(None))[()]
            center_chan = [center_chan_dim0, center_chan_dim1]

            path = param_tpl.format('pixelsize_dim0')
            pixel_size_dim0 = master_h5.get(path, np.array(None))[()]
            path = param_tpl.format('pixelsize_dim1')
            pixel_size_dim1 = master_h5.get(path, np.array(None))[()]
            pixel_size = [pixel_size_dim0, pixel_size_dim1]

            path = param_tpl.format('detector_orient')
            det_orient = master_h5.get(path, np.array(None))[()]

            angle = master_h5.get(angle_path, np.array(None))[()]##angle_path = np.float64(positioners['eta'][()])

            n_images.append(n_image)
            n_positions.append(n_position)
            img_sizes.append(img_size)
            beam_energies.append(beam_energy)
            chan_per_degs.append(chan_per_deg)
            center_chans.append(center_chan)
            pixel_sizes.append(pixel_size)
            det_orients.append(det_orient)
            angles.append(angle)
    result = {scan:dict(scans=entries[idx],
                        n_images=n_images[idx],
                        n_positions=n_positions[idx],
                        img_size=img_sizes[idx],
                        beam_energy=beam_energies[idx],
                        chan_per_deg=chan_per_degs[idx],
                        center_chan=center_chans[idx],
                        pixelsize=pixel_sizes[idx],
                        detector_orient=det_orients[idx],
                        angle=angles[idx])
              for idx, scan in enumerate(entries)}
    return result


def img_2_qspace(data_h5f,
                 output_f,
                 qspace_size,
                 beam_energy=None,
                 center_chan=None,
                 chan_per_deg=None,
                 nav=(4, 4),
                 pos_indices=None,
                 n_proc=None,
                 overwrite=False):
    """
    :param data_h5f: path to the HDF5 file containing the scan counters
        and images
    :type data_h5f: `str`

    :param output_dir: folder name into which output data (as well as
        temporary files) will be written (unused at the moment).
    :type output_dir: `str`

    :param qspace_size: number of "bins" for the qspace cube (TODO : rephrase)
    :type qspace_size: `array_like`

    :param output_f: Name of the output file the results will written to. This
        file will be created in *output_dir*. If not set, the file will be
        named 'qspace.h5'. This file will be overwritten if it already exists.
    :type output_f: *optional* str

    :param beam_energy: energy (in ...) of the beam used during the data
        acquisition. If set, this will overwrite the one found (if any) in
        the HDF5 file.
    :type beam_energy: *optional* numeric

    :param center_chan: direct beam position in the detector coordinates
        If set, this will overwrite the one found (if any) in the HDF5 file.
    :type center_chan: *optional* `array_like`

    :param chan_per_deg: number of detector chanels per degre. If set,
        this will overwrite the values found (if any) in the HDF5 file.
    :type chan_per_deg: *optional* `array_like`

    :param nav: size of the averaging window to use when downsampling
        the images (TODO : rephrase)
    :type nav: *optional* `array_like`

    :param pos_indices: indices of the positions (on the sample) that have
        to be converted to qspace. E.g : if the array [1, 2, 3] is provided,
        only the first 3 sample scans positions will be converted to qspace.
    :type pos_indices: *optional* `array_like`

    :param n_proc: number of process to use. If None, the number of process
        used will be the one returned by multiprocessing.cpu_count().
    :type n_proc: `int`
    """
    converter = RecipSpaceConverter(data_h5f,
                                    output_f)
    converter.qspace_size = qspace_size
    converter.merge(overwrite=overwrite, pos_indices=pos_indices)

def _img_2_qspace(data_h5f,
                  output_f,
                  qspace_size,
                  beam_energy=None,
                  chan_per_deg=None,
                  center_chan=None,
                  detector_orient=None,
                  pixelsize=None,  # unused at the moment
                  image_binning=(1, 1),
                  pos_indices=None,
                  n_proc=None,
                  overwrite=False,
                  shared_progress=None,
                  manager=None,
                  term_evt=None):

    """
    TODO : put this in _ConvertThread::run
    TODO : detector_orient is NOT supported YET.
    This function does NOT :
    - check input file consistency
        (see RecipSpaceConverter.check_consistency)
    - check if output file already exists
        (see RecipSpaceConverter.check_overwrite)
    It DOES :
    - some minimal checks on the input parameters and data
        (format, existence)
        (see also RecipSpaceConverter.check_parameters)
    """
    
    ta = time.time()

    if len(qspace_size) != 3:
        raise ValueError('<qspace_size> must be a 3-elements array.')

    if min(qspace_size) <= 0:
        raise ValueError('<qspace_size> values must be strictly positive.')

    if len(image_binning) != 2:
        raise ValueError('<image_binning> must be a 2-elements array.')

    if min(image_binning) <= 0:
        raise ValueError('<image_binning> values must be strictly positive.')

    params = _get_all_params(data_h5f)

    entries = sorted(params.keys())
    n_entries = len(entries)

    first_param = params[entries[0]]

    if beam_energy is None:
        beam_energy = first_param['beam_energy']
    if beam_energy is None:
        raise ValueError('Invalid/missing beam energy : {0}.'
                         ''.format(beam_energy))

    if chan_per_deg is None:
        chan_per_deg = first_param['chan_per_deg']
    if beam_energy is None or len(chan_per_deg) != 2:
        raise ValueError('Invalid/missing chan_per_deg value : {0}.'
                         ''.format(chan_per_deg))

    if center_chan is None:
        center_chan = first_param['center_chan']
    if beam_energy is None or len(center_chan) != 2:
        raise ValueError('Invalid/missing center_chan value : {0}.'
                         ''.format(center_chan))

    if detector_orient is None:
        detector_orient = first_param['detector_orient']
    if detector_orient is None:
        raise ValueError('Invalid/missing detector_orient value : {0}'
                         ''.format(detector_orient))

    n_images = first_param['n_images']
    if n_images is None or n_images == 0:
        raise ValueError('Data does not contain any images (n_images={0}).'
                         ''.format(n_images))

    img_size = first_param['img_size']
    if img_size is None or 0 in img_size:
        raise ValueError('Invalid image size (img_size={0}).'
                         ''.format(img_size))

    # TODO : implement ROI
    roi = [0, img_size[0], 0, img_size[1]]

    # TODO value testing
    if pos_indices is None:
        pos_indices = np.arange(n_images)
    else:
        n_images = len(pos_indices)
    n_xy = len(pos_indices)

    print('Parameters :')
    print('\t- beam energy  : {0}'.format(beam_energy))
    print('\t- center chan  : {0}'.format(center_chan))
    print('\t- chan per deg : {0}'.format(chan_per_deg))
    print('\t- img binning : {0}'.format(image_binning))
    print('\t- qspace size : {0}'.format(qspace_size))

    # TODO : make this editable?
    nx, ny, nz = qspace_size
    qconv = xu.experiment.QConversion(['y-'],
                                      ['z+', 'y-'],
                                      [1, 0, 0])

    # convention for coordinate system:
    # x downstream
    # z upwards
    # y to the "outside"
    # (righthanded)
    hxrd = xu.HXRD([1, 0, 0],
                   [0, 0, 1],
                   en=beam_energy,
                   qconv=qconv)

    hxrd.Ang2Q.init_area('z-',
                         'y+',
                         cch1=center_chan[0],
                         cch2=center_chan[1],
                         Nch1=img_size[0],
                         Nch2=img_size[1],
                         chpdeg1=chan_per_deg[0],
                         chpdeg2=chan_per_deg[1],
                         Nav=image_binning)

    # shape of the array that will store the qx/qy/qz for all
    # rocking angles
    q_shape = (n_entries,
               (img_size[0] // image_binning[0]) * (img_size[1] // image_binning[1]),
               3)

    # then the array
    q_ar = np.zeros(q_shape, dtype=np.float64)

    img_dtype = None

    with h5py.File(data_h5f, 'r') as master_h5:

        entry_files = []

        measurement = master_h5[measurement_tpl.format(entries[0])]
        sample_x = measurement['adcX'][:]
        sample_y = measurement['adcY'][:]

        # this has to be done otherwise h5py complains about not being
        # able to open compressed datasets from other processes
        del measurement

        for entry_idx, entry in enumerate(entries):
            entry_file = master_h5[entry].file.filename
            if not os.path.isabs(entry_file):
                entry_file = os.path.abspath(os.path.join(base_dir,
                                                          entry_file))
            entry_files.append(entry_file)

            positioners = master_h5[positioners_tpl.format(entry)]

            eta = np.float64(positioners['eta'][()])
            nu = np.float64(positioners['nu'][()])
            delta = np.float64(positioners['del'][()])

            qx, qy, qz = hxrd.Ang2Q.area(eta, nu, delta)
            q_ar[entry_idx, :, 0] = qx.reshape(-1)
            q_ar[entry_idx, :, 1] = qy.reshape(-1)
            q_ar[entry_idx, :, 2] = qz.reshape(-1)

            entry_dtype = master_h5[img_data_tpl.format(entry)].dtype

            if img_dtype is None:
                img_dtype = entry_dtype
            elif img_dtype != entry_dtype:
                raise TypeError('All images in the input HDF5 files should '
                                'be of the same type. Found {0} and {1}.'
                                ''.format(img_dtype, entry_dtype))

            del positioners

    # custom bins range to have the same histo as xrayutilities.gridder3d
    # bins centered around the qx, qy, qz
    # bins will be like :
    # bin_1 = [min - step/2, min + step/2[
    # bin_2 = [min - step/2, min + 3*step/2]
    # ...
    # bin_N = [max - step/2, max + step/2]
    qx_min = q_ar[:, :, 0].min()
    qy_min = q_ar[:, :, 1].min()
    qz_min = q_ar[:, :, 2].min()
    qx_max = q_ar[:, :, 0].max()
    qy_max = q_ar[:, :, 1].max()
    qz_max = q_ar[:, :, 2].max()

    step_x = (qx_max - qx_min)/(nx-1.)
    step_y = (qy_max - qy_min)/(ny-1.)
    step_z = (qz_max - qz_min)/(nz-1.)

    bins_rng_x = ([qx_min - step_x/2., qx_min +
                  (qx_max - qx_min + step_x) - step_x/2.])
    bins_rng_y = ([qy_min - step_y/2., qy_min +
                  (qy_max - qy_min + step_y) - step_y/2.])
    bins_rng_z = ([qz_min - step_z/2., qz_min +
                  (qz_max - qz_min + step_z) - step_z/2.])
    bins_rng = [bins_rng_x, bins_rng_y, bins_rng_z]

    qx_idx = qx_min + step_x * np.arange(0, nx, dtype=np.float64)
    qy_idx = qy_min + step_y * np.arange(0, ny, dtype=np.float64)
    qz_idx = qz_min + step_z * np.arange(0, nz, dtype=np.float64)

    # TODO : on windows we may be forced to use shared memory
    # TODO : find why we use more memory when using shared arrays
    #       this shouldnt be the case (use the same amount as non shared mem)
    # on linux apparently we dont because when fork() is called data is
    # only copied on write.
    # shared histo used by all processes
    #histo_shared = mp_sharedctypes.RawArray(ctypes.c_int32, nx * ny * nz)
    #histo = np.frombuffer(histo_shared, dtype='int32')
    #histo.shape = nx, ny, nz
    #histo[:] = 0
    histo = np.zeros(qspace_size, dtype=np.int32)

    # shared LUT used by all processes
    #h_lut = None
    #h_lut_shared = None
    h_lut = []

    for h_idx in range(n_entries):
        lut = histogramnd_get_lut(q_ar[h_idx, ...],
                                  bins_rng,
                                  [nx, ny, nz],
                                  last_bin_closed=True)

        #if h_lut_shared is None:
            #lut_dtype = lut[0].dtype
            #if lut_dtype == np.int16:
                #lut_ctype = ctypes.c_int16
            #elif lut_dtype == np.int32:
                #lut_ctype = ctypes.c_int32
            #elif lut_dtype == np.int64:
                #lut_ctype == ctypes.c_int64
            #else:
                #raise TypeError('Unknown type returned by '
                                #'histogramnd_get_lut : {0}.'
                                #''.format(lut.dtype))
            #h_lut_shared = mp_sharedctypes.RawArray(lut_ctype,
                                                    #n_images * lut[0].size)
            #h_lut = np.frombuffer(h_lut_shared, dtype=lut_dtype)
            #h_lut.shape = (n_images, -1)

        #h_lut[h_idx, ...] = lut[0]
        h_lut.append(lut[0])
        histo += lut[1]

    del lut
    del q_ar

    # TODO : split the output file into several files? speedup?
    output_shape = (n_images,) + histo.shape

    chunks = (1,
              max(output_shape[1]//4, 1),
              max(output_shape[2]//4, 1),
              max(output_shape[3]//4, 1),)
    _create_result_file(output_f,
                        output_shape,
                        np.float64,
                        sample_x[pos_indices],
                        sample_y[pos_indices],
                        qx_idx,
                        qy_idx,
                        qz_idx,
                        histo,
                        compression='lzf',
                        chunks=chunks,
                        overwrite=overwrite)

    if manager is None:
        manager = mp.Manager()

    if term_evt is None:
        term_evt = manager.Event()

    write_lock = manager.Lock()
    idx_queue = manager.Queue()

    if n_proc is None:
        n_proc = mp.cpu_count()

    pool = mp.Pool(n_proc,
                   initializer=_init_thread,
                   initargs=(idx_queue,
                             write_lock,
                             bins_rng,
                             qspace_size,
                             h_lut, #_shared,
                             None, #lut_dtype,
                             n_xy,
                             histo, #_shared,))
                             shared_progress,
                             term_evt,))

    res_list = []

    if disp_times:
        class myTimes(object):
            def __init__(self):
                self.t_histo = 0.
                self.t_sum = 0.
                self.t_mask = 0.
                self.t_read = 0.
                self.t_dnsamp = 0.
                self.t_medfilt = 0.
                self.t_write = 0.
                self.t_w_lock = 0.

            def update(self, arg):
                (t_read_, t_dnsamp_, t_medfilt_, t_histo_,
                 t_mask_, t_sum_, t_write_, t_w_lock_) = arg[2]
                self.t_histo += t_histo_
                self.t_sum += t_sum_
                self.t_mask += t_mask_
                self.t_read += t_read_
                self.t_dnsamp += t_dnsamp_
                self.t_medfilt += t_medfilt_
                self.t_write += t_write_
                self.t_w_lock += t_w_lock_
        res_times = myTimes()
        callback = res_times.update
    else:
        callback = None

    # creating the processes
    for th_idx in range(n_proc):
        arg_list = (th_idx,
                    entry_files,
                    entries,
                    img_size,
                    output_f,
                    image_binning,
                    img_dtype)
        res = pool.apply_async(_to_qspace, args=arg_list, callback=callback)
        res_list.append(res)

    # sending the image indices
    for pos_idx in pos_indices:
        idx_queue.put(pos_idx)

    # sending the None value to let the threads know that they should return
    for th_idx in range(n_proc):
        idx_queue.put(None)

    pool.close()
    pool.join()

    for res in res_list:
        res_val = res.get()
        if isinstance(res_val, Exception):
            raise res_val

    tb = time.time()

    if(disp_times):
        print('TOTAL {0}'.format(tb - ta))
        print('Read {0}'.format(res_times.t_read))
        print('Dn Sample {0}'.format(res_times.t_dnsamp))
        print('Medfilt {0}'.format(res_times.t_medfilt))
        print('Histo {0}'.format(res_times.t_histo))
        print('Mask {0}'.format(res_times.t_mask))
        print('Sum {0}'.format(res_times.t_sum))
        print('Write {0}'.format(res_times.t_write))
        print('(lock : {0})'.format(res_times.t_w_lock))

def _init_thread(idx_queue_,
                 write_lock_,
                 bins_rng_,
                 qspace_size_,
                 h_lut_shared_,
                 h_lut_dtype_,
                 n_xy_,
                 histo_shared_,
                 shared_progress_,
                 term_evt_):

        global idx_queue,\
            write_lock,\
            bins_rng,\
            qspace_size,\
            h_lut_shared,\
            h_lut_dtype,\
            n_xy,\
            histo_shared, \
            shared_progress, \
            term_evt

        idx_queue = idx_queue_
        write_lock = write_lock_
        bins_rng = bins_rng_
        qspace_size = qspace_size_
        h_lut_shared = h_lut_shared_
        h_lut_dtype = h_lut_dtype_
        n_xy = n_xy_
        histo_shared = histo_shared_
        shared_progress = shared_progress_
        term_evt = term_evt_


def _create_result_file(h5_fn,
                        shape,
                        dtype,
                        pos_x,
                        pos_y,
                        bins_x,
                        bins_y,
                        bins_z,
                        histo,
                        compression='lzf',
                        chunks=None,
                        overwrite=False):

    if not overwrite:
        mode = 'w-'
    else:
        mode = 'w'

    dir_name = os.path.dirname(h5_fn)
    if len(dir_name) > 0 and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with h5py.File(h5_fn, mode) as h5f:
        h5f.create_dataset('data/qspace', shape, dtype=dtype,
                           shuffle=True, compression=compression,
                           chunks=chunks)
        h5f.create_dataset('data/sum', (len(pos_x),), dtype=np.float64,
                           shuffle=True, compression=compression)
        h5f.create_dataset('bins_edges/x', data=bins_x)
        h5f.create_dataset('bins_edges/y', data=bins_y)
        h5f.create_dataset('bins_edges/z', data=bins_z)
        h5f.create_dataset('geom/x', data=pos_x)
        h5f.create_dataset('geom/y', data=pos_y)
        h5f.create_dataset('histo', data=histo)


def _to_qspace(th_idx,
               entry_files,
               entries,
               img_size,
               output_fn,
               image_binning,
               img_dtype):

    print('Thread {0} started.'.format(th_idx))

    t_histo = 0.
    t_fit = 0.
    t_mask = 0.
    t_sum = 0.
    t_read = 0.
    t_dnsamp = 0.
    t_medfilt = 0.
    t_write = 0.
    t_w_lock = 0.
    
    if shared_progress is not None:
        progress_np = np.frombuffer(shared_progress, dtype='int32')
        progress_np[th_idx] = 0
    else:
        progress_np = None

    #histo = np.frombuffer(histo_shared, dtype='int32')
    #histo.shape = qspace_size
    histo = histo_shared
    mask = histo > 0
    
    #h_lut = np.frombuffer(h_lut_shared, dtype=h_lut_dtype)
    #h_lut.shape = (n_xy, -1)
    h_lut = h_lut_shared

    img = np.ascontiguousarray(np.zeros(img_size), dtype=img_dtype)

    # TODO : handle case when nav is not a multiple of img_size!!
    # TODO : find why the first version is faster than the second one
    img_shape_1 = img_size[0]//image_binning[0], image_binning[0], img_size[1]
    img_shape_2 = (img_shape_1[0], img_shape_1[2]//image_binning[1],
                   image_binning[1])
    sum_axis_1 = 1
    sum_axis_2 = 2
    # img_shape_1 = img_size[0], img_size[1]/nav[1], nav[1]
    # img_shape_2 = img_size[0]//nav[0], nav[0], img_shape_1[1]
    # sum_axis_1 = 2
    # sum_axis_2 = 1
    avg_weight = 1./(image_binning[0]*image_binning[1])

    is_done = False

    try:
        while True:
            if term_evt.is_set():  # noqa
                raise Exception('Thread #{0} : conversion aborted.'
                                ''.format(th_idx))

            image_idx = idx_queue.get()
            if image_idx is None:
                is_done = True
                break

            if image_idx % 100 == 0:
                print('#{0}/{1}'.format(image_idx, n_xy))

            cumul = None
            # histo = None

            for entry_idx, entry in enumerate(entries):

                t0 = time.time()

                try:
                    with h5py.File(entry_files[entry_idx], 'r') as entry_h5:
                        img_data = entry_h5[img_data_tpl.format(entry)]
                        img_data.read_direct(img,
                                             source_sel=np.s_[image_idx],
                                             dest_sel=None)
                        #img = img_data[image_idx].astype(np.float64)
                except Exception as ex:
                    raise RuntimeError('Error in proc {0} while reading '
                                       'img {1} from entry {2} ({3}) : {4}.'
                                       ''.format(th_idx, image_idx, entry_idx,
                                                 entry, ex))

                t_read += time.time() - t0
                t0 = time.time()

                if image_binning[0] != 1 or image_binning[1] != 1:
                    intensity = img.reshape(img_shape_1).\
                        sum(axis=sum_axis_1, dtype=np.uint32).reshape(img_shape_2).\
                        sum(axis=sum_axis_2, dtype=np.uint32) *\
                        avg_weight
                    # intensity = xu.blockAverage2D(img, nav[0], nav[1], roi=roi)
                else:
                    intensity = img

                t_dnsamp += time.time() - t0
                t0 = time.time()

                # intensity = medfilt2d(intensity, 3)
                intensity = medfilt2D(intensity, kernel=[3, 3], n_threads=None)

                t_medfilt += time.time() - t0
                t0 = time.time()

                try:
                    cumul = histogramnd_from_lut(intensity.reshape(-1),
                                                 h_lut[entry_idx],
                                                 shape=qspace_size,
                                                 weighted_histo=cumul,
                                                 dtype=np.float64)
                except Exception as ex:
                    print('EX {0}'.format(str(ex)))
                    raise ex

                t_histo += time.time() - t0

            t0 = time.time()
            cumul_sum = cumul.sum(dtype=np.float64)
            t_sum += time.time() - t0

            t0 = time.time()
            #cumul[mask] = cumul[mask]/histo[mask]
            t_mask += time.time() - t0

            t0 = time.time()
            write_lock.acquire()
            t_w_lock += time.time() - t0
            t0 = time.time()
            try:
                with h5py.File(output_fn, 'r+') as output_h5:
                    output_h5['data/qspace'][image_idx] = cumul
                    output_h5['data/sum'][image_idx] = cumul_sum
            except Exception as ex:
                raise RuntimeError('Error in proc {0} while writing result '
                                   'for img {1} : {2}.'
                                   ''.format(th_idx, image_idx, ex))
            write_lock.release()

            if progress_np is not None:
                progress_np[th_idx] = round(100. * (image_idx + 1.) / n_xy)

            t_write += time.time() - t0
    except Exception as ex:
        print(str(ex))
        term_evt.set()
        is_done = False
        
    if disp_times:
        print('Thread {0} is done. Times={1}'
              ''.format(th_idx, (t_read, t_dnsamp,
                                 t_medfilt, t_histo,
                                 t_mask, t_sum, t_write, t_w_lock)))
    return [is_done, '', (t_read, t_dnsamp,
                       t_medfilt, t_histo,
                       t_mask, t_sum, t_write, t_w_lock,)]
