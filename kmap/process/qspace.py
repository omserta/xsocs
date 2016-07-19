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
                 output_f):
        super(RecipSpaceConverter, self).__init__()

        self.reset(data_h5f, output_f)

    def reset(self,
              data_h5f,
              output_f):

        self.__data_h5f = data_h5f
        self.__output_f = output_f
        self.__thread = None

        self.__qspace_size = None
        self.__img_binning = [1, 1]

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

    def is_running(self):
        return self.__thread and self.__thread.is_alive()

    def convert(self,
                blocking=True,
                overwrite=False,
                callback=None,
                pos_indices=None):

        thread = self.__thread
        if thread is not None and thread.is_running():
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
                                self.__img_binning,
                                callback=callback,
                                overwrite=overwrite,
                                pos_indices=pos_indices)

        self.__thread = thread

        thread.start()

        if blocking:
            self.wait()

    def wait(self):
        if self.__thread is not None:
            self.__thread.wait()

    def __on_conversion_done(self, callback):
        self.__results = self.__thread.results()
        if callback:
            callback()

    def check_overwrite(self):
        """
        Checks if the output file(s) already exist(s).
        """
        if os.path.exists(self.__output_f):
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
        if self.__img_binning is None:
            errors.append('invalid "image binning"')
        if self.__qspace_size is None:
            errors.append('invalid "qspace size"')
        return errors

    img_binning = property(lambda self: self.__img_binning)

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
            values = dic[key]
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

        n_images = params['n_images'][0]
        n_positions = params['n_positions'][0]
        if n_images != n_positions:
            errors.append('number of images != number of X/Y coordinates'
                          'on sample :'
                          '{0} != {1}'.format(n_images, n_positions))

        return errors

    @img_binning.setter
    def img_binning(self, img_binning):
        """
        Binning applied to the image before converting to qspace
        """
        if len(img_binning) != 2:
            raise ValueError('img_binning must be a two elements array.')
        self.__img_binning = [int(img_binning[0]), int(img_binning[1])]

    qspace_size = property(lambda self: self.__qspace_size)

    @qspace_size.setter
    def qspace_size(self, qspace_size):
        """
        Size of the qspace volume.
        """
        if len(qspace_size) != 3:
            raise ValueError('qspace_size must be a three elements array.')
        self.__qspace_size = [int(qspace_size[0]),
                              int(qspace_size[1]),
                              int(qspace_size[2])]

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

    output_f = property(__get_output_f)


class _ConvertThread(Thread):
    def __init__(self,
                 data_h5f,
                 output_f,
                 qspace_size,
                 img_binning,
                 n_proc=None,
                 callback=None,
                 overwrite=False,
                 pos_indices=None):
        super(_ConvertThread, self).__init__()

        self.__data_h5f = data_h5f
        self.__output_f = output_f
        self.__qspace_size = qspace_size
        self.__img_binning = img_binning
        self.__callback = callback
        self.__n_proc = n_proc
        self.__overwrite = overwrite
        self.__pos_indices = pos_indices

        self.__results = None

    def results(self):
        return self.__results

    def progress(self):
        return None

    def wait(self):
        self.join()

    def run(self):
        _img_2_qspace(self.__data_h5f,
                      self.__output_f,
                      self.__qspace_size,
                      overwrite=self.__overwrite,
                      pos_indices=self.__pos_indices,
                      img_binning=self.__img_binning)

        if self.__callback:
            self.__callback()


def _get_all_params(data_h5f):
    """
    Read the whole data and returns the parameters for each entry.
    Returns a dictionary will the following fields :
    scans, n_images, n_positions, img_size, beam_energy, chan_per_deg,
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

    with h5py.File(data_h5f, 'r') as master_h5:
        entries = sorted(master_h5.keys())
        entry_files = []

        n_entries = len(entries)

        for entry_idx, entry in enumerate(entries):
            imgnr_tpl = measurement_tpl.format(entry) + '/imgnr'
            param_tpl = detector_tpl.format(entry) + '/{0}'

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

            n_images.append(n_image)
            n_positions.append(n_position)
            img_sizes.append(img_size)
            beam_energies.append(beam_energy)
            chan_per_degs.append(chan_per_deg)
            center_chans.append(center_chan)
            pixel_sizes.append(pixel_size)
            det_orients.append(det_orient)
    result = dict(scans=entries,
                  n_images=n_images,
                  n_positions=n_positions,
                  img_size=img_sizes,
                  beam_energy=beam_energies,
                  chan_per_deg=chan_per_degs,
                  center_chan=center_chans,
                  pixelsize=pixel_sizes,
                  detector_orient=det_orients)
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
                  img_binning=(1, 1),
                  pos_indices=None,
                  n_proc=None,
                  overwrite=False):

    """
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

    if len(img_binning) != 2:
        raise ValueError('<img_binning> must be a 2-elements array.')

    if min(img_binning) <= 0:
        raise ValueError('<img_binning> values must be strictly positive.')

    params = _get_all_params(data_h5f)

    if beam_energy is None:
        beam_energy = params['beam_energy'][0]
    if beam_energy is None:
        raise ValueError('Invalid/missing beam energy : {0}.'
                         ''.format(beam_energy))

    if chan_per_deg is None:
        chan_per_deg = params['chan_per_deg'][0]
    if beam_energy is None or len(chan_per_deg) != 2:
        raise ValueError('Invalid/missing chan_per_deg value : {0}.'
                         ''.format(chan_per_deg))

    if center_chan is None:
        center_chan = params['center_chan'][0]
    if beam_energy is None or len(center_chan) != 2:
        raise ValueError('Invalid/missing center_chan value : {0}.'
                         ''.format(center_chan))

    if detector_orient is None:
        detector_orient = params['detector_orient'][0]
    if detector_orient is None:
        raise ValueError('Invalid/missing detector_orient value : {0}'
                         ''.format(detector_orient))

    n_images = params['n_images'][0]
    if n_images is None or n_images == 0:
        raise ValueError('Data does not contain any images (n_images={0}).'
                         ''.format(n_images))

    img_size = params['img_size'][0]
    if img_size is None or 0 in img_size:
        raise ValueError('Invalid image size (img_size={0}).'
                         ''.format(img_size))

    entries = params['scans']
    n_entries = len(entries)

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
    print('\t- center chan  : [{0}, {1}]'.format(*center_chan))
    print('\t- chan per deg : [{0}, {1}]'.format(*chan_per_deg))

    # TODO : make this editable?
    nx, ny, nz = qspace_size
    qconv = xu.experiment.QConversion(['y-'],
                                      ['z+', 'y-'],
                                      [1, 0, 0])

    # convention for coordinate system:
    # - x downstream
    # - z upwards
    # - y to the "outside"
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
                         Nav=img_binning)

    # shape of the array that will store the qx/qy/qz for all
    # rocking angles
    q_shape = (n_entries,
               img_size[0] // img_binning[0] * img_size[1] // img_binning[1],
               3)

    # then the array
    q_ar = np.zeros(q_shape, dtype=np.float64)

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

    histo = np.zeros([nx, ny, nz], dtype=np.int32)
    h_lut = []

    for h_idx in range(n_entries):
        lut = histogramnd_get_lut(q_ar[h_idx, ...],
                                  bins_rng,
                                  [nx, ny, nz],
                                  last_bin_closed=True)

        h_lut.append(lut[0])
        histo += lut[1]

    del q_ar

    # TODO : split the output file into several files? speedup?
    output_shape = (n_images,) + histo.shape

    chunks = (1, output_shape[1]//4, output_shape[2]//4, output_shape[3]//4,)
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

    manager = mp.Manager()
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
                             h_lut,
                             n_xy,
                             histo,))

    res_list = []

    if disp_times:
        class myTimes(object):
            def __init__(self):
                self.t_histo = 0.
                self.t_fit = 0.
                self.t_mask = 0.
                self.t_read = 0.
                self.t_dnsamp = 0.
                self.t_medfilt = 0.
                self.t_write = 0.
                self.t_w_lock = 0.

            def update(self, arg):
                (t_read_, t_dnsamp_, t_medfilt_, t_histo_,
                 t_mask_, t_fit_, t_write_, t_w_lock_) = arg
                self.t_histo += t_histo_
                self.t_fit += t_fit_
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
                    img_binning)
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
        print('Fit {0}'.format(res_times.t_fit))
        print('Write {0}'.format(res_times.t_write))
        print('(lock : {0})'.format(res_times.t_w_lock))

def _init_thread(idx_queue_,
                 write_lock_,
                 bins_rng_,
                 qspace_size_,
                 h_lut_,
                 n_xy_,
                 histo_):

        global idx_queue,\
            write_lock,\
            bins_rng,\
            qspace_size,\
            h_lut,\
            n_xy,\
            histo

        idx_queue = idx_queue_
        write_lock = write_lock_
        bins_rng = bins_rng_
        qspace_size = qspace_size_
        h_lut = h_lut_
        n_xy = n_xy_
        histo = histo_


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
        h5f.create_dataset('qspace', shape, dtype=dtype,
                           shuffle=True, compression=compression,
                           chunks=chunks)
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
               img_binning):

    print('Thread {0} started.'.format(th_idx))

    t_histo = 0.
    t_fit = 0.
    t_mask = 0.
    t_read = 0.
    t_dnsamp = 0.
    t_medfilt = 0.
    t_write = 0.
    t_w_lock = 0.

    mask = histo > 0

    img = np.ascontiguousarray(np.zeros((516, 516)), dtype=np.float64)

    # TODO : handle case when nav is not a multiple of img_size!!
    # TODO : find why the first version is faster than the second one
    img_shape_1 = img_size[0]//img_binning[0], img_binning[0], img_size[1]
    img_shape_2 = img_shape_1[0], img_shape_1[2]//img_binning[1], img_binning[1]
    sum_axis_1 = 1
    sum_axis_2 = 2
    # img_shape_1 = img_size[0], img_size[1]/nav[1], nav[1]
    # img_shape_2 = img_size[0]//nav[0], nav[0], img_shape_1[1]
    # sum_axis_1 = 2
    # sum_axis_2 = 1
    avg_weight = 1./(img_binning[0]*img_binning[1])

    while True:
        image_idx = idx_queue.get()
        if image_idx is None:
            print('Thread {0} is done. Times={1}'
                  ''.format(th_idx, (t_read, t_dnsamp,
                                     t_medfilt, t_histo,
                                     t_mask, t_fit, t_write, t_w_lock)))
            if disp_times:
                return (t_read, t_dnsamp,
                        t_medfilt, t_histo,
                        t_mask, t_fit, t_write, t_w_lock,)
            else:
                return None

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
                print('Error in proc {0} while reading img {1} from entry '
                      '{2} ({3}) : {4}.'
                      ''.format(th_idx, image_idx, entry_idx, entry, ex))

            t_read += time.time() - t0
            t0 = time.time()

            if img_binning[0] != 1 or img_binning[1] != 1:
                intensity = img.reshape(img_shape_1).\
                    sum(axis=sum_axis_1).reshape(img_shape_2).\
                    sum(axis=sum_axis_2) *\
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
                print 'EX', ex
                raise ex

            t_histo += time.time() - t0

        t0 = time.time()

        cumul[mask] = cumul[mask]/histo[mask]

        t_mask += time.time() - t0
        
        t0 = time.time()
        write_lock.acquire()
        t_w_lock += time.time() - t0
        try:
            with h5py.File(output_fn, 'r+') as output_h5:
                output_h5['qspace'][image_idx] = cumul
        except Exception as ex:
            print('Error in proc {0} while writing result for img {1} : {2}.'
                  ''.format(th_idx, image_idx, ex))
        write_lock.release()

        t_write += time.time() - t0
