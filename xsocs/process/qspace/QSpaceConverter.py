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
import time
import ctypes
from threading import Thread
import multiprocessing as mp
import multiprocessing.sharedctypes as mp_sharedctypes

import numpy as np
import xrayutilities as xu

# from scipy.signal import medfilt2d

from ...util.filt_utils import medfilt2D
from ...util.histogramnd_lut import histogramnd_get_lut, histogramnd_from_lut
# from silx.math import histogramnd
from ...io import XsocsH5, QSpaceH5

disp_times = False


class QSpaceConverter(object):
    (READY, RUNNING, DONE,
     ERROR, CANCELED, UNKNOWN) = __STATUSES = range(6)
    """ Available status codes """

    status = property(lambda self: self.__status)
    """ Current status code of this instance """

    status_msg = property(lambda self: self.__status_msg)
    """ Status message if any, or None """

    results = property(lambda self: self.__results)
    """ Parse results. KmapParseResults instance. """

    xsocsH5_f = property(lambda self: self.__xsocsH5_f)
    """ Input file name. """

    output_f = property(lambda self: self.__output_f)
    """ Output file name. """

    qspace_dims = property(lambda self: self.__params['qspace_dims'])
    """ dimensions of the Q Space (i.e : number of bins). """

    image_binning = property(lambda self: self.__params['image_binning'])
    """ Binning applied to the images before conversion. """

    sample_indices = property(lambda self: self.__params['sample_indices'])
    """ Indices of sample positions that will be converted. """

    n_proc = property(lambda self: self.__n_proc)
    """ Number of processes to use. Will use cpu_count() if None or 0. """

    roi = property(lambda self: self.__params['roi'])
    """ Selected ROI in sample coordinates : [xmin, xmax, ymin, ymax] """

    def __init__(self,
                 xsocsH5_f,
                 qspace_dims=None,
                 img_binning=None,
                 output_f=None,
                 roi=None,
                 entries=None,
                 callback=None):
        """
        Merger for the Kmap SPEC and EDF files. This loads a spech5 file,
             converts it to HDF5 and then tries to match scans and edf image
             files.
        :param xsocsH5_f: path to the input XsocsH5 file.
        :param qspace_dims: dimensions of the qspace volume
        :param img_binning: binning to apply to the images before conversion.
            Default : (1, 1)
        :param output_f: path to the output file that will be created.
        :param roi: Roi in sample coordinates (xMin, xMax, yMin, yMax)
        :param entries: a list of entry names to convert to qspace. If None,
            all entries found in the xsocsH5File will be used.
        :param callback: callback to call when the parsing is done.
        """
        super(QSpaceConverter, self).__init__()

        self.__status = None

        self.__set_status(self.UNKNOWN, 'Init')

        self.__xsocsH5_f = xsocsH5_f
        self.__output_f = output_f

        xsocsH5 = XsocsH5.XsocsH5(xsocsH5_f)
        # checking entries
        if entries is None:
            entries = xsocsH5.entries()
        else:
            diff = set(entries) - set(xsocsH5.entries())
            if len(diff) > 0:
                raise ValueError('The following entries were not found in '
                                 'the input file :\n - {0}'
                                 ''.format('\n -'.join(diff)))

        self.__params = {'qspace_dims': None,
                         'image_binning': None,
                         'sample_indices': None,
                         'roi': None,
                         'entries': sorted(entries)}

        self.__callback = callback
        self.__n_proc = None
        self.__overwrite = False

        self.__shared_progress = None
        self.__results = None
        self.__term_evt = None

        self.__thread = None

        self.image_binning = img_binning
        self.qspace_dims = qspace_dims
        self.roi = roi

        self.__set_status(self.READY)

    def __get_scans(self):
        """
        Returns the entries that will be converted.
        """
        return self.__params['entries']

    scans = property(__get_scans)
    """ Returns the scans found in the input file. """

    def __set_status(self, status, msg=None):
        """
        Sets the status of this instance.
        :param status:
        :param msg:
        :return:
        """
        assert status in self.__STATUSES
        self.__status = status
        self.__status_msg = msg

    def convert(self,
                overwrite=False,
                blocking=True,
                callback=None,
                check_consistency=True):
        """
        Starts the conversion.
        :param overwrite: if False raises an exception if some files already
        exist.
        :param blocking: if False, the merge will be done in a separate
         thread and this method will return immediately.
        :param callback: callback that will be called when the merging is done.
        It overwrites the one passed the constructor.
        :param check_consistency: set to False to ignore any incensitencies
        in the input entries (e.g : different counters, ...).
        :return:
        """

        if self.is_running():
            raise RuntimeError('This QSpaceConverter instance is already '
                               'parsing.')

        self.__set_status(self.RUNNING)

        errors = self.check_parameters()

        if len(errors) > 0:
            msg = 'Invalid parameters.\n{0}'.format('\n'.join(errors))
            raise ValueError(msg)

        errors = self.check_consistency()

        if len(errors) > 0:
            msg = 'Inconsistent input data.\n{0}'.format('\n'.join(errors))

            if check_consistency:
                raise ValueError(msg)
            else:
                print('==============.')
                print('==============.')
                print('WARNING.')
                print(msg)

        output_f = self.__output_f
        if output_f is None:
            self.__set_status(self.ERROR)
            raise ValueError('Output file name (output_f) has not been set.')

        output_dir = os.path.dirname(output_f)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not overwrite:
            if len(self.check_overwrite()):
                    self.__set_status(self.ERROR)
                    raise RuntimeError('Some files already exist. Use the '
                                       'overwrite keyword to ignore this '
                                       'warning.')

        self.__results = None
        self.__overwrite = overwrite

        if callback is not None:
            self.__callback = callback

        if blocking:
            self.__run_convert()
        else:
            thread = self.__thread = Thread(target=self.__run_convert)
            thread.start()

    @qspace_dims.setter
    def qspace_dims(self, qspace_dims):
        """
        Sets the dimensions of the qspace volume (i.e. number of bins).
        """

        if qspace_dims is None or None in qspace_dims:
            self.__params['qspace_dims'] = None
            return

        qspace_dims = np.array(qspace_dims, ndmin=1).astype(np.int32)

        if qspace_dims.ndim != 1 or qspace_dims.size != 3:
            raise ValueError('qspace_dims must be a three elements array.')

        if not np.all(qspace_dims > 1):
            raise ValueError('<qspace_dims> values must be strictly'
                             ' greater than one.')
        self.__params['qspace_dims'] = qspace_dims

    @image_binning.setter
    def image_binning(self, image_binning):
        """
        Binning applied to the image before converting to qspace
        """
        err = False
        if image_binning is None:
            self.__params['image_binning'] = (1, 1)
            return

        image_binning_int = None
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
        self.__params['image_binning'] = np.array(image_binning_int,
                                                  dtype=np.int32)

    # @sample_indices.setter
    # def sample_indices(self, sample_indices):
    #     """
    #     Binning applied to the image before converting to qspace
    #     """
    #     if sample_indices is None:
    #         self.__params['sample_indices'] = None
    #         return
    #
    #     sample_indices = np.array(sample_indices, ndmin=1).astype(np.long)
    #
    #     if sample_indices.ndim != 1:
    #         raise ValueError('sample_indices must be a 1D array.')
    #
    #     if len(sample_indices) == 0:
    #         self.__params['sample_indices'] = None
    #         return
    #
    #     # TODO : check values
    #     self.__params['sample_indices'] = np.array(sample_indices,
    #                                                dtype=np.int32)

    @roi.setter
    def roi(self, roi):
        """
        Sets the roi. Set to None to unset it. To change an already set roi
        the previous one has to be unset first.
        :param roi: roi coordinates in sample coordinates.
            Four elements array : (xmin, xmax, ymin, ymax)
        :return:
        """
        if self.roi is False:
            raise ValueError('Cannot set a rectangular ROI, pos_indices are '
                             'already set, remove them first.')
        self.__params['roi'] = roi
        self.__params['sample_indices'] = self.__indices_from_roi()

    def __indices_from_roi(self):
        # TODO : check all positions
        # at the moment using only the first scan's positions
        with XsocsH5.XsocsH5(self.__xsocsH5_f) as xsocsH5:
            entries = xsocsH5.entries()
            positions = xsocsH5.scan_positions(entries[0])
            x_pos = positions.pos_0
            y_pos = positions.pos_1

        roi = self.roi
        if self.roi is None:
            return np.arange(len(x_pos))

        x_min = roi[0]
        x_max = roi[1]
        y_min = roi[2]
        y_max = roi[3]

        # we cant do this because the points arent perfectly aligned!
        # we could end up with non rectangular rois
        pos_indices = np.where((x_pos >= x_min) & (x_pos <= x_max) &
                               (y_pos >= y_min) & (y_pos <= y_max))[0]
        # # TODO : rework this
        # n_x = scan_params['motor_0_steps']
        # n_y = scan_params['motor_1_steps']
        # steps_0 = scan_params['motor_0_steps']
        # steps_1 = scan_params['motor_1_steps']
        # x = np.linspace(scan_params['motor_0_start'],
        #                 scan_params['motor_0_end'], steps_0, endpoint=False)
        # y = np.linspace(scan_params['motor_1_start'],
        #                 scan_params['motor_1_end'], steps_1, endpoint=False)


        # x_pos = x_pos[]
        #
        # x_pos.shape = (n_y, n_x)
        # y_pos.shape = (n_y, n_x)
        # pos_indices_2d = np.where((x_pos >= x_min) & (x_pos <= x_max) &
        #                           (y_pos >= y_min) & (y_pos <= y_max))[0]
        return pos_indices  # pos_indices_2d.shape

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
        image_binning = self.image_binning
        qspace_dims = self.qspace_dims

        if (image_binning is None
                or None in image_binning
                or len(image_binning) != 2
                or image_binning.min() <= 0):
            errors.append('- "image binning" : must be an array of two'
                          ' strictly positive integers.')

        if (qspace_dims is None
                or None in qspace_dims
                or len(qspace_dims) != 3
                or qspace_dims.min() <= 0):
            errors.append('- "qspace size" must be an array of three'
                          ' strictly positive integers.')
        return errors

    def check_consistency(self):
        """
        Check if all entries have the same values plus some other
        MINIMAL checks.
        This does not check if the parameter values are valid.
        Returns a list of strings describing those errors, if any,
        or an empty list.
        """
        errors = []

        params = _get_all_params(self.__xsocsH5_f)

        def check_values(dic, key, description):
            values = [dic[scan][key] for scan in sorted(dic.keys())]
            if isinstance(values[0], (list, tuple)):
                values = [tuple(val) for val in values]
            values_set = set(values)
            if len(values_set) != 1:
                errors.append('Parameter inconsistency : '
                              '"{0}" : {1}.'
                              ''.format(description, '; '.join(str(m)
                                        for m in values_set)))

        check_values(params, 'n_images', 'Number of images')
        check_values(params, 'n_positions', 'Number of X/Y positions')
        check_values(params, 'img_size', 'Images size')
        check_values(params, 'beam_energy', 'Beam energy')
        check_values(params, 'chan_per_deg', 'Chan. per deg.')
        check_values(params, 'center_chan', 'Center channel')

        keys = list(params.keys())
        n_images = params[keys[0]]['n_images']
        n_positions = params[keys[0]]['n_positions']
        if n_images != n_positions:
            errors.append('number of images != number of X/Y coordinates '
                          'on sample : '
                          '{0} != {1}'.format(n_images, n_positions))

        return errors

    def scan_params(self, scan):
        """ Returns the scan parameters (filled during acquisition). """
        params = _get_all_params(self.__xsocsH5_f)
        return params[scan]

    def __run_convert(self):
        """
        Performs the conversion.
        :return:
        """

        self.__set_status(self.RUNNING)

        image_binning = self.image_binning
        qspace_dims = self.qspace_dims
        xsocsH5_f = self.xsocsH5_f
        output_f = self.output_f
        sample_roi = self.__params['roi']

        try:
            ta = time.time()

            params = _get_all_params(xsocsH5_f)

            entries = self.__get_scans()
            n_entries = len(entries)

            first_param = params[entries[0]]

            beam_energy = first_param['beam_energy']
            if beam_energy is None:
                raise ValueError('Invalid/missing beam energy : {0}.'
                                 ''.format(beam_energy))

            chan_per_deg = first_param['chan_per_deg']
            if beam_energy is None or len(chan_per_deg) != 2:
                raise ValueError('Invalid/missing chan_per_deg value : {0}.'
                                 ''.format(chan_per_deg))

            center_chan = first_param['center_chan']
            if beam_energy is None or len(center_chan) != 2:
                raise ValueError('Invalid/missing center_chan value : {0}.'
                                 ''.format(center_chan))

            n_images = first_param['n_images']
            if n_images is None or n_images == 0:
                raise ValueError(
                    'Data does not contain any images (n_images={0}).'
                    ''.format(n_images))

            img_size = first_param['img_size']
            if img_size is None or 0 in img_size:
                raise ValueError('Invalid image size (img_size={0}).'
                                 ''.format(img_size))

            # TODO value testing
            sample_indices = self.sample_indices
            if sample_indices is None:
                sample_indices = np.arange(n_images)
            else:
                n_images = len(sample_indices)

            n_xy = len(sample_indices)

            print('Parameters :')
            print('\t- beam energy  : {0}'.format(beam_energy))
            print('\t- center chan  : {0}'.format(center_chan))
            print('\t- chan per deg : {0}'.format(chan_per_deg))
            print('\t- img binning : {0}'.format(image_binning))
            print('\t- qspace size : {0}'.format(qspace_dims))

            # TODO : make this editable?
            nx, ny, nz = qspace_dims
            qconv = xu.experiment.QConversion(['y-', 'z-'],
                                              ['z-', 'y-'],
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
                       (img_size[0] // image_binning[0]) * (
                           img_size[1] // image_binning[1]),
                       3)

            # then the array
            q_ar = np.zeros(q_shape, dtype=np.float64)

            img_dtype = None

            with XsocsH5.XsocsH5(xsocsH5_f, mode='r') as master_h5:

                entry_files = []

                all_entries = set(master_h5.entries())

                positions = master_h5.scan_positions(entries[0])
                sample_x = positions.pos_0
                sample_y = positions.pos_1

                for entry_idx, entry in enumerate(entries):
                    entry_file = master_h5.entry_filename(entry)
                    if not os.path.isabs(entry_file):
                        base_dir = os.path.dirname(xsocsH5_f)
                        entry_file = os.path.abspath(os.path.join(base_dir,
                                                                  entry_file))
                    entry_files.append(entry_file)

                    phi = np.float64(master_h5.positioner(entry, 'phi'))
                    eta = np.float64(master_h5.positioner(entry, 'eta'))
                    nu = np.float64(master_h5.positioner(entry, 'nu'))
                    delta = np.float64(master_h5.positioner(entry, 'del'))

                    qx, qy, qz = hxrd.Ang2Q.area(eta, phi, nu, delta)
                    q_ar[entry_idx, :, 0] = qx.reshape(-1)
                    q_ar[entry_idx, :, 1] = qy.reshape(-1)
                    q_ar[entry_idx, :, 2] = qz.reshape(-1)

                    entry_dtype = master_h5.image_dtype(entry=entry)

                    if img_dtype is None:
                        img_dtype = entry_dtype
                    elif img_dtype != entry_dtype:
                        raise TypeError(
                            'All images in the input HDF5 files should '
                            'be of the same type. Found {0} and {1}.'
                            ''.format(img_dtype, entry_dtype))

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

            step_x = (qx_max - qx_min) / (nx - 1.)
            step_y = (qy_max - qy_min) / (ny - 1.)
            step_z = (qz_max - qz_min) / (nz - 1.)

            bins_rng_x = ([qx_min - step_x / 2., qx_min +
                           (qx_max - qx_min + step_x) - step_x / 2.])
            bins_rng_y = ([qy_min - step_y / 2., qy_min +
                           (qy_max - qy_min + step_y) - step_y / 2.])
            bins_rng_z = ([qz_min - step_z / 2., qz_min +
                           (qz_max - qz_min + step_z) - step_z / 2.])
            bins_rng = [bins_rng_x, bins_rng_y, bins_rng_z]

            qx_idx = qx_min + step_x * np.arange(0, nx, dtype=np.float64)
            qy_idx = qy_min + step_y * np.arange(0, ny, dtype=np.float64)
            qz_idx = qz_min + step_z * np.arange(0, nz, dtype=np.float64)

            # TODO : on windows we may be forced to use shared memory
            # TODO : find why we use more memory when using shared arrays
            #        this shouldnt be the case
            #        (use the same amount as non shared mem)
            # on linux apparently we dont because when fork() is called data is
            # only copied on write.
            # shared histo used by all processes
            # histo_shared = mp_sharedctypes.RawArray(ctypes.c_int32,
            #                                         nx * ny * nz)
            # histo = np.frombuffer(histo_shared, dtype='int32')
            # histo.shape = nx, ny, nz
            # histo[:] = 0
            histo = np.zeros(qspace_dims, dtype=np.int32)

            # shared LUT used by all processes
            # h_lut = None
            # h_lut_shared = None
            h_lut = []
            lut = None

            for h_idx in range(n_entries):
                lut = histogramnd_get_lut(q_ar[h_idx, ...],
                                          bins_rng,
                                          [nx, ny, nz],
                                          last_bin_closed=True)

                # if h_lut_shared is None:
                #     lut_dtype = lut[0].dtype
                #     if lut_dtype == np.int16:
                #         lut_ctype = ctypes.c_int16
                #     elif lut_dtype == np.int32:
                #         lut_ctype = ctypes.c_int32
                #     elif lut_dtype == np.int64:
                #         lut_ctype == ctypes.c_int64
                #     else:
                #         raise TypeError('Unknown type returned by '
                #                         'histogramnd_get_lut : {0}.'
                #                         ''.format(lut.dtype))
                #     h_lut_shared = mp_sharedctypes.RawArray(lut_ctype,
                #                                       n_images * lut[0].size)
                #     h_lut = np.frombuffer(h_lut_shared, dtype=lut_dtype)
                #     h_lut.shape = (n_images, -1)
                #
                # h_lut[h_idx, ...] = lut[0]
                h_lut.append(lut[0])
                histo += lut[1]

            del lut
            del q_ar

            # TODO : split the output file into several files? speedup?
            output_shape = histo.shape

            chunks = (1,
                      max(output_shape[0] // 4, 1),
                      max(output_shape[1] // 4, 1),
                      max(output_shape[2] // 4, 1),)
            qspace_sum_chunks = max(n_images // 10, 1),

            discarded_entries = sorted(all_entries - set(entries))

            _create_result_file(output_f,
                                output_shape,
                                image_binning,
                                sample_roi,
                                sample_x[sample_indices],
                                sample_y[sample_indices],
                                qx_idx,
                                qy_idx,
                                qz_idx,
                                histo,
                                selected_entries=entries,
                                discarded_entries=discarded_entries,
                                compression='lzf',
                                qspace_chunks=chunks,
                                qspace_sum_chunks=qspace_sum_chunks,
                                overwrite=self.__overwrite)

            manager = mp.Manager()
            self.__term_evt = term_evt = manager.Event()

            write_lock = manager.Lock()
            idx_queue = manager.Queue()

            n_proc = self.n_proc
            if n_proc is None or n_proc <= 0:
                n_proc = mp.cpu_count()

            self.__shared_progress = mp_sharedctypes.RawArray(ctypes.c_int32,
                                                              n_proc)
            np.frombuffer(self.__shared_progress, dtype='int32')[:] = 0

            pool = mp.Pool(n_proc,
                           initializer=_init_thread,
                           initargs=(idx_queue,
                                     write_lock,
                                     bins_rng,
                                     qspace_dims,
                                     h_lut,  # _shared,
                                     None,  # lut_dtype,
                                     n_xy,
                                     histo,  # _shared,))
                                     self.__shared_progress,
                                     term_evt,))

            if disp_times:
                class myTimes(object):
                    def __init__(self):
                        self.t_histo = 0.
                        self.t_sum = 0.
                        self.t_mask = 0.
                        self.t_read = 0.
                        self.t_context = 0.
                        self.t_dnsamp = 0.
                        self.t_medfilt = 0.
                        self.t_write = 0.
                        self.t_w_lock = 0.

                    def update(self, arg):
                        (t_read_, t_context_, t_dnsamp_, t_medfilt_, t_histo_,
                         t_mask_, t_sum_, t_write_, t_w_lock_) = arg[2]
                        self.t_histo += t_histo_
                        self.t_sum += t_sum_
                        self.t_mask += t_mask_
                        self.t_read += t_read_
                        self.t_context += t_context_
                        self.t_dnsamp += t_dnsamp_
                        self.t_medfilt += t_medfilt_
                        self.t_write += t_write_
                        self.t_w_lock += t_w_lock_

                res_times = myTimes()
                callback = res_times.update
            else:
                callback = None

            # creating the processes
            results = []
            for th_idx in range(n_proc):
                arg_list = (th_idx,
                            entry_files,
                            entries,
                            img_size,
                            output_f,
                            image_binning,
                            img_dtype)
                res = pool.apply_async(_to_qspace, args=arg_list,
                                       callback=callback)
                results.append(res)

            # sending the image indices
            for result_idx, pos_idx in enumerate(sample_indices):
                idx_queue.put((result_idx, pos_idx))

            # sending the None value to let the threads know that they
            # should return
            for th_idx in range(n_proc):
                idx_queue.put(None)

            pool.close()
            pool.join()

            tb = time.time()

            if disp_times:
                print('TOTAL {0}'.format(tb - ta))
                print('Read {0}'.format(res_times.t_read))
                print('Context {0}'.format(res_times.t_context))
                print('Dn Sample {0}'.format(res_times.t_dnsamp))
                print('Medfilt {0}'.format(res_times.t_medfilt))
                print('Histo {0}'.format(res_times.t_histo))
                print('Mask {0}'.format(res_times.t_mask))
                print('Sum {0}'.format(res_times.t_sum))
                print('Write {0}'.format(res_times.t_write))
                print('(lock : {0})'.format(res_times.t_w_lock))

            proc_results = [result.get() for result in results]
            proc_codes = np.array([proc_result[0]
                                   for proc_result in proc_results])

            rc = self.DONE
            if not np.all(proc_codes == self.DONE):
                if self.ERROR in proc_codes:
                    rc = self.ERROR
                elif self.CANCELED in proc_codes:
                    rc = self.CANCELED
                else:
                    raise ValueError('Unknown return code.')

            if rc != self.DONE:
                errMsg = 'Conversion failed. Process status :'
                for th_idx, result in enumerate(proc_results):
                    errMsg += ('\n- Proc {0} : rc={1}; {2}'
                               ''.format(th_idx, result[0], result[1]))
                self.__set_status(rc, errMsg)
            else:
                self.__set_status(rc)

        except Exception as ex:
            self.__set_status(self.ERROR, str(ex))
        else:
            self.__results = self.output_f

        # TODO : catch exception?
        if self.__callback:
            self.__callback()

        return self.__results

    def wait(self):
        """
        Waits until parsing is done, or returns if it is not running.
        :return:
        """
        if self.__thread:
            self.__thread.join()

    def __running_exception(self):
        """ Raises an exception if a conversion is in progress. """
        if self.is_running():
            raise RuntimeError('Operation not permitted while '
                               'a parse or merge in running.')

    def is_running(self):
        """ Returns True if a conversion is in progress. """
        return self.status == QSpaceConverter.RUNNING
        #self.__thread and self.__thread.is_alive()

    @output_f.setter
    def output_f(self, output_f):
        """ Sets the output file. """
        if not isinstance(output_f, str):
            raise TypeError('output_f must be a string. Received {0}'
                            ''.format(type(output_f)))
        self.__output_f = output_f

    @n_proc.setter
    def n_proc(self, n_proc):
        """ Sets the number of processes to use. If None or 0 the number of
            processes used will be the number returned by
            multiprocessing.cpu_count.
        """
        if n_proc is None:
            self.__n_proc = None
            return

        n_proc = int(n_proc)
        if n_proc <= 0:
            self.__n_proc = None
        else:
            self.__n_proc = n_proc

    def abort(self, wait=True):
        """
        Aborts the current conversion, if any.
        :param wait: set to False to return immediatly without waiting for the
        processes to return.
        :return:
        """
        if self.is_running():
            self.__term_evt.set()
            if wait:
                self.wait()

    def progress(self):
        """
        Returns the progress of the conversion.
        :return:
        """
        if self.__shared_progress:
            progress = np.frombuffer(self.__shared_progress, dtype='int32')
            return progress.max()
        return 0


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
    global idx_queue, \
        write_lock, \
        bins_rng, \
        qspace_size, \
        h_lut_shared, \
        h_lut_dtype, \
        n_xy, \
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
                        qspace_dims,
                        image_binning,
                        sample_roi,
                        pos_x,
                        pos_y,
                        q_x,
                        q_y,
                        q_z,
                        histo,
                        selected_entries,
                        discarded_entries=None,
                        compression='lzf',
                        qspace_chunks=None,
                        qspace_sum_chunks=None,
                        overwrite=False):
    """
    Initializes the output file.
    :param h5_fn: name of the file to initialize
    :param qspace_dims: dimensions of the q space
    :param image_binning: binning applied to the images
    :param pos_x: sample X positions (one for each qspace cube)
    :param pos_y: sample Y positions (one for each qspace cube)
    :param q_x: X coordinates of the qspace cube
    :param q_y: Y coordinates of the qspace cube
    :param q_z: Z coordinates of the qspace cube
    :param histo: histogram (number of hits per element of the qspace elements)
    :param selected_entries: list of input entries used for the conversion
    :param discarded_entries: list of input entries discarded, or None
    :param compression: datasets compression
    :param qspace_chunks: qspace chunking
    :param qspace_sum_chunks:
    :param overwrite: True to force overwriting the file if it already exists.
    :return:
    """

    if not overwrite:
        mode = 'w-'
    else:
        mode = 'w'

    dir_name = os.path.dirname(h5_fn)
    if len(dir_name) > 0 and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    qspace_h5 = QSpaceH5.QSpaceH5Writer(h5_fn, mode=mode)
    qspace_h5.init_file(len(pos_x),
                        qspace_dims,
                        qspace_chunks=qspace_chunks,
                        qspace_sum_chunks=qspace_sum_chunks,
                        compression=compression)
    qspace_h5.set_histo(histo)
    qspace_h5.set_sample_x(pos_x)
    qspace_h5.set_sample_y(pos_y)
    qspace_h5.set_qx(q_x)
    qspace_h5.set_qy(q_y)
    qspace_h5.set_qz(q_z)
    qspace_h5.set_entries(selected_entries, discarded=discarded_entries)
    qspace_h5.set_image_binning(image_binning)
    qspace_h5.set_sample_roi(sample_roi)


def _to_qspace(th_idx,
               entry_files,
               entries,
               img_size,
               output_fn,
               image_binning,
               img_dtype):
    """
    Fonction running in a process. Performs the conversion.
    :param th_idx:
    :param entry_files:
    :param entries:
    :param img_size:
    :param output_fn:
    :param image_binning:
    :param img_dtype:
    :return:
    """
    print('Thread {0} started.'.format(th_idx))

    t_histo = 0.
    t_mask = 0.
    t_sum = 0.
    t_read = 0.
    t_dnsamp = 0.
    t_medfilt = 0.
    t_write = 0.
    t_w_lock = 0.
    t_context = 0.

    output_h5 = QSpaceH5.QSpaceH5Writer(output_fn, mode='r+')

    if shared_progress is not None:
        progress_np = np.frombuffer(shared_progress, dtype='int32')
        progress_np[th_idx] = 0
    else:
        progress_np = None

    # histo = np.frombuffer(histo_shared, dtype='int32')
    # histo.shape = qspace_size
    histo = histo_shared
    mask = histo > 0

    # h_lut = np.frombuffer(h_lut_shared, dtype=h_lut_dtype)
    # h_lut.shape = (n_xy, -1)
    h_lut = h_lut_shared

    img = np.ascontiguousarray(np.zeros(img_size), dtype=img_dtype)

    # TODO : handle case when nav is not a multiple of img_size!!
    # TODO : find why the first version is faster than the second one
    img_shape_1 = img_size[0] // image_binning[0], image_binning[0], img_size[1]
    img_shape_2 = (img_shape_1[0], img_shape_1[2] // image_binning[1],
                   image_binning[1])
    sum_axis_1 = 1
    sum_axis_2 = 2
    # img_shape_1 = img_size[0], img_size[1]/nav[1], nav[1]
    # img_shape_2 = img_size[0]//nav[0], nav[0], img_shape_1[1]
    # sum_axis_1 = 2
    # sum_axis_2 = 1
    avg_weight = 1. / (image_binning[0] * image_binning[1])

    rc = None
    errMsg = None
    try:
        while True:
            if term_evt.is_set():  # noqa
                rc = QSpaceConverter.CANCELED
                raise Exception('conversion aborted')

            next_data = idx_queue.get()
            if next_data is None:
                rc = QSpaceConverter.DONE
                break

            result_idx, image_idx = next_data
            if result_idx % 100 == 0:
                print('#{0}/{1}'.format(result_idx, n_xy))

            cumul = None
            # histo = None

            for entry_idx, entry in enumerate(entries):

                t0 = time.time()

                try:
                    # TODO : there s room for improvement here maybe
                    # (recreating a XsocsH5 instance each time slows down
                    # slows down things a big, not much tho)
                    # TODO : add a lock on the files if there is no SWMR
                    # test if it slows down things much
                    with XsocsH5.XsocsH5(entry_files[entry_idx],
                                         mode='r').image_dset_ctx(entry) \
                            as img_data:  # noqa
                        t1 = time.time()
                        img_data.read_direct(img,
                                             source_sel=np.s_[image_idx],
                                             dest_sel=None)
                        t_context = time.time() - t1
                        # img = img_data[image_idx].astype(np.float64)
                except Exception as ex:
                    raise RuntimeError('Error in proc {0} while reading '
                                       'img {1} from entry {2} ({3}) : {4}.'
                                       ''.format(th_idx, image_idx, entry_idx,
                                                 entry, ex))

                t_read += time.time() - t0
                t0 = time.time()

                if image_binning[0] != 1 or image_binning[1] != 1:
                    intensity = img.reshape(img_shape_1). \
                                    sum(axis=sum_axis_1,
                                        dtype=np.uint32).reshape(img_shape_2). \
                                    sum(axis=sum_axis_2, dtype=np.uint32) * \
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
                    print('EX2 {0}'.format(str(ex)))
                    raise ex

                t_histo += time.time() - t0

            t0 = time.time()
            cumul_sum = cumul.sum(dtype=np.float64)
            t_sum += time.time() - t0

            t0 = time.time()
            # cumul[mask] = cumul[mask]/histo[mask]
            t_mask += time.time() - t0

            t0 = time.time()
            write_lock.acquire()
            t_w_lock += time.time() - t0
            t0 = time.time()
            try:
                output_h5.set_position_data(result_idx, cumul, cumul_sum)
            except Exception as ex:
                raise RuntimeError('Error in proc {0} while writing result '
                                   'for img {1} (idx = {3}) : {2}.)'
                                   ''.format(th_idx, image_idx, ex, result_idx))
            write_lock.release()

            if progress_np is not None:
                progress_np[th_idx] = round(100. * (result_idx + 1.) / n_xy)

            t_write += time.time() - t0
    except Exception as ex:
        if rc is None:
            rc = QSpaceConverter.ERROR
        errMsg = 'In thread {0} : {1}.'.format(th_idx, str(ex))
        term_evt.set()

    if rc is None:
        rc = QSpaceConverter.DONE

    if disp_times:
        print('Thread {0} is done. Times={1}'
              ''.format(th_idx, (t_read, t_context, t_dnsamp,
                                 t_medfilt, t_histo,
                                 t_mask, t_sum, t_write, t_w_lock)))
    return [rc, errMsg, (t_read, t_context, t_dnsamp,
                         t_medfilt, t_histo,
                         t_mask, t_sum, t_write, t_w_lock,)]


def _get_all_params(data_h5f):
    """
    Read the whole data and returns the parameters for each entry.
    Returns a dictionary will the scans as keys and the following fields :
    n_images, n_positions, img_size, beam_energy, chan_per_deg,
    center_chan
    Each of those fields are N elements arrays, where N is the number of
    scans found in the file.
    """
    n_images = []
    n_positions = []
    img_sizes = []
    beam_energies = []
    center_chans = []
    chan_per_degs = []
    angles = []

    with XsocsH5.XsocsH5(data_h5f, mode='r') as master_h5:
        entries = master_h5.entries()

        for entry_idx, entry in enumerate(entries):
            n_image = master_h5.n_images(entry=entry)
            img_size = master_h5.image_size(entry=entry)

            imgnr = master_h5.measurement(entry, 'imgnr')
            n_position = len(imgnr) if imgnr is not None else None

            beam_energy = master_h5.beam_energy(entry=entry)
            chan_per_deg = master_h5.chan_per_deg(entry=entry)
            center_chan = master_h5.direct_beam(entry=entry)

            angle = master_h5.positioner(entry, 'eta')

            n_images.append(n_image)
            n_positions.append(n_position)
            img_sizes.append(img_size)
            beam_energies.append(beam_energy)
            chan_per_degs.append(chan_per_deg)
            center_chans.append(center_chan)
            angles.append(angle)

    result = dict([(scan, dict(scans=entries[idx],
                               n_images=n_images[idx],
                               n_positions=n_positions[idx],
                               img_size=img_sizes[idx],
                               beam_energy=beam_energies[idx],
                               chan_per_deg=chan_per_degs[idx],
                               center_chan=center_chans[idx],
                               angle=angles[idx]))
                   for idx, scan in enumerate(entries)])
    return result


if __name__ == '__main__':
    pass

