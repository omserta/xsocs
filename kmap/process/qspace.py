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
import multiprocessing as mp
import multiprocessing.sharedctypes as mp_sharedctypes

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
img_data_tpl = '/{0}/measurement/image_data/data'
measurement_tpl = '/{0}/measurement'
detector_tpl = '/{0}/instrument/image_detector'

# 1d Gaussian func
gauss_fit = lambda p, x: (p[0] * (1 / np.sqrt(2 * np.pi * (p[2]**2))) *
                          np.exp(-(x - p[1])**2/(2 * p[2]**2)))
# 1d Gaussian fit
e_gauss_fit = lambda p, x, y: (gauss_fit(p, x) - y)


def img_2_qpeak(data_h5f,
                output_dir,
                n_bins,
                beam_energy=None,
                center_chan=None,
                chan_per_deg=None,
                nav=(4, 4),
                img_indices=None,
                n_threads=None):
    """
    TODO : roi parameter
    TODO : use histogram_lut when available in silx
    Creates a "master" HDF5 file and one HDF5 per scan. Those scan HDF5 files
    contain spec data (from *spec_fname*) as well as the associated
    image data. This file will either contain all valid scans or the one
    selected using the scan_ids parameter. A valid scan is a scan associated
    with an (existing) image file. Existing output files will be
    overwritten.

    :param data_h5f: path to the HDF5 file containing the scan counters
        and images
    :type data_h5f: `str`

    :param output_dir: folder name into which output data (as well as
        temporary files) will be written (unused at the moment).
    :type output_dir: `str`

    :param n_bins: number of "bins" for the qspace cube (TODO : rephrase)
    :type n_bins: `array_like`

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

    :param img_indices: indices of the images for which the qx/qy/qz peaks
        coordinates will be computed. E.g : if the array [1, 2, 3] is provided,
        only the first 3 acquisitions of each scans will be used.
        (TODO : give example)
    :type img_indices: *optional* `array_like`

    :param n_threads: number of threads to use. If None, the number of threads
        used will be the one returned by multiprocessing.cpu_count().
    :type n_threads: `int`

    :returns: a list of tuples (x_pos, y_pos, qx_peak, qy_peak, qz_peak,
        ||q||, i_peak)
    :rtype: *list*
    """
    # TODO : put beam_energy/center_chan at top level or check
    # that all values are identical
    ta = time.time()

    base_dir = os.path.dirname(data_h5f)

    with h5py.File(data_h5f, 'r') as master_h5:

        entries = sorted(master_h5.keys())
        entry_files = []

        n_entries = len(entries)

        n_xy_pos = None
        n_images = None

        # retrieving some info from the first image to initialize some arrays
        img_data = master_h5[img_data_tpl.format(entries[0])]

        roi = [0, img_data.shape[1], 0, img_data.shape[2]]
#        if roi is None:
#            in_roi = [0, img_data.shape[1], 0, img_data.shape[2]]
#        else:
#            #TODO : values check
#            in_roi = roi
#            img_slice = (slice(roi[0], roi[1]), slice(roi[2], roi[3]))

        measurement = master_h5[measurement_tpl.format(entries[0])]

        n_xy_pos = len(measurement['imgnr'])
        n_images = img_data.shape[0]
        img_x, img_y = img_data.shape[1:3]

        detector = master_h5[detector_tpl.format(entries[0])]

        if beam_energy is None:
            try:
                beam_energy = detector.get('beam_energy')[()]
            except:
                beam_energy = None

            if beam_energy is None:
                raise ValueError('The beam_energy value found in the '
                                 'file is None (or missing).')

        if chan_per_deg is None:
            try:
                chan_per_deg_dim0 = detector.get('chan_per_deg_dim0')[()]
            except:
                chan_per_deg_dim0 = None

            try:
                chan_per_deg_dim1 = detector.get('chan_per_deg_dim1')[()]
            except:
                chan_per_deg_dim1 = None

            if chan_per_deg_dim0 is None:
                raise ValueError('The chan_per_deg_dim0 value found in the '
                                 'file is None (or missing).')
            if chan_per_deg_dim1 is None:
                raise ValueError('The chan_per_deg_dim1 value found in the '
                                 'file is None (or missing).')
        else:
            chan_per_deg_dim0 = chan_per_deg[0]
            chan_per_deg_dim1 = chan_per_deg[1]

        if center_chan is None:
            try:
                center_chan_dim0 = detector.get('center_chan_dim0')[()]
            except:
                center_chan_dim0 = None

            try:
                center_chan_dim1 = detector.get('center_chan_dim1')[()]
            except:
                center_chan_dim1 = None

            if center_chan_dim0 is None:
                raise ValueError('The center_chan_dim0 value found in the '
                                 'file is None (or missing).')
            if center_chan_dim1 is None:
                raise ValueError('The center_chan_dim1 value found in the '
                                 'file is None (or missing).')
        else:
            center_chan_dim0 = center_chan[0]
            center_chan_dim1 = center_chan[1]

        # TODO value testing
        if img_indices is None:
            img_indices = np.arange(n_images)

        nx, ny, nz = n_bins

        qconv = xu.experiment.QConversion(['y-'],
                                          ['z+', 'y-'],
                                          [1, 0, 0])

        print('Parameters :')
        print('\t- beam energy  : {0}'.format(beam_energy))
        print('\t- center chan  : [{0}, {1}]'.format(center_chan_dim0,
                                                     center_chan_dim1))
        print('\t- chan per deg : [{0}, {1}]'.format(chan_per_deg_dim0,
                                                     chan_per_deg_dim1))

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
                             cch1=center_chan_dim0,
                             cch2=center_chan_dim1,
                             Nch1=img_x,
                             Nch2=img_y,
                             chpdeg1=chan_per_deg_dim0,
                             chpdeg2=chan_per_deg_dim1,
                             Nav=nav,
                             roi=roi)

        # shape of the array that will store the qx/qy/qz for all
        # rocking angles
        q_shape = (n_entries,
                   img_data.shape[1] // nav[0] * img_data.shape[2] // nav[1],
                   3)

        # then the array
        q_ar = np.zeros(q_shape, dtype=np.float64)

        for entry_idx, entry in enumerate(entries):
            # TODO : handle the case when all the data is contained
            # in a single file?
            entry_file = master_h5[entry].file.filename
            if not os.path.isabs(entry_file):
                entry_file = os.path.abspath(os.path.join(base_dir,
                                                          entry_file))
            entry_files.append(entry_file)

            positioners = master_h5[positioners_tpl.format(entry)]
            img_data = master_h5[img_data_tpl.format(entry)]
            measurement = master_h5[measurement_tpl.format(entry)]

            n_xy = len(measurement['imgnr'])
            n_img = img_data.shape[0]
            img_shape = img_data.shape

            # some minimal checks
            # TODO : are we sure the number of images will always be the same
            #   (e.g : 1 failed x,y scan that is done a second time)?
            if n_xy != n_xy_pos:
                raise ValueError('TODO')

            # some minimal checks
            if n_img != n_images:
                raise ValueError('TODO')
            if n_img != n_xy_pos:
                raise ValueError('TODO')
            if img_shape[1] != img_x:
                raise ValueError('TODO')
            if img_shape[2] != img_y:
                raise ValueError('TODO')

            eta = np.float64(positioners['eta'][()])
            nu = np.float64(positioners['nu'][()])
            delta = np.float64(positioners['del'][()])

            qx, qy, qz = hxrd.Ang2Q.area(eta, nu, delta)
            q_ar[entry_idx, :, 0] = qx.reshape(-1)
            q_ar[entry_idx, :, 1] = qy.reshape(-1)
            q_ar[entry_idx, :, 2] = qz.reshape(-1)

        # custom bins range to have the same histo as xrayutilities.gridder3d
        # the last bin extends beyond q_max
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

        # TODO : find why the first version is faster than the second one
        img_shape_1 = img_shape[1]//nav[0], nav[0], img_shape[2]
        img_shape_2 = img_shape_1[0], img_shape_1[2]//nav[1], nav[1]
        sum_axis_1 = 1
        sum_axis_2 = 2
        # img_shape_1 = img_shape[1], img_shape[2]/nav[1], nav[1]
        # img_shape_2 = img_shape[1]//nav[0], nav[0], img_shape_1[1]
        # sum_axis_1 = 2
        # sum_axis_2 = 1
        avg_weight = 1./(nav[0]*nav[1])

        # h_lut = None
        histo = np.zeros([nx, ny, nz], dtype=np.int32)
        h_lut = []

        for h_idx in range(n_entries):
            lut = histogramnd_get_lut(q_ar[h_idx, ...],
                                      bins_rng,
                                      [nx, ny, nz],
                                      last_bin_closed=True)

            h_lut.append(lut[0])
            histo += lut[1]

        mask = histo > 0

        measurement = master_h5[measurement_tpl.format(entries[0])]
        sample_x = measurement['adcX'][:]
        sample_y = measurement['adcY'][:]

    manager = mp.Manager()

    # array to store the results
    # qx_peak, qy_peak, qz_peak, ||q||, I_peak
    shared_res = mp_sharedctypes.RawArray(ctypes.c_double, n_xy_pos*5)

    entry_locks = [manager.Lock() for n in range(n_entries)]
    idx_queue = manager.Queue()

    if n_threads is None:
        n_threads = mp.cpu_count()

    pool = mp.Pool(n_threads,
                   initializer=_init_thread,
                   initargs=(shared_res,
                             idx_queue,
                             entries,
                             entry_files,
                             entry_locks,
                             q_ar,
                             bins_rng,
                             n_bins,
                             histo,
                             h_lut,
                             mask,
                             img_shape_1,
                             img_shape_2,
                             sum_axis_1,
                             sum_axis_2,
                             avg_weight,
                             n_xy,
                             qx,
                             qy,
                             qz,
                             qx_idx,
                             qy_idx,
                             qz_idx,
                             n_xy_pos,))

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

            def update(self, arg):
                (t_read_, t_dnsamp_, t_medfilt_, t_histo_,
                 t_mask_, t_fit_, t_write_) = arg
                self.t_histo += t_histo_
                self.t_fit += t_fit_
                self.t_mask += t_mask_
                self.t_read += t_read_
                self.t_dnsamp += t_dnsamp_
                self.t_medfilt += t_medfilt_
                self.t_write += t_write_
        res_times = myTimes()
        callback = res_times.update
    else:
        callback = None

    # creating the processes
    for th_idx in range(n_threads):
        arg_list = (th_idx,)
        res = pool.apply_async(_get_q_peak, args=arg_list, callback=callback)
        res_list.append(res)

    # sending the image indices
    for image_idx in img_indices:
        idx_queue.put(image_idx)

    # sending the None value to let the threads know that they should return
    for th_idx in range(n_threads):
        idx_queue.put(None)

    pool.close()
    pool.join()

    for res in res_list:
        res_val = res.get()
        if isinstance(res_val, Exception):
            raise res_val

    t0 = time.time()
    final_results = np.ndarray((n_xy_pos, 7), dtype=np.float64)
    results = np.frombuffer(shared_res).copy()
    results.shape = n_xy_pos, 5
    final_results[:, 0] = sample_x
    final_results[:, 1] = sample_y
    final_results[:, 2:] = results
    print('final time', time.time() - t0)

    tb = time.time()

    if(disp_times):
        print('TOTAL', tb - ta)
        print('Read', res_times.t_read)
        print('Dn Sample', res_times.t_dnsamp)
        print('Medfilt', res_times.t_medfilt)
        print('Histo', res_times.t_histo)
        print('Mask', res_times.t_mask)
        print('Fit', res_times.t_fit)
        print('Write', res_times.t_write)

    return final_results


def _init_thread(shared_res_,
                 idx_queue_,
                 entries_,
                 entry_files_,
                 entry_locks_,
                 q_ar_,
                 bins_rng_,
                 n_bins_,
                 histo_,
                 h_lut_,
                 mask_,
                 img_shape_1_,
                 img_shape_2_,
                 sum_axis_1_,
                 sum_axis_2_,
                 avg_weight_,
                 n_xy_,
                 qx_,
                 qy_,
                 qz_,
                 qx_idx_,
                 qy_idx_,
                 qz_idx_,
                 n_xy_pos_):

        global g_shared_res,\
            idx_queue,\
            entries,\
            entry_files,\
            entry_locks,\
            q_ar,\
            bins_rng,\
            n_bins,\
            histo,\
            h_lut,\
            mask,\
            img_shape_1,\
            img_shape_2,\
            sum_axis_1,\
            sum_axis_2,\
            avg_weight,\
            n_xy,\
            qx,\
            qy,\
            qz,\
            qx_idx,\
            qy_idx,\
            qz_idx,\
            n_xy_pos
        g_shared_res = shared_res_
        idx_queue = idx_queue_
        entries = entries_
        entry_files = entry_files_
        entry_locks = entry_locks_
        q_ar = q_ar_
        bins_rng = bins_rng_
        n_bins = n_bins_
        histo = histo_
        h_lut = h_lut_
        mask = mask_
        img_shape_1 = img_shape_1_
        img_shape_2 = img_shape_2_
        sum_axis_1 = sum_axis_1_
        sum_axis_2 = sum_axis_2_
        avg_weight = avg_weight_
        n_xy = n_xy_
        qx = qx_
        qy = qy_
        qz = qz_
        qx_idx = qx_idx_
        qy_idx = qy_idx_
        qz_idx = qz_idx_
        n_xy_pos = n_xy_pos_


def _get_q_peak(th_idx):

    print('Thread {0} started.'.format(th_idx))

    t_histo = 0.
    t_fit = 0.
    t_mask = 0.
    t_read = 0.
    t_dnsamp = 0.
    t_medfilt = 0.
    t_write = 0.

    while True:
        image_idx = idx_queue.get()
        if image_idx is None:
            print('Thread {0} is done. Times={1}'
                  ''.format(th_idx, (t_read, t_dnsamp,
                                     t_medfilt, t_histo,
                                     t_mask, t_fit, t_write)))
            if disp_times:
                return (t_read, t_dnsamp,
                        t_medfilt, t_histo,
                        t_mask, t_fit, t_write,)
            else:
                return None

        if image_idx % 100 == 0:
            print('#{0}/{1}'.format(image_idx, n_xy))

        cumul = None
        # histo = None

        for entry_idx, entry in enumerate(entries):

            t0 = time.time()

            entry_locks[entry_idx].acquire()
            try:
                with h5py.File(entry_files[entry_idx], 'r') as entry_h5:
                    img_data = entry_h5[img_data_tpl.format(entry)]
                    img = img_data[image_idx].astype(np.float64)
            except Exception as ex:
                print('Error in proc {0} while reading img {1} from entry '
                      '{2} ({3}) : {4}.'
                      ''.format(th_idx, image_idx, entry_idx, entry, ex))
            entry_locks[entry_idx].release()

            t_read += time.time() - t0
            t0 = time.time()

            intensity = img.reshape(img_shape_1).\
                sum(axis=sum_axis_1).reshape(img_shape_2).\
                sum(axis=sum_axis_2) *\
                avg_weight
            # intensity = xu.blockAverage2D(img, nav[0], nav[1], roi=roi)

            t_dnsamp += time.time() - t0
            t0 = time.time()

            # intensity = medfilt2d(intensity, 3)
            intensity = medfilt2D(intensity, kernel=[3, 3], n_threads=None)

            t_medfilt += time.time() - t0
            t0 = time.time()

            try:
                cumul = histogramnd_from_lut(intensity.reshape(-1),
                                             h_lut[entry_idx],
                                             shape=n_bins,
                                             weighted_histo=cumul,
                                             dtype=np.float64)
            except Exception as ex:
                print 'EX', ex
                raise ex

#            histo, cumul = histogramnd(q_ar[entry_idx, ...],
#                                       bins_rng,
#                                       n_bins,
#                                       weights=intensity.reshape(-1),
#                                       cumul=cumul,
#                                       histo=histo,
#                                       last_bin_closed=True)

            t_histo += time.time() - t0

        t0 = time.time()

        # mask = histo > 0

        cumul[mask] = cumul[mask]/histo[mask]

        t_mask += time.time() - t0

        t0 = time.time()

        v0 = [1.0, qz.mean(), 1.0]
        qz_peak = leastsq(e_gauss_fit,
                          v0[:],
                          args=(qz_idx, (cumul.sum(axis=0)).sum(axis=0)),
                          maxfev=100000,
                          full_output=1)[0][1]
        v0 = [1.0, qy.mean(), 1.0]
        qy_peak = leastsq(e_gauss_fit,
                          v0[:],
                          args=(qy_idx, (cumul.sum(axis=2)).sum(axis=0)),
                          maxfev=100000,
                          full_output=1)[0][1]
        v0 = [1.0, qx.mean(), 1.0]
        qx_peak = leastsq(e_gauss_fit,
                          v0[:],
                          args=(qx_idx, (cumul.sum(axis=2)).sum(axis=1)),
                          maxfev=100000,
                          full_output=1)[0][1]
        i_peak = leastsq(e_gauss_fit,
                         v0[:],
                         args=(qx_idx, (cumul.sum(axis=2)).sum(axis=1)),
                         maxfev=100000,
                         full_output=1)[0][0]
        t_fit += time.time() - t0

        q = np.sqrt(qx_peak**2 + qy_peak**2 + qz_peak**2)

        t0 = time.time()
        results = np.frombuffer(g_shared_res)
        results.shape = n_xy_pos, 5
        results[image_idx] = (qx_peak,
                              qy_peak,
                              qz_peak,
                              q,
                              i_peak)
        t_write += time.time() - t0
