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
__date__ = "01/06/2016"
__license__ = "MIT"

import os
import time
import ctypes
import multiprocessing as mp
import multiprocessing.sharedctypes as mp_sharedctypes

import h5py
import numpy as np

from scipy.optimize import leastsq

disp_times = False

class FitTypes(object):
    ALLOWED = range(2)
    GAUSSIAN, CENTROID = ALLOWED


# 1d Gaussian func
_gauss_fn = lambda p, x: (p[0] * (1 / np.sqrt(2 * np.pi * (p[2]**2))) *
                          np.exp(-(x - p[1])**2/(2 * p[2]**2)))
# 1d Gaussian fit
_gauss_fit_err = lambda p, x, y: (_gauss_fn(p, x) - y)


def _qspace_gauss_fit(x, y, v0):
    # TODO : throw exception if fit failed
    result = leastsq(_gauss_fit_err,
                     v0,
                     args=(x, y,),
                     maxfev=100000,
                     full_output=True)
    if result[4] not in [1, 2, 3, 4]:
        raise ValueError('Failed to fit : {0}.'.format(result[3]))

    return result[0]


def get_peaks(qspace_f,
              fit_type=FitTypes.GAUSSIAN,
              indices=None,
              n_proc=None):
        #:returns: a list of tuples (x_pos, y_pos, qx_peak, qy_peak, qz_peak,
        #||q||, i_peak)
    #:rtype: *list*

    t_total = time.time()

    if fit_type not in FitTypes.ALLOWED:
        raise ValueError('Unknown fit type : {0}')
        
    if fit_type == FitTypes.GAUSSIAN:
        fit_fn = _qspace_gauss_fit

    with h5py.File(qspace_f, 'r') as qspace_h5:

        q_x = qspace_h5['bins_edges/x'][:]
        q_y = qspace_h5['bins_edges/y'][:]
        q_z = qspace_h5['bins_edges/z'][:]
        qdata = qspace_h5['qspace']

        n_points = qdata.shape[0]
        
        if indices is None:
            indices = range(n_points)

        n_indices = len(indices)
        
        shared_res = mp_sharedctypes.RawArray(ctypes.c_double, n_indices * 9)
        # TODO : find something better
        shared_success = mp_sharedctypes.RawArray(ctypes.c_bool, n_indices)

        qdata = 0

    manager = mp.Manager()

    read_lock = manager.Lock()
    idx_queue = manager.Queue()

    if n_proc is None:
        n_proc = mp.cpu_count()

    pool = mp.Pool(n_proc,
               initializer=_init_thread,
               initargs=(shared_res,
                         shared_success,
                         fit_fn,
                         (n_indices, 9),
                         idx_queue,
                         qspace_f,
                         read_lock))

    if disp_times:
        class myTimes(object):
            def __init__(self):
                self.t_fit = 0.
                self.t_read = 0.
                self.t_lock = 0.

            def update(self, arg):
                (t_read_, t_lock_, t_fit_) = arg
                self.t_fit += t_fit_
                self.t_read += t_read_
                self.t_lock += t_lock_
        res_times = myTimes()
        callback = res_times.update
    else:
        callback = None

    # creating the processes
    res_list = []
    for th_idx in range(n_proc):
        arg_list = (th_idx,)
        res = pool.apply_async(_fit_process, args=arg_list, callback=callback)
        res_list.append(res)

    # sending the image indices
    for i_cube in indices:
        idx_queue.put(i_cube)

    # sending the None value to let the threads know that they should return
    for th_idx in range(n_proc):
        idx_queue.put(None)
        
    pool.close()
    pool.join()

    #t0 = time.time()
    fit_x = np.ndarray((n_indices, 3), dtype=np.float64)
    fit_y = np.ndarray((n_indices, 3), dtype=np.float64)
    fit_z = np.ndarray((n_indices, 3), dtype=np.float64)
    results = np.frombuffer(shared_res)
    results.shape = n_indices, 9
    fit_x = results[..., 0:3]
    fit_y = results[..., 3:6]
    fit_z = results[..., 6:9]
    #success = np.frombuffer(shared_success)
    success = np.ndarray((n_indices,), dtype=np.bool)
    
    t_total = time.time() - t_total
    if disp_times:
        print('Total : {0}.'.format(t_total))
        print('Read {0} (lock : {1})'.format(res_times.t_read, res_times.t_lock))
        print('Fit {0}'.format(res_times.t_fit))
    return fit_x, fit_y, fit_z, success
        
def _init_thread(shared_res_,
                 shared_success_,
                 fit_fn_,
                 result_shape_,
                 idx_queue_,
                 qspace_f_,
                 read_lock_):

        global shared_res,\
               shared_success,\
               fit_fn,\
               result_shape,\
               idx_queue,\
               qspace_f,\
               read_lock

        shared_res = shared_res_
        shared_success = shared_success_
        fit_fn = fit_fn_
        result_shape = result_shape_
        idx_queue = idx_queue_
        qspace_f = qspace_f_
        read_lock = read_lock_

def _fit_process(th_idx):

    try:
        t_read = 0.
        t_lock = 0.
        t_fit = 0.
        
        results = np.frombuffer(shared_res)
        results.shape = result_shape
        success = np.frombuffer(shared_success, dtype=bool)

        #TODO : timeout to check if it has been canceled
        read_lock.acquire()
        with h5py.File(qspace_f, 'r') as qspace_h5:
            q_x = qspace_h5['bins_edges/x'][:]
            q_y = qspace_h5['bins_edges/y'][:]
            q_z = qspace_h5['bins_edges/z'][:]
            q_shape = qspace_h5['qspace'].shape
            q_dtype = qspace_h5['qspace'].dtype
        read_lock.release()

        cube = np.ascontiguousarray(np.zeros(q_shape[1:]),
                                    dtype=q_dtype)

        x_0 = [1.0, q_x.mean(), 1.0]
        y_0 = [1.0, q_y.mean(), 1.0]
        z_0 = [1.0, q_z.mean(), 1.0]
        
        while True:
            # TODO : timeout
            i_cube = idx_queue.get()

            if i_cube is None:
                break

            if i_cube % 100 == 0:
                print('Processing cube {0}/{1}.'.format(i_cube, result_shape[0]))

            t0 = time.time()
            read_lock.acquire()
            t_lock = time.time() - t0
            t0 = time.time()
            with h5py.File(qspace_f, 'r') as qspace_h5:
                qspace_h5['qspace'].read_direct(cube,
                                                source_sel=np.s_[i_cube],
                                                dest_sel=None)
            read_lock.release()
            t_read += time.time() - t0

            t0 = time.time()

            success_x = True
            success_y = True
            success_z = True

            z_sum = cube.sum(axis=0).sum(axis=0)
            try:
                fit_z = fit_fn(q_z, z_sum, z_0)
            except Exception as ex:
                print 'failed z', ex
                success_z = False

            cube_sum_z = cube.sum(axis=2)

            y_sum = cube_sum_z.sum(axis=0)
            try:
                fit_y = fit_fn(q_y, y_sum, y_0)
            except:
                success_y = False

            x_sum = cube_sum_z.sum(axis=1)
            try:
                fit_x = fit_fn(q_x, x_sum, x_0)
            except:
                success_x = False

            t_fit += time.time() - t0

            success[i_cube] = True

            if success_x:
                results[i_cube, 0:3] = fit_x
            else:
                results[i_cube, 0:3] = np.nan
                success[i_cube] = False

            if success_y:
                results[i_cube, 3:6] = fit_y
            else:
                results[i_cube, 3:6] = np.nan
                success[i_cube] = False

            if success_z:
                results[i_cube, 6:9] = fit_z
            else:
                results[i_cube, 6:9] = np.nan
                success[i_cube] = False
    except Exception as ex:
        print 'EX', ex
        
    times = (t_read, t_lock, t_fit)
    print('Thread {0} done ({1}).'.format(th_idx, times))
    return times
#t_total = time.time() - t_total
#if disp_times:
    #print('Times : total={t_total}, read={t_read}, fit={t_fit}.'
          #''.format(t_total=t_total, t_read=t_read, t_fit=t_fit))

#return res_x, res_y, res_z, success

    #t0 = time.time()

    #v0 = [1.0, qz.mean(), 1.0]
    #qz_peak = leastsq(e_gauss_fit,
                      #v0[:],
                      #args=(qz_idx, (cumul.sum(axis=0)).sum(axis=0)),
                      #maxfev=100000,
                      #full_output=1)[0][1]
    #v0 = [1.0, qy.mean(), 1.0]
    #qy_peak = leastsq(e_gauss_fit,
                      #v0[:],
                      #args=(qy_idx, (cumul.sum(axis=2)).sum(axis=0)),
                      #maxfev=100000,
                      #full_output=1)[0][1]
    #v0 = [1.0, qx.mean(), 1.0]
    #qx_peak = leastsq(e_gauss_fit,
                      #v0[:],
                      #args=(qx_idx, (cumul.sum(axis=2)).sum(axis=1)),
                      #maxfev=100000,
                      #full_output=1)[0][1]
    #i_peak = leastsq(e_gauss_fit,
                     #v0[:],
                     #args=(qx_idx, (cumul.sum(axis=2)).sum(axis=1)),
                     #maxfev=100000,
                     #full_output=1)[0][0]
    #t_fit += time.time() - t0

    #q = np.sqrt(qx_peak**2 + qy_peak**2 + qz_peak**2)

    #t0 = time.time()
    #results = np.frombuffer(g_shared_res)
    #results.shape = n_xy_pos, 5
    #results[image_idx] = (qx_peak,
                          #qy_peak,
                          #qz_peak,
                          #q,
                          #i_peak)
