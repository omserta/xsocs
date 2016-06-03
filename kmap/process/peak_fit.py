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


def get_peaks(qspace_f, fit_type=FitTypes.GAUSSIAN, indices=None):
        #:returns: a list of tuples (x_pos, y_pos, qx_peak, qy_peak, qz_peak,
        #||q||, i_peak)
    #:rtype: *list*
    
    t_read = 0.
    t_fit = 0.
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

        cube = np.ascontiguousarray(np.zeros(qdata.shape[1:]),
                                    dtype=qdata.dtype)

        x_0 = [1.0, q_x.mean(), 1.0]
        y_0 = [1.0, q_y.mean(), 1.0]
        z_0 = [1.0, q_z.mean(), 1.0]

        res_x = np.full((n_indices, 3), np.nan, dtype=np.float64)
        res_y = np.full((n_indices, 3), np.nan, dtype=np.float64)
        res_z = np.full((n_indices, 3), np.nan, dtype=np.float64)

        success = np.full((n_indices,), True, dtype=bool)

        for i_cube in indices:

            if i_cube % 100 == 0:
                print('Processed cube {0}/{1}.'.format(i_cube, n_points))

            t0 = time.time()
            qdata.read_direct(cube,
                              source_sel=np.s_[i_cube],
                              dest_sel=None)

            t_read += time.time() - t0
            t0 = time.time()

            z_sum = cube.sum(axis=0).sum(axis=0)
            try:
                fit_z = fit_fn(q_z, z_sum, z_0)
            except:
                success[i_cube] = False
            else:
                res_z[i_cube] = fit_z

            
            y_sum = cube.sum(axis=2).sum(axis=0)
            try:
                fit_y = fit_fn(q_y, y_sum, y_0)
            except:
                success[i_cube] = False
            else:
                res_y[i_cube] = fit_y

            x_sum = cube.sum(axis=2).sum(axis=1)
            try:
                fit_x = fit_fn(q_x, x_sum, x_0)
            except:
                success[i_cube] = False
            else:
                res_x[i_cube] = fit_x

            t_fit += time.time() - t0
    
    t_total = time.time() - t_total
    if disp_times:
        print('Times : total={t_total}, read={t_read}, fit={t_fit}.'
              ''.format(t_total=t_total, t_read=t_read, t_fit=t_fit))

    return res_x, res_y, res_z, success

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
        
