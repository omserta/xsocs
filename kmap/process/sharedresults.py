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

from __future__ import absolute_import

__authors__ = ["D. Naudet"]
__date__ = "01/06/2016"
__license__ = "MIT"


import ctypes
import multiprocessing.sharedctypes as mp_sharedctypes

import numpy as np

from .fitresults import FitResult


class FitTypes(object):
    ALLOWED = range(2)
    GAUSSIAN, CENTROID = ALLOWED


class FitSharedResults(object):
    def __init__(self,
                 n_points=None,
                 n_params=None,
                 shared_results=None,
                 shared_status=None):
        super(FitSharedResults, self).__init__()
        assert n_points is not None, n_params is not None

        self._shared_qx_results = None
        self._shared_qy_results = None
        self._shared_qz_results = None

        self._shared_qx_status = None
        self._shared_qy_status = None
        self._shared_qz_status = None

        self._npy_qx_results = None
        self._npy_qy_results = None
        self._npy_qz_results = None

        self._npy_qx_status = None
        self._npy_qy_status = None
        self._npy_qz_status = None

        self._n_points = n_points
        self._n_params = n_params

        self._init_shared_results(shared_results)
        self._init_shared_status(shared_status)
        self._init_npy_results()
        self._init_npy_status()

    def _init_shared_results(self, shared_results=None):
        if shared_results is None:
            self._shared_qx_results = mp_sharedctypes.RawArray(
                ctypes.c_double, self._n_points * self._n_params)
            self._shared_qy_results = mp_sharedctypes.RawArray(
                ctypes.c_double, self._n_points * self._n_params)
            self._shared_qz_results = mp_sharedctypes.RawArray(
                ctypes.c_double, self._n_points * self._n_params)
        else:
            self._shared_qx_results = shared_results[0]
            self._shared_qy_results = shared_results[1]
            self._shared_qz_results = shared_results[2]

    def _init_shared_status(self, shared_status=None):
        if shared_status is None:
            self._shared_qx_status = mp_sharedctypes.RawArray(
                ctypes.c_bool, self._n_points)
            self._shared_qy_status = mp_sharedctypes.RawArray(
                ctypes.c_bool, self._n_points)
            self._shared_qz_status = mp_sharedctypes.RawArray(
                ctypes.c_bool, self._n_points)
        else:
            self._shared_qx_status = shared_status[0]
            self._shared_qy_status = shared_status[1]
            self._shared_qz_status = shared_status[2]

    def _init_npy_results(self):
        self._npy_qx_results = np.frombuffer(self._shared_qx_results)
        self._npy_qx_results.shape = self._n_points, self._n_params
        self._npy_qy_results = np.frombuffer(self._shared_qy_results)
        self._npy_qy_results.shape = self._n_points, self._n_params
        self._npy_qz_results = np.frombuffer(self._shared_qz_results)
        self._npy_qz_results.shape = self._n_points, self._n_params

    def _init_npy_status(self):
        self._npy_qx_status = np.frombuffer(self._shared_qx_status,
                                            dtype=bool)
        self._npy_qy_status = np.frombuffer(self._shared_qy_status,
                                            dtype=bool)
        self._npy_qz_status = np.frombuffer(self._shared_qz_status,
                                            dtype=bool)

    def set_qx_results(self, idx, results, status):
        self._npy_qx_results[idx] = results
        self._npy_qx_status[idx] = status

    def set_qy_results(self, idx, results, status):
        self._npy_qy_results[idx] = results
        self._npy_qy_status[idx] = status

    def set_qz_results(self, idx, results, status):
        self._npy_qz_results[idx] = results
        self._npy_qz_status[idx] = status

    def local_copy(self):
        shared_results = (self._shared_qx_results,
                          self._shared_qy_results,
                          self._shared_qz_results)
        shared_status = (self._shared_qx_status,
                         self._shared_qy_status,
                         self._shared_qz_status)
        return FitSharedResults(n_points=self._n_points,
                                n_params=self._n_params,
                                shared_results=shared_results,
                                shared_status=shared_status)

    def fit_results(self, *args, **kwargs):
        raise NotImplementedError('')


class GaussianResults(FitSharedResults):
    def __init__(self,
                 n_points=None,
                 shared_results=None,
                 shared_status=None):
        super(GaussianResults, self).__init__(n_points=n_points,
                                              n_params=3,
                                              shared_results=shared_results,
                                              shared_status=shared_status)

    def fit_results(self, *args, **kwargs):
        qx_results = self._npy_qx_results
        qy_results = self._npy_qy_results
        qz_results = self._npy_qz_results

        qx_status = self._npy_qx_status
        qy_status = self._npy_qy_status
        qz_status = self._npy_qz_status

        fit_name = 'Gaussian'
        results = FitResult(fit_name, *args, **kwargs)
        results.add_qx_result('gaussian', 'intensity', qx_results[:, 0].ravel())
        results.add_qx_result('gaussian', 'position', qx_results[:, 1].ravel())
        results.add_qx_result('gaussian', 'width', qx_results[:, 2].ravel())
        results.set_qx_status('gaussian', qx_status)

        results.add_qy_result('gaussian', 'intensity', qy_results[:, 0].ravel())
        results.add_qy_result('gaussian', 'position', qy_results[:, 1].ravel())
        results.add_qy_result('gaussian', 'width', qy_results[:, 2].ravel())
        results.set_qy_status('gaussian', qy_status)

        results.add_qz_result('gaussian', 'intensity', qz_results[:, 0].ravel())
        results.add_qz_result('gaussian', 'position', qz_results[:, 1].ravel())
        results.add_qz_result('gaussian', 'width', qz_results[:, 2].ravel())
        results.set_qz_status('gaussian', qz_status)

        return results


class CentroidResults(FitSharedResults):
    def __init__(self,
                 n_points=None,
                 shared_results=None,
                 shared_status=None):
        super(CentroidResults, self).__init__(n_points=n_points,
                                              n_params=3,
                                              shared_results=shared_results,
                                              shared_status=shared_status)

    def fit_results(self, *args, **kwargs):
        qx_results = self._npy_qx_results
        qy_results = self._npy_qy_results
        qz_results = self._npy_qz_results

        qx_status = self._npy_qx_status
        qy_status = self._npy_qy_status
        qz_status = self._npy_qz_status

        fit_name = 'Centroid'
        results = FitResult(fit_name, *args, **kwargs)

        results.add_qx_result('centroid', 'I', qx_results[:, 0].ravel())
        results.add_qx_result('centroid', 'COM',
                              qx_results[:, 1].ravel())
        results.add_qx_result('centroid', 'Max',
                              qx_results[:, 2].ravel())
        results.set_qx_status('centroid', qx_status)

        results.add_qy_result('centroid', 'I', qy_results[:, 0].ravel())
        results.add_qy_result('centroid',
                              'COM',
                              qy_results[:, 1].ravel())
        results.add_qy_result('centroid', 'Max',
                              qy_results[:, 2].ravel())
        results.set_qy_status('centroid', qy_status)

        results.add_qz_result('centroid', 'I', qz_results[:, 0].ravel())
        results.add_qz_result('centroid',
                              'COM',
                              qz_results[:, 1].ravel())
        results.add_qz_result('centroid', 'Max',
                              qz_results[:, 2].ravel())
        results.set_qz_status('centroid', qz_status)

        return results

if __name__ == '__main__':
    pass
