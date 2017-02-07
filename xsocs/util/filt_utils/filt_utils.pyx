# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
# ############################################################################*/

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "20/04/2016"

import time as _time
cimport cython
import numpy as _np
cimport numpy as _np
from libc.stdio cimport printf
from cython.parallel import prange, threadid

ctypedef fused image_t:
    _np.uint16_t
    _np.uint32_t
    _np.int64_t
    _np.float64_t
    _np.float32_t
    _np.int32_t

def medfilt2D(image,
              kernel=(3, 3),
              n_threads=None):
                  
    #TODO make sure it s a 2d array
    #TODO check for odd kernel value
    
    kernel = _np.array(kernel, ndmin=1, dtype=image.dtype)
    
    tmp_arrays = _np.zeros((image.shape[0], kernel.prod()), dtype=image.dtype)
    
    result = _np.zeros(image.shape, dtype=image.dtype)
    
    image_c = _np.ascontiguousarray(image.reshape(-1))
    
    kernel_c = _np.ascontiguousarray(kernel.reshape(-1),
                                    dtype=_np.int32)

    result_c = _np.ascontiguousarray(result.reshape(-1))
    
    tmp_arrays_c = _np.ascontiguousarray(tmp_arrays.reshape(-1))
    
    n_x = image.shape[0]
    n_y = image.shape[1]
    
    if n_threads is None:
        n_threads = 4
    
    run_medfilt(image_c,
                         n_x,
                         n_y,
                         kernel_c,
                         tmp_arrays_c,
                         n_threads,
                         result_c)
                         
#    _medfilt2D_fused(image_c,
#                         n_x,
#                         n_y,
##                         0, n_x,
##                         0, n_y,
#                         kernel_c,
#                         tmp_arrays_c,
#                         result_c)
                         
    return result
  
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def run_medfilt(image_t[::1] i_image,
                     int n_x,
                     int n_y,
                     int[::1] i_kernel,
                     image_t[::1] tmp_arrays,
                     int n_threads,
                     image_t[::1] o_image):
    cdef:
        int th

    for th in prange(0, n_x, 1,
                     nogil=True,
                     chunksize=1,
                     schedule='static',
                     num_threads=n_threads):
        _medfilt2D_fused_mp(i_image,
                         n_x,
                         n_y,
                         th, th+1,
                         0, n_y,
                         i_kernel,
                         &(tmp_arrays[th*i_kernel[0]*i_kernel[1]]),#tmp_arrays[i_kernel[1]*th:(i_kernel[1]+1)*th],# + th * i_kernel[1],
                         o_image)
    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _swap(image_t* array, int i, int j) nogil:
    cdef:
        image_t tmp

    tmp = array[i]
    array[i] = array[j]
    array[j] = tmp



#@cython.wraparound(False)
#@cython.boundscheck(False)
#@cython.initializedcheck(False)
#@cython.nonecheck(False)
#@cython.cdivision(True)
#cdef int _q_select_2(image_t[::1] array,
#                   int size,
#                   int pos_idx) nogil:

#    cdef:
#        image_t pival
#        int piv_idx
#        int left
#        int right
#        int l_idx
    
#    left = 0
#    right = size - 1
    
#    while True:
#        if left == right:
#            return array[pos_idx]
            
#        piv_idx = (left + right) // 2
        
#        # =========
#        # partition
#        # =========
#        pival = array[piv_idx]
#        _swap(array, piv_idx, right)
        
#        l_idx = left
#        for i in range(left, right):
#            if array[i] < pival:
#                _swap(array, l_idx, i)
#                l_idx += 1
                
#        _swap(array, right, l_idx)
        
#        # =========
#        # =========        
        
#        if l_idx == pos_idx:
#            return array[pos_idx]
            
#        if pos_idx < l_idx:
#            right = l_idx - 1
#        else:
#            left = l_idx + 1
            
            
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef image_t _q_select(image_t* array,
                   int size,
                   int pos_idx) nogil:
                         
    cdef:
        image_t pival
        int piv_idx = 0
        int left = 0
        int right = 0
        int l_idx = 0
        int h_idx = 0
        int middle = 0
        
    left = 0
    right = size - 1
    middle = 0
    l_idx = 0
    h_idx = 0
    
#    median = idx(low + high) // 2

    while True:
        if right <= left:
            return array[pos_idx]
        if right == (left + 1):
            if array[left] > array[right]:
                _swap(array, left, right)
            return array[pos_idx]
        
        middle = (left + right) // 2
        
        if array[middle] > array[right]: _swap(array, middle, right)
        if array[left] > array[right]: _swap(array, left, right)
        if array[middle] > array[left]: _swap(array, middle, left)
        
        _swap(array, middle, left + 1)
        
        l_idx = left + 1
        h_idx = right
        
        while True:
            while True:
                l_idx += 1
                if array[left] <= array[l_idx]: break
            while True:
                h_idx -= 1
                if array[h_idx] <= array[left]: break
            if h_idx < l_idx:
                break
            _swap(array, l_idx, h_idx)
            
        _swap(array, left, h_idx)
    
        if h_idx <= pos_idx:
            left = l_idx
        if h_idx >= pos_idx:
            right = h_idx - 1

#@cython.wraparound(False)
#@cython.boundscheck(False)
#@cython.initializedcheck(False)
#@cython.nonecheck(False)
#@cython.cdivision(True)
#cdef void _medfilt2D_fused(image_t[::1] i_image,
#                     int n_x,
#                     int n_y,
#                     int[::1] i_kernel,
#                     image_t[::1] tmp_arrays,
#                     image_t[::1] o_image) nogil:

#    cdef:
#        int k_x = i_kernel[0]
#        int k_x_half = (i_kernel[0] / 2)
#        int k_y = i_kernel[1]
#        int k_y_half = (i_kernel[1] / 2)
#        int k_size = i_kernel[0] * i_kernel[1]
#        int k_half = (k_size / 2)
#        image_t tmp = 0
#        bint swap = False
#        int x_idx, y_idx = 0
#        int row_idx = 0
#        int col_idx = 0
#        int tmp_idx = 0
#        int tmp_row_offset = 0
#        int med_idx_offset = 0
#        int start_offset = 0
#        int img_offset = 0
#        int row_start, row_end
#        int col_start, col_end

#    # corners are always 0, since there are k_half + 1 elements
#    #   outside the image (i.e : k_half + 1 elements equal to 0)
#    #   and thus the median value is always 0
    
#    med_idx_offset =  k_half - k_size
    
#    start_offset = -n_y
    
#    for x_idx in range(0, n_x):
        
#        if x_idx < k_x_half:
#            row_start = 0
#            row_end = x_idx + k_x_half + 1
#        elif x_idx >= n_x - k_x_half:
#            row_start = -k_x_half
#            row_end = n_x - x_idx
#        else:
#            row_start = -k_x_half
#            row_end = k_x_half + 1
        
#        start_offset += n_y
        
#        for y_idx in range(0, n_y):
            
#            if y_idx < k_y_half:
#                col_start = 0
#                col_end = y_idx + k_y_half + 1
#                if (col_end - col_start) * (row_end - row_start) <= k_half:
#                    o_image[x_idx * n_y + y_idx] = 0
#                    continue
#            elif y_idx >= n_y - k_y_half:
#                col_start = -k_y_half
#                col_end = n_y - y_idx
#                if (col_end - col_start) * (row_end - row_start) <= k_half:
#                    o_image[x_idx * n_y + y_idx] = 0
#                    continue
#            else:
#                col_start = -k_y_half
#                col_end = k_y_half + 1
                
#            tmp_idx = 0
#            img_offset = start_offset + row_start * n_y + y_idx
            
#            for row_idx in range(row_start, row_end):
#                for col_idx in range(col_start, col_end):
#                    tmp_arrays[tmp_idx] = i_image[img_offset + col_idx]
#                    tmp_idx += 1
                    
#                img_offset += n_y

#            o_image[x_idx * n_y + y_idx] = _q_select(tmp_arrays,
#                                                     tmp_idx,
#                                                     med_idx_offset + tmp_idx)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _medfilt2D_fused_mp(image_t[::1] i_image,
                     int n_x,
                     int n_y,
                     int x_start,
                     int x_end,
                     int y_start,
                     int y_end,
                     int[::1] i_kernel,
                     image_t * tmp_arrays,
                     image_t[::1] o_image) nogil:

    cdef:
        int k_x = i_kernel[0]
        int k_x_half = (i_kernel[0] / 2)
        int k_y = i_kernel[1]
        int k_y_half = (i_kernel[1] / 2)
        int k_size = i_kernel[0] * i_kernel[1]
        int k_half = (k_size / 2)
        image_t tmp = 0
        bint swap = False
        int x_idx, y_idx = 0
        int row_idx = 0
        int col_idx = 0
        int tmp_idx = 0
        int tmp_row_offset = 0
        int med_idx_offset = 0
        int start_offset = 0
        int img_offset = 0
        int row_start, row_end
        int col_start, col_end

    # corners are always 0, since there are k_half + 1 elements
    #   outside the image (i.e : k_half + 1 elements equal to 0)
    #   and thus the median value is always 0
    
    med_idx_offset =  k_half - k_size
    
    start_offset = (x_start-1) * n_y
    
    for x_idx in range(x_start, x_end):
        
        if x_idx < k_x_half:
            row_start = 0
            row_end = x_idx + k_x_half + 1
        elif x_idx >= n_x - k_x_half:
            row_start = -k_x_half
            row_end = n_x - x_idx
        else:
            row_start = -k_x_half
            row_end = k_x_half + 1
        
        start_offset += n_y
        
        for y_idx in range(y_start, y_end):
            
            if y_idx < k_y_half:
                col_start = 0
                col_end = y_idx + k_y_half + 1
                if (col_end - col_start) * (row_end - row_start) <= k_half:
                    o_image[x_idx * n_y + y_idx] = 0
                    continue
            elif y_idx >= n_y - k_y_half:
                col_start = -k_y_half
                col_end = n_y - y_idx
                if (col_end - col_start) * (row_end - row_start) <= k_half:
                    o_image[x_idx * n_y + y_idx] = 0
                    continue
            else:
                col_start = -k_y_half
                col_end = k_y_half + 1
                
            tmp_idx = 0
            img_offset = start_offset + row_start * n_y + y_idx
            
            for row_idx in range(row_start, row_end):
                for col_idx in range(col_start, col_end):
                    tmp_arrays[tmp_idx] = i_image[img_offset + col_idx]
                    tmp_idx += 1
                    
                img_offset += n_y

            o_image[x_idx * n_y + y_idx] = _q_select(tmp_arrays,
                                                     tmp_idx,
                                                     med_idx_offset + tmp_idx)
