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
__date__ = "15/09/2016"

import weakref
from contextlib import contextmanager

import numpy as _np

from .XsocsH5Base import XsocsH5Base


class QSpaceH5(XsocsH5Base):
    qspace_path = 'Data/qspace'
    qx_path = 'Data/qx'
    qy_path = 'Data/qy'
    qz_path = 'Data/qz'
    histo_path = 'Data/histo'
    sample_x_path = 'Data/sample_x'
    sample_y_path = 'Data/sample_y'
    qspace_sum_path = 'Data/qspace_sum'
    image_shape_path = 'Data/image_shape'
    params_path = 'Params/'
    entries_path = 'params/entries'

    def __init__(self, h5_f, mode='r'):
        super(QSpaceH5, self).__init__(h5_f, mode=mode)

    @contextmanager
    def qspace_dset_ctx(self):
        """
        Context manager for the image dataset.
        WARNING: only to be used as a context manager!
        WARNING: the data set must exist. see also QSpaceH5Writer.init_cube
        """
        with self._get_file() as h5_file:
            qspace_dset = h5_file[QSpaceH5.qspace_path]
            yield weakref.proxy(qspace_dset)
            del qspace_dset

    @contextmanager
    def qspace_sum_dset_ctx(self):
        """
        Context manager for the image dataset.
        WARNING: only to be used as a context manager!
        WARNING: the data set must exist. see also QSpaceH5Writer.init_cube
        """
        with self._get_file() as h5_file:
            qspace_sum_dset = h5_file[QSpaceH5.qspace_sum_path]
            yield weakref.proxy(qspace_sum_dset)
            del qspace_sum_dset

    qspace = property(lambda self: self._get_array_data(QSpaceH5.qspace_path))

    def qspace_slice(self, index):
        with self.item_context(self.qspace_path) as dset:
            return dset[index]

    qx = property(lambda self: self._get_array_data(QSpaceH5.qx_path))

    qy = property(lambda self: self._get_array_data(QSpaceH5.qy_path))

    qz = property(lambda self: self._get_array_data(QSpaceH5.qz_path))

    sample_x = property(lambda self:
                        self._get_array_data(QSpaceH5.sample_x_path))

    sample_y = property(lambda self:
                        self._get_array_data(QSpaceH5.sample_y_path))

    histo = property(lambda self: self._get_array_data(QSpaceH5.histo_path))

    qspace_sum = property(lambda self:
                          self._get_array_data(QSpaceH5.qspace_sum_path))

    image_shape = property(lambda self:
                           self._get_array_data(QSpaceH5.image_shape_path))

    @property
    def selected_entries(self):
        """
        Returns the input entries used for the conversion.
        :return:
        """
        path = self.entries_path + '/selected'
        entries = self._get_array_data(path)
        if entries is not None:
            return [entry.decode() for entry in entries]
        return []

    @property
    def discarded_entries(self):
        """
        Returns the input entries that were not used for the conversion.
        :return:
        """
        path = self.entries_path + '/discarded'
        entries = self._get_array_data(path)
        if entries is not None:
            return [entry.decode() for entry in entries]
        return []

    @property
    def image_binning(self):
        """
        Returns the image binning used when converting to q space.
        :return:
        """
        path = self.params_path + '/image_binning'
        return self._get_array_data(path)

    @property
    def sample_roi(self):
        """
        Returns the sample area selected for conversion (sample coordinates).
        :return: 4 elements array : xMin, xMax, yMin, yMax
        """
        path = self.params_path + '/sample_roi'
        sample_roi = self._get_array_data(path)
        if sample_roi is None:
            return [_np.nan, _np.nan, _np.nan, _np.nan]
        return sample_roi

    @property
    def qspace_dimensions(self):
        """
        Returns the dimensions of the qspace cubes
        :return:
        """
        with self.qspace_dset_ctx() as dset:
            return dset.shape[1:]


class QSpaceH5Writer(QSpaceH5):
    cube_dtype = _np.float32
    histo_dtype = _np.int32
    position_dtype = _np.float32
    q_bins_dtype = _np.float64

    def __init__(self, h5_f, mode='a', **kwargs):
        self.mode = mode
        super(QSpaceH5Writer, self).__init__(h5_f, mode=mode, **kwargs)
        self.__cube_init = False

    def init_file(self,
                  n_positions,
                  qspace_shape,
                  qspace_chunks=None,
                  qspace_sum_chunks=None,
                  compression='lzf'):
        # TODO : mode this to XsocsH5Base ('init_dataset')
        if not self.__cube_init:
            with self._get_file() as h5f:
                shapes = [(n_positions,) + qspace_shape,
                          qspace_shape[0:1],
                          qspace_shape[1:2],
                          qspace_shape[2:3],
                          qspace_shape,
                          (n_positions,),
                          (n_positions,),
                          (n_positions,),
                          (2,)]
                paths = [QSpaceH5.qspace_path,
                         QSpaceH5.qx_path,
                         QSpaceH5.qy_path,
                         QSpaceH5.qz_path,
                         QSpaceH5.histo_path,
                         QSpaceH5.sample_x_path,
                         QSpaceH5.sample_y_path,
                         QSpaceH5.qspace_sum_path,
                         QSpaceH5.image_shape_path]
                dtypes = [QSpaceH5Writer.cube_dtype,
                          QSpaceH5Writer.q_bins_dtype,
                          QSpaceH5Writer.q_bins_dtype,
                          QSpaceH5Writer.q_bins_dtype,
                          QSpaceH5Writer.histo_dtype,
                          QSpaceH5Writer.position_dtype,
                          QSpaceH5Writer.position_dtype,
                          QSpaceH5Writer.cube_dtype,
                          int]
                chunks = [qspace_chunks,
                          None, None, None, None, None, None,
                          qspace_sum_chunks, None]
                params = zip(shapes, paths, dtypes, chunks)
                for shape, path, dtype, chunk in params:
                    h5f.require_dataset(path,
                                        shape=shape,
                                        dtype=dtype,
                                        compression=compression,
                                        chunks=chunk)

    def set_qx(self, qx):
        self._set_array_data(QSpaceH5.qx_path, qx)

    def set_qy(self, qy):
        self._set_array_data(QSpaceH5.qy_path, qy)

    def set_qz(self, qz):
        self._set_array_data(QSpaceH5.qz_path, qz)

    def set_sample_x(self, sample_x):
        self._set_array_data(QSpaceH5.sample_x_path, sample_x)

    def set_sample_y(self, sample_y):
        self._set_array_data(QSpaceH5.sample_y_path, sample_y)

    def set_histo(self, histo):
        self._set_array_data(QSpaceH5.histo_path, histo)

    def set_qspace_sum(self, qspace_sum):
        self._set_array_data(QSpaceH5.qspace_sum_path, qspace_sum)

    def set_position_data(self, pos_idx, qspace, qspace_sum):
        with self._get_file() as h5f:
            h5f[QSpaceH5.qspace_path][pos_idx] = qspace
            h5f[QSpaceH5.qspace_sum_path][pos_idx] = qspace_sum

    def set_image_shape(self, image_shape):
        self._set_array_data(QSpaceH5.image_shape_path, image_shape)

    def set_entries(self, selected, discarded=None):
        """
        Sets the input entries that were converted to qspace.
        :param selected: Selected entry names
        :param discarded: List of input entries that were discarded, or None.
        :return:
        """
        path = self.entries_path + '/selected'
        selected = _np.array(selected, dtype=_np.string_)
        self._set_array_data(path, selected)
        path = self.entries_path + '/discarded'
        discarded = _np.array((discarded is not None and discarded) or [],
                              dtype=_np.string_)
        self._set_array_data(path, discarded)

    def set_image_binning(self, image_binning):
        """
        Stores the image binning used when converting to q space
        :param image_binning: a 2 elements array.
        :return:
        """
        path = self.params_path + '/image_binning'
        if image_binning is None or len(image_binning) != 2:
            raise ValueError('image_binning must be a 2 elements array : '
                             '{0}.'.format(image_binning))
        self._set_array_data(path, _np.array(image_binning))

    def set_sample_roi(self, sample_roi):
        """
        Stores the sample area selected for conversion (sample coordinates).
        :param sample_roi: 4 elements array : xMin, xMax, yMin, yMax
        :return:
        """
        path = self.params_path + '/sample_roi'
        if sample_roi is None:
            sample_roi = [_np.nan, _np.nan, _np.nan, _np.nan]
        elif len(sample_roi) != 4:
            raise ValueError('sample_roi must be a 4 elements array (or None):'
                             ' {0}.'.format(sample_roi))
        self._set_array_data(path, _np.array(sample_roi))

if __name__ == '__main__':
    pass
