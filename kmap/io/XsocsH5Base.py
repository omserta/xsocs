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
from functools import partial
from contextlib import contextmanager

import h5py as _h5py
import numpy as _np


class XsocsH5Base(object):
    # TODO : mechanism to test file type (isValid whatever)

    def __init__(self, h5_f, mode='r'):
        self.mode = mode
        self.__h5_f = h5_f

        self.__file = None
        self.__file_count = 0

        # opening the file the first time
        # (creating it if necessary)
        # all subsequent access will use the mode 'r' or 'a'
        with self._get_file() as h5_f:
            pass

        # setting the mode to append if mode was 'w' (so we don't erase it
        # when opening it later)
        self.mode = (self.mode == 'w' and 'a') or self.mode

    filename = property(lambda self: self.__h5_f)

    def _path_exists(self, path):
        with self._get_file() as h5f:
            return path in h5f

    def set_attribute(self, path, name, value):
        with self._get_file() as h5f:
            h5f[path].attrs[name] = value

    def attribute(self, path, name):
        with self._get_file() as h5f:
            return h5f[path].attrs.get(name)

    @contextmanager
    def _get_file(self, mode=None):
        """
        This protected context manager opens the hdf5 file if it isn't already
        opened (i.e : if the XsocsH5Base isn't already used as a context
        manager).
        """
        # TODO : lots of tests...
        prev_mode = None
        if mode is not None:
            if self.__file is not None:
                if self.__file.mode != mode:
                    raise ValueError('File is already opened with '
                                     'a different mode.\n'
                                     'File : {0}, current mode : {1}, '
                                     'requested mode : {2}'
                                     ''.format(self.filename,
                                               self.__file.mode,
                                               mode))
            else:
                prev_mode = self.mode
                self.mode = mode
        with self:
            yield self.__file
        if prev_mode is not None:
            self.mode = prev_mode

    def _open(self):
        if self.__file is None:
            self.__file = _h5py.File(self.__h5_f, self.mode)
        self.__file_count += 1

    def _close(self):
        self.__file_count -= 1
        if self.__file_count == 0 and self.__file:
            self.__file.close()
            self.__file = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self._close()

    def _get_scalar_data(self, path):
        with self._get_file() as h5_file:
            try:
                return h5_file.get(path, _np.array(None))[()]
            except KeyError:
                return None

    def _get_array_data(self, path, shape=False, dtype=False):
        """
        Returns the array contained in the dataset.
        Returns only the shape of the dataset if shape is True.
        Returns only the dtype of the dataset if dtype is True.
        Keyword order of priority : shape takes precedence over dtype.
        """
        with self._get_file() as h5_file:
            try:
                if shape:
                    return h5_file[path].shape
                if dtype:
                    return h5_file[path].dtype
                return h5_file[path][:]
            except KeyError as ex:
                print ex
                return None

    def _set_scalar_data(self, path, value):
        with self._get_file() as h5_f:
            value_np = _np.array(value)
            dset = h5_f.require_dataset(path,
                                        shape=value_np.shape,
                                        dtype=value_np.dtype)
            dset[()] = value

    def _set_array_data(self, path, value):
        with self._get_file() as h5_f:
            dset = h5_f.require_dataset(path,
                                        shape=value.shape,
                                        dtype=value.dtype)
            dset[:] = value

    def add_file_link(self, in_path, file_name, ext_path):
        with self._get_file() as h5_file:
            h5_file[in_path] = _h5py.ExternalLink(file_name, ext_path)

    def add_soft_link(self, from_path, target_path):
        with self._get_file() as h5_file:
            h5_file[from_path] = _h5py.SoftLink(target_path)

    @contextmanager
    def item_context(self, item_path, **kwargs):
        """
        Context manager for the image dataset.
        WARNING: only to be used as a context manager!
        WARNING: the data set must exist. see also QSpaceH5Writer.init_cube
        """
        no_proxy = kwargs.get('no_proxy') is not None
        with self._get_file() as h5_file:
            item = h5_file[item_path]
            if no_proxy:
                yield item
            else:
                yield weakref.proxy(item)
            del item

    def copy_group(self, src_h5f, src_path, dest_path):
        """
        Recursively copies an object from one HDF5 file to another.
        Warning : it fails if it finds a conflict with an already existing
        dataset.
        """
        # We have to work around a limitation of the h5py.Group.copy method
        # that fails when a group already exists in the destination file.
        def _copy_obj(name, obj, src_grp=None, dest_grp=None):
            if isinstance(obj, _h5py.Group):
                dest_grp.require_group(name)
            else:
                src_grp.copy(name,
                             dest_grp,
                             name=name,
                             shallow=False,
                             expand_soft=True,
                             expand_external=True,
                             expand_refs=True,
                             without_attrs=False)

        with _h5py.File(src_h5f, 'r') as src_h5:
            with self._get_file() as h5_file:
                src_grp = src_h5[src_path]
                dest_grp = h5_file.require_group(dest_path)
                src_grp.visititems(partial(_copy_obj,
                                           src_grp=src_grp,
                                           dest_grp=dest_grp))

    def object_filename(self, path):
        with self._get_file() as h5f:
            obj = h5f.get(path)
            if obj is None:
                return None
            return obj.file.filename
