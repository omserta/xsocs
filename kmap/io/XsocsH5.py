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
from collections import OrderedDict
from contextlib import contextmanager

import h5py as _h5py
import numpy as _np

from .XsocsH5Base import XsocsH5Base


class InvalidEntryError(Exception):
    pass


class XsocsH5(XsocsH5Base):

    TOP_ENTRY = 'global'
    positioners_tpl = '/{0}/instrument/positioners'
    img_data_tpl = '/{0}/measurement/image/data'
    entry_cumul_tpl = '/processed/{0}/cumul'
    entry_processed_tpl = '/processed/{0}/'
    processed_grp = '/processed'
    measurement_tpl = '/{0}/measurement'
    measurement_command_tpl = '/processed/{0}/command'
    detector_tpl = '/{0}/instrument/detector'

    def __init__(self, h5_f, mode='r'):
        super(XsocsH5, self).__init__(h5_f, mode=mode)

        self.__entries = None

    def title(self, entry):
        with self._get_file() as h5_file:
            path = entry + '/title'
            return h5_file[path]

    def entry_filename(self, entry):
        with self._get_file() as h5_file:
            return h5_file[entry].file.filename

    def _update_entries(self):
        with self._get_file() as h5_file:
            # TODO : this isnt pretty but for some reason the attrs.get() fails
            # when there is no attribute NX_class (should return the default
            # None)
            self.__entries = sorted([key for key in h5_file
                                     if ('NX_class' in h5_file[key].attrs and
                                         h5_file[key].attrs['NX_class'] == 'NXentry')])  # noqa

    def entries(self):
        if self.__entries is None:
            self._update_entries()
        return self.__entries[:]

    def get_entry_name(self, entry_idx):
        """
        Get the entry found at position *entry_idx* (entries names sorted
        alphabeticaly).
        Raises InvalidEntryError if the entry is not found.
        """
        try:
            return self.entries()[entry_idx]
        except IndexError:
            raise InvalidEntryError('Entry not found (entry_idx={0}).'
                                    ''.format(entry_idx))

    def __detector_params(self, entry, param_names):
        with self._get_file() as h5_file:
            path = self.detector_tpl.format(entry) + '/{0}'
            if isinstance(param_names, (list, set, tuple)):
                return [h5_file.get(path.format(param), _np.array(None))[()]
                        for param in param_names]
            return h5_file.get(path.format(param_names), _np.array(None))[()]

    def __command_params(self, entry, param_names):
        with self._get_file() as h5_file:
            path = self.measurement_command_tpl.format(entry) + '/{0}'
            if isinstance(param_names, (list, set, tuple)):
                return OrderedDict([(param, h5_file.get(path.format(param),
                                                      _np.array(None))[()])
                                    for param in param_names])
            return {param_names: h5_file.get(path.format(param_names),
                                             _np.array(None))[()]}

    def beam_energy(self, entry):
        return self.__detector_params(entry, 'beam_energy')

    def direct_beam(self, entry):
        return self.__detector_params(entry, ['center_chan_dim0',
                                              'center_chan_dim1'])

    def pixel_size(self, entry):
        return self.__detector_params(entry, ['pixelsize_dim0',
                                              'pixelsize_dim1'])

    def chan_per_deg(self, entry):
        return self.__detector_params(entry, ['chan_per_deg_dim0',
                                              'chan_per_deg_dim1'])

    def detector_orient(self, entry):
        return self.__detector_params(entry, 'detector_orient')

    def n_images(self, entry):
        # TODO : make sure that data.ndims = 3
        path = self.img_data_tpl.format(entry)
        return self._get_array_data(path, shape=True)[0]

    def image_size(self, entry):
        # TODO : make sure that data.ndims = 3
        path = self.img_data_tpl.format(entry)
        return self._get_array_data(path, shape=True)[1:3]

    def image_dtype(self, entry):
        path = self.img_data_tpl.format(entry)
        return self._get_array_data(path, dtype=True)

    def dset_shape(self, path):
        return self._get_array_data(path, shape=True)

    def image_cumul(self, entry):
        """
        Returns the summed intensity for each image.
        """
        #if entry == self.TOP_ENTRY:
            #path = self.full_cumul_tpl
        #else:
        path = self.entry_cumul_tpl.format(entry)
        cumul = self._get_array_data(path)
        return cumul

    def scan_positions(self, entry):
        # TODO : check the motors : could by x/y x/z y/z
        path = self.measurement_tpl.format(entry)
        x_pos = self._get_array_data(path + '/adcX')
        y_pos = self._get_array_data(path + '/adcY')
        return (x_pos, y_pos)

    def scan_params(self, entry):
        return self.__command_params(entry,
                                     ['motor_0', 'motor_0_start',
                                      'motor_0_end', 'motor_0_steps',
                                      'motor_1', 'motor_1_start',
                                      'motor_1_end', 'motor_1_steps',
                                      'delay'])


    def positioner(self, entry, positioner):
        path = self.positioners_tpl.format(entry) + '/' + positioner
        return self._get_scalar_data(path)

    def measurement(self, entry, measurement):
        path = self.measurement_tpl.format(entry) + '/' + measurement
        return self._get_array_data(path)

    @contextmanager
    def image_dset_ctx(self,
                       entry,
                       create=False,
                       **kwargs):
        """
        Context manager for the image dataset.
        WARNING: only to be used as a context manager!
        """
        dset_path = self.img_data_tpl.format(entry)
        with self._get_file() as h5_file:
            if create:
                try:
                    image_dset = h5_file.require_dataset(dset_path,
                                                         **kwargs)
                except TypeError:
                    image_dset = h5_file.create_dataset(dset_path,
                                                        **kwargs)
            else:
                image_dset = h5_file[dset_path]
            yield weakref.proxy(image_dset)
            del image_dset


class XsocsH5_Writer(XsocsH5):

    def __init__(self, h5_f, mode='a', **kwargs):
        self.mode = mode
        super(XsocsH5_Writer, self).__init__(h5_f, mode=mode, **kwargs)

    def __set_detector_params(self, entry, params):
        with self._get_file() as h5_file:
            path = self.detector_tpl.format(entry) + '/{0}'
            for param_name, param_value in params.items():
                self._set_scalar_data(path.format(param_name), param_value)

    def __set_measurement_params(self, entry, params):
        with self._get_file() as h5_file:
            path = self.measurement_command_tpl.format(entry) + '/{0}'
            for param_name, param_value in params.items():
                self._set_scalar_data(path.format(param_name), param_value)

    def set_beam_energy(self, beam_energy, entry):
        return self.__set_detector_params(entry, {'beam_energy': beam_energy})

    def set_direct_beam(self, direct_beam, entry):
        value = {'center_chan_dim0': direct_beam[0],
                 'center_chan_dim1': direct_beam[1]}
        return self.__set_detector_params(entry, value)

    def set_pixel_size(self, pixel_size, entry):
        value = {'pixelsize_dim0': pixel_size[0],
                 'pixelsize_dim1': pixel_size[1]}
        return self.__set_detector_params(entry, value)

    def set_chan_per_deg(self, chan_per_deg, entry):
        value = {'chan_per_deg_dim0': chan_per_deg[0],
                 'chan_per_deg_dim1': chan_per_deg[1]}
        return self.__set_detector_params(entry, value)

    def set_detector_orient(self, detector_orient, entry):
        value = {'detector_orient': _np.string_(detector_orient)}
        return self.__set_detector_params(entry, value)

    def set_scan_params(self,
                        entry,
                        motor_0,
                        motor_0_start,
                        motor_0_end,
                        motor_0_steps,
                        motor_1,
                        motor_1_start,
                        motor_1_end,
                        motor_1_steps,
                        delay,
                        **kwargs):
        self.__set_measurement_params(entry,
                                      {'motor_0': _np.string_(motor_0),
                                       'motor_0_start': float(motor_0_start),
                                       'motor_0_end': float(motor_0_end),
                                       'motor_0_steps': int(motor_0_steps),
                                       'motor_1': _np.string_(motor_1),
                                       'motor_1_start': float(motor_1_start),
                                       'motor_1_end': float(motor_1_end),
                                       'motor_1_steps': int(motor_1_steps),
                                       'delay': float(delay)})

    def create_entry(self, entry):
        with self._get_file() as h5_file:
            entry_grp = h5_file.require_group(entry)
            entry_grp.attrs['NX_class'] = _np.string_('NXentry')

            # creating mandatory groups and setting their Nexus attributes
            grp = entry_grp.require_group('measurement/image')
            grp.attrs['interpretation'] = _np.string_('image')

            # setting the nexus classes
            entry_grp.attrs['NX_class'] = _np.string_('NXentry')

            grp = entry_grp.require_group('instrument')
            grp.attrs['NX_class'] = _np.string_('NXinstrument')

            grp = entry_grp.require_group('instrument/detector')
            grp.attrs['NX_class'] = _np.string_('NXdetector')

            grp = entry_grp.require_group('instrument/positioners')
            grp.attrs['NX_class'] = _np.string_('NXcollection')

            grp = entry_grp.require_group('measurement')
            grp.attrs['NX_class'] = _np.string_('NXcollection')

            grp = entry_grp.require_group('measurement/image')
            grp.attrs['NX_class'] = _np.string_('NXcollection')

            ## creating some links
            grp = entry_grp.require_group('measurement/image')
            det_grp = entry_grp.require_group('instrument/detector')
            grp['info'] = det_grp
            det_grp['data'] = _h5py.SoftLink(self.img_data_tpl.format(entry))

        self._update_entries()

    def set_image_cumul(self,
                        entry,
                        cumul,
                        **kwargs):
        with self._get_file() as h5_file:
            #if entry == self.TOP_ENTRY:
                #path = self.full_cumul_tpl
            #else:
            path = self.entry_cumul_tpl.format(entry)
            dset = h5_file.require_dataset(path,
                                           shape=cumul.shape,
                                           dtype=cumul.dtype,
                                           **kwargs)
            dset[:] = cumul
            del dset


class XsocsH5_Master_Writer(XsocsH5_Writer):

    def add_entry_file(self, entry, entry_file):
        with self._get_file() as h5_file:
            h5_file[entry] = _h5py.ExternalLink(entry_file, entry)
            processed_grp = h5_file.require_group(self.processed_grp)
            grp_path = self.entry_processed_tpl.format(entry)
            processed_grp[grp_path] = _h5py.ExternalLink(entry_file,
                                                         grp_path)
