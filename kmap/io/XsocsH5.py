import weakref
from collections import OrderedDict
from functools import partial
from contextlib import contextmanager

import h5py as _h5py
import numpy as _np

class InvalidEntryError(Exception):
    pass

class XsocsH5Base(object):
    def __init__(self, h5_f, mode='r'):
        self.mode = mode
        self.__h5_f = h5_f

        self.__file = None
        self.__file_count = 0

        # opening the file the first time if necessary
        # (creating it if necessary)
        # all subsequent access will use the mode 'r' or 'a'
        if mode == 'w':
            with self._get_file() as h5_f:
                pass

        # setting the mode to append if mode was 'w' (so we don't erase it
        # when opening it later)
        self.mode = (self.mode == 'w' and 'a') or self.mode

    filename = property(lambda self: self.__h5_f)

    @contextmanager
    def _get_file(self):
        """
        This protected context manager opens the hdf5 file if it isn't already
        opened (i.e : if the XsocsH5Base isn't already used as a context
        manager).
        """
        with self:
            yield self.__file

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

class XsocsH5(XsocsH5Base):

    TOP_ENTRY = 'global'
    positioners_tpl = '/{0}/instrument/positioners'
    img_data_tpl = '/{0}/measurement/image/data'
    entry_cumul_tpl = '/processed/{0}/cumul'
    entry_processed_tpl = '/processed/{0}/'
    processed_grp = '/processed'
    #full_cumul_tpl = '/processed/total/cumul'
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
