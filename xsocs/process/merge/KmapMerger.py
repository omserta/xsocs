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

import re
import copy
import os.path

import ctypes
from threading import Thread
import multiprocessing.sharedctypes as mp_sharedctypes
from multiprocessing import Pool, cpu_count, Manager

import h5py
import numpy as np
from silx.third_party import EdfFile

from ...io import XsocsH5


class KmapMerger(object):
    (READY, RUNNING, DONE,
     ERROR, CANCELED, UNKNOWN) = __STATUSES = range(6)
    """ Available status codes """

    status = property(lambda self: self.__status)
    """ Current status code of this instance """

    statusMsg = property(lambda self: self.__status_msg)
    """ Status message if any, or None """

    results = property(lambda self: self.__results)
    """ Parse results. KmapParseResults instance. """

    no_match_ids = property(lambda self: self.__no_match_ids)
    """ Scans that werent matched during the parsing """

    no_img_ids = property(lambda self: self.__no_img_ids)
    """ Scans that didnt have the image info line in the spec file """

    on_error_ids = property(lambda self: self.__on_error_ids)
    """ Scans for which an error occured during the parsing """

    matched_ids = property(lambda self: self.__matched_ids)
    """ Scans that have a matching image file """

    selected_ids = property(lambda self: sorted(self.__selected_ids))
    """ Selected scans for merging """

    def __init__(self,
                 spec_h5,
                 match_info,
                 output_dir=None,
                 callback=None):
        """
        Merger for the Kmap SPEC and EDF files. This loads a spech5 file,
             converts it to HDF5 and then tries to match scans and edf image
             files.
        :param spec_h5: Name of the spech5.
        .. seealso : silx.io.convert_spec_h5.convert
        :param match_info: instance of KmapMatchResults.
        :param output_dir: output directory for the merged files.
        :param callback: callback to call when the parsing is done.
        """
        super(KmapMerger, self).__init__()

        self.__status = None

        self.__set_status(self.UNKNOWN, 'Init')

        self.__output_dir = output_dir

        self.__params = {'prefix': None,
                         'beam_energy': None,
                         'center_channel': None,
                         'chan_per_deg': None}

        self.__spec_h5 = spec_h5
        self.__callback = callback
        self.__n_proc = None
        self.__compression = 'lzf'
        self.__prefix = 'prefix'
        self.__overwrite = False

        self.__matched_scans = copy.deepcopy(match_info.matched)
        self.__no_match_scans = copy.deepcopy(match_info.not_matched)
        self.__no_img_scans = copy.deepcopy(match_info.no_img_info)
        self.__on_error_scans = copy.deepcopy(match_info.error)
        self.__selected_ids = set(self.__matched_scans.keys())
        self.__matched_ids = sorted(self.__matched_scans.keys())

        self.__no_match_ids = sorted(self.__no_match_scans.keys())
        self.__no_img_ids = sorted(self.__no_img_scans)
        self.__on_error_ids = sorted(self.__on_error_scans)

        self.__results = None
        self.__master = None
        self.__shared_progress = None
        self.__proc_indices = None
        self.__term_evt = None

        self.__thread = None

        self.prefix = None

        self.__set_status(self.READY)

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

    def merge(self,
              overwrite=False,
              blocking=True,
              callback=None):
        """
        Starts the merge.
        :param overwrite: if False raises an exception if some files already
        exist.
        :param blocking: if False, the merge will be done in a separate
         thread and this method will return immediately.
        :param callback: callback that will be called when the merging is done.
        It overwrites the one passed the constructor.
        :return:
        """

        self.__set_status(self.RUNNING)

        if self.is_running():
            raise RuntimeError('This KmapSpecParser instance is already '
                               'parsing.')

        self.check_parameters()

        output_dir = self.__output_dir
        if output_dir is None:
            self.__set_status(self.ERROR)
            raise ValueError('output_dir has not been set.')

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
            self.__run_merge()
            if self.__callback:
                self.__callback()
        else:
            thread = self.__thread = Thread(target=self.__run_merge)
            thread.start()

    def __run_merge(self):
        """
        Runs the merge.
        :return:
        """

        self.__set_status(self.RUNNING)

        self.__master = None

        selected_scans = self.selected_ids
        matched_scans = self.__matched_scans

        output_files = self.summary()
        master_f = output_files['master']
        del output_files['master']

        scans = {scan_id: {'image': matched_scans[scan_id]['image'],
                           'output': output_files[scan_id]}
                 for scan_id in selected_scans}

        print('Merging scan IDs : {}.'
              ''.format(', '.join(self.selected_ids)))

        try:
            manager = Manager()
            self.__term_evt = term_evt = manager.Event()

            self.__shared_progress = mp_sharedctypes.RawArray(ctypes.c_int32,
                                                              len(scans))

            master_f = os.path.join(self.__output_dir, master_f)

            if not self.__overwrite:
                mode = 'w-'
            else:
                mode = 'w'

            # trying to access the file (erasing it if necessary)
            with XsocsH5.XsocsH5MasterWriter(master_f, mode=mode):
                pass

            if self.__n_proc is None:
                n_proc = cpu_count()
            else:
                n_proc = self.__n_proc

            def init(term_evt_, shared_progress_):
                global g_term_evt
                global g_shared_progress
                g_term_evt = term_evt_
                g_shared_progress = shared_progress_

            # setting progress to 0
            np.frombuffer(self.__shared_progress, dtype='int32')[:] = 0

            pool = Pool(n_proc,
                        initializer=init,
                        initargs=(term_evt,
                                  self.__shared_progress),
                        maxtasksperchild=2)

            def callback(result_):
                scan, finished, info = result_
                print('{0} finished.'.format(scan))
                if not finished:
                    term_evt.set()

            results = {}
            self.__proc_indices = proc_indices = {}
            for proc_idx, (scan_id, infos) in enumerate(scans.items()):
                args = (scan_id,
                        proc_idx,
                        self.__spec_h5,
                        self.__output_dir,
                        infos['output'],
                        infos['image'],
                        self.beam_energy,
                        self.chan_per_deg,
                        self.center_chan,
                        self.compression)
                results[scan_id] = pool.apply_async(_add_edf_data,
                                                    args,
                                                    callback=callback)
                proc_indices[scan_id] = proc_idx

            pool.close()

            self.__proc_indices = proc_indices
            pool.join()

            proc_results = [result.get() for result in results.values()]
            proc_codes = np.array([proc_result[1]
                                   for proc_result in proc_results])

            rc = self.DONE
            if not np.all(proc_codes == self.DONE):
                if self.ERROR in proc_codes:
                    rc = self.ERROR
                elif self.CANCELED in proc_codes:
                    rc = self.CANCELED
                else:
                    raise ValueError('Unknown return code.')

            if rc == self.DONE:
                with XsocsH5.XsocsH5MasterWriter(master_f, mode='a') as m_h5f:
                    items = scans.items()
                    for proc_idx, (scan_id, infos) in enumerate(items):
                        entry_fn = infos['output']
                        entry = entry_fn.rpartition('.')[0]
                        m_h5f.add_entry_file(entry, entry_fn)

            self.__set_status(rc)

        except Exception as ex:
            self.__set_status(self.ERROR, str(ex))
        else:
            self.__results = master_f

        self.prefix = None
        self.__master = master_f

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
        if self.is_running():
            raise RuntimeError('Operation not permitted while '
                               'a parse or merge in running.')

    def is_running(self):
        return self.__thread and self.__thread.is_alive()

    output_dir = property(lambda self: self.__output_dir)

    @output_dir.setter
    def output_dir(self, output_dir):
        if not isinstance(output_dir, str):
            raise TypeError('output_dir must be a string. Received {0}'
                            ''.format(type(output_dir)))
        self.__output_dir = output_dir

    def select(self, scan_ids, clear=False):

        if scan_ids is None:
            scan_ids = list(self.__matched_scans.keys())

        if not isinstance(scan_ids, (list, tuple)):
            scan_ids = [scan_ids]

        scan_ids = set(scan_ids)
        unknown_scans = scan_ids - set(self.__matched_scans.keys())

        if len(unknown_scans) != 0:
            err_ids = '; '.join(str(scan) for scan in unknown_scans)
            raise ValueError('Unknown scan IDs : {0}.'.format(err_ids))

        if clear:
            self.__selected_ids = scan_ids
        else:
            self.__selected_ids |= scan_ids

    def unselect(self, scan_ids):

        if scan_ids is None:
            return

        if not isinstance(scan_ids, (list, tuple)):
            scan_ids = [scan_ids]

        self.__selected_ids -= set(scan_ids)

    def get_scan_command(self, scan_id):
        with h5py.File(self.__spec_h5, 'r') as spec_h5:
            try:
                scan = spec_h5[scan_id]
            except KeyError:
                raise ValueError('Scan ID {0} not found.')

            command = scan['title'][()].decode()

        try:
            return parse_scan_command(command)
        except:
            raise ValueError(
                'Failed to parse command line for scan ID {0} : '
                '{1}.'.format(scan_id, command))

    def check_overwrite(self):
        """
        Returns a list of the files that already exist and that will be
        overwritten.
        """
        files = self.summary(fullpath=True)
        exist = [f for f in files.values() if os.path.exists(f)]
        return exist

    def check_consistency(self):
        """
        Checks if all selected scans have the same command.
        """
        errors = []
        scan_ids = self.selected_ids
        n_scans = len(scan_ids)
        if n_scans == 0:
            return None
        commands = [self.get_scan_command(scan_id)
                    for scan_id in scan_ids]

        def check_key(key):
            values = [command[key] for command in commands]
            values_set = set(values)
            if len(values_set) != 1:
                errors.append('Selected scans command inconsistency : '
                              '"{0}" : {1}.'
                              ''.format(key, '; '.join(str(m)
                                                       for m in
                                                       values_set)))

        check_key('motor_0')
        check_key('motor_1')
        check_key('motor_0_start')
        check_key('motor_1_start')
        check_key('motor_0_end')
        check_key('motor_1_end')
        check_key('motor_0_steps')
        check_key('motor_1_steps')
        check_key('delay')

        return errors

    def check_parameters(self):
        errors = []
        if self.beam_energy is None:
            errors.append('invalid "Beam energy"')
        if self.output_dir is None:
            errors.append('invalid "Output directory"')
        if self.center_chan is None:
            errors.append('invalid "Center channel"')
        if self.chan_per_deg is None:
            errors.append('invalid "Channel per degree"')
        return errors

    beam_energy = property(lambda self: self.__params['beam_energy'])

    @beam_energy.setter
    def beam_energy(self, beam_energy):
        self.__params['beam_energy'] = float(beam_energy)

    chan_per_deg = property(lambda self: self.__params['chan_per_deg'])

    @chan_per_deg.setter
    def chan_per_deg(self, chan_per_deg):
        if len(chan_per_deg) != 2:
            raise ValueError('chan_per_deg must be a two elements array.')
        self.__params['chan_per_deg'] = [float(chan_per_deg[0]),
                                           float(chan_per_deg[1])]

    center_chan = property(lambda self: self.__params['center_chan'])

    @center_chan.setter
    def center_chan(self, center_chan):
        if len(center_chan) != 2:
            raise ValueError('center_chan must be a two elements array.')
        self.__params['center_chan'] = [float(center_chan[0]),
                                        float(center_chan[1])]

    compression = property(lambda self: self.__compression)

    @compression.setter
    def compression(self, compression):
        if not isinstance(compression, str):
            raise ValueError('compression must be a string.')
        self.__compression = compression

    n_proc = property(lambda self: self.__n_proc)

    @n_proc.setter
    def n_proc(self, n_proc):
        if n_proc is None:
            self.__n_proc = None
            return

        n_proc = int(n_proc)
        if n_proc <= 0:
            self.__n_proc = None
        else:
            self.__n_proc = n_proc

    def get_imagefile_info(self, scan_id, key=None):

        try:
            scan_info = self.__matched_scans[scan_id]
        except KeyError:
            try:
                scan_info = self.__no_match_scans[scan_id]
            except KeyError:
                raise ValueError('No imageFile line found in the SPEC file'
                                 ' for scan ID {0}.'
                                 ''.format(scan_id))

        if key is not None:
            return copy.deepcopy(scan_info['spec'][key])
        else:
            return copy.deepcopy(scan_info['spec'])

    def get_scan_image(self, scan_id):

        try:
            scan_info = self.__matched_scans[scan_id]
        except KeyError:
            raise ValueError('Scan ID {0} is not one of the valid scans.'
                             ''.format(scan_id))
        return copy.deepcopy(scan_info['image'])

    def common_prefix(self):
        scan_ids = self.__selected_ids

        if len(scan_ids) == 0:
            return ''

        prefixes = [self.__matched_scans[scan_id]['spec']['prefix']
                    for scan_id in scan_ids]
        common = os.path.commonprefix(prefixes)

        # this has to be tests and made more robust
        if len(prefixes[0]) > len(common) or common.endswith('_'):
            common = common.rpartition('_')[0]

        return common

    master_file = property(lambda self: self.__master)

    prefix = property(lambda self: self.__prefix)

    @prefix.setter
    def prefix(self, prefix):

        if prefix is None or len(prefix) == 0:
            prefix = self.common_prefix()
            if not prefix:
                self.__prefix = 'prefix'
            else:
                self.__prefix = prefix

        elif isinstance(prefix, str):
            self.__prefix = prefix
        else:
            raise TypeError('prefix must be a string, or None. This '
                            'is q {0}.'.format(type(prefix)))

    def __gen_scan_filename(self, scan_id, fullpath=False):
        pattern = '{img_file}_{scan_id}.h5'
        img_file = self.__matched_scans[scan_id]['image']
        img_file = os.path.basename(img_file).split('.')[0]
        merged_file = pattern.format(img_file=img_file, scan_id=scan_id)

        if fullpath:
            merged_file = os.path.join(self.output_dir, merged_file)
        return merged_file

    def __gen_master_filename(self, fullpath=False):
        prefix = self.prefix
        # if not master.endswith('.h5'):
        master = prefix + '.h5'
        if fullpath:
            master = os.path.join(self.output_dir, master)
        return master

    def summary(self, fullpath=False):
        if self.__output_dir is None:
            raise ValueError('output_summary() cannot be called '
                             'before an output directory has been set. '
                             'Please call set_output_dir() first.')

        master = self.__gen_master_filename(fullpath=fullpath)
        files = {'master': master}

        sel_ids = list(self.__selected_ids)
        for scan_id in sel_ids:
            files.update({scan_id: self.__gen_scan_filename(scan_id,
                                                            fullpath=fullpath)})

        return files

    def abort(self, wait=True):
        if self.is_running():
            self.__term_evt.set()
            if wait:
                self.wait()

    def progress(self):
        # TODO : rework
        if self.__shared_progress is not None:
            progress = np.frombuffer(self.__shared_progress, dtype='int32')
            proc_indices = self.__proc_indices
            if proc_indices:
                merge_progress = dict([(scan_id, progress[proc_idx])
                                       for scan_id, proc_idx in
                                       proc_indices.items()])
            else:
                merge_progress = dict([(scan_id, 0)
                                       for scan_id in self.selected_ids])
            return merge_progress
        return None


# #######################################################################
# #######################################################################


def parse_scan_command(command):
    _COMMAND_LINE_PATTERN = ('^(?P<id>[0-9]*) '
                             '(?P<command>[^ ]*) '
                             '(?P<motor_0>[^ ]*) '
                             '(?P<motor_0_start>[^ ]*) '
                             '(?P<motor_0_end>[^ ]*) '
                             '(?P<motor_0_steps>[^ ]*) '
                             '(?P<motor_1>[^ ]*) '
                             '(?P<motor_1_start>[^ ]*) '
                             '(?P<motor_1_end>[^ ]*) '
                             '(?P<motor_1_steps>[^ ]*) '
                             '(?P<delay>[^ ]*)\s*'
                             '.*'
                             '$')
    cmd_rgx = re.compile(_COMMAND_LINE_PATTERN)
    cmd_match = cmd_rgx.match(command)
    if cmd_match is None:
        raise ValueError('Failed to parse command line : "{0}".'
                         ''.format(command))
    cmd_dict = cmd_match.groupdict()
    cmd_dict.update(full=command)
    return cmd_dict


# #######################################################################
# #######################################################################


def _add_edf_data(scan_id,
                  proc_idx,
                  spec_h5_fn,
                  output_dir,
                  output_f,
                  img_f,
                  beam_energy,
                  chan_per_deg,
                  center_chan,
                  compression):

    """
    Creates an entry_*.h5 file with scan data from the provided
    "*spec*" HDF5 files, and adds the image data from the associated
    image file. This function is meant to be called in from _merge_data.
    """

    global g_term_evt
    global g_master_lock
    global g_shared_progress

    entry = output_f.rpartition('.')[0]
    entry_fn = os.path.join(output_dir, output_f)

    progress = np.frombuffer(g_shared_progress, dtype='int32')  # noqa
    progress[proc_idx] = 0

    rc = None

    try:
        print('Merging scan ID {0}'.format(scan_id))

        if g_term_evt.is_set():  # noqa
            rc = KmapMerger.CANCELED
            raise Exception('Merge of scan {0} aborted.'.format(scan_id))

        with XsocsH5.XsocsH5Writer(entry_fn, mode='w') as entry_h5f:
            entry_h5f.create_entry(entry)
            progress[proc_idx] = 1

            entry_h5f.copy_group(spec_h5_fn, scan_id, entry)
            with h5py.File(spec_h5_fn) as spec_h5:
                command = spec_h5[scan_id]['title'][()].decode()
                command_params = parse_scan_command(command)

            entry_h5f.set_scan_params(entry, **command_params)

            if beam_energy is not None:
                entry_h5f.set_beam_energy(float(beam_energy), entry)

            if chan_per_deg is not None:
                entry_h5f.set_chan_per_deg([float(chan_per_deg[0]),
                                            float(chan_per_deg[1])],
                                           entry=entry)

            if center_chan is not None:
                entry_h5f.set_direct_beam([float(center_chan[0]),
                                           float(center_chan[1])],
                                          entry=entry)

            progress[proc_idx] = 2
            edf_file = EdfFile.EdfFile(img_f, access='r', fastedf=True)
            progress[proc_idx] = 5

            n_images = edf_file.GetNumImages()

            image = edf_file.GetData(0)
            dtype = image.dtype
            img_shape = image.shape
            dset_shape = (n_images, img_shape[0], img_shape[1])
            chunks = (1, dset_shape[1]//4, dset_shape[2]//4)

            with entry_h5f.image_dset_ctx(entry=entry,
                                          create=True,
                                          shape=dset_shape,
                                          dtype=dtype,
                                          chunks=chunks,
                                          compression=compression,
                                          shuffle=True) as image_dset:
                for i in range(n_images):
                    if i % 500 == 0:
                        progress[proc_idx] = round(5. + (95.0 * i) / n_images)
                        if g_term_evt.is_set():  # noqa
                            raise Exception('Merge of scan {0} aborted.'
                                            ''.format(scan_id))

                    data = edf_file.GetData(i)
                    image_dset[i, :, :] = data

    except Exception as ex:
        print(ex)
        if rc is None:
            rc = KmapMerger.ERROR
        result = (scan_id, rc, str(ex))
    else:
        print('Entry {0} merged.'.format(entry))
        result = (scan_id, KmapMerger.DONE, None)

    if result[1] == KmapMerger.DONE:
        progress[proc_idx] = 100
    else:
        progress[proc_idx] = -1
    return result


if __name__ == '__main__':
    pass
