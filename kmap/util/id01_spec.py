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
import glob
import copy
import os.path

import ctypes
from threading import Thread
from functools import partial
import multiprocessing.sharedctypes as mp_sharedctypes
from multiprocessing import Pool, cpu_count, Manager

import h5py
import numpy as np
from silx.third_party import EdfFile
from silx.io import spectoh5


# regular expression matching the imageFile comment line
_IMAGEFILE_LINE_PATTERN = ('^#C imageFile '
                           'dir\[(?P<dir>[^\]]*)\] '
                           'prefix\[(?P<prefix>[^\]]*)\] '
                           '(idxFmt\[(?P<idxFmt>[^\]]*)\] ){0,1}'
                           'nextNr\[(?P<nextNr>[^\]]*)\] '
                           'suffix\[(?P<suffix>[^\]]*)\]$')


# #######################################################################
# #######################################################################
# #######################################################################


class Id01DataMerger(object):
    """
    NOT thread safe
    """

    def __init__(self,
                 spec_fname,
                 work_dir,
                 img_dir=None,
                 output_dir=None,
                 version=1):

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        spec_h5 = os.path.join(work_dir, 'temp_spec.h5')
        self.__spec_h5 = spec_h5

        self.__merge_thread = None
        self.__parse_thread = None

        self.reset(spec_fname,
                   img_dir=img_dir,
                   output_dir=output_dir,
                   version=version)

    def reset(self,
              spec_fname,
              img_dir=None,
              output_dir=None,
              version=1):
        self.__running_exception()

        self.__spec_fname = spec_fname
        self.__img_dir = img_dir
        self.__version = version
        self.__master_file = 'master.h5'

        self.__parsed = False
        self.__merged = False

        self.__master = ''
        self.__output_dir = output_dir

        self.__compression = 'lzf'

        self.__n_proc = None

        self.__merge_thread = None
        self.__parse_thread = None

        self.__beam_energy = None
        self.__center_chan = None
        self.__chan_per_deg = None
        self.__pixelsize = None
        self.__detector_orient = None

        self.__set_parse_results(reset=True)

    def __on_merge_done(self, callback=None):
        self.__merged = True
        if callback:
            callback()

    def __on_parse_done(self, callback=None):
        self.__set_parse_results()
        if callback:
            callback()

    def __set_parse_results(self, reset=False):
        if reset is False and self.__parse_thread is None:
            # shouldnt even be here
            raise RuntimeError('This should be called from an active'
                               '_ParseThread.')
        if reset:
            self.__matched_scans = None
            self.__no_match_scans = None
            self.__no_img_scans = None
            self.__on_error_scans = None
            self.__selected_ids = set()

            self.__matched_ids = []
            self.__no_match_ids = []
            self.__no_img_ids = []
            self.__on_error_ids = []
            self.__parsed = False
        else:
            match_results = self.__parse_thread.results()
            self.__matched_scans = match_results[0]
            self.__no_match_scans = match_results[1]
            self.__no_img_scans = match_results[2]
            self.__on_error_scans = match_results[3]
            self.__selected_ids = set(self.__matched_scans.keys())

            self.__matched_ids = sorted(self.__matched_scans.keys())
            self.__no_match_ids = sorted(self.__no_match_scans.keys())
            self.__no_img_ids = sorted(self.__no_img_scans)
            self.__on_error_ids = sorted(self.__on_error_scans)

            self.__parsed = True

        self.__merged = False
        self.master_file = None

    def __running_exception(self):
        if self.is_running():
            raise RuntimeError('Operation not permitted while '
                               'a parse or merge in running.')

    def is_running(self):
        return ((self.__merge_thread and self.__merge_thread.is_alive()) or
                self.__parse_thread and self.__parse_thread.is_alive())

    def parse(self, blocking=True, callback=None):

        if self.__parse_thread is not None and self.__parse_thread.is_alive():
            raise RuntimeError('A parse is already in progress.')

        if self.__merge_thread is not None and self.__merge_thread.is_alive():
            raise RuntimeError('A merge is already in progress.')

        self.__parsed = False
        self.__merged = False

        callback = partial(self.__on_parse_done,
                           callback=callback)
        self.__parse_thread = _ParseThread(self.__spec_fname,
                                           self.__spec_h5,
                                           img_dir=self.__img_dir,
                                           version=self.__version,
                                           callback=callback)

        self.__parse_thread.start()

        if blocking:
            self.wait()

    def __check_parsed(self):
        if not self.__parsed:
            raise ValueError('Error : parse() has not been called yet.')

    output_dir = property(lambda self: self.__output_dir)

    @output_dir.setter
    def output_dir(self, output_dir):
        if not isinstance(output_dir, str):
            raise TypeError('output_dir must be a valid path.')
        self.__output_dir = output_dir

    def merge(self,
              blocking=True,
              overwrite=False,
              callback=None):

        if self.__parse_thread is not None and self.__parse_thread.is_alive():
            raise RuntimeError('A parse is already in progress.')

        if self.__merge_thread is not None and self.__merge_thread.is_alive():
            raise RuntimeError('A merge is already in progress.')

        self.__check_parsed()

        if not overwrite:
            if len(self.check_overwrite()):
                raise RuntimeError('Some files already exist. Use the '
                                   'overwrite keyword to ignore this '
                                   'warning.')

        self.__merged = False

        if len(self.__selected_ids) == 0:
            raise ValueError('No scans selected for merge.')

        output_dir = self.__output_dir

        if output_dir is None:
            raise ValueError('output_dir has not been set.')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        selected_scans = self.selected_ids
        matched_scans = self.__matched_scans

        output_files = self.summary()
        master_f = output_files['master']
        del output_files['master']

        scans_infos = {scan_id: {'image': matched_scans[scan_id]['image'],
                                 'output': output_files[scan_id]}
                       for scan_id in selected_scans}

        print('Merging scan IDs : {}.'
              ''.format(', '.join(self.selected_ids)))

        callback = partial(self.__on_merge_done,
                           callback=callback)

        self.__merge_thread = _MergeThread(self.__output_dir,
                                           self.__spec_h5,
                                           scans_infos,
                                           self.__beam_energy,
                                           self.__chan_per_deg,
                                           self.__pixelsize,
                                           self.__center_chan,
                                           self.__detector_orient,
                                           master_f=master_f,
                                           overwrite=overwrite,
                                           n_proc=self.__n_proc,
                                           compression=self.__compression,
                                           callback=callback)
        self.__merge_thread.start()

        if blocking:
            self.wait()

    def wait(self):
        if self.__merge_thread is not None:
            self.__merge_thread.wait()
        if self.__parse_thread is not None:
            self.__parse_thread.wait()

    def abort_merge(self, wait=True):
        if self.__merge_thread is not None:
            self.__merge_thread.abort(wait=wait)

    def merge_results(self):
        if self.__merge_thread is not None:
            return self.__merge_thread.results()
        else:
            return None

    def merge_progress(self):
        if self.__merge_thread is not None:
            return self.__merge_thread.progress()
        return None

    def select(self, scan_ids, clear=False):

        self.__check_parsed()

        if not isinstance(scan_ids, (list, tuple)):
            scan_ids = [scan_ids]

        scan_ids = set(scan_ids)
        unknown_scans = scan_ids - set(self.__matched_scans)

        if len(unknown_scans) != 0:
            err_ids = '; '.join('{0}'.format(scan for scan in unknown_scans))
            raise ValueError('Unknown scan IDs : {0}.'.format(err_ids))

        if clear:
            self.__selected_ids = scan_ids
        else:
            self.__selected_ids |= scan_ids

    def unselect(self, scan_ids):

        self.__check_parsed()

        if not issubclass(scan_ids, (list, tuple)):
            scan_ids = [scan_ids]

        self.__selected_ids -= set(scan_ids)

    def get_scan_command(self, scan_id):
        with h5py.File(self.__spec_h5, 'r') as spec_h5:
            try:
                scan = spec_h5[scan_id]
            except KeyError:
                raise ValueError('Scan ID {0} not found.')

            command = scan['title'][()].decode()

        _COMMAND_LINE_PATTERN = ('^(?P<id>[0-9]*) '
                                 '(?P<command>[^ ]*) '
                                 '(?P<motor_0>[^ ]*) '
                                 '(?P<motor_0_start>[^ ]*) '
                                 '(?P<motor_0_end>[^ ]*) '
                                 '(?P<motor_0_step>[^ ]*) '
                                 '(?P<motor_1>[^ ]*) '
                                 '(?P<motor_1_start>[^ ]*) '
                                 '(?P<motor_1_end>[^ ]*) '
                                 '(?P<motor_1_step>[^ ]*) '
                                 '(?P<delay>[^ ]*)\s*'
                                 '$')
        cmd_rgx = re.compile(_COMMAND_LINE_PATTERN)
        cmd_match = cmd_rgx.match(command)
        if cmd_match is None:
            raise ValueError('Failed to parse command line for scan ID {0} : '
                             '{1}.'.format(scan_id, command))
        cmd_dict = cmd_match.groupdict()
        cmd_dict.update(full=command)
        return cmd_dict

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

        def check_key(dic, key, error):
            values = [command[key] for command in commands]
            values_set = set(values)
            if len(values_set) != 1:
                errors.append('Selected scans command inconsistency : '
                              '"{0}" : {1}.'
                              ''.format(key, '; '.join(str(m)
                                        for m in values_set)))

        check_key(commands, 'motor_0', errors)
        check_key(commands, 'motor_1', errors)
        check_key(commands, 'motor_0_start', errors)
        check_key(commands, 'motor_1_start', errors)
        check_key(commands, 'motor_0_end', errors)
        check_key(commands, 'motor_1_end', errors)
        check_key(commands, 'motor_0_step', errors)
        check_key(commands, 'motor_1_step', errors)
        check_key(commands, 'delay', errors)

        return errors

    def check_parameters(self):
        errors = []
        if self.__beam_energy is None:
            errors.append('invalid "Beam energy"')
        if self.__output_dir is None:
            errors.append('invalid "Output directory"')
        if self.__center_chan is None:
            errors.append('invalid "Center channel"')
        if self.__chan_per_deg is None:
            errors.append('invalid "Channel per degree"')
        if self.__pixelsize is None:
            errors.append('invalid "Pixel size"')
        if self.__detector_orient is None:
            errors.append('invalid "Detector orientation"')
        return errors

    def get_imagefile_info(self, scan_id, key=None):
        self.__check_parsed()

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
        self.__check_parsed()

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

    @master_file.setter
    def master_file(self, master):
        # self.__check_parsed()

        if master is None or len(master) == 0:
            master = self.common_prefix()
            if len(master) == 0:
                self.__master = 'master'
            else:
                self.__master = master + '_master'

        elif isinstance(master, str):
            self.__master = master
        else:
            raise TypeError('master must be a string, or None.')

    def __gen_scan_filename(self, scan_id, fullpath=False):
        pattern = '{img_file}_{scan_id}.h5'
        img_file = self.__matched_scans[scan_id]['image']
        img_file = os.path.basename(img_file).split('.')[0]
        merged_file = pattern.format(img_file=img_file, scan_id=scan_id)

        if fullpath:
            merged_file = os.path.join(self.output_dir, merged_file)
        return merged_file

    def __gen_master_filename(self, fullpath=False):
        master = self.__master
        if not master.endswith('.h5'):
            master += '.h5'
        if fullpath:
            master = os.path.join(self.output_dir, master)
        return master

    def summary(self, fullpath=False):
        self.__check_parsed()
        if self.__output_dir is None:
            raise ValueError('output_summary() cannot be called '
                             'before an output directory has been set. '
                             'Please call set_output_dir() first.')

        master = self.__gen_master_filename(fullpath=fullpath)
        files = {'master': master}

        sel_ids = list(self.__selected_ids)
        {files.update({scan_id: self.__gen_scan_filename(scan_id,
                                                         fullpath=fullpath)
                       for scan_id in sel_ids})}

        return files

    beam_energy = property(lambda self: self.__beam_energy)

    @beam_energy.setter
    def beam_energy(self, beam_energy):
        self.__beam_energy = float(beam_energy)

    pixelsize = property(lambda self: self.__pixelsize)

    @pixelsize.setter
    def pixelsize(self, pixelsize):
        if len(pixelsize) != 2:
            raise ValueError('pixelsize must be a two elements array.')
        self.__pixelsize = [float(pixelsize[0]), float(pixelsize[1])]

    chan_per_deg = property(lambda self: self.__chan_per_deg)

    @chan_per_deg.setter
    def chan_per_deg(self, chan_per_deg):
        if len(chan_per_deg) != 2:
            raise ValueError('chan_per_deg must be a two elements array.')
        self.__chan_per_deg = [float(chan_per_deg[0]), float(chan_per_deg[1])]

    center_chan = property(lambda self: self.__center_chan)

    @center_chan.setter
    def center_chan(self, center_chan):
        if len(center_chan) != 2:
            raise ValueError('center_chan must be a two elements array.')
        self.__center_chan = [float(center_chan[0]), float(center_chan[1])]

    detector_orient = property(lambda self: self.__detector_orient)

    @detector_orient.setter
    def detector_orient(self, detector_orient):
        if not isinstance(detector_orient, str):
            raise ValueError('detector_orient must be a string.')
        self.__detector_orient = detector_orient

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

    matched_ids = property(lambda self: self.__matched_ids)
    selected_ids = property(lambda self: sorted(self.__selected_ids))
    no_match_ids = property(lambda self: self.__no_match_ids)
    no_img_ids = property(lambda self: self.__no_img_ids)
    on_error_ids = property(lambda self: self.__on_error_ids)
    parsed = property(lambda self: self.__parsed)
    merged = property(lambda self: self.__merged)


# #######################################################################
# #######################################################################
# #######################################################################


def merge_scan_data(output_dir,
                    spec_fname,
                    beam_energy=None,
                    chan_per_deg=None,
                    pixelsize=None,
                    center_chan=None,
                    detector_orient='',
                    scan_ids=None,
                    master_f=None,
                    img_dir=None,
                    n_proc=None,
                    version=1,
                    compression='lzf',
                    overwrite=False):
    """
    Creates a "master" HDF5 file and one HDF5 per scan. Those scan HDF5 files
    contain spec data (from *spec_fname*) as well as the associated
    image data. This file will either contain all valid scans or the one
    selected using the scan_ids parameter. A valid scan is a scan associated
    with an (existing) image file. Existing output files will be
    overwritten.

    :param output_dir: folder name into which output data (as well as
        temporary files) will be written.
    :type output_dir: str

    :param spec_fname: path to the spec file.
    :type output_dir: str

    :param beam_energy: beam energy in ....
    :type beam_energy: numeric

    :param chan_per_deg: 2 elements array containing the number of channels
        per degree (as defined by xrayutilitied, used when converting to
        reciprocal space coordinates).
    :type chan_per_deg: array_like

    :param pixelsize: 2 elements array containing the pixel size of the,
        detector, in TBD.
    :type pixelsize: *optional* array_like

    :param center_chan: 2 elements array containing the coordinates of the
        direct beam position in the detector coordinates.
    :type center_chan: *optional* array_like

    :param scan_ids: array of scan numbers to add to the merged file. If
        None, all valid scans will be merged.
    :type scan_ids: *optional* array of int

    :param master_f: name of the "master" (top level) HDF5 file.
        If None, the file will be named master.h5. This file is created in the
        folder pointed to by output_dir.
    :type master_f: *optional* str

    :param img_dir: directory path. If provided the image files will be
        looked for into that folder instead of the one found in the scan
        headers.
    :type img_dir: *optional* str

    :param n_proc: Number of threads to use when merging files. If None, the
        number of threads used will be the value returned by the function
        `multiprocessing.cpu_count()`
    :type n_proc: *optional* str

    :param version: version of the spec file. It is currently used to get
    the offset and padding to apply to the nextNr value found in the spec scan
    headers. This nextNr is then used to generate the image file name. Set it
    to 0 if you are merging data generated before April 2016 (TBC).
    :type img_dir: *optional* int

    :returns: a list of scan IDs that were merged
    :rtype: *list*
    """

    id01_merger = Id01DataMerger(spec_fname,
                                 output_dir,
                                 img_dir=img_dir,
                                 version=version)

    id01_merger.parse()

    id01_merger.output_dir = output_dir

    id01_merger.beam_energy = beam_energy
    id01_merger.center_chan = center_chan
    id01_merger.chan_per_deg = chan_per_deg
    id01_merger.pixelsize = pixelsize
    id01_merger.detector_orient = detector_orient
    id01_merger.n_proc = n_proc
    id01_merger.compression = compression

    merged = id01_merger.merge(overwrite=overwrite)

    return merged

# #######################################################################
# #######################################################################
# #######################################################################


def _spec_to_h5(spec_filename, h5_filename):
    """
    Converts a spec file into a HDF5 file.

    :param spec_filename: name of the spec file to convert to HDF5.
    :type spec_filename: str

    :param h5_filename: name of the HDF5 file to create.
    :type h5_filename: str

    .. seealso : silx.io.convert_spec_h5.convert
    """
    spectoh5.convert(spec_filename,
                     h5_filename,
                     mode='w')


# ########################################################################
# ########################################################################


def _spec_get_img_filenames(spec_h5_filename):
    """
    Parsed spec scans headers to retrieve the associated image files.

    :param spec_h5_filename: name of the HDF5 file containing spec data.
    :type spec_h5_filename: str

    .. todo : can we suppose that there's
        only one scan per scan number?
        i.e : no 0.2, 1.2, ...
    .. todo : expecting only one imageFile comment line per scan. Is this
        always true?

    :return: 3 elements tuple : a dict containing the scans that have valid
        image file info, a list with the scans that dont have any image files,
        and a list of the scans that have more that one image files.
    :rtype: *list* (*dict*, *list*, *list*)
    """
    with h5py.File(spec_h5_filename, 'r') as h5_f:

        # scans for which a file name was found
        with_file = {}

        # scans for which no file name was found
        without_file = []

        # scans for which more than one file name was found
        # -> this case is not expected/handled atm
        error_scans = []

        # regular expression to find the imagefile line
        regx = re.compile(_IMAGEFILE_LINE_PATTERN)

        for k_scan, v_scan in h5_f.items():
            header = [s.decode()
                      for s in v_scan['instrument/specfile/scan_header']]
            imgfile_match = [m for line in header
                             if line.startswith('#C imageFile')
                             for m in [regx.match(line.strip())] if m]

            # TODO : provide some more info
            # expecting only one imagefile line per scan
            if len(imgfile_match) > 1:
                error_scans.append(k_scan)
                continue

            # if no imagefile line
            if len(imgfile_match) == 0:
                without_file.append(k_scan)
                continue

            # extracting the named subgroups
            imgfile_grpdict = imgfile_match[0].groupdict()
            with_file[k_scan] = imgfile_grpdict

        return with_file, without_file, error_scans


# ########################################################################
# ########################################################################


def _find_scan_img_files(spec_h5_filename,
                         img_dir=None,
                         version=1):
    """
    Parses the provided "*spec*" HDF5 file and tries to find the edf file
    associated  with each scans. will look for the files in img_dir if
    provided (instead of looking for the files in the path written
    in the spec file).

    :param spec_h5_filename: name of the HDF5 file containing spec data.
    :type spec_h5_filename: str

    :param img_dir: directory path. If provided the image files will be
        looked for into that folder instead of the one found in the scan
        headers.
    :type img_dir: *optional* str

    :param version: version of the spec file. It is currently used to get
    the offset and padding to apply to the nextNr value found in the spec scan
    headers. This nextNr is then used to generate the image file name. Set it
    to 0 if you are merging data generated before April 2016 (TBC).
    :type img_dir: *optional* int

    :returns: 4 elements tuple : a dict containing the scans whose image file
        has been found, a dict containing the scans that have that have
        valid image file info in the scan header but whose image file has not
        been found, a list with the scans that dont have any image file info,
        and a list of the scans that have more that one image file info line.
    :rtype: *list* (*dict*, *dict*, *list*, *list*)
    """

    if img_dir and not os.path.exists(img_dir):
        raise ValueError('Image folder not found : {0}'
                         ''.format(img_dir))

    imgfile_info = _spec_get_img_filenames(spec_h5_filename)
    with_files = imgfile_info[0]
    complete_scans = {}
    incomplete_scans = {}

    if version == 0:
        nextnr_ofst = -1
        nextnr_pattern = '{0:0>4}'
    else:
        nextnr_ofst = 0
        nextnr_pattern = '{0:0>5}'

    if img_dir:
        img_dir = os.path.expanduser(os.path.expandvars(img_dir))

    for scan_id, infos in with_files.items():
        parsed_fname = (infos['prefix'] +
                        nextnr_pattern.format(int(infos['nextNr']) +
                                              nextnr_ofst) +
                        infos['suffix'])
        img_file = None

        if not img_dir:
            parsed_fname = os.path.join(infos['dir'], parsed_fname)
            if os.path.isfile(parsed_fname):
                img_file = parsed_fname
        else:
            edf_fullpath = glob.glob(os.path.join(img_dir, parsed_fname))
            if edf_fullpath:
                img_file = edf_fullpath[0]

        if img_file:
            complete_scans[scan_id] = {'spec': infos, 'image': img_file}
        else:
            incomplete_scans[scan_id] = {'spec': infos, 'image': None}

    result = [complete_scans, incomplete_scans]
    result.extend(elem for elem in imgfile_info[1:])

    return tuple(result)


# #######################################################################
# #######################################################################
# #######################################################################


class _ParseThread(Thread):
    def __init__(self,
                 spec_fname,
                 spec_h5,
                 img_dir,
                 version,
                 callback=None):
        super(_ParseThread, self).__init__()

        self.__spec_fname = spec_fname
        self.__spec_h5 = spec_h5
        self.__callback = callback
        self.__img_dir = img_dir
        self.__version = version

        self.__results = None

    def run(self):
        _spec_to_h5(self.__spec_fname, self.__spec_h5)

        self.__results = _find_scan_img_files(self.__spec_h5,
                                              img_dir=self.__img_dir,
                                              version=self.__version)

        if self.__callback:
            self.__callback()

    def results(self):
        return self.__results

    def progress(self):
        return None

    def wait(self):
        self.join()


# #######################################################################
# #######################################################################
# #######################################################################


class _MergeThread(Thread):
    def __init__(self,
                 output_dir,
                 spec_h5_fname,
                 scans,
                 beam_energy,
                 chan_per_deg,
                 pixelsize,
                 center_chan,
                 detector_orient,
                 master_f,
                 overwrite=True,
                 n_proc=None,
                 compression='lzf',
                 callback=None):
        super(_MergeThread, self).__init__()
        self.__output_dir = output_dir
        self.__scans = scans
        self.__spec_h5_fname = spec_h5_fname
        self.__output_dir = output_dir
        self.__beam_energy = beam_energy
        self.__chan_per_deg = chan_per_deg
        self.__pixelsize = pixelsize
        self.__center_chan = center_chan
        self.__detector_orient = detector_orient
        self.__n_proc = n_proc
        self.__compression = compression
        self.__master_f = master_f
        self.__overwrite = overwrite
        self.__callback = callback

        self.__results = None
        self.__proc_indices = None
        self.__shared_progress = mp_sharedctypes.RawArray(ctypes.c_int32,
                                                          len(scans))

        manager = Manager()
        self.__term_evt = manager.Event()
        self.__manager = manager

    def run(self):
        master_f = os.path.join(self.__output_dir, self.__master_f)

        if not self.__overwrite:
            mode = 'w-'
        else:
            mode = 'w'

        # trying to access the file (erasing it if necessary)
        with h5py.File(master_f, mode) as m_h5f:
            pass

        if self.__n_proc is None:
            n_proc = cpu_count()

        def init(term_evt_, shared_progress_):
            global g_term_evt
            global g_shared_progress
            g_term_evt = term_evt_
            g_shared_progress = shared_progress_

        np.frombuffer(self.__shared_progress, dtype='int32')[:] = 0

        pool = Pool(n_proc,
                    initializer=init,
                    initargs=(self.__term_evt,
                              self.__shared_progress),
                    maxtasksperchild=2)

        def callback(result_):
            scan, finished, info = result_
            print('{0} finished.'.format(scan))
            if not finished:
                self.__term_evt.set()

        results = {}
        proc_indices = {}
        for proc_idx, (scan_id, infos) in enumerate(self.__scans.items()):
            args = (scan_id,
                    proc_idx,
                    self.__spec_h5_fname,
                    self.__output_dir,
                    infos['output'], infos['image'],
                    self.__beam_energy, self.__chan_per_deg,
                    self.__pixelsize, self.__center_chan,
                    self.__detector_orient,
                    self.__compression, master_f,
                    mode)
            results[scan_id] = pool.apply_async(_add_edf_data,
                                                args,
                                                callback=callback)
            proc_indices[scan_id] = proc_idx

        pool.close()

        self.__results = results
        self.__proc_indices = proc_indices
        pool.join()

        valid = all(result[1] for result in results)
        if valid:
            with h5py.File(master_f, 'a') as m_h5f:
                items = self.__scans.items()
                for proc_idx, (scan_id, infos) in enumerate(items):
                    entry_fn = infos['output']
                    entry = entry_fn.rpartition('.')[0]
                    m_h5f[entry] = h5py.ExternalLink(entry_fn, entry)

        if self.__callback:
            self.__callback()

    def wait(self):
        self.join()

    def abort(self, wait=True):
        if self.is_alive():
            self.__term_evt.set()
            if wait:
                self.wait()

    def progress(self):
        progress = np.frombuffer(self.__shared_progress, dtype='int32')
        proc_indices = self.__proc_indices
        if proc_indices:
            merge_progress = {scan_id: progress[proc_idx]
                              for scan_id, proc_idx in proc_indices.items()}
        else:
            merge_progress = {scan_id: 0 for scan_id in self.__scans.keys()}
        return merge_progress

    def results(self, wait=True):
        if self.is_alive():
            if not wait:
                raise RuntimeError('Merge is still running. Please call '
                                   'wait() or abort() before asking for the '
                                   'results.')
            else:
                self.wait()

        if self.__results is None:
            raise RuntimeError('No restults available : start() has to be '
                               'called first.')

        errors = []
        completed = []
        for scan_id, async_res in self.__results.items():
            scan, finished, info = async_res.get()
            if finished:
                completed.append(scan)
            else:
                errors.append((scan, info))

        return completed, errors


# #######################################################################
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
                  pixelsize,
                  center_chan,
                  detector_orient,
                  compression,
                  master_f,
                  mode):

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

    if pixelsize is None:
        pixelsize = [-1., -1.]

    progress = np.frombuffer(g_shared_progress, dtype='int32')  # noqa
    progress[proc_idx] = 0

    try:
        print('Merging scan ID {0}'.format(scan_id))

        if g_term_evt.is_set():  # noqa
            raise Exception('Merge of scan {0} aborted.'.format(scan_id))
        with h5py.File(entry_fn, 'w') as entry_h5f:
            progress[proc_idx] = 1
            entry_grp = entry_h5f.create_group(entry)
            with h5py.File(spec_h5_fn, 'r') as s_h5f:
                scan_grp = s_h5f[scan_id]

                for grp_name in scan_grp:
                    scan_grp.copy(grp_name,
                                  entry_grp,
                                  shallow=False,
                                  expand_soft=True,
                                  expand_external=True,
                                  expand_refs=True,
                                  without_attrs=False)
            img_det_grp = entry_grp.require_group('instrument/image_detector')
            img_data_grp = entry_grp.require_group('measurement/image')

            img_det_grp.create_dataset('pixelsize_dim0',
                                       data=float(pixelsize[0]))
            img_det_grp.create_dataset('pixelsize_dim1',
                                       data=float(pixelsize[1]))
            if beam_energy is not None:
                img_det_grp.create_dataset('beam_energy',
                                           data=float(beam_energy))
            if chan_per_deg is not None:
                img_det_grp.create_dataset('chan_per_deg_dim0',
                                           data=float(chan_per_deg[0]))
                img_det_grp.create_dataset('chan_per_deg_dim1',
                                           data=float(chan_per_deg[1]))
            if center_chan is not None:
                img_det_grp.create_dataset('center_chan_dim0',
                                           data=float(center_chan[0]))
                img_det_grp.create_dataset('center_chan_dim1',
                                           data=float(center_chan[1]))
            if detector_orient is not None:
                img_det_grp.create_dataset('detector_orient',
                                           data=np.string_(detector_orient))

            progress[proc_idx] = 2
            edf_file = EdfFile.EdfFile(img_f, access='r', fastedf=True)
            progress[proc_idx] = 5

            n_images = edf_file.GetNumImages()

            image = edf_file.GetData(0)
            dtype = image.dtype
            img_shape = image.shape
            dset_shape = (n_images, img_shape[0], img_shape[1])
            chunks = (1, dset_shape[1]//4, dset_shape[2]//4)

            image_dset = img_data_grp.create_dataset('data',
                                                     shape=dset_shape,
                                                     dtype=dtype,
                                                     chunks=chunks,
                                                     compression=compression,
                                                     shuffle=True)

            # creating some links
            img_data_grp['info'] = img_det_grp
            img_det_grp['data'] = image_dset

            # attributes
            grp = entry_grp.require_group('measurement/image')
            grp.attrs['interpretation'] = np.string_('image')

            # setting the nexus classes
            grp = entry_grp.require_group('instrument')
            grp.attrs['NX_class'] = np.string_('NXinstrument')

            grp = entry_grp.require_group('instrument/image_detector')
            grp.attrs['NX_class'] = np.string_('NXdetector')

            grp = entry_grp.require_group('instrument/positioners')
            grp.attrs['NX_class'] = np.string_('NXcollection')

            grp = entry_grp.require_group('measurement')
            grp.attrs['NX_class'] = np.string_('NXcollection')

            grp = entry_grp.require_group('measurement/image')
            grp.attrs['NX_class'] = np.string_('NXcollection')

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
        result = (scan_id, False, str(ex))
    else:
        print('Entry {0} merged.'.format(entry))
        result = (scan_id, True, None)

    if result[1]:
        progress[proc_idx] = 100
    else:
        progress[proc_idx] = -1
    return result


if __name__ == '__main__':
    pass
