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
import os.path
from collections import namedtuple

from threading import Thread

import h5py
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


KmapParseResults = namedtuple('KmapParseResults',
                              ['img_info', 'no_img_info', 'error', 'spec_h5'])

KmapMatchResults = namedtuple('KmapMatchResults',
                              ['matched', 'not_matched',
                               'no_img_info', 'error',
                               'spec_h5'])


# #######################################################################
# #######################################################################


class KmapSpecParser(object):
    (READY, RUNNING, DONE,
     ERROR, CANCELED, UNKNOWN) = __STATUSES = range(6)
    """ Available status codes """

    status = property(lambda self: self.__status)
    """ Current status code of this instance """

    statusMsg = property(lambda self: self.__status_msg)
    """ Status message if any, or None """

    results = property(lambda self: self.__results)
    """ Parse results. KmapParseResults instance. """

    def __init__(self,
                 spec_fname,
                 spec_h5,
                 img_dir=None,
                 version=1,
                 nr_padding=None,
                 nr_offset=None,
                 callback=None):
        """
        Parser for the Kmap SPEC files. This loads a spec file, converts
            it to HDF5 and then tries to match scans and edf image files.
        :param spec_fname: Path to the spec file.
        :param spec_h5: Name of the HDF5 file that will be created and filled
            with the contents of the SPEC file
        .. seealso : silx.io.convert_spec_h5.convert
        :param img_dir: directory where the images are stored, if they are not
        stored where the path (#C imageFile line) in the SPEC file points to.
        :param version: version of the spec file. It is currently used to get
            the offset and padding to apply to the nextNr value found in the
            spec scan headers. This nextNr is then used to generate the image
            file name. Set it to 0 if you are merging data generated
            before April 2016 (TBC).
        :param nr_padding: zero padding to apply to the nextNr number found
            in the SPEC file.
        :param nr_offset: offset to apply to the nextNr number found
            in the SPEC file.
        :param callback: callback to call when the parsing is done
        """
        super(KmapSpecParser, self).__init__()

        self.__status = None

        self.__set_status(self.UNKNOWN, 'Init')

        self.__results = None

        self.__thread = None

        self.__spec_fname = spec_fname
        self.__spec_h5 = spec_h5
        self.__callback = callback
        self.__img_dir = img_dir
        self.__version = version

        self.__nr_padding = nr_padding
        self.__nr_offset = nr_offset

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

    def parse(self, blocking=True):
        """
        Starts the parsing.
        :param blocking: if False, the parsing will be done in a separate
         thread and this method will return immediately.
        :return:
        """

        if self.__thread and self.__thread.is_alive():
            raise RuntimeError('This KmapSpecParser instance is already '
                               'parsing.')

        self.__results = None

        if blocking:
            self.__run_parse()
        else:
            thread = self.__thread = Thread(target=self.__run_parse)
            thread.start()

    def __run_parse(self):
        """
        Runs the parsing.
        :return:
        """

        self.__set_status(self.RUNNING)

        try:
            _spec_to_h5(self.__spec_fname, self.__spec_h5)

            self.__results = _find_scan_img_files(self.__spec_h5,
                                                  img_dir=self.__img_dir,
                                                  version=self.__version,
                                                  nr_padding=self.__nr_padding,
                                                  nr_offset=self.__nr_offset)
            self.__set_status(self.DONE)
        except Exception as ex:
            self.__set_status(self.ERROR, str(ex))

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
            header = v_scan['instrument/specfile/scan_header'][()]
            header = header.decode().split('\n')
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

        return KmapParseResults(img_info=with_file,
                                no_img_info=without_file,
                                error=error_scans,
                                spec_h5=spec_h5_filename)


# ########################################################################
# ########################################################################


def _find_scan_img_files(spec_h5_filename,
                         img_dir=None,
                         version=1,
                         nr_padding=None,
                         nr_offset=None):
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
    with_files = imgfile_info.img_info
    complete_scans = {}
    incomplete_scans = {}

    if version == 0:
        nextnr_ofst = -1
        nextnr_pattern = '{0:0>4}'
    else:
        nextnr_ofst = 0
        nextnr_pattern = '{0:0>5}'

    if nr_padding is not None:
        nextnr_pattern = '{{0:0>{0}}}'.format(nr_padding)

    if nr_offset is not None:
        nextnr_ofst = nr_offset

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

    result = KmapMatchResults(matched=complete_scans,
                              not_matched=incomplete_scans,
                              no_img_info=imgfile_info.no_img_info,
                              error=imgfile_info.error,
                              spec_h5=spec_h5_filename)

    return result


if __name__ == '__main__':
    pass
