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


import os

import numpy as np

from .ProjectDef import ProcessId
from .HybridItem import HybridItem
from ...io.XsocsH5 import XsocsH5
from .ProjectItem import ProjectItem
from .ProjectDef import ItemClassDef


@ItemClassDef('SourceItem')
class SourceItem(ProjectItem):
    XSocsFilePath = 'input'
    AcqParamsPath = 'AcqParams'
    DataPath = 'Data'
    EntriesPath = 'Entries'

    def __init__(self, *args, **kwargs):
        super(SourceItem, self).__init__(*args, **kwargs)
        self.__xsocsFile = None

    @property
    def xsocsFile(self):
        """ The name of the input data file. """
        if self.__xsocsFile is None:
            with self._get_file() as h5f:
                path = self.path + '/' + SourceItem.XSocsFilePath
                group = h5f.get(path)
                if group:
                    self.__xsocsFile = group.file.filename
                del group
        return self.__xsocsFile

    @xsocsFile.setter
    def xsocsFile(self, xsocs_f):
        """ Set the input data file for this Xsocs workspace. The input data
         file can only be set once. To use a different data file you have to
         create a new workspace. """
        # TODO : make sure file exists and is readable
        if self.xsocsFile is not None:
            raise ValueError('Xsocs input file is already set.')

        # adding a link to the source file
        xsocsH5 = XsocsH5(xsocs_f)
        self.__xsocsFile = xsocs_f
        path = '/'.join([self.path, SourceItem.XSocsFilePath])
        self.add_file_link(path, xsocs_f, '/')

        # adding parameter values to the source folder
        entries = xsocsH5.entries()
        # TODO : make sure that all parameters are consistent
        scan_params = xsocsH5.scan_params(entries[0])
        path_tpl = '{0}/{1}/{{0}}'.format(self.path, SourceItem.AcqParamsPath)

        for key, value in scan_params.items():
            self._set_scalar_data(path_tpl.format(key), value)

        path_tpl = '{0}/{1}/{{0}}'.format(self.path, SourceItem.DataPath)
        globalIntensityGrp = HybridItem(self.filename,
                                        path_tpl.format('intensity'),
                                        processLevel=ProcessId.Input)

        gIntensity = None
        gPos_0 = None
        gPos_1 = None
        gParams = None
        gSteps_0 = None
        gSteps_1 = None

        xsocs_f_prefix = os.path.basename(xsocs_f).rsplit('.')[0]
        # adding misc. data
        path_tpl = '{0}/{1}//{2}/{{0}}/{{1}}'.format(self.path,
                                                     SourceItem.DataPath,
                                                     SourceItem.EntriesPath)
        for entry in entries:
            entry_stripped = entry.lstrip(xsocs_f_prefix)
            dataGrp = HybridItem(self.filename,
                                 path_tpl.format(entry_stripped,
                                                 'intensity'),
                                 processLevel=ProcessId.Input)
            data = xsocsH5.image_cumul(entry)
            pos_0, pos_1 = xsocsH5.scan_positions(entry)

            # intensity as a scatter plot
            dataGrp.setScatter(pos_0, pos_1, data)

            # intensity as an image
            scan_params = xsocsH5.scan_params(entry)
            # xSlice = np.s_[0:scan_params['motor_0_steps']:1]
            # ySlice = np.s_[0::scan_params['motor_0_steps']]
            # dataGrp.setImageFromScatter(xSlice, ySlice)
            steps_0 = scan_params['motor_0_steps']
            steps_1 = scan_params['motor_1_steps']
            x = np.linspace(scan_params['motor_0_start'],
                            scan_params['motor_0_end'], steps_0, endpoint=False)
            y = np.linspace(scan_params['motor_1_start'],
                            scan_params['motor_1_end'], steps_1, endpoint=False)

            # TODO : check overflow
            if gIntensity is None:
                # TODO : see if we want to keep the first entry positions,
                # or an avg of all of them or ...
                gIntensity = data.copy()
                gPos_0 = pos_0
                gPos_1 = pos_1
                gParams = scan_params
                gSteps_0 = steps_0
                gSteps_1 = steps_1
            else:
                gIntensity += data

            data = data.reshape(steps_1, steps_0)
            dataGrp.setImage(x, y, data)
            del dataGrp

        globalIntensityGrp.setScatter(gPos_0, gPos_1, gIntensity)
        x = np.linspace(gParams['motor_0_start'],
                        gParams['motor_0_end'], gSteps_0, endpoint=False)
        y = np.linspace(gParams['motor_1_start'],
                        gParams['motor_1_end'], gSteps_1, endpoint=False)
        gIntensity = gIntensity.reshape(gSteps_1, gSteps_0)
        globalIntensityGrp.setImage(x, y, gIntensity)
        del globalIntensityGrp
