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
from .ProjectDef import ItemClassDef
from .ProjectItem import ProjectItem
from .AcqDataGroup import AcqDataGroup
from .IntensityGroup import IntensityGroup
from .QSpaceGroup import QSpaceGroup


@ItemClassDef('XsocsProject')
class XsocsProject(ProjectItem):
    AcquisitionGroupPath = '/Acquisition'
    # ScanPositionsPath = '/Positions'
    IntensityGroupPath = '/Intensity'
    QSpaceGroupPath = '/QSpace'

    XsocsNone, XsocsInput, XsocsQSpace, XsocsFit = range(4)

    def __init__(self, *args, **kwargs):
        super(XsocsProject, self).__init__(*args, **kwargs)
        self.__xsocsFile = None
        self.__xsocsH5 = None
        self.__projectModel = None

    workdir = property(lambda self: os.path.dirname(self.filename))

    def _createItem(self):
        AcqDataGroup(self.filename,
                     self.AcquisitionGroupPath,
                     mode=self.mode,
                     gui=self.gui)
        IntensityGroup(self.filename,
                       self.IntensityGroupPath,
                       mode=self.mode,
                       gui=self.gui)
        QSpaceGroup(self.filename,
                    self.QSpaceGroupPath,
                    mode=self.mode,
                    gui=self.gui)

    def positions(self, entry):
        with self.xsocsH5 as xsocsH5:
            if entry == 'Total':
                entry = xsocsH5.entries()[0]
            return xsocsH5.scan_positions(entry)

    def shortName(self, entry):
        if entry == 'Total':
            return entry
        with self.xsocsH5 as xsocsH5:
            return str(xsocsH5.scan_angle(entry))

    def qspaceGroup(self, mode=None):
        mode = mode or self.mode
        return QSpaceGroup(self.filename,
                           self.QSpaceGroupPath,
                           mode=mode)
