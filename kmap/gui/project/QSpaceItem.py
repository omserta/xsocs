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


import h5py

from .ProjectDef import ProcessId
from .HybridItem import HybridItem
from ...io.QSpaceH5 import QSpaceH5
from .ProjectItem import ProjectItem
from .ProjectDef import ItemClassDef


@ItemClassDef('QSpaceItem')
class QSpaceItem(ProjectItem):
    QSpaceFilePath = 'File'
    AcqParamsPath = 'AcqParams'
    SumPath = 'Sum'

    def __init__(self, *args, **kwargs):
        super(QSpaceItem, self).__init__(*args, **kwargs)
        self.__qspaceFile = None

    @property
    def qspaceFile(self):
        """ The name of the input data file. """
        if self.__qspaceFile is None:
            with self._get_file() as h5f:
                path = self.path + '/' + QSpaceItem.QSpaceFilePath
                group = h5f.get(path)
                if group:
                    self.__qspaceFile = group.file.filename
                del group
        return self.__qspaceFile

    @qspaceFile.setter
    def qspaceFile(self, qspace_f):
        # TODO : make sure file exists and is readable
        if self.qspaceFile is not None:
            raise ValueError('Xsocs input file is already set.')

        # adding a link to the source file
        qspaceH5 = QSpaceH5(qspace_f)
        self.__qspaceFile = qspace_f
        qspaceRoot = '/' + '/'.join([self.path, QSpaceItem.QSpaceFilePath])
        self.add_file_link(qspaceRoot, qspace_f, '/')

        sumItemPath = '/'.join([self.path, QSpaceItem.SumPath])
        intensityGrp = HybridItem(self.filename,
                                  sumItemPath,
                                  processLevel=ProcessId.QSpace)
        sumPath = qspaceRoot + '/' + QSpaceH5.qspace_sum_path
        xPath = qspaceRoot + '/' + QSpaceH5.sample_x_path
        yPath = qspaceRoot + '/' + QSpaceH5.sample_y_path

        sumLink = h5py.SoftLink(sumPath)
        xLink = h5py.SoftLink(xPath)
        yLink = h5py.SoftLink(yPath)
        with qspaceH5:
            intensityGrp.setScatter(xLink,
                                    yLink,
                                    sumLink)


