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

# from .ProjectDef import ProcessId
# from .HybridItem import HybridItem
from ...io.QSpaceH5 import QSpaceH5
from .ProjectItem import ProjectItem
from .ProjectDef import ItemClassDef


@ItemClassDef('QSpaceGroup')
class QSpaceGroup(ProjectItem):
    def addQSpace(self, qspaceFile):
        itemName = os.path.basename(qspaceFile).rsplit('.')[0]
        itemPath = self.path + '/' + itemName
        item = QSpaceItem(self.filename, itemPath, mode='a')
        item.qspaceFile = qspaceFile
        return item


@ItemClassDef('QSpaceItem')
class QSpaceItem(ProjectItem):
    QSpaceH5FilePath = 'input'

    def __init__(self, *args, **kwargs):
        self.__qspaceFile = None
        super(QSpaceItem, self).__init__(*args, **kwargs)

    @property
    def qspaceFile(self):
        """ The name of the input data file. """
        if self.__qspaceFile is None:
            with self._get_file() as h5f:
                path = self.path + '/' + QSpaceItem.QSpaceH5FilePath
                if path in h5f:
                    group = h5f.get(path)
                    self.__qspaceFile = group.file.filename
                    del group
        return self.__qspaceFile

    @qspaceFile.setter
    def qspaceFile(self, qspace_f):
        """ Set the qspace data file for this item. The qspace data
         file can only be set once. To use a different data file you have to
         create a new project. """
        # TODO : make sure file exists and is readable
        if self.qspaceFile is not None:
            raise ValueError('Xsocs input file is already set.')

        # Adding the external link to the file
        self.__qspaceFile = qspace_f
        path = self.path + '/' + QSpaceItem.QSpaceH5FilePath
        self.add_file_link(path, qspace_f, '/')
        self.setHidden(True, path)

        self._createItem()

    def _createItem(self):
        qspaceFile = self.qspaceFile
        if qspaceFile is None:
            return

        with QSpaceH5(qspaceFile) as qspaceH5:
            with self:
                pathTpl = self.path + '/info/{0}'
                with qspaceH5.qspace_dset_ctx() as ctx:
                    shape = np.array(ctx.shape)
                itemPath = pathTpl.format('#')
                self._set_scalar_data(itemPath, shape[0])
                itemPath = pathTpl.format('shape')
                self._set_array_data(itemPath, shape[1:])
                qx = qspaceH5.qx
                qy = qspaceH5.qy
                qz = qspaceH5.qz
                itemPath = pathTpl.format('qx range')
                self._set_array_data(itemPath, np.array([qx[0], qx[-1]]))
                itemPath = pathTpl.format('qy range')
                self._set_array_data(itemPath, np.array([qy[0], qy[-1]]))
                itemPath = pathTpl.format('qz range')
                self._set_array_data(itemPath, np.array([qz[0], qz[-1]]))



# @ItemClassDef('QSpaceItem')
# class QSpaceItem(ProjectItem):
#     QSpaceFilePath = 'File'
#     AcqParamsPath = 'AcqParams'
#     SumPath = 'Sum'
#
#     def __init__(self, *args, **kwargs):
#         super(QSpaceItem, self).__init__(*args, **kwargs)
#         self.__qspaceFile = None
#
#     @property
#     def qspaceFile(self):
#         """ The name of the input data file. """
#         if self.__qspaceFile is None:
#             with self._get_file() as h5f:
#                 path = self.path + '/' + QSpaceItem.QSpaceFilePath
#                 group = h5f.get(path)
#                 if group:
#                     self.__qspaceFile = group.file.filename
#                 del group
#         return self.__qspaceFile
#
#     @qspaceFile.setter
#     def qspaceFile(self, qspace_f):
#         # TODO : make sure file exists and is readable
#         if self.qspaceFile is not None:
#             raise ValueError('Xsocs input file is already set.')
#
#         # adding a link to the source file
#         qspaceH5 = QSpaceH5(qspace_f)
#         self.__qspaceFile = qspace_f
#         qspaceRoot = '/' + '/'.join([self.path, QSpaceItem.QSpaceFilePath])
#         self.add_file_link(qspaceRoot, qspace_f, '/')
#
#         sumItemPath = '/'.join([self.path, QSpaceItem.SumPath])
#         intensityGrp = HybridItem(self.filename,
#                                   sumItemPath,
#                                   processLevel=ProcessId.QSpace)
#         sumPath = qspaceRoot + '/' + QSpaceH5.qspace_sum_path
#         xPath = qspaceRoot + '/' + QSpaceH5.sample_x_path
#         yPath = qspaceRoot + '/' + QSpaceH5.sample_y_path
#
#         sumLink = h5py.SoftLink(sumPath)
#         xLink = h5py.SoftLink(xPath)
#         yLink = h5py.SoftLink(yPath)
#         with qspaceH5:
#             intensityGrp.setScatter(xLink,
#                                     yLink,
#                                     sumLink)


