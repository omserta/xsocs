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

import numpy as np
from silx.gui import qt as Qt
from silx.gui.hdf5 import Hdf5TreeModel
from ...io import XsocsH5 as _XsocsH5
from .ProjectModel import ProjectModel
from .ProjectView import ProjectView
from .HybridItem import HybridItem
from .ProjectItem import ProjectItem


class XsocsProject(_XsocsH5.XsocsH5Base):
    H5_SOURCE_F = '/source/file'
    H5_SCAN_PARAMS = '/source/params'
    H5_INPUT_DATA = '/input/'
    GLOBAL_ENTRY = '_global'

    def __init__(self, *args, **kwargs):
        super(XsocsProject, self).__init__(*args, **kwargs)
        self.__xsocsFile = None
        self.__xsocsH5 = None
        self.__proxyModel = None

    xsocsH5 = property(lambda self: _XsocsH5.XsocsH5(self.xsocsFile)
                       if self.xsocsFile else None)
    """ Returns an XsocsH5 instance if """

    def __model(self):
        """

        """
        if self.__proxyModel:
            return self.__proxyModel
        self.__sourceModel = Hdf5TreeModel()
        self.__sourceModel.appendFile(self.filename)
        self.__proxyModel = ProjectModel()
        self.__proxyModel.setSourceModel(self.__sourceModel)
        return self.__proxyModel

    def view(self, parent=None):
        view = ProjectView(parent)
        view.setModel(self.__model())
        root = view.model().index(0, 0, Qt.QModelIndex())
        view.setRootIndex(root)
        inputNode = view.model().index(0, 0, root)
        view.expand(inputNode)
        # view.header().setResizeMode(Qt.QHeaderView.ResizeToContents)

        indices = view.model().match(view.model().index(0, 0, inputNode),
                                     ProjectModel.IsXsocsNode,
                                     1,
                                     hits=-1,
                                     flags=(Qt.Qt.MatchExactly |
                                            Qt.Qt.MatchRecursive))
        for index in indices:
            # had to do this otherwise the openPersistentEditor wouldnt work
            idx = view.model().index(index.row(), 1, index.parent())
            view.openPersistentEditor(idx)
        return view

    @property
    def xsocsFile(self):
        """ The name of the input data file. """
        if self.__xsocsFile is None:
            with self._get_file() as h5f:
                group = h5f.get(XsocsProject.H5_SOURCE_F)
                if group:
                    self.__xsocsFile = group.file.filename
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
        self.__xsocsH5 = h5f = _XsocsH5.XsocsH5(xsocs_f)
        self.__xsocsFile = xsocs_f
        self.add_file_link(XsocsProject.H5_SOURCE_F, xsocs_f, '/')

        # adding parameter values to the source folder
        entries = h5f.entries()
        # TODO : make sure that all parameters are consistent
        scan_params = h5f.scan_params(entries[0])
        path_tpl = '{0}/{{0}}'.format(XsocsProject.H5_SCAN_PARAMS)

        for key, value in scan_params.items():
            self._set_scalar_data(path_tpl.format(key), value)

        path_tpl = '{0}/{{0}}'.format(XsocsProject.H5_INPUT_DATA)
        globalIntensityGrp = HybridItem(self.filename,
                                        path_tpl.format('intensity'),
                                        processLevel=ProjectItem.XsocsInput)

        gIntensity = None
        gPos_0 = None
        gPos_1 = None
        gParams = None
        gSteps_0 = None
        gSteps_1 = None

        # adding misc. data
        path_tpl = '{0}/entries/{{0}}/{{1}}'.format(XsocsProject.H5_INPUT_DATA)
        for entry in entries:
            dataGrp = HybridItem(self.filename,
                                 path_tpl.format(entry,
                                                 'intensity'),
                                 processLevel=ProjectItem.XsocsInput)
            data = h5f.image_cumul(entry)
            pos_0, pos_1 = h5f.scan_positions(entry)

            # intensity as a scatter plot
            dataGrp.setScatter(pos_0, pos_1, data)

            # intensity as an image
            scan_params = h5f.scan_params(entry)
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
