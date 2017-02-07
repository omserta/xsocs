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
from ...io.FitH5 import FitH5
from .ProjectItem import ProjectItem
from .ProjectDef import ItemClassDef


@ItemClassDef('FitGroup')
class FitGroup(ProjectItem):
    def addFitFile(self, fitFile):
        itemName = os.path.basename(fitFile).rsplit('.')[0]
        itemPath = self.path + '/' + itemName
        item = FitItem(self.filename, itemPath, mode='a')
        item.fitFile = fitFile
        return item

    def getFitItems(self):
        return self.children(classinfo=FitItem)


@ItemClassDef('FitItem')
class FitItem(ProjectItem):
    FitH5FilePath = 'input'

    def __init__(self, *args, **kwargs):
        self.__fitFile = None
        super(FitItem, self).__init__(*args, **kwargs)

    fitH5 = property(lambda self: FitH5(self.fitFile)
                     if self.fitFile else None)

    @property
    def fitFile(self):
        """ The name of the input data file. """
        if self.__fitFile is None:
            with self._get_file() as h5f:
                path = self.path + '/' + FitItem.FitH5FilePath
                if path in h5f:
                    group = h5f.get(path)
                    self.__fitFile = group.file.filename
                    del group
        return self.__fitFile

    @fitFile.setter
    def fitFile(self, fitFile):
        """ Set the fit data file for this item. The fit data
         file can only be set once. To use a different data file you have to
         create a new project. """
        # TODO : make sure file exists and is readable
        if self.fitFile is not None:
            raise ValueError('Fit input file is already set.')

        # Adding the external link to the file
        self.__fitFile = fitFile
        path = self.path + '/' + FitItem.FitH5FilePath
        self.add_file_link(path, fitFile, '/')
        self.setHidden(True, path)

        self._createItem()

    def _createItem(self):
        fitFile = self.fitFile
        if fitFile is None:
            return
