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
import numpy as np

from .ProjectDef import getItemClass
from ...io.XsocsH5Base import XsocsH5Base


class ProjectItem(XsocsH5Base):
    itemName = None

    def __init__(self, h5File, nodePath, mode='r+', processLevel=None):
        # TODO : check if parent already has a child with the same name
        super(ProjectItem, self).__init__(h5File, mode=mode)
        self.__nodePath = nodePath
        self.__processLevel = None
        self.__processLevelIn = processLevel

    path = property(lambda self: self.__nodePath)

    @property
    def processLevel(self):
        if self.__processLevel is None:
            if self.__processLevelIn is None:
                with self._get_file() as h5f:
                    processLevel = h5f[self.__nodePath].attrs.get('XsocsLevel')
                    self.__processLevel = processLevel
            else:
                self.__processLevel = self.__processLevelIn
        return self.__processLevel

    def _commit(self):
        with self._get_file() as h5f:
            grp = h5f.require_group(self.path)
            grp.attrs['XsocsType'] = np.string_(self.itemName)
            if self.__processLevelIn is not None:
                grp.attrs['XsocsLevel'] = self.__processLevelIn
            del grp

    @classmethod
    def load(cls, h5File, groupPath):
        with h5py.File(h5File, 'r') as h5f:
            grp = h5f[groupPath]
            xsocsType = grp.attrs.get('XsocsType')
            del grp
            if xsocsType is None:
                return None
            klass = getItemClass(xsocsType)
            if klass is None:
                return None
        instance = klass(h5File, groupPath)
        return instance
