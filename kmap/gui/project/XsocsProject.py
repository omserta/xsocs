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
from ...io import XsocsH5 as _XsocsH5
from ..model.ProjectModel import ProjectModel
from ..model.ProjectView import ProjectView
from .SourceItem import SourceItem


class XsocsProject(_XsocsH5.XsocsH5Base):
    InputItemPath = '/Source'

    XsocsNone, XsocsInput, XsocsQSpace, XsocsFit = range(4)

    def __init__(self, *args, **kwargs):
        super(XsocsProject, self).__init__(*args, **kwargs)
        self.__xsocsFile = None
        self.__xsocsH5 = None
        self.__projectModel = None

    xsocsH5 = property(lambda self: _XsocsH5.XsocsH5(self.xsocsFile)
                       if self.xsocsFile else None)
    """ Returns an XsocsH5 instance if """

    def __model(self):
        """

        """
        if self.__projectModel is None:
            self.__projectModel = ProjectModel(self.filename)

        return self.__projectModel

    def view(self, parent=None):
        view = ProjectView(parent)
        view.setModel(self.__model())
        return view

    workdir = property(lambda self: os.path.dirname(self.filename))

    @property
    def xsocsFile(self):
        with self._get_file() as h5f:
            if h5f.get(self.InputItemPath) is None:
                return None
        return SourceItem(self.filename, self.InputItemPath).xsocsFile

    @xsocsFile.setter
    def xsocsFile(self, xsocs_f):
        item = SourceItem(self.filename, self.InputItemPath)
        item.xsocsFile = xsocs_f
