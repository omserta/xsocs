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
from .ModelDef import NodeClassDef, ModelColumns, ModelRoles
from .ProjectNode import ProjectNode


@NodeClassDef(nodeType='RootNode')
class RootNode(ProjectNode):
    pass


@NodeClassDef(nodeType=h5py.Group, icon='folder', editor=None)
class GroupNode(ProjectNode):
    pass


class XsocsNode(ProjectNode):
    def __init__(self, *args, **kwargs):
        super(XsocsNode, self).__init__(*args, **kwargs)
        self.setData(ModelColumns.NameColumn,
                     data=True,
                     role=ModelRoles.IsXsocsNodeRole)
        with h5py.File(self.projectFile) as h5f:
            item = h5f[self.path]
            processId = item.attrs.get('XsocsLevel')
            del item
        for colIdx in range(ModelColumns.ColumnMax):
            self.setData(colIdx,
                         data=processId,
                         role=ModelRoles.XsocsProcessId)
