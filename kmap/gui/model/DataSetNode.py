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
from silx.gui import qt as Qt, icons
from .ModelDef import NodeClassDef, ModelColumns
from .ProjectNode import ProjectNode


@NodeClassDef(nodeType=h5py.Dataset, icon=None, editor=None)
class DatasetNode(ProjectNode):

    def __init__(self, *args, **kwargs):
        super(DatasetNode, self).__init__(*args, **kwargs)
        iconTpl = 'item-{0}dim'
        with h5py.File(self.projectFile) as h5f:
            item = h5f[self.path]
            ndims = len(item.shape)
            if ndims == 0:
                text = str(item[()])
            else:
                text = '...'
            del item

        icon = iconTpl.format(ndims)
        try:
            icon = icons.getQIcon(icon)
        except ValueError:
            icon = icons.getQIcon('item-ndim')

        self.setData(ModelColumns.NameColumn, icon, Qt.Qt.DecorationRole)

        self.setData(ModelColumns.ValueColumn,
                     text,
                     role=Qt.Qt.DisplayRole)

    def childCount(self):
        return 0
