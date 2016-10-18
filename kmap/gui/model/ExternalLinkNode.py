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
import h5py
from silx.gui import qt as Qt
from .ModelDef import NodeClassDef, ModelColumns
from .ProjectNode import DelegateEvent, NodeDelegate
from .Nodes import XsocsNode


class ExternalLinkEvent(DelegateEvent):
    pass


class ExternalLinkDelegate(NodeDelegate):

    def __init__(self, parent, option, index):
        super(ExternalLinkDelegate, self).__init__(parent, option, index)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        style = Qt.QApplication.style()
        icon = style.standardIcon(Qt.QStyle.SP_FileDialogContentsView)
        bn = Qt.QToolButton()
        bn.setIcon(icon)
        layout.addWidget(bn, Qt.Qt.AlignLeft)
        bn.clicked.connect(self.__onClicked)
        layout.addStretch(1)

    def __onClicked(self):
        persistentIndex = self.index
        index = persistentIndex.model().index(persistentIndex.row(),
                                              persistentIndex.column(),
                                              persistentIndex.parent())
        node = index.internalPointer()
        event = ExternalLinkEvent(self.index, data=node.externalFile)
        self.sigEditorEvent.emit(event)


@NodeClassDef(nodeType=h5py.ExternalLink,
              icon=Qt.QStyle.SP_FileLinkIcon,
              editor=ExternalLinkDelegate)
class ExternalLinkNode(XsocsNode):
    def __init__(self, *args, **kwargs):
        super(ExternalLinkNode, self).__init__(*args, **kwargs)
        with h5py.File(self.projectFile) as h5f:
            item = h5f[self.path]
            self.__externalFile = filename = item.file.filename
            followLink = item.attrs.get('XsocsExpand')
            del item
        self.__followLink = followLink if followLink is not None else False
        basename = os.path.basename(filename).rpartition('.')[0]
        self.setData(ModelColumns.NameColumn,
                     basename,
                     role=Qt.Qt.DisplayRole)
        self.setData(ModelColumns.NameColumn,
                     filename,
                     role=Qt.Qt.ToolTipRole)

    def childCount(self):
        if not self.__followLink:
            return 0
        return super(ExternalLinkNode, self).childCount()

    externalFile = property(lambda self: self.__externalFile)
