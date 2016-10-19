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

from functools import partial
from collections import namedtuple

from silx.gui import qt as Qt, icons

from .Nodes import XsocsNode
from ..project.HybridItem import HybridItem
from .ModelDef import NodeClassDef, ModelRoles
from .ProjectNode import DelegateEvent, NodeDelegate


class HybridItemEvent(DelegateEvent):
    HybridEventData = namedtuple('HybridEventData', ['evtType', 'path'])


class HybridItemDelegate(NodeDelegate):

    def __init__(self, parent, option, index):
        super(HybridItemDelegate, self).__init__(parent, option, index)
        if not index.isValid():
            return
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        node = index.data(ModelRoles.InternalDataRole)
        item = HybridItem(node.projectFile, node.path)

        if item.hasScatter():
            icon = icons.getQIcon('item-1dim')
            bn = Qt.QToolButton()
            bn.setIcon(icon)
            bn.clicked.connect(partial(self.__onClicked, eventType='scatter'))
            layout.addWidget(bn, Qt.Qt.AlignLeft)
        if item.hasImage():
            icon = icons.getQIcon('item-2dim')
            bn = Qt.QToolButton()
            bn.setIcon(icon)
            bn.clicked.connect(partial(self.__onClicked, eventType='image'))
            layout.addWidget(bn, Qt.Qt.AlignLeft)
        layout.addStretch(1)

    def __onClicked(self, eventType=None):
        persistentIndex = self.index
        index = persistentIndex.model().index(persistentIndex.row(),
                                              persistentIndex.column(),
                                              persistentIndex.parent())
        node = index.internalPointer()
        data = HybridItemEvent.HybridEventData(evtType=eventType,
                                               path=node.path)
        event = HybridItemEvent(self.index, data=data)
        self.sigEditorEvent.emit(event)


@NodeClassDef(nodeType='HybridItem',
              icon='item-ndim',
              editor=HybridItemDelegate)
class HybridNode(XsocsNode):
    def childCount(self):
        return 0
