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
from functools import partial

import h5py
from silx.gui import qt as Qt, icons
from .ModelDef import NodeClassDef, ModelColumns, ModelRoles
from .ProjectNode import ProjectNode, DelegateEvent


@NodeClassDef(nodeType='RootNode')
class RootNode(ProjectNode):
    pass


@NodeClassDef(nodeType=h5py.Group, icon='folder', editor=None)
class GroupNode(ProjectNode):
    pass


class HybridItemEvent(DelegateEvent):
    def plotData(self):
        eventType = self.type
        if eventType == 'scatter':
            return self.item.getScatter()
        if eventType == 'image':
            return self.item.getImage()
        return None


class HybridItemDelegate(Qt.QWidget):
    sigEditorEvent = Qt.Signal(object)

    def __init__(self, parent, option, index):
        super(HybridItemDelegate, self).__init__(parent)
        self.__index = Qt.QPersistentModelIndex(index)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon = icons.getQIcon('item-1dim')
        bn = Qt.QToolButton()
        bn.setIcon(icon)
        bn.clicked.connect(partial(self.__onClicked, eventType='scatter'))
        layout.addWidget(bn, Qt.Qt.AlignLeft)
        icon = icons.getQIcon('item-2dim')
        bn = Qt.QToolButton()
        bn.setIcon(icon)
        bn.clicked.connect(partial(self.__onClicked, eventType='image'))
        layout.addWidget(bn, Qt.Qt.AlignLeft)
        layout.addStretch(1)
        # self.setAutoFillBackground(True)
        # layout.setSizeConstraint(Qt.QLayout.SetMinimumSize)

    def __onClicked(self, eventType=None):
        print 'CLICKED', self, eventType
        # obj = self.__index.data(Hdf5TreeModel.H5PY_OBJECT_ROLE)
        # instance = ProjectItem.load(obj.file.filename, obj.name)
        # event = HybridItemEvent(instance, type)
        # self.sigEditorEvent.emit(event)

    def sizeHint(self):
        return Qt.QSize(0, 0)


class XsocsNode(ProjectNode):
    def __init__(self, *args, **kwargs):
        super(XsocsNode, self).__init__(*args, **kwargs)
        self.setData(ModelColumns.NameColumn,
                     data=True,
                     role=ModelRoles.IsXsocsNodeRole)


@NodeClassDef(nodeType='HybridItem',
              icon='item-ndim', editor=HybridItemDelegate)
class HybridNode(XsocsNode):
    def childCount(self):
        return 0


class ExternalLinkDelegate(Qt.QWidget):
    sigEditorEvent = Qt.Signal(object)

    def __init__(self, parent, option, index):
        super(ExternalLinkDelegate, self).__init__(parent)
        self.__index = Qt.QPersistentModelIndex(index)
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
        print 'CLICKED', self
        # obj = self.__index.data(Hdf5TreeModel.H5PY_OBJECT_ROLE)
        # instance = ProjectItem.load(obj.file.filename, obj.name)
        # event = HybridItemEvent(instance, type)
        # self.sigEditorEvent.emit(event)

    def sizeHint(self):
        return Qt.QSize(0, 0)


@NodeClassDef(nodeType=h5py.ExternalLink,
              icon=Qt.QStyle.SP_FileLinkIcon,
              editor=ExternalLinkDelegate)
class ExternalLinkNode(XsocsNode):
    def __init__(self, *args, **kwargs):
        super(ExternalLinkNode, self).__init__(*args, **kwargs)
        with h5py.File(self.projectFile) as h5f:
            item = h5f[self.path]
            filename = item.file.filename
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
