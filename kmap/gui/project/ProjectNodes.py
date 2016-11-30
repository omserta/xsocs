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
__date__ = "01/11/2016"

import os

from silx.gui import qt as Qt, icons

from ..model.ModelDef import ModelColumns
from .IntensityGroup import IntensityGroup, IntensityItem
from ..model.NodeEditor import EditorMixin
from .Hdf5Nodes import H5GroupNode, H5NodeClassDef, H5DatasetNode
from .XsocsH5Factory import h5NodeToProjectItem
from ..model.Node import Node


class ScatterPlotButton(EditorMixin, Qt.QWidget):
    persistent = True

    sigValueChanged = Qt.Signal()

    def __init__(self, parent, option, index):
        super(ScatterPlotButton, self).__init__(parent, option, index)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon = icons.getQIcon('item-1dim')
        button = Qt.QToolButton()
        button.setIcon(icon)
        layout.addWidget(button)
        layout.addStretch(1)

        button.clicked.connect(self.__clicked)

    def __clicked(self):
        # node = self.node
        event = {'event': 'scatter'}
        self.notifyView(event)


class QSpaceButton(EditorMixin, Qt.QWidget):
    persistent = True

    sigValueChanged = Qt.Signal()

    def __init__(self, parent, option, index):
        super(QSpaceButton, self).__init__(parent, option, index)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon = icons.getQIcon('item-ndim')
        button = Qt.QToolButton()
        button.setIcon(icon)
        layout.addWidget(button)
        layout.addStretch(1)

        button.clicked.connect(self.__clicked)

    def __clicked(self):
        # node = self.node
        event = {'event': 'qspace'}
        self.notifyView(event)


@H5NodeClassDef('IntensityGroupNode',
                attribute=('XsocsClass', 'IntensityGroup'))
class IntensityGroupNode(H5GroupNode):
    editors = ScatterPlotButton


@H5NodeClassDef('IntensityNode',
                attribute=('XsocsClass', 'IntensityItem'))
class IntensityNode(H5DatasetNode):
    # editors = ScatterPlotButton

    def _setupNode(self):
        with IntensityItem(self.h5File, self.h5Path, mode='r') as item:
            self.setData(ModelColumns.NameColumn,
                         str(item.projectRoot().shortName(item.entry)),
                         Qt.Qt.DisplayRole)


@H5NodeClassDef('QSpaceItem',
                attribute=('XsocsClass', 'QSpaceItem'))
class QSpaceItemNode(H5GroupNode):
    editors = QSpaceButton


class FitButton(EditorMixin, Qt.QWidget):
    persistent = True

    sigValueChanged = Qt.Signal()

    def __init__(self, parent, option, index):
        super(FitButton, self).__init__(parent, option, index)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon = icons.getQIcon('item-1dim')
        button = Qt.QToolButton()
        button.setIcon(icon)
        button.clicked.connect(self.__clicked)
        layout.addWidget(button)

        button = Qt.QToolButton()
        style = Qt.QApplication.style()
        icon = style.standardIcon(Qt.QStyle.SP_DialogSaveButton)
        button.setIcon(icon)
        button.clicked.connect(self.__export)
        layout.addWidget(button)
        layout.addStretch(1)

    def __clicked(self):
        # node = self.node
        event = {'event': 'fit'}
        self.notifyView(event)

    def __export(self):
        fitItem = h5NodeToProjectItem(self.node)
        workdir = fitItem.projectRoot().workdir
        itemBasename = os.path.basename(fitItem.fitFile).rsplit('.')[0]
        itemBasename += '.txt'
        dialog = Qt.QFileDialog(self, 'Export fit results.')
        dialog.setFileMode(Qt.QFileDialog.AnyFile)
        dialog.selectFile(os.path.join(workdir, itemBasename))
        if dialog.exec_():
            csvPath = dialog.selectedFiles()[0]
            fitItem.fitH5.export_txt(csvPath)


class FitHeightNode(Node):
    className = 'height'


class FitCenterNode(Node):
    className = 'center'


class FitWidthNode(Node):
    className = 'width'


@H5NodeClassDef('FitItem',
                attribute=('XsocsClass', 'FitItem'))
class FitItemNode(H5GroupNode):
    editors = FitButton
    groupClasses = [(None, FitHeightNode),
                    (None, FitCenterNode),
                    (None, FitWidthNode)]