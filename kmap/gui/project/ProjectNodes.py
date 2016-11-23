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

from silx.gui import qt as Qt, icons


from ..model.ModelDef import ModelColumns
from .IntensityGroup import IntensityGroup, IntensityItem
from ..model.NodeEditor import EditorMixin
from .Hdf5Nodes import H5GroupNode, H5NodeClassDef, H5DatasetNode


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
