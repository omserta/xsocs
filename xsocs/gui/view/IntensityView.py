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

import numpy as np

from silx.gui import qt as Qt
from silx.gui.plot import PlotWindow

from matplotlib import cm

from ..model.TreeView import TreeView
from ..model.Model import Model, RootNode, Node
from ..model.ModelDef import ModelColumns, ModelRoles
from ..project.XsocsH5Factory import h5NodeToProjectItem
from ..project.Hdf5Nodes import H5GroupNode
from ..project.IntensityGroup import IntensityGroup
from ..widgets.XsocsPlot2D import XsocsPlot2D
from ..widgets.Input import StyledLineEdit
from ..widgets.Buttons import FixedSizePushButon
from ..widgets.Containers import GroupBox

try:
    from silx.gui.plot.ImageRois import ImageRoiManager
except ImportError:
    # TODO remove this import once the ROIs are added to the silx release.
    from ..silx_imports.ImageRois import ImageRoiManager


class RectRoiWidget(Qt.QWidget):
    sigRoiApplied = Qt.Signal(object)

    def __init__(self, roiManager, parent=None):
        # TODO :
        # support multiple ROIs then batch them
        super(RectRoiWidget, self).__init__(parent)

        self.__roiToolBar = roiToolBar = roiManager.toolBar(rois=['rectangle'],
                                        options=['show'])
        roiToolBar.setMovable(False)

        topLayout = Qt.QVBoxLayout(self)

        grpBox = GroupBox('ROI')
        layout = Qt.QGridLayout(grpBox)

        row = 0
        layout.addWidget(roiToolBar, row, 0, 1, 2, Qt.Qt.AlignTop)

        row += 1
        self._xEdit = edit = StyledLineEdit(nChar=6)
        edit.setReadOnly(True)
        layout.addWidget(Qt.QLabel('x='), row, 0, Qt.Qt.AlignTop)
        layout.addWidget(edit, row, 1, Qt.Qt.AlignTop)

        row += 1
        self._yEdit = edit = StyledLineEdit(nChar=6)
        edit.setReadOnly(True)
        layout.addWidget(Qt.QLabel('y='), row, 0, Qt.Qt.AlignTop)
        layout.addWidget(edit, row, 1, Qt.Qt.AlignTop)

        row += 1
        self._wEdit = edit = StyledLineEdit(nChar=6)
        edit.setReadOnly(True)
        layout.addWidget(Qt.QLabel('w='), row, 0, Qt.Qt.AlignTop)
        layout.addWidget(edit, row, 1, Qt.Qt.AlignTop)

        row += 1
        self._hEdit = edit = StyledLineEdit(nChar=6)
        edit.setReadOnly(True)
        layout.addWidget(Qt.QLabel('h='), row, 0, Qt.Qt.AlignTop)
        layout.addWidget(edit, row, 1, Qt.Qt.AlignTop)

        row += 1

        hLayout = Qt.QHBoxLayout()

        style = Qt.QApplication.style()

        icon = style.standardIcon(Qt.QStyle.SP_DialogApplyButton)
        self.__applyBn = applyBn = Qt.QToolButton()
        applyBn.setToolTip('Apply ROI')
        applyBn.setStatusTip('Apply ROI')
        applyBn.setIcon(icon)
        applyBn.setToolButtonStyle(Qt.Qt.ToolButtonTextBesideIcon)
        applyBn.setText('To Q Space')
        applyBn.setEnabled(False)
        hLayout.addWidget(applyBn)
        applyBn.clicked.connect(self.__applyRoi)

        icon = style.standardIcon(Qt.QStyle.SP_DialogCloseButton)
        self.__discardBn = discardBn = Qt.QToolButton()
        discardBn.setToolTip('Discard ROI')
        discardBn.setStatusTip('Discard ROI')
        discardBn.setIcon(icon)
        discardBn.setEnabled(False)
        hLayout.addWidget(discardBn, Qt.Qt.AlignRight)
        discardBn.clicked.connect(self.__discardRoi)

        layout.addLayout(hLayout, row, 0, 1, 2, Qt.Qt.AlignCenter)

        # topLayout.setSizeConstraint(Qt.QLayout.SetMinimumSize)

        topLayout.addWidget(grpBox)
        topLayout.addStretch(100)

        # TODO : weakref
        self.__roiManager = roiManager
        roiManager.sigRoiDrawingFinished.connect(self.__roiDrawingFinished,
                                                 Qt.Qt.QueuedConnection)
        roiManager.sigRoiRemoved.connect(self.__roiRemoved,
                                         Qt.Qt.QueuedConnection)
        roiManager.sigRoiMoved.connect(self.__roiMoved,
                                       Qt.Qt.QueuedConnection)

    def sizeHint(self):
        return Qt.QSize(self.__roiToolBar.sizeHint().width() + 10, 0)

    def __discardRoi(self, checked):
        self.__roiManager.clear()

    def __applyRoi(self, checked):
        # At the moment we only support one roi at a time.
        roi = self.__roiManager.rois
        roiItem = self.__roiManager.roiItem(roi[0])
        xMin = roiItem.pos[0]
        xMax = xMin + roiItem.width
        yMin = roiItem.pos[1]
        yMax = yMin + roiItem.height
        self.sigRoiApplied.emit([xMin, xMax, yMin, yMax])

    def __roiDrawingFinished(self, event):
        self.__display(event['xdata'], event['ydata'])
        self.__discardBn.setEnabled(True)
        self.__applyBn.setEnabled(True)

    def __clear(self):
        self._xEdit.clear()
        self._yEdit.clear()
        self._wEdit.clear()
        self._hEdit.clear()

    def __display(self, xData, yData):
        xMin, xMax = xData[0], xData[2]
        if xMax < xMin:
            xMin, xMax = xMax, xMin
        yMin, yMax = yData[0], yData[1]
        if yMax < yMin:
            yMin, yMax = yMax, yMin
        self._xEdit.setText(str(xMin))
        self._yEdit.setText(str(yMin))
        self._wEdit.setText(str(xMax - xMin))
        self._hEdit.setText(str(yMax - yMin))

    def __roiRemoved(self, name):
        self.__clear()
        self.__discardBn.setEnabled(False)
        self.__applyBn.setEnabled(False)

    def __roiMoved(self, event):
        self.__display(event['xdata'], event['ydata'])


class IntensityTotalNode(H5GroupNode):
    """
    Node displaying info about the number of entries selected.
    """

    total = property(lambda self: self.__total)
    samplePos = property(lambda self: self.__samplePos)
    icons = None

    def __init__(self, **kwargs):
        super(IntensityTotalNode, self).__init__(**kwargs)
        # TODO : check item type
        self.nodeName = 'Total'
        self.__total = None
        self.__samplePos = None
        self.__notifyModel = True

    def _loadChildren(self):
        iGroup = IntensityGroup(self.h5File, self.h5Path)
        iItems = iGroup.getIntensityItems()

        children = []
        for iItem in iItems:
            if iItem.entry == 'Total':
                continue
            iNode = IntensityViewItemNode(iItem)
            children.append(iNode)
        self.nodeName = 'Total {0} / {0}'.format(len(children))
        return children

    def _childInternalDataChanged(self, sender, *args):
        super(IntensityTotalNode, self)._childInternalDataChanged(sender,
                                                                  *args)
        if sender.parent() != self:
            return

        if self.__notifyModel:
            self.__getTotal()
            self.sigInternalDataChanged.emit([0])

    def __getTotal(self):
        total = None
        samplePos = None
        nSelected = 0

        childCount = self.childCount()

        for childIdx in range(childCount):
            child = self.child(childIdx)

            if child.checkState == Qt.Qt.Unchecked:
                continue

            nSelected += 1

            intensity, pos = child.item.getScatterData()
            if total is None:
                total = intensity
                samplePos = pos
            else:
                total += intensity

        blocked = self.blockSignals(True)
        self.nodeName = 'Total {0} / {1}'.format(nSelected, childCount)
        self.__total = total
        self.__samplePos = samplePos
        self.blockSignals(blocked)

    def scatterData(self):
        if self.total is None:
            self.__getTotal()
        return self.total, self.samplePos

    def getSelectedEntries(self):
        """
        Returns the list of entries, the list of selected entries indices,
        and the list of unselected entries indices.
        :return:
        """
        selected = []
        unselected = []
        entries = []
        for childIdx in range(self.childCount()):
            child = self.child(childIdx)

            entries.append(child.item.entry)

            if child.checkState == Qt.Qt.Unchecked:
                unselected.append(childIdx)
            else:
                selected.append(childIdx)

        return entries, selected, unselected

    def selectAll(self):
        """
        Selects all entries.
        :return:
        """
        # blocked = self.blockSignals(True)
        self.__notifyModel = False
        for childIdx in range(self.childCount()):
            child = self.child(childIdx)
            child.setCheckState(Qt.Qt.Checked)
        # self.blockSignals(blocked)
        self.__getTotal()
        self.__notifyModel = True
        self._notifyDataChange()

    def unselectAll(self):
        """
        Unselects all entries.
        :return:
        """
        # blocked = self.blockSignals(True)
        self.__notifyModel = False
        for childIdx in range(self.childCount()):
            child = self.child(childIdx)
            child.setCheckState(Qt.Qt.Unchecked)
        # self.blockSignals(blocked)
        self.__getTotal()
        self.__notifyModel = True
        self._notifyDataChange()


class IntensityViewItemNode(Node):
    checkable = True

    item = property(lambda self: self.__item)

    def __init__(self, iItem, **kwargs):
        super(IntensityViewItemNode, self).__init__(**kwargs)
        # TODO : check item type
        self.__item = iItem
        self.nodeName = str(iItem.projectRoot().shortName(iItem.entry))
        self.setCheckState(Qt.Qt.Checked)
        self.setData(IntensityModelColumns.AngleColumn,
                     iItem.entry,
                     Qt.Qt.ToolTipRole)

    def scatterData(self):
        return self.item.getScatterData()


class IntensityModelColumns(object):
    AngleColumn = range(1)
    ColumnNames = ['Angle']


class IntensityRootNode(RootNode):
    """
    Root node for the FitModel
    """
    ColumnNames = IntensityModelColumns.ColumnNames


class IntensityModel(Model):
    """
    Model displaying a FitH5 file contents.
    """
    RootNode = IntensityRootNode
    ModelColumns = IntensityModelColumns
    ColumnsWithDelegates = None

    def __iTotalNode(self):
        iTotalIndex = self.index(0, 0)
        return iTotalIndex.data(ModelRoles.InternalDataRole)

    def getSelectedEntries(self):
        iTotalNode = self.__iTotalNode()
        if iTotalNode is None:
            return []
        return iTotalNode.getSelectedEntries()

    def selectAll(self):
        iTotalNode = self.__iTotalNode()
        if iTotalNode is None:
            return []
        iTotalNode.selectAll()

    def unselectAll(self):
        iTotalNode = self.__iTotalNode()
        if iTotalNode is None:
            return []
        iTotalNode.unselectAll()


class IntensityTree(TreeView):
    sigCurrentChanged = Qt.Signal(object)

    def __init__(self, intensityGroupItem, **kwargs):
        super(IntensityTree, self).__init__(**kwargs)

        model = IntensityModel()
        iGroupNode = IntensityTotalNode(h5File=intensityGroupItem.filename,
                                        h5Path=intensityGroupItem.path)
        model.appendGroup(iGroupNode)
        self.setModel(model)
        self.setShowUniqueGroup(True)
        self.setExpanded(self.model().index(0, 0), True)
        model.startModel()

    def currentChanged(self, current, previous):
        super(IntensityTree, self).currentChanged(current, previous)
        node = current.data(ModelRoles.InternalDataRole)
        if not node:
            return
        self.sigCurrentChanged.emit(node)


class IntensityView(Qt.QMainWindow):
    sigProcessApplied = Qt.Signal(object)

    plot = property(lambda self: self.__plotWindow)

    def __init__(self,
                 parent,
                 model,
                 node,
                 **kwargs):
        super(IntensityView, self).__init__(parent=parent)

        self.setWindowTitle('[XSOCS] {0}'.format(node.h5Path))

        self.__displayedNode = None
        self.__selectedPoint = None

        self.__plotWindow = plotWindow = XsocsPlot2D()
        plotWindow.setShowMousePosition(True)
        plotWindow.setShowSelectedCoordinates(True)
        plotWindow.sigPointSelected.connect(self.__slotPointSelected)

        selector = Qt.QWidget()
        layout = Qt.QVBoxLayout(selector)

        # TODO : check item type
        self.__iGroup = intensityGroup = h5NodeToProjectItem(node)
        self.__tree = tree = IntensityTree(intensityGroup, parent=self)
        tree.model().dataChanged.connect(self.__slotModelDataChanged)
        tree.sigCurrentChanged.connect(self.__slotItemSelected)
        layout.addWidget(tree)

        bnLayout = Qt.QHBoxLayout()
        selAllBn = FixedSizePushButon('Select All')
        selNoneBn = FixedSizePushButon('Clear')
        selAllBn.clicked.connect(tree.model().selectAll)
        selNoneBn.clicked.connect(tree.model().unselectAll)
        bnLayout.addWidget(selAllBn)
        bnLayout.addWidget(selNoneBn)
        layout.addLayout(bnLayout)

        dock = Qt.QDockWidget(self)
        dock.setWidget(selector)
        features = dock.features() ^ Qt.QDockWidget.DockWidgetClosable
        dock.setFeatures(features)
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, dock)

        self.__roiManager = roiManager = ImageRoiManager(plotWindow)

        rectRoiWidget = RectRoiWidget(roiManager)
        rectRoiWidget.sigRoiApplied.connect(self.__slotRoiApplied)

        dock = Qt.QDockWidget(self)
        dock.setWidget(rectRoiWidget)
        features = dock.features() ^ Qt.QDockWidget.DockWidgetClosable
        dock.setFeatures(features)
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, dock)

        self.__profilePlot = profilePlot = XsocsPlot2D()
        profilePlot.setKeepDataAspectRatio(False)
        dock = Qt.QDockWidget(self)
        dock.setWidget(profilePlot)
        features = dock.features() ^ Qt.QDockWidget.DockWidgetClosable
        dock.setFeatures(features)
        self.addDockWidget(Qt.Qt.BottomDockWidgetArea, dock)

        self.setCentralWidget(plotWindow)

    def __slotModelDataChanged(self, topLeft, bottomRight, roles=None):
        nodeL = topLeft.data(ModelRoles.InternalDataRole)
        nodeR = bottomRight.data(ModelRoles.InternalDataRole)

        if nodeL != nodeR:
            print('Multiple selection not supported yet.')
            return

        if nodeL is None:
            return

        if not isinstance(nodeL, IntensityTotalNode):
            return
        # else: the total intensity has changed.

        self.__slotPointSelected(self.__selectedPoint)

        if nodeL == self.__displayedNode:
            self.__slotItemSelected(nodeL)

    def __slotItemSelected(self, node):
        """
        Slot called when an item is selected in the tree view. Updates the
        scatter plot accordingly.
        :param item:
        :return:
        """
        self.__displayedNode = node
        intensity, positions = node.scatterData()
        title = node.nodeName
        if intensity is None:
            self.__plotWindow.clear()
            self.__plotWindow.setGraphTitle(title)
            return
        self.setPlotData(positions.pos_0, positions.pos_1, intensity, title)

    def setPlotData(self, x, y, data, title=None):
        """
        Displays the scatter plot.
        :param x:
        :param y:
        :param data:
        :return:
        """
        plot = self.__plotWindow
        plot.setPlotData(x, y, data)
        plot.setGraphTitle(title)

    def __slotPointSelected(self, point):
        """
        Slot called when a point is selected on the intensity map.
        :param point:
        :return:
        """
        plot = self.__profilePlot
        plot.setGraphTitle(None)
        plot.clear()

        if point is None:
            return

        iGroup = self.__iGroup
        entries, selected, unselected = self.__tree.model().getSelectedEntries()
        nEntries = len(entries)

        xsocsH5 = iGroup.projectRoot().xsocsH5

        angles = np.ndarray(shape=(nEntries, ))
        intensities = np.ndarray(shape=(nEntries,))
        # TODO : error if selected not in iItems (isnt supposed to happen...)
        # TODO : make sure the angles are sorted?
        for entryIdx, entry in enumerate(entries):
            item = iGroup.getIntensityItem(entry)
            intensities[entryIdx] = item.getPointValue(point.xIdx)
            angles[entryIdx] = xsocsH5.scan_angle(entry)

        title = 'Profile @ ({0:.7g}, {1:.7g})'.format(point.x, point.y)
        plot.addCurve(angles, intensities, legend='i_entries')
        plot.setGraphTitle(title)

        if selected:
            plot.addCurve(angles[selected],
                          intensities[selected],
                          legend='i_selected')

        for unselIdx in unselected:
            plot.addXMarker(angles[unselIdx],
                            legend=entries[unselIdx],
                            color='red')

        self.__selectedPoint = point

    def __slotRoiApplied(self, roi):
        """
        Slot called when the ROI is applied.
        :param roi:
        :return:
        """
        self.sigProcessApplied.emit(roi)


if __name__ == '__main__':
    pass
