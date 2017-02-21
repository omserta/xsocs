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
import weakref

from silx.gui import qt as Qt, icons

from ..model.Node import Node
from ..model.ModelDef import ModelColumns
from ..model.NodeEditor import EditorMixin

from .IntensityGroup import IntensityItem
from .XsocsH5Factory import h5NodeToProjectItem
from .Hdf5Nodes import H5GroupNode, H5NodeClassDef, H5DatasetNode

from ..view.FitView import FitView
from ..view.QspaceView import QSpaceView
from ..view.intensity.IntensityView import IntensityView

from ..project.QSpaceGroup import QSpaceItem
from ..project.ProjectItem import ProjectItem
from ..project.IntensityGroup import IntensityGroup



class ScatterPlotButton(EditorMixin, Qt.QWidget):
    persistent = True

    sigValueChanged = Qt.Signal()

    def __init__(self, parent, option, index):
        super(ScatterPlotButton, self).__init__(parent, option, index)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon = icons.getQIcon('plot-widget')
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
                attribute=('XsocsClass', 'IntensityGroup'),
                icons='math-sigma')
class IntensityGroupNode(H5GroupNode):
    editors = ScatterPlotButton

    def __init__(self, *args, **kwargs):
        super(IntensityGroupNode, self).__init__(*args, **kwargs)

        self.__viewWidget = None

    def getView(self, parent=None):
        """
        Returns a IntensityView for this item's data.
        :param parent:
        :return:
        """

        view = self.__viewWidget
        if view is None or view() is None:
            iGroup = IntensityGroup(self.h5File, nodePath=self.h5Path)
            view = weakref.ref(IntensityView(iGroup, parent))
            self.__viewWidget = view
        return view()

    def _loadChildren(self):
        return []


@H5NodeClassDef('IntensityNode',
                attribute=('XsocsClass', 'IntensityItem'))
class IntensityNode(H5DatasetNode):
    # editors = ScatterPlotButton

    def _setupNode(self):
        with IntensityItem(self.h5File, self.h5Path, mode='r') as item:
            self.setData(ModelColumns.NameColumn,
                         str(item.projectRoot().shortName(item.entry)),
                         Qt.Qt.DisplayRole)


class QSpaceInfoNode(Node):
    """
    Simple node displaying the qspace conversion parameters.
    """
    icons = Qt.QStyle.SP_FileDialogInfoView

    def _loadChildren(self):
        # This node is created by a QSpaceItemNode, which is a H5Node
        # and H5Node have themselves as subject, and the groupClasses
        # inherit their parent's subject.
        qspaceNode = self.subject
        qspaceItem = ProjectItem(qspaceNode.h5File, qspaceNode.h5Path).cast()

        children = []

        if not isinstance(qspaceItem, QSpaceItem):
            node = Node(nodeName='Error, invalid file.')
            icon = Qt.qApp.style().standardIcon(
                Qt.QStyle.SP_MessageBoxCritical)
            node.setData(0, icon, Qt.Qt.DecorationRole)
            return [node]

        qspaceH5 = qspaceItem.qspaceH5

        ##################################################
        # Adding selected/discarded entries.
        ##################################################

        selected = qspaceH5.selected_entries
        discarded = qspaceH5.discarded_entries

        # support for previous versions.
        # TODO : remove sometimes...
        if selected is None or len(selected) == 0:
            selected = qspaceItem.projectRoot().xsocsH5.entries()
        if discarded is None:
            discarded = []

        nSelected = len(selected)
        nDiscarded = len(discarded)

        selectedList = Node(nodeName='Selected entries')
        selectedList.setData(0,
                             'Entries used for the conversion.',
                             role=Qt.Qt.ToolTipRole)
        selectedList.setData(ModelColumns.ValueColumn,
                             '{0}'.format(nSelected))
        for entry in selected:
            node = Node(nodeName=entry)
            selectedList.appendChild(node)
        children.append(selectedList)

        discardedList = Node(nodeName='Discarded entries')
        discardedList.setData(0,
                              'Discarded input entries.',
                              role=Qt.Qt.ToolTipRole)
        discardedList.setData(ModelColumns.ValueColumn,
                              '{0}'.format(nDiscarded))
        for entry in discarded:
            node = Node(nodeName=entry)
            discardedList.appendChild(node)
        children.append(discardedList)

        ##################################################
        # Adding ROI info
        ##################################################

        sampleRoi = qspaceH5.sample_roi
        toolTip = """<ul>
                    <li>xMin : {0:.7g}
                    <li>xMax : {1:.7g}
                    <li>yMin : {2:.7g}
                    <li>yMax : {3:.7g}
                    </ul>
                  """.format(*sampleRoi)
        roiNode = Node(nodeName='Roi')
        text = '{0:6g}, {1:6g}, {2:6g}, {3:6g}'.format(*sampleRoi)
        roiNode.setData(ModelColumns.ValueColumn, text)
        roiNode.setData(ModelColumns.NameColumn, toolTip, Qt.Qt.ToolTipRole)
        node = Node(nodeName='xMin')
        node.setData(ModelColumns.ValueColumn, '{0:.7g}'.format(sampleRoi[0]))
        roiNode.appendChild(node)
        node = Node(nodeName='xMax')
        node.setData(ModelColumns.ValueColumn, '{0:.7g}'.format(sampleRoi[1]))
        roiNode.appendChild(node)
        node = Node(nodeName='yMin')
        node.setData(ModelColumns.ValueColumn, '{0:.7g}'.format(sampleRoi[2]))
        roiNode.appendChild(node)
        node = Node(nodeName='yMax')
        node.setData(ModelColumns.ValueColumn, '{0:.7g}'.format(sampleRoi[3]))
        roiNode.appendChild(node)

        children.append(roiNode)

        ##################################################
        # Adding image binning.
        ##################################################
        node = Node(nodeName='Image binning')
        imageBinning = qspaceH5.image_binning
        # support for previous versions
        # TODO : remove eventualy
        if imageBinning is None:
            text = 'unavailable'
        else:
            text = '{0}x{1}'.format(*imageBinning)
        node.setData(ModelColumns.ValueColumn, text)
        children.append(node)

        ##################################################
        # Adding qspace dims.
        ##################################################
        qspaceDimsNode = Node(nodeName='Qspace size')
        qspaceDims = qspaceH5.qspace_dimensions
        # support for previous versions
        # TODO : remove eventualy
        text = '{0}x{1}x{2} (qx, qy, qz)'.format(*qspaceDims)
        qspaceDimsNode.setData(ModelColumns.ValueColumn, text)
        children.append(qspaceDimsNode)

        return children


@H5NodeClassDef('QSpaceItem',
                attribute=('XsocsClass', 'QSpaceItem'))
class QSpaceItemNode(H5GroupNode):
    editors = QSpaceButton
    groupClasses = [('Infos', QSpaceInfoNode)]

    def __init__(self, *args, **kwargs):
        super(QSpaceItemNode, self).__init__(*args, **kwargs)
        self.__viewWidget = None

    def getView(self, parent=None):
        """
        Returns a QSpaceView for this item's data.
        :param parent:
        :return:
        """
        view = self.__viewWidget
        if view is None or view() is None:
            view = weakref.ref(QSpaceView(parent,
                                          self.model,
                                          self))
            self.__viewWidget = view
        return view()

    def _loadChildren(self):
        # dirty hack to remove a legacy group from appearing in the tree
        # TODO : to be removed eventualy
        children = super(QSpaceItemNode, self)._loadChildren()
        filtered = [child for child in children
                    if os.path.basename(child.h5Path) != 'info']
        return filtered


class FitButton(EditorMixin, Qt.QWidget):
    persistent = True

    sigValueChanged = Qt.Signal()

    def __init__(self, parent, option, index):
        super(FitButton, self).__init__(parent, option, index)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon = icons.getQIcon('plot-widget')
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
        event = {'event': 'fit'}
        self.notifyView(event)

    def __export(self):
        fitItem = h5NodeToProjectItem(self.node)
        workdir = fitItem.projectRoot().workdir
        itemBasename = os.path.basename(fitItem.fitFile).rsplit('.')[0]
        itemBasename += '.txt'
        dialog = Qt.QFileDialog(self, 'Export fit results.')
        dialog.setFileMode(Qt.QFileDialog.AnyFile)
        dialog.setAcceptMode(Qt.QFileDialog.AcceptSave)
        dialog.selectFile(os.path.join(workdir, itemBasename))
        if dialog.exec_():
            csvPath = dialog.selectedFiles()[0]
            fitItem.fitH5.export_csv(fitItem.fitH5.entries()[0], csvPath)


@H5NodeClassDef('FitGroup',
                attribute=('XsocsClass', 'FitGroup'),
                icons='math-fit')
class FitGroupNode(H5GroupNode):
    pass


@H5NodeClassDef('FitItem',
                attribute=('XsocsClass', 'FitItem'),
                icons='math-fit')
class FitItemNode(H5GroupNode):
    editors = FitButton

    def _loadChildren(self):
        return []

    def __init__(self, *args, **kwargs):
        super(FitItemNode, self).__init__(*args, **kwargs)
        self.__viewWidget = None

    def getView(self, parent=None):
        """
        Returns a FitView for this item's data.
        :param parent:
        :return:
        """
        view = self.__viewWidget
        if view is None or view() is None:
            view = weakref.ref(FitView(parent,
                                       self.model,
                                       self))
            self.__viewWidget = view
        return view()


if __name__ == '__main__':
    pass
