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

import numpy as np
from matplotlib import cm


from silx.gui import qt as Qt
from silx.gui.plot import PlotActions, PlotToolButtons, PlotWidget
from silx.gui.icons import getQIcon
from plot3d.ScalarFieldView import ScalarFieldView
from plot3d.SFViewParamTree import TreeView as SFViewParamTree

from ..model.TreeView import TreeView
from ..widgets.XsocsPlot2D import XsocsPlot2D
from ..process.FitWidget import FitWidget
from ..project.XsocsH5Factory import h5NodeToProjectItem

from ..Utils import nextFileName


class QSpaceTree(TreeView):
    pass


class PlotIntensityMap(XsocsPlot2D):
    """Plot intensities as a scatter plot

    :param parent: QWidget's parent
    """

    def __init__(self, parent=None, **kwargs):
        super(PlotIntensityMap, self).__init__(**kwargs)
        self.setMinimumSize(150, 150)

        self.setDataMargins(0.2, 0.2, 0.2, 0.2)
        self.setShowMousePosition(True)
        self.setShowSelectedCoordinates(True)

    def sizeHint(self):
        return Qt.QSize(200, 200)


class ROIPlotIntensityMap(PlotIntensityMap):
    """Plot ROI intensities with an update button to compute it in a thread"""

    _DEFAULT_TOOLTIP = 'Intensity Map: sum of the whole QSpace'
    _ROI_TOOLTIP = ('ROI Intensity Map: sum of the Region of Interest:\n' +
                    'qx = [%f, %f]\nqy = [%f, %f]\nqz = [%f, %f]')

    def __init__(self, parent, qspaceH5):
        self.__roiSlices = None  # qz, qy, qx ROI slices or None
        self.__roiQRange = None  # qx, qy, qz ROI range in Q space or None
        self.__qspaceH5 = qspaceH5
        super(ROIPlotIntensityMap, self).__init__(parent)
        self.setGraphTitle('ROI Intensity Map')
        self.setToolTip(self._DEFAULT_TOOLTIP)

        self.__updateButton = Qt.QPushButton(self)
        self.__updateButton.setText('Update')
        self.__updateButton.setIcon(getQIcon('view-refresh'))
        self.__updateButton.setToolTip('Compute the intensity map for the current ROI')
        self.__updateButton.clicked.connect(self.__updateClicked)

        toolBar = Qt.QToolBar('ROI Intensity Update', parent=self)
        toolBar.addWidget(self.__updateButton)
        self.addToolBar(Qt.Qt.BottomToolBarArea, toolBar)

    def roiChanged(self, selectedRegion):
        """To call when ROI has changed"""
        if selectedRegion is not None:
            self.__roiSlices = selectedRegion.getArraySlices()
            self.__roiQRange = selectedRegion.getDataRange()
        else:
            self.__roiSlices = None
            self.__roiQRange = None
        self.__updateButton.setEnabled(True)

    def __updateClicked(self, checked=False):
        """Handle button clicked"""

        if self.__roiSlices is None:
            # No ROI, use sum for the whole QSpace
            with self.__qspaceH5 as qspaceH5:
                intensities = np.array(qspaceH5.qspace_sum, copy=True)

        else:
            # Compute sum for QSpace ROI
            # This is performed as a co-routine using a QTimer

            # Show dialog
            dialog = Qt.QDialog(self)
            dialog.setWindowTitle('ROI Intensity Map')
            layout = Qt.QVBoxLayout(dialog)
            progress = Qt.QProgressBar()
            layout.addWidget(progress)

            btnBox = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Abort)
            btnBox.rejected.connect(dialog.reject)
            layout.addWidget(btnBox)

            dialog.setModal(True)
            dialog.show()

            qapp = Qt.QApplication.instance()
            with self.__qspaceH5 as qspaceH5:
                intensities = np.zeros((qspaceH5.qspace_sum.size,), dtype=np.float64)
                progress.setRange(0, qspaceH5.qspace_sum.size - 1)

                zslice, yslice, xslice = self.__roiSlices

                for index in range(qspaceH5.qspace_sum.size):
                    qspace = qspaceH5.qspace_slice(index)
                    intensities[index] = np.sum(qspace[xslice, yslice, zslice])
                    progress.setValue(index)
                    qapp.processEvents()
                    if not dialog.isVisible():
                        break  # It has been rejected by the abort button
                else:
                    dialog.accept()

            if dialog.result() == Qt.QDialog.Rejected:
                return  # Aborted, stop here

            intensities = np.array(intensities, copy=True)

        # Reset plot
        self.__updateButton.setEnabled(False)
        self.remove(kind='curve')

        # Update plot
        with self.__qspaceH5 as qsp:
            sampleX = qsp.sample_x
            sampleY = qsp.sample_y
            self.setPlotData(sampleX, sampleY, intensities)

        if self.__roiQRange is None:
            self.setToolTip(self._DEFAULT_TOOLTIP)
        else:
            self.setToolTip(
                self._ROI_TOOLTIP % tuple(self.__roiQRange.ravel()))


class CutPlanePlotWindow(PlotWidget):
    """Plot cut plane as an image

    :param parent: QWidget's parent
    """

    def __init__(self, parent=None):
        super(CutPlanePlotWindow, self).__init__(parent=parent)
        self.setMinimumSize(150, 150)

        # Create toolbar
        toolbar = Qt.QToolBar('Cut Plane Plot', self)
        self.addToolBar(toolbar)

        self.__resetZoomAction = PlotActions.ResetZoomAction(parent=self, plot=self)
        toolbar.addAction(self.__resetZoomAction)
        self.__colormapAction = PlotActions.ColormapAction(parent=self, plot=self)
        toolbar.addAction(self.__colormapAction)
        toolbar.addWidget(PlotToolButtons.AspectToolButton(
            parent=self, plot=self))
        toolbar.addWidget(PlotToolButtons.YAxisOriginToolButton(
            parent=self, plot=self))
        toolbar.addSeparator()
        self.__copyAction = PlotActions.CopyAction(parent=self, plot=self)
        toolbar.addAction(self.__copyAction)
        self.__saveAction = PlotActions.SaveAction(parent=self, plot=self)
        toolbar.addAction(self.__saveAction)
        self.__printAction = PlotActions.PrintAction(parent=self, plot=self)
        toolbar.addAction(self.__printAction)

        self.setKeepDataAspectRatio(True)
        self.setActiveCurveHandling(False)

    def sizeHint(self):
        return Qt.QSize(200, 200)


class QSpaceView(Qt.QMainWindow):
    """
    Window displaying the 3D q space isosurfaces.
    """

    sigFitDone = Qt.Signal(object, object)

    plot = property(lambda self: self.__plotWindow)

    def __init__(self,
                 parent,
                 model,
                 node):
        """

        :param parent: parent widget
        :param model: XsocsModel
        :param node: QspaceItem node
        """

        super(QSpaceView, self).__init__(parent)

        self.setWindowTitle('[XSOCS] {0}'.format(node.h5Path))

        self.__projectItem = item = h5NodeToProjectItem(node)

        self.__qspaceH5 = item.qspaceH5

        # plot window displaying the intensity map
        self.__plotWindow = plotWindow = PlotIntensityMap(parent=self)
        plotWindow.setToolTip('Intensity Map integrated on whole QSpaces')
        plotWindow.setPointSelectionEnabled(True)
        plotWindow.sigPointSelected.connect(self.__pointSelected)

        self.__roiPlotWindow = roiPlotWindow = ROIPlotIntensityMap(
            parent=self, qspaceH5=item.qspaceH5)
        roiPlotWindow.setPointSelectionEnabled(True)
        roiPlotWindow.sigPointSelected.connect(self.__pointSelected)

        self.__planePlotWindow = planePlotWindow = CutPlanePlotWindow(self)

        with item.qspaceH5 as qspaceH5:
            sampleX = qspaceH5.sample_x
            sampleY = qspaceH5.sample_y
            self.__setPlotData(sampleX,
                               sampleY,
                               qspaceH5.qspace_sum)
            self.__qx = qspaceH5.qx
            self.__qy = qspaceH5.qy
            self.__qz = qspaceH5.qz
            firstX = sampleX[0]
            firstY = sampleY[1]

        self.__node = node

        # setting up the plot3D and its param tree
        self.__view3d = view3d = ScalarFieldView()
        view3d.addIsosurface(0., '#FF000060')
        view3d.addIsosurface(
            lambda data: np.mean(data) + np.std(data),
            '#00FF00FF')
        view3d.setMinimumSize(400, 400)
        view3d.setAxesLabels('qx', 'qy', 'qz')
        self.setCentralWidget(view3d)
        sfTree = SFViewParamTree()
        sfTree.setSfView(view3d)

        # Register ROIPlotIntensity
        view3d.sigSelectedRegionChanged.connect(roiPlotWindow.roiChanged)

        # Store the cut plane signals connection state
        self.__connectedToCutPlane = True
        view3d.getCutPlanes()[0].sigPlaneChanged.connect(self.__cutPlaneChanged)
        view3d.getCutPlanes()[0].sigDataChanged.connect(self.__cutPlaneChanged)

        self.__fitWidget = fitWidget = FitWidget(self.__qspaceH5.filename)
        fitWidget.roiWidget().sigRoiChanged.connect(self.__slotRoiChanged)
        fitWidget.roiWidget().sigRoiToggled.connect(self.__slotRoiToggled)
        fitWidget.sigProcessStarted.connect(self.__slotFitProcessStarted)
        fitWidget.sigProcessDone.connect(self.__slotFitProcessDone)
        self.__nextFitFile()
        fitDock = Qt.QDockWidget()
        fitDock.setWindowTitle('Fit')
        fitDock.setWidget(fitWidget)
        features = fitDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        fitDock.setFeatures(features)
        view3d.addDockWidget(Qt.Qt.RightDockWidgetArea, fitDock)

        # widget that are to be disabled when the fit is running
        self.__lockWidgets = lockWidgets = []

        sfDock = Qt.QDockWidget()
        sfDock.setWindowTitle('Isosurface options')
        sfDock.setWidget(sfTree)
        features = sfDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        sfDock.setFeatures(features)
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, sfDock)
        lockWidgets.append(sfDock)

        planePlotDock = Qt.QDockWidget('Cut Plane', self)
        planePlotDock.setWidget(planePlotWindow)
        features = planePlotDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        planePlotDock.setFeatures(features)
        planePlotDock.visibilityChanged.connect(
            self.__planePlotDockVisibilityChanged)
        self.splitDockWidget(sfDock, planePlotDock, Qt.Qt.Vertical)
        lockWidgets.append(planePlotDock)

        roiPlotDock = Qt.QDockWidget('ROI Intensity', self)
        roiPlotDock.setWidget(roiPlotWindow)
        features = roiPlotDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        roiPlotDock.setFeatures(features)
        self.splitDockWidget(sfDock, roiPlotDock, Qt.Qt.Vertical)
        self.tabifyDockWidget(planePlotDock, roiPlotDock)
        lockWidgets.append(roiPlotDock)

        plotDock = Qt.QDockWidget('Intensity', self)
        plotDock.setWidget(plotWindow)
        features = plotDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        plotDock.setFeatures(features)
        self.tabifyDockWidget(roiPlotDock, plotDock)
        lockWidgets.append(plotDock)

        self.__showIsoView(firstX, firstY)

    def __setPlotData(self, x, y, data):
        """
        Sets the intensity maps data.
        :param x:
        :param y:
        :param data:
        :return:
        """
        self.__plotWindow.setPlotData(x, y, data)
        self.__plotWindow.resetZoom()
        self.__roiPlotWindow.setPlotData(x, y, data)
        self.__roiPlotWindow.resetZoom()

    def selectPoint(self, x, y):
        """
        Displays the q space closest to sample coordinates x and y.
        :param x:
        :param y:
        :return:
        """
        self.__showIsoView(x, y)

    def __pointSelected(self, point):
        """
        Slot called each time a point is selected on one of the intensity maps.
        Displays the corresponding q space cube.
        :param point:
        :return:
        """
        xIdx = point.xIdx
        x = point.x
        y = point.y

        self.__showIsoView(x, y, xIdx)

    def __showIsoView(self, x, y, idx=None):
        """
        Displays the q space closest to sample coordinates x and y.
        If idx is provided, x and y are ignored. If idx is not provided, the
        closest point to x and y is selected.
        :param x: sample x coordinate of the point to select
        :param y: sample y coordinate of the point to select
        :param idx: index of the point to select in the array of sample
        coordinates.
        :return:
        """
        isoView = self.__view3d

        if self.sender() == self.__roiPlotWindow:
            self.__plotWindow.selectPoint(x, y)
        elif self.sender() == self.__roiPlotWindow:
            self.__roiPlotWindow.selectPoint(x, y)
        else:
            self.__plotWindow.selectPoint(x, y)
            self.__roiPlotWindow.selectPoint(x, y)

        with self.__qspaceH5 as qspaceH5:
            if idx is None:
                sampleX = qspaceH5.sample_x
                sampleY = qspaceH5.sample_y

                idx = ((sampleX - x)**2 + (sampleY - y)**2).argmin()

                x = sampleX[idx]
                y = sampleY[idx]

            qspace = qspaceH5.qspace_slice(idx)

            # Set scale and translation
            # Do it before setting data as corresponding
            # nodes in the SFViewParamTree are updated on sigDataChanged
            qxLen, qyLen, qzLen = qspace.shape
            qxMin, qxMax = min(self.__qx), max(self.__qx)
            qyMin, qyMax = min(self.__qy), max(self.__qy)
            qzMin, qzMax = min(self.__qz), max(self.__qz)
            isoView.setScale((qxMax - qxMin) / (qxLen - 1),
                             (qyMax - qyMin) / (qyLen - 1),
                             (qzMax - qzMin) / (qzLen - 1))
            isoView.setTranslation(qxMin, qyMin, qzMin)

            isoView.setData(qspace.swapaxes(0, 2))

            z_sum = qspace.sum(axis=0).sum(axis=0)
            cube_sum_z = qspace.sum(axis=2)
            y_sum = cube_sum_z.sum(axis=0)
            x_sum = cube_sum_z.sum(axis=1)

            colors = cm.jet(np.arange(255))
            cmap = [Qt.QColor.fromRgbF(*c).rgba() for c in colors]

            roiWidget = self.__fitWidget.roiWidget()
            roiWidget.xSlider().setSliderProfile(x_sum, colormap=cmap)
            roiWidget.ySlider().setSliderProfile(y_sum, colormap=cmap)
            roiWidget.zSlider().setSliderProfile(z_sum, colormap=cmap)

    def __nextFitFile(self):
        """
        Temporary method that generated a new file name for the Fit results.
        :return:
        """
        project = self.__projectItem.projectRoot()
        xsocsFile = os.path.basename(project.xsocsFile)
        xsocsPrefix = xsocsFile.rpartition('.')[0]
        template = '{0}_fit_{{0:>04}}.h5'.format(xsocsPrefix)
        output_f = nextFileName(project.workdir, template)
        self.__fitWidget.setOutputFile(output_f)

    def __slotFitProcessStarted(self):
        """
        Slot called when a fit is started. Locks all widgets that reads
        the XsocsProject file. This is necessary because that file is
        is opened in write mode when the fit is done, to write the fit result.
        :return:
        """
        for widget in self.__lockWidgets:
            widget.setEnabled(False)

    def __slotFitProcessDone(self, event):
        """
        Slot called when the fit is done. Event is the name of the FitH5 file
        that has been created.
        Unlocks the widgets locked in __slotFitProcessStarted.
        Emits QSpaceView.sigFitDone.
        :param event:
        :return:
        """
        if event is not None:
            self.sigFitDone.emit(self.__node, event)
            self.__nextFitFile()

        for widget in self.__lockWidgets:
            widget.setEnabled(True)

    def __slotRoiToggled(self, on):
        """
        Slot called when the Roi selection is enabled.
        :param on:
        :return:
        """
        view3d = self.__view3d
        region = view3d.getSelectedRegion()

        if not on:
            # Reset selection region
            view3d.setSelectedRegion()
        else:
            if region is None:  # Init region
                data = view3d.getData(copy=False)

                if data is not None:
                    depth, height, width = data.shape
                    view3d.setSelectedRegion(zrange=(0, depth),
                                             yrange=(0, height),
                                             xrange_=(0, width))
                    region = view3d.getSelectedRegion()

        if on and region:
            ([xLeft, xRight],
             [yLeft, yRight],
             [zLeft, zRight]) = region.getDataRange()
        else:
            ([zLeft, zRight],
             [yLeft, yRight],
             [xLeft, xRight]) = ([None, None], [None, None], [None, None])

        roiWidget = self.__fitWidget.roiWidget()
        roiWidget.xSlider().setSliderValues(xLeft, xRight)
        roiWidget.ySlider().setSliderValues(yLeft, yRight)
        roiWidget.zSlider().setSliderValues(zLeft, zRight)

    def __slotRoiChanged(self, event):
        """
        Slot called each time the ROI is modified
        :param event:
        :return:
        """

        region = self.__view3d.getSelectedRegion()
        if region is None:
            return

        xState = event['x']
        yState = event['y']
        zState = event['z']
        xRoi = xState.leftIndex, xState.rightIndex + 1
        yRoi = yState.leftIndex, yState.rightIndex + 1
        zRoi = zState.leftIndex, zState.rightIndex + 1

        self.__view3d.setSelectedRegion(zrange=zRoi, yrange=yRoi, xrange_=xRoi)

    def __planePlotDockVisibilityChanged(self, visible):
        cutPlane = self.__view3d.getCutPlanes()[0]
        if visible:
            if not self.__connectedToCutPlane:  # Prevent multiple connect
                self.__connectedToCutPlane = True
                cutPlane.sigPlaneChanged.connect(self.__cutPlaneChanged)
                cutPlane.sigDataChanged.connect(self.__cutPlaneChanged)
                self.__cutPlaneChanged()  # To sync
        else:
            if self.__connectedToCutPlane:  # Prevent multiple disconnect
                self.__connectedToCutPlane = False
                cutPlane.sigPlaneChanged.disconnect(self.__cutPlaneChanged)
                cutPlane.sigDataChanged.disconnect(self.__cutPlaneChanged)

    def __cutPlaneChanged(self):
        plane = self.__view3d.getCutPlanes()[0]

        if plane.isVisible() and plane.isValid():
            planeImage = plane.getImageData()
            if planeImage.isValid():
                self.__planePlotWindow.setGraphXLabel(planeImage.getXLabel())
                self.__planePlotWindow.setGraphYLabel(planeImage.getYLabel())
                title = (planeImage.getNormalLabel() +
                         ' = %f' % planeImage.getPosition())
                self.__planePlotWindow.setGraphTitle(title)
                self.__planePlotWindow.addImage(
                    planeImage.getData(copy=False),
                    legend='cutting plane',
                    origin=planeImage.getTranslation(),
                    scale=planeImage.getScale(),
                    resetzoom=True)
