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
from matplotlib import cm


from silx.gui import qt as Qt
from silx.gui.plot import PlotActions, PlotToolButtons, PlotWidget, PlotWindow
from silx.gui.icons import getQIcon
from plot3d.ScalarFieldView import ScalarFieldView
from plot3d.SFViewParamTree import TreeView as SFViewParamTree

from ..model.TreeView import TreeView
from ..widgets.Containers import GroupBox
from ..widgets.RangeSlider import RangeSlider
from ..widgets.Input import StyledLineEdit
from ..project.XsocsH5Factory import h5NodeToProjectItem


class QSpaceTree(TreeView):
    pass


class RoiAxisWidget(Qt.QWidget):
    sigSliderMoved = Qt.Signal(object)

    slider = property(lambda self: self.__slider)

    def __init__(self, label=None, **kwargs):
        super(RoiAxisWidget, self).__init__(**kwargs)

        layout = Qt.QGridLayout(self)
        label = Qt.QLabel(label)
        slider = self.__slider = RangeSlider()
        leftEdit = self.__leftEdit = StyledLineEdit(nChar=5)
        rightEdit = self.__rightEdit = StyledLineEdit(nChar=5)
        leftEdit.setReadOnly(True)
        rightEdit.setReadOnly(True)

        layout.addWidget(label, 0, 0)
        layout.addWidget(slider, 0, 1, 1, 2)
        layout.addWidget(leftEdit, 1, 1)
        layout.addWidget(rightEdit, 1, 2)

        slider.sigSliderMoved.connect(self.__sliderMoved)
        slider.sigSliderMoved.connect(self.sigSliderMoved)

    def __sliderMoved(self, event):
        self.__leftEdit.setText('{0:6g}'.format(event.left))
        self.__rightEdit.setText('{0:6g}'.format(event.right))


class PlotIntensityMap(PlotWindow):
    """Plot intensities as a scatter plot

    :param parent: QWidget's parent
    """

    def __init__(self, parent=None):
        super(PlotIntensityMap, self).__init__(
            parent=parent, backend=None,
            resetzoom=True, autoScale=False,
            logScale=False, grid=True,
            curveStyle=False, colormap=False,
            aspectRatio=True, yInverted=False,
            copy=True, save=True, print_=True,
            control=False, position=False,
            roi=False, mask=False, fit=False)
        self.setMinimumSize(150, 150)

        self.setKeepDataAspectRatio(True)
        self.setActiveCurveHandling(False)
        self.setDataMargins(0.2, 0.2, 0.2, 0.2)

    def setSelectedPosition(self, x, y):
        """Set the selected position.

        :param float x:
        :param float y:
        """
        self.addXMarker(x, legend='Xselection', color='pink')
        self.addYMarker(y, legend='Yselection', color='pink')

    def sizeHint(self):
        return Qt.QSize(200, 200)

    def setPlotData(self, x, y, data):
        """Set the scatter plot.

        This is removing previous scatter plot

        :param numpy.ndarray x: X coordinates of the points
        :param numpy.ndarray y: Y coordinates of the points
        :param numpy.ndarray data: Values associated to points
        """
        min_, max_ = data.min(), data.max()
        colormap = cm.jet
        colors = colormap((data.astype(np.float64) - min_) / (max_ - min_))
        self.addCurve(x, y,
                      legend='intensities',
                      color=colors,
                      symbol='s',
                      linestyle='',
                      resetzoom=False)


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
            with self.__qspaceH5 as qspaceH5:
                progress.setRange(0, qspaceH5.qspace_sum.size - 1)
            layout.addWidget(progress)

            self.__aborted = False
            btnBox = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Abort)
            btnBox.rejected.connect(self.__abortComputeIntensities)
            layout.addWidget(btnBox)

            timer = Qt.QTimer(self)
            timer.timeout.connect(self.__stepComputeIntensities)
            timer.intensities = []
            timer.progressBar = progress
            timer.dialog = dialog
            timer.start()

            if dialog.exec_() == Qt.QDialog.Rejected:
                return  # Aborted, stop here

            intensities = np.array(timer.intensities, copy=True)

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

    def __abortComputeIntensities(self):
        """Abort computation of ROI intensities
        """
        self.__aborted = True

    def __stepComputeIntensities(self):
        """Step in ROI Intensity map computation

        This is intended to be called by a timer in __updateClicked
        """
        if self.__aborted:
            self.sender().stop()  # Stop timer
            self.sender().dialog.reject()  # Abort dialog
            return

        intensities = self.sender().intensities

        with self.__qspaceH5 as qspaceH5:
            if len(intensities) >= qspaceH5.qspace_sum.size:
                self.sender().stop()  # Stop timer
                self.sender().dialog.accept()  # Closes dialog
            else:
                qspace = qspaceH5.qspace_slice(len(intensities))
                zslice, yslice, xslice = self.__roiSlices
                intensities.append(np.sum(qspace[xslice, yslice, zslice]))
                self.sender().progressBar.setValue(len(intensities))


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
    sigProcessApplied = Qt.Signal(object, object)

    plot = property(lambda self: self.__plotWindow)

    def __init__(self,
                 parent,
                 model,
                 node,
                 **kwargs):
        super(QSpaceView, self).__init__(parent)

        item = h5NodeToProjectItem(node)

        # plot window displaying the intensity map
        self.__plotWindow = plotWindow = PlotIntensityMap(parent=self)
        plotWindow.setGraphTitle('Intensity Map')
        plotWindow.setToolTip('Intensity Map integrated on whole QSpaces')
        plotWindow.sigPlotSignal.connect(self.__plotSignal)

        self.__roiPlotWindow = roiPlotWindow = ROIPlotIntensityMap(
            parent=self, qspaceH5=item.qspaceH5)
        roiPlotWindow.sigPlotSignal.connect(self.__plotSignal)

        self.__planePlotWindow = planePlotWindow = CutPlanePlotWindow(self)

        with item.qspaceH5 as qspaceH5:
            sampleX = qspaceH5.sample_x
            sampleY = qspaceH5.sample_y
            self.__setPlotData(sampleX,
                               sampleY,
                               qspaceH5.qspace_sum)
            qx = self.__qx = qspaceH5.qx
            qy = self.__qy = qspaceH5.qy
            qz = self.__qz = qspaceH5.qz

            firstX = sampleX[0]
            firstY = sampleX[1]

        self.__node = node

        # setting up the plot3D and its param tree
        self.__view3d = view3d = ScalarFieldView()
        view3d.addIsosurface(
            lambda data: np.mean(data) + np.std(data),
            '#FF0000FF')
        view3d.setMinimumSize(400, 400)
        view3d.setAxesLabels('qx', 'qy', 'qz')
        self.setCentralWidget(view3d)
        sfTree = SFViewParamTree()
        sfTree.setSfView(view3d)

        # Register ROIPlotIntensity
        view3d.sigSelectedRegionChanged.connect(roiPlotWindow.roiChanged)

        # Store the cut plane signals connection state
        self.__connectedToCutPlane = False
        view3d.getCutPlanes()[0].sigPlaneChanged.connect(self.__cutPlaneChanged)
        view3d.getCutPlanes()[0].sigDataChanged.connect(self.__cutPlaneChanged)

        # the widget containing :
        # - the ROI sliders
        # - the Fit button
        # - the Plot3d param tree
        controlWid = Qt.QWidget()
        controlLayout = Qt.QVBoxLayout(controlWid)

        roiWidget = GroupBox('Roi')
        roiWidget.setCheckable(True)
        roiWidget.setChecked(False)
        roiWidget.toggled.connect(self.__roiGroupToggled)
        layout = Qt.QVBoxLayout(roiWidget)

        xRoiWid = self.__xRoiWid = RoiAxisWidget('X')
        yRoiWid = self.__yRoiWid = RoiAxisWidget('Y')
        zRoiWid = self.__zRoiWid = RoiAxisWidget('Z')
        xRoiWid.slider.setRange([qx[0], qx[-1]])
        yRoiWid.slider.setRange([qy[0], qy[-1]])
        zRoiWid.slider.setRange([qz[0], qz[-1]])
        layout.addWidget(xRoiWid)
        layout.addWidget(yRoiWid)
        layout.addWidget(zRoiWid)

        xRoiWid.sigSliderMoved.connect(self.__roiChanged)
        yRoiWid.sigSliderMoved.connect(self.__roiChanged)
        zRoiWid.sigSliderMoved.connect(self.__roiChanged)

        icon = getQIcon('math-fit')
        fitButton = Qt.QPushButton('Fit')
        fitButton.setIcon(icon)
        fitButton.setToolTip('Start fitting')
        fitButton.clicked.connect(self.__roiApplied)

        controlLayout.addWidget(roiWidget, alignment=Qt.Qt.AlignCenter)
        controlLayout.addWidget(fitButton, alignment=Qt.Qt.AlignCenter)
        controlLayout.addWidget(sfTree)

        sfDock = Qt.QDockWidget()
        sfDock.setWidget(controlWid)
        features = sfDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        sfDock.setFeatures(features)
        view3d.addDockWidget(Qt.Qt.RightDockWidgetArea, sfDock)

        treeDock = Qt.QDockWidget(self)
        tree = QSpaceTree(self, model=model)
        index = node.index()
        tree.setRootIndex(index)
        treeDock.setWidget(tree)
        features = treeDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        treeDock.setFeatures(features)
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, treeDock)

        planePlotDock = Qt.QDockWidget('Cut Plane', self)
        planePlotDock.setWidget(planePlotWindow)
        features = planePlotDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        planePlotDock.setFeatures(features)
        planePlotDock.visibilityChanged.connect(self.__planePlotDockVisibilityChanged)
        self.splitDockWidget(treeDock, planePlotDock, Qt.Qt.Vertical)

        roiPlotDock = Qt.QDockWidget('ROI Intensity', self)
        roiPlotDock.setWidget(roiPlotWindow)
        features = roiPlotDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        roiPlotDock.setFeatures(features)
        self.splitDockWidget(treeDock, roiPlotDock, Qt.Qt.Vertical)
        self.tabifyDockWidget(planePlotDock, roiPlotDock)

        plotDock = Qt.QDockWidget('Intensity', self)
        plotDock.setWidget(plotWindow)
        features = plotDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        plotDock.setFeatures(features)
        self.tabifyDockWidget(roiPlotDock, plotDock)

        self.__showIsoView(firstX, firstY)

    # TODO : refactor this in a common base with RealSpaceViewWidget
    def __setPlotData(self, x, y, data):
        self.__plotWindow.setPlotData(x, y, data)
        self.__plotWindow.resetZoom()
        self.__roiPlotWindow.setPlotData(x, y, data)
        self.__roiPlotWindow.resetZoom()

    def __roiApplied(self):
        region = self.__view3d.getSelectedRegion()
        if region:
            zRoi, yRoi, xRoi = region.getArrayRange()
            roi = [xRoi, yRoi, zRoi]
        else:
            roi = None

        self.sigProcessApplied.emit(self.__node, roi)

    def __plotSignal(self, event):
        if event['event'] not in ('mouseClicked'):
            return
        x, y = event['x'], event['y']

        self.__showIsoView(x, y)

    def __showIsoView(self, x, y):
        isoView = self.__view3d
        item = h5NodeToProjectItem(self.__node)

        with item.qspaceH5 as qspaceH5:
            sampleX = qspaceH5.sample_x
            sampleY = qspaceH5.sample_y

            xIdx = ((sampleX - x)**2 + (sampleY - y)**2).argmin()

            x = sampleX[xIdx]
            y = sampleY[xIdx]

            self.__plotWindow.setSelectedPosition(x, y)
            self.__roiPlotWindow.setSelectedPosition(x, y)

            qspace = qspaceH5.qspace_slice(xIdx)

            # Set scale and translation
            # Do it before setting data as corresponding
            # nodes in the SFViewParamTree are updated on sigDataChanged
            qxLen, qyLen, qzLen = qspace.shape
            qxMin, qxMax = min(self.__qx), max(self.__qx)
            qyMin, qyMax = min(self.__qy), max(self.__qy)
            qzMin, qzMax = min(self.__qz), max(self.__qz)
            isoView.setScale((qxMax - qxMin) / qxLen,
                             (qyMax - qyMin) / qyLen,
                             (qzMax - qzMin) / qzLen)
            isoView.setTranslation(qxMin, qyMin, qzMin)

            isoView.setData(qspace.swapaxes(0, 2))

            z_sum = qspace.sum(axis=0).sum(axis=0)
            cube_sum_z = qspace.sum(axis=2)
            y_sum = cube_sum_z.sum(axis=0)
            x_sum = cube_sum_z.sum(axis=1)

            colors = cm.jet(np.arange(255))
            cmap = [Qt.QColor.fromRgbF(*c).rgba() for c in colors]

            self.__xRoiWid.slider.setSliderProfile(x_sum, colormap=cmap)
            self.__yRoiWid.slider.setSliderProfile(y_sum, colormap=cmap)
            self.__zRoiWid.slider.setSliderProfile(z_sum, colormap=cmap)

    def __roiGroupToggled(self, on):
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
            ([zLeft, zRight],
             [yLeft, yRight],
             [xLeft, xRight]) = region.getArrayRange()
        else:
            ([xLeft, xRight],
             [yLeft, yRight],
             [zLeft, zRight]) = ([None, None], [None, None], [None, None])

        self.__xRoiWid.slider.setSliderValues(xLeft, xRight)
        self.__yRoiWid.slider.setSliderValues(yLeft, yRight)
        self.__zRoiWid.slider.setSliderValues(zLeft, zRight)

    def __roiChanged(self, event):
        sender = self.sender()

        region = self.__view3d.getSelectedRegion()
        if region is None:
            return

        zRoi, yRoi, xRoi = region.getArrayRange()
        if sender == self.__xRoiWid:
            xRoi = event.leftIndex, event.rightIndex + 1
        elif sender == self.__yRoiWid:
            yRoi = event.leftIndex, event.rightIndex + 1
        else:
            zRoi = event.leftIndex, event.rightIndex + 1
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
