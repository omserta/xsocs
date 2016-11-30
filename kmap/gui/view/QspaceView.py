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
from silx.gui.plot import PlotWindow
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
        self.__plotWindow = plotWindow = PlotWindow(aspectRatio=True,
                                                    curveStyle=False,
                                                    mask=False,
                                                    roi=False,
                                                    **kwargs)
        plotWindow.sizeHint = lambda: Qt.QSize(200, 200)
        plotWindow.setMinimumSize(150, 150)

        plotWindow.sigPlotSignal.connect(self.__plotSignal)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)

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
        view3d.setMinimumSize(400, 400)
        view3d.setAxesLabels('qx', 'qy', 'qz')
        self.setCentralWidget(view3d)
        sfTree = SFViewParamTree()
        sfTree.setSfView(view3d)

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

        plotDock = Qt.QDockWidget(self)
        plotDock.setWidget(plotWindow)
        features = plotDock.features() ^ Qt.QDockWidget.DockWidgetClosable
        plotDock.setFeatures(features)
        self.splitDockWidget(treeDock, plotDock, Qt.Qt.Vertical)

        self.__showIsoView(firstX, firstY)

    # TODO : refactor this in a common base with RealSpaceViewWidget
    def __setPlotData(self, x, y, data):
        plot = self.__plotWindow
        # scatter
        min_, max_ = data.min(), data.max()
        colormap = cm.jet
        colors = colormap((data.astype(np.float64) - min_) / (max_ - min_))
        plot.addCurve(x, y,
                      color=colors,
                      symbol='s',
                      linestyle='')

    def __roiApplied(self):
        region = self.__view3d.getSelectedRegion()
        if region:
            zRoi, yRoi, xRoi = region.getArrayRange()
            # xRoi = self.__qx[[xRoi[0], xRoi[1] - 1]]
            # yRoi = self.__qy[[yRoi[0], yRoi[1] - 1]]
            # zRoi = self.__qz[[zRoi[0], zRoi[1] - 1]]
            roi = [xRoi, yRoi, zRoi]
        else:
            roi = None

        self.sigProcessApplied.emit(self.__node, roi)

    def __plotSignal(self, event):
        if event['event'] not in ('curveClicked',): # , 'mouseClicked'):
            return
        x, y = event['xdata'], event['ydata']

        self.__showIsoView(x, y)

    def __showIsoView(self, x, y):
        isoView = self.__view3d
        plot = self.__plotWindow
        item = h5NodeToProjectItem(self.__node)

        with item.qspaceH5 as qspaceH5:
            sampleX = qspaceH5.sample_x
            sampleY = qspaceH5.sample_y

            # TODO : better
            try:
                xIdx = (np.abs(sampleX - x) + np.abs(sampleY - y)).argmin()
            except:
                xIdx = (np.abs(sampleX - x[0]) + np.abs(sampleY - y[0])).argmin()

            x = sampleX[xIdx]
            y = sampleY[xIdx]

            plot.addXMarker(x, legend='Xselection')
            plot.addYMarker(y, legend='Yselection')

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
            isoView.setOffset(qxMin, qyMin, qzMin)

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