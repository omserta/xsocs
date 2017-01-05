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


import weakref
from collections import namedtuple, OrderedDict

import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


from silx.gui import qt as Qt
from silx.io.utils import savetxt
from silx.gui.icons import getQIcon
from silx.gui.plot import PlotWindow
from silx.math.histogram import Histogramnd

from ..widgets.Containers import GroupBox
from ..widgets.RangeSlider import RangeSlider
from ..widgets.PointWidget import PointWidget
from ...gui.icons import getQIcon as getKmapIcon
from ..widgets.Input import StyledLineEdit, StyledLabel


_defaultNColors = 256

XsocsPlot2DColormap = namedtuple('XsocsPlot2DColormap',
                                 ['colormap', 'minVal', 'maxVal', 'nColors'])


def _arrayToIndexedPixmap(vector, cmap, nColors=256):
    """

    :param vector:
    :param cmap:
    :param nColors:
    :return:
    """
    assert vector.ndim <= 2

    # colors = cmap(np.arange(nColors))  # .to_rgba(np.arange(255))

    sm = cm.ScalarMappable(cmap=cmap)
    colors = sm.to_rgba(np.arange(nColors))

    cmap = [Qt.QColor.fromRgbF(*c).rgba() for c in colors]

    vMin = vector.min()
    vMax = vector.max()
    image = np.round((nColors - 1.) * (vector - vMin) / (vMax - vMin))

    image = image.astype(np.int8)

    if image.ndim == 1:
        height = 1
        width = image.shape[0]
    else:
        height = image.shape[0]
        width = image.shape[1]

    qImage = Qt.QImage(image.data,
                       width,
                       height,
                       Qt.QImage.Format_Indexed8)

    qImage.setColorTable(cmap)
    return Qt.QPixmap.fromImage(qImage)


def _applyColormap(colormap,
                   values,
                   minVal=None,
                   maxVal=None,
                   clip=True):
    """

    :param colormap: An Xsocs2DColormap instance
    :param values: Values to convert to color values.
    :param minVal: clips the values to minVal. If None, it will take the value
        found in colormap if that one is not None
    :param maxVal: clips the values to maxVal. If None, it will take the value
        found in colormap if that one is not None
    :param clip: clips the data to minVal and maxVal
    :return:
    """
    if minVal is None:
        minVal = colormap.minVal

    if maxVal is None:
        maxVal = colormap.maxVal

    sm = cm.ScalarMappable(norm=Normalize(vmin=minVal, vmax=maxVal),
                           cmap=colormap.colormap)
    colors = sm.to_rgba(values)

    return colors


class XsocsPlot2DColorDialog(Qt.QDialog):
    """
    Color dialog for the XsocsPlot2D.
    Right now only supports one scatter plot!
    """
    colormaps = OrderedDict([('jet', cm.jet),
                             ('afmhot', cm.afmhot)])

    def __init__(self, plot, curve, **kwargs):

        super(XsocsPlot2DColorDialog, self).__init__(plot, **kwargs)

        colormap = plot.getPlotColormap(curve)

        self.__plot = weakref.ref(plot)
        self.__curve = curve
        self.__histogram = histo = plot.getHistogram(curve, colormap.nColors)

        if colormap is None:
            minVal = histo.edges[0][0]
            maxVal = histo.edges[0][-1]
            cmap = cm.jet
            nColors = _defaultNColors
        else:
            minVal = colormap.minVal
            maxVal = colormap.maxVal
            cmap = colormap.colormap
            if cmap not in self.colormaps:
                self.colormaps[cmap.name] = cmap
            nColors = colormap.nColors
            if minVal is None:
                minVal = histo.edges[0][0]
            if maxVal is None:
                maxVal = histo.edges[0][-1]

        index = self.colormaps.keys().index(cmap.name)

        self.__colormap = XsocsPlot2DColormap(colormap=cmap,
                                              minVal=minVal,
                                              maxVal=maxVal,
                                              nColors=nColors)

        layout = Qt.QGridLayout(self)

        grpBox = GroupBox('Colormap')
        grpBoxLayout = Qt.QGridLayout(grpBox)
        self.__cmapCBox = cmapCBox = Qt.QComboBox()
        for key, value in self.colormaps.items():
            cmapCBox.addItem(key, userData=value)
        cmapCBox.setCurrentIndex(index)
        grpBoxLayout.addWidget(cmapCBox, 0, 0, Qt.Qt.AlignCenter)

        cmapCBox.currentIndexChanged.connect(self.__cmapCBoxChanged)

        self.__colorLabel = colorLabel = Qt.QLabel()
        colorLabel.setFrameStyle(Qt.QFrame.Panel | Qt.QFrame.Sunken)
        colorLabel.setLineWidth(2)
        colorLabel.setMidLineWidth(2)
        grpBoxLayout.addWidget(colorLabel, 1, 0)
        grpBox.setSizePolicy(Qt.QSizePolicy.Fixed, Qt.QSizePolicy.Fixed)
        layout.addWidget(grpBox, 0, 0)

        grpBox = GroupBox('Range')
        grpBoxLayout = Qt.QGridLayout(grpBox)
        self.__rngSlider = rngSlider = RangeSlider()
        grpBoxLayout.addWidget(rngSlider, 0, 0, 1, 2)

        self.__filledProfile = filledProfile = ColorFilledProfile()
        filledProfile.setFixedHeight(100)
        grpBoxLayout.addWidget(filledProfile, 1, 0, 1, 2)

        self.__minEdit = minEdit = StyledLineEdit(nChar=6)
        self.__maxEdit = maxEdit = StyledLineEdit(nChar=6)
        minEdit.setValidator(Qt.QDoubleValidator())
        maxEdit.setValidator(Qt.QDoubleValidator())
        minEdit.editingFinished.connect(self.__lineEditFinished)
        maxEdit.editingFinished.connect(self.__lineEditFinished)
        grpBoxLayout.addWidget(minEdit, 2, 0)
        grpBoxLayout.addWidget(maxEdit, 2, 1)
        grpBox.setSizePolicy(Qt.QSizePolicy.Fixed, Qt.QSizePolicy.Fixed)
        layout.addWidget(grpBox, 1, 0, Qt.Qt.AlignCenter)

        bnBox = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Close)
        bnBox.button(Qt.QDialogButtonBox.Close).clicked.connect(self.accept)

        layout.addWidget(bnBox, 2, 0)

        self.__setupWidgets()

        rngSlider.sigSliderMoved.connect(self.__rngSliderMoved)

    def __cmapCBoxChanged(self):
        """
        Slot for the currentIndexChanged signal (colormap combobox).
        :return:
        """
        self.__setColormapPixmap()
        self.__applyColormap()

    def __rngSliderMoved(self, event):
        """
        Slot for the sliderMoved signal.
        :param event:
        :return:
        """
        blockedMin = self.__minEdit.blockSignals(True)
        blockedMax = self.__maxEdit.blockSignals(True)
        self.__minEdit.setText('{0:6g}'.format(event.left))
        self.__maxEdit.setText('{0:6g}'.format(event.right))
        self.__minEdit.blockSignals(blockedMin)
        self.__maxEdit.blockSignals(blockedMax)
        self.__applyColormap()

    def __lineEditFinished(self):
        """
        Slot for the editingFinished signal.
        :return:
        """
        minVal = float(self.__minEdit.text())
        maxVal = float(self.__maxEdit.text())
        blocked = self.__rngSlider.blockSignals(True)
        self.__rngSlider.setSliderValues(minVal, maxVal)
        self.__rngSlider.blockSignals(blocked)
        self.__applyColormap()

    def __applyColormap(self):
        """
        Applies the colormap to the plot.
        :return:
        """
        plot = self.__plot()
        if plot is None:
            return
        minVal = float(self.__minEdit.text())
        maxVal = float(self.__maxEdit.text())
        plotColormap = self.__colormap
        curve = self.getCurve()

        colormap = self.__cmapCBox.itemData(self.__cmapCBox.currentIndex())

        plotColormap = XsocsPlot2DColormap(colormap=colormap,
                                       minVal=minVal,
                                       maxVal=maxVal,
                                       nColors=plotColormap.nColors)
        self.__colormap = plotColormap
        plot.setPlotColormap(curve, plotColormap)
        self.__drawHistogram()

    def getCurve(self):
        """
        Returns the curves managed by this widget.
        :return:
        """
        return self.__curve

    def __setupWidgets(self):
        """
        Sets up the dialog.
        :return:
        """
        self.__setColormapPixmap()

        plot = self.__plot()
        if plot is None:
            return

        curve = self.getCurve()
        if curve is None:
            return

        histo = self.__histogram
        rngSlider = self.__rngSlider

        colormap = self.__colormap

        self.__minEdit.setText('{0:6g}'.format(colormap.minVal))
        self.__maxEdit.setText('{0:6g}'.format(colormap.maxVal))
        rngSlider.setRange([histo.edges[0][0], histo.edges[0][-1]])
        rngSlider.setSliderValues(colormap.minVal, colormap.maxVal)

        self.__drawHistogram()

    def __drawHistogram(self):
        histo = self.__histogram
        self.__filledProfile.setProfile(histo.edges[0][0:-1],
                                        histo.histo,
                                        self.__colormap)

    def getColormap(self):
        """
        Returns the currently selected colormap.
        :return:
        """
        return self.__colormap

    def __setColormapPixmap(self):
        """
        Sets the colormap preview label.
        :return:
        """
        print 'set in'
        style = Qt.QApplication.style()
        size = style.pixelMetric(Qt.QStyle.PM_SmallIconSize)

        colorLabel = self.__colorLabel
        plotColormap = self.__colormap
        colormap = self.__cmapCBox.itemData(self.__cmapCBox.currentIndex())

        image = np.tile(np.arange(plotColormap.nColors,
                                  dtype=np.uint8),
                        (size, 1))
        print colormap.name
        pixmap = _arrayToIndexedPixmap(image,
                                       colormap,
                                       nColors=plotColormap.nColors)
        print pixmap.isNull()
        colorLabel.setPixmap(pixmap)
        print 'set out'


XsocsPlot2DPoint = namedtuple('XsocsPlot2DPoint', ['x', 'y', 'xIdx', 'yIdx'])


class DoublePointDock(Qt.QDockWidget):
    """
    Widget for displaying selected point and mouse coordinate
    """

    mousePoint = property(lambda self: self.__mousePoint)
    selectedPoint = property(lambda self: self.__selectedPoint)

    def __init__(self, *args, **kwargs):
        super(DoublePointDock, self).__init__(*args, **kwargs)

        widget = Qt.QWidget()
        layout = Qt.QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.__mouseVisible = True
        self.__selectedVisible = True

        self.__mousePoint = mousePoint = PointWidget()
        self.__mouseLabel = mouseLabel = Qt.QLabel('Mouse')
        mousePoint.setFrameStyle(Qt.QFrame.Box)
        self.__selectedPoint = selectedPoint = PointWidget()
        self.__selectedLabel = selectedLabel = Qt.QLabel('Selected')
        selectedPoint.setFrameStyle(Qt.QFrame.Box)

        layout.addWidget(mouseLabel, 0, 0, Qt.Qt.AlignLeft)
        layout.addWidget(mousePoint, 0, 1, Qt.Qt.AlignLeft)
        layout.addWidget(selectedLabel, 1, 0, Qt.Qt.AlignLeft)
        layout.addWidget(selectedPoint, 1, 1, Qt.Qt.AlignLeft)

        layout.setColumnStretch(2, 1)

        self.setWidget(widget)

    def setShowSelectedPoint(self, show):
        """
        Shows/hides the PointWidget displaying the currently selected point.
        :param show:
        :return:
        """
        if show != self.__selectedVisible:
            layout = self.widget().layout()
            if show:
                layout.addWidget(self.__selectedLabel, 1, 0, Qt.Qt.AlignLeft)
                layout.addWidget(self.__selectedPoint, 1, 1, Qt.Qt.AlignLeft)
            else:
                layout.takeAt(layout.indexOf(self.__selectedPoint))
                self.__selectedPoint.setParent(None)
                layout.takeAt(layout.indexOf(self.__selectedLabel))
                self.__selectedLabel.setParent(None)
        self.__selectedVisible = show
        self.widget().updateGeometry()
        self.widget().adjustSize()

    def setShowMousePoint(self, show):
        """
        Shows/Hides the PointWidget displaying the mouse position.
        :param show:
        :return:
        """
        if show != self.__mouseVisible:
            layout = self.widget().layout()
            if show:
                layout.addWidget(self.__mouseLabel, 0, 0)
                layout.addWidget(self.__mousePoint, 0, 1)
            else:
                layout.takeAt(layout.indexOf(self.__mousePoint))
                self.__mousePoint.setParent(None)
                layout.takeAt(layout.indexOf(self.__mouseLabel))
                self.__mouseLabel.setParent(None)
        self.__mouseVisible = show
        self.widget().updateGeometry()
        self.widget().adjustSize()


class XsocsPlot2D(PlotWindow):
    """
    Base class for the 2D scatter plot.
    The colormap widget only supports one plot at the moment.
    """

    sigPointSelected = Qt.Signal(object)

    def __init__(self, **kwargs):
        super(XsocsPlot2D, self).__init__(**kwargs)

        self.setActiveCurveHandling(False)
        self.setKeepDataAspectRatio(True)

        self.__sigPlotConnected = False
        self.__pointSelectionEnabled = False
        self.__showSelectedCoordinates = False
        self.__showMousePosition = False

        self.__logScatter = False
        self.__colormap = cm.jet
        self.__values = {}
        self.__cmapDialog = None

        pointDock = self.__pointWidget = DoublePointDock()

        features = Qt.QDockWidget.DockWidgetVerticalTitleBar | Qt.QDockWidget.DockWidgetClosable
        pointDock.setFeatures(features)
        pointDock.sizeHint = lambda: Qt.QSize()
        self.addDockWidget(Qt.Qt.BottomDockWidgetArea, pointDock)
        pointDockAction = pointDock.toggleViewAction()
        pointDockAction.setIcon(getQIcon('crosshair'))
        pointDockAction.setIconVisibleInMenu(True)
        pointDock.setShowMousePoint(self.__showMousePosition)
        pointDock.setShowSelectedPoint(self.__showSelectedCoordinates)

        pointDockBn = Qt.QToolButton()
        pointDockBn.setDefaultAction(pointDockAction)
        closeButton = Qt.QToolButton()
        style = Qt.QApplication.style()
        icon = style.standardIcon(Qt.QStyle.SP_TitleBarCloseButton)
        closeButton.setIcon(icon)
        closeButton.setFixedSize(closeButton.iconSize())
        closeButton.clicked.connect(pointDockAction.trigger)
        pointDock.setTitleBarWidget(closeButton)

        toolbars = self.findChildren(Qt.QToolBar)
        for toolbar in toolbars:
            toolbar.hide()

        centralWid = self.centralWidget()

        centralWid.installEventFilter(self)

        self.__optionsBase = optionsBase = Qt.QWidget(centralWid)
        optionsLayout = Qt.QHBoxLayout(optionsBase)
        optionsLayout.setContentsMargins(0, 0, 0, 0)
        optionsLayout.setSpacing(0)

        style = Qt.QApplication.style()
        size = style.pixelMetric(Qt.QStyle.PM_SmallIconSize)
        icon = style.standardIcon(Qt.QStyle.SP_ArrowRight)
        showBarBn = Qt.QToolButton()
        showBarBn.setIcon(icon)
        showBarBn.setFixedWidth(size)
        showBarBn.clicked.connect(self.__showBarBnClicked)
        showBarBn.setAutoFillBackground(True)
        self.__hidden = True
        self.__firstShow = True

        self.__optionsBaseA = optionsBaseA = Qt.QWidget()
        optionsLayoutA = Qt.QHBoxLayout(optionsBaseA)
        optionsLayoutA.setContentsMargins(0, 0, 0, 0)
        optionsLayoutA.setSpacing(0)
        rstZoomAction = self.resetZoomAction
        rstToolBn = Qt.QToolButton()
        rstToolBn.setDefaultAction(rstZoomAction)

        optionsLayoutA.addWidget(rstToolBn)
        optionsLayoutA.addWidget(showBarBn)

        self.__optionsBaseB = optionsBaseB = Qt.QWidget()
        optionsLayoutB = Qt.QHBoxLayout(optionsBaseB)
        optionsLayoutB.setContentsMargins(0, 0, 0, 0)
        optionsLayoutB.setSpacing(0)

        # coordinates dock action
        pointDockBn = Qt.QToolButton()
        pointDockBn.setDefaultAction(pointDockAction)
        optionsLayoutB.addWidget(pointDockBn)

        # colormap dialog action
        self.__colormapBn = colormapBn = Qt.QToolButton()
        colormapBn.setDisabled(True)
        colormapBn.setIcon(getQIcon('colormap'))
        colormapBn.clicked.connect(self.__colormapBnClicked)
        optionsLayoutB.addWidget(colormapBn)

        icon = getKmapIcon('gears')
        options = Qt.QToolButton()

        options.setToolTip('Options')
        options.setIcon(icon)
        options.setPopupMode(Qt.QToolButton.InstantPopup)

        menu = Qt.QMenu()

        # save scatter as 3D
        self.__save2DAction = action = Qt.QAction(
            getKmapIcon('save_2dscatter'), 'Save "2D" scatter', self)
        action.setIconVisibleInMenu(True)
        action.triggered.connect(self.__save2DTriggered)
        action.setEnabled(False)
        menu.addAction(action)

        # save to file action
        self.saveAction.setIconVisibleInMenu(True)
        menu.addAction(self.saveAction)

        # print action
        self.printAction.setIconVisibleInMenu(True)
        menu.addAction(self.printAction)

        # screenshot action
        self.copyAction.setIconVisibleInMenu(True)
        menu.addAction(self.copyAction)

        # grid action
        self.gridAction.setIconVisibleInMenu(True)
        menu.addAction(self.gridAction)

        # crosshair action
        self.crosshairAction.setIconVisibleInMenu(True)
        menu.addAction(self.crosshairAction)

        # pan action
        self.panWithArrowKeysAction.setIconVisibleInMenu(True)
        menu.addAction(self.panWithArrowKeysAction)

        # # x log scale
        # self.xAxisLogarithmicAction.setIconVisibleInMenu(True)
        # menu.addAction(self.xAxisLogarithmicAction)
        #
        # # y log scale
        # self.yAxisLogarithmicAction.setIconVisibleInMenu(True)
        # menu.addAction(self.yAxisLogarithmicAction)

        # x autoscale action
        self.xAxisAutoScaleAction.setIconVisibleInMenu(True)
        menu.addAction(self.xAxisAutoScaleAction)

        # y autoscale action
        self.yAxisAutoScaleAction.setIconVisibleInMenu(True)
        menu.addAction(self.yAxisAutoScaleAction)

        # curvestyle action
        self.curveStyleAction.setIconVisibleInMenu(True)
        menu.addAction(self.curveStyleAction)

        # aspect ratio action
        aspectMenu = self.keepDataAspectRatioButton.menu()
        if aspectMenu is not None:
            action = aspectMenu.menuAction()
            action.setIconVisibleInMenu(True)
            menu.addAction(action)
        else:
            self.keepDataAspectRatioAction.setIconVisibleInMenu(True)
            menu.addAction(self.keepDataAspectRatioAction)

        options.setMenu(menu)

        optionsLayoutB.addWidget(options)

        optionsLayout.addWidget(optionsBaseB)
        optionsLayout.addWidget(optionsBaseA)

    def __drawSelectedPosition(self, x, y):
        """Set the selected position.

        :param float x:
        :param float y:
        """
        self.addXMarker(x, legend='Xselection', color='pink')
        self.addYMarker(y, legend='Yselection', color='pink')

    def __displaySelectedCoordinates(self, x, y):
        if self.__showSelectedCoordinates:
            self.__pointWidget.selectedPoint.setPoint(x, y)

    def __displayMousePosition(self, x, y):
        self.__pointWidget.mousePoint.setPoint(x, y)

    def __onPlotSignal(self, event):
        if self.__pointSelectionEnabled or self.__showSelectedCoordinates:
            if event['event'] == 'mouseClicked':
                x, y = event['x'], event['y']
                self.selectPoint(x, y)

                curves = self.getAllCurves(just_legend=True)
                if not curves:
                    xIdx = None
                    yIdx = None
                else:
                    curveX, curveY = self.getCurve(curves[0])[0:2]
                    xIdx = ((curveX - x) ** 2 + (curveY - y) ** 2).argmin()
                    yIdx = xIdx
                    x = curveX[xIdx]
                    y = curveY[xIdx]

                point = XsocsPlot2DPoint(x=x, y=y, xIdx=xIdx, yIdx=yIdx)

                self.selectPoint(x, y)

                self.sigPointSelected.emit(point)

        if self.__showMousePosition:
            if event['event'] == 'mouseMoved':
                self.__displayMousePosition(event['x'], event['y'])

    def selectPoint(self, x, y):
        """
        Called when PointSelectionEnabled is True and the mouse is clicked.
        The default implementation just draws a crosshair at the position
        of the closest point of the 2D scatter plot, if any, or at the
        position x,y if an image is plotted. sigPointSelected is not emitted.
        :param x:
        :param y:
        :return:
        """
        self.__displaySelectedCoordinates(x, y)
        self.__drawSelectedPosition(x, y)

    def setShowSelectedCoordinates(self, show):
        """
        Controls the visibility of the selected position widget.
        :param show:
        :return:
        """
        self.__pointWidget.setShowSelectedPoint(show)
        self.__connectPlotSignal(showSelectedCoordinates=show)

    def getShowSelectedCoordinates(self):
        """
        Returns True if the selected position is shown.
        :return:
        """
        return self.__showSelectedCoordinates

    def setShowMousePosition(self, show):
        """
        Controls the visibility of the mouse position widget.
        :param show:
        :return:
        """
        self.__pointWidget.setShowMousePoint(show)
        self.__connectPlotSignal(showMousePosition=show)

    def getShowMousePosition(self):
        """
        Returns True if the mouse position is shown.
        :return:
        """
        return self.__showMousePosition

    def setPointSelectionEnabled(self, enabled):
        """
        Controls the point selection behaviour.
        :param enabled: True to allow point selection.
        :return:
        """
        self.__connectPlotSignal(pointSelection=enabled)

    def isPointSelectionEnabled(self):
        """
        Returns True if plot points selection is enabled.
        :return:
        """
        return self.__pointSelectionEnabled

    def __connectPlotSignal(self,
                            pointSelection=None,
                            showSelectedCoordinates=None,
                            showMousePosition=None):
        """
        Connects/disconnects the plot signal as needed.
        :param pointSelection:
        :param showSelectedCoordinates:
        :param showMousePosition:
        :return:
        """
        currentState = self.__sigPlotConnected

        if pointSelection is not None:
            self.__pointSelectionEnabled = pointSelection

        if showSelectedCoordinates is not None:
            self.__showSelectedCoordinates = showSelectedCoordinates

        if showMousePosition is not None:
            self.__showMousePosition = showMousePosition

        newState = (self.__pointSelectionEnabled |
                    self.__showSelectedCoordinates |
                    self.__showMousePosition)

        if currentState != newState:
            if newState:
                self.sigPlotSignal.connect(self.__onPlotSignal)
            else:
                self.sigPlotSignal.disconnect(self.__onPlotSignal)
            self.__sigPlotConnected = newState

    def __save2DTriggered(self):
        # TODO : support more that one curve
        if not self.__values:
            return
        if len(self.__values)> 1:
            raise ValueError('Export : only one 2D scatter plot '
                             'supported at the moment.')
        legend = self.__values.keys()[0]
        values = self.__values[legend][0]
        curve = self.getCurve(legend)
        x, y = curve[0:2]

        xlabel = curve[4]['xlabel']
        if xlabel is None:
            xlabel = self.getGraphXLabel()
        ylabel = curve[4]['ylabel']
        if ylabel is None:
            ylabel = self.getGraphYLabel()

        dialog = Qt.QFileDialog(self)
        dialog.setWindowTitle("Output File Selection")
        dialog.setModal(True)
        dialog.setNameFilters(['Curve as Raw ASCII (*.txt)'])

        dialog.setFileMode(dialog.AnyFile)
        dialog.setAcceptMode(dialog.AcceptSave)

        if not dialog.exec_():
            return False

        # nameFilter = dialog.selectedNameFilter()
        filename = dialog.selectedFiles()[0]
        dialog.close()

        stack = np.vstack((x, y, values)).transpose()

        delimiter = ";"
        header = '{col0} {delimiter} {col1} {delimiter} {col2}'\
                 '\n'.format(col0=xlabel,
                             col1=ylabel,
                             col2='values',
                             delimiter=delimiter)
        savetxt(filename, stack, header=header, delimiter=delimiter)

    def showEvent(self, event):
        super(XsocsPlot2D, self).showEvent(event)
        if self.__firstShow:
            self.__moveOptionBar()
            self.__firstShow = False

    def getHistogram(self, curve, nBins):
        values = self.getPlotValues(curve)
        if values is None:
            return None

        vMin = values.min()
        vMax = values.max()

        # we have to convert to a format supported by Histogramnd
        # TODO : issue a warning or something
        histo = Histogramnd(values.astype(np.float32),
                            [vMin, vMax],
                            nBins,
                            last_bin_closed=True)
        return histo

    def eventFilter(self, obj, event):
        if event.type() == Qt.QEvent.Resize:
            self.__moveOptionBar()
        return super(XsocsPlot2D, self).eventFilter(obj, event)

    def __moveOptionBar(self):
        optionsBase = self.__optionsBase
        optionsBaseB = self.__optionsBaseB
        geom = self.centralWidget().geometry()

        if self.__hidden:
            newPos = Qt.QPoint(0 - optionsBaseB.width(),
                               geom.height() - optionsBase.height())
        else:
            newPos = Qt.QPoint(0,
                               geom.height() - optionsBase.height())

        if optionsBase.pos() != newPos:
            optionsBase.move(newPos)

    def __showBarBnClicked(self):
        style = Qt.QApplication.style()

        optionsBase = self.__optionsBase
        optionsBaseB = self.__optionsBaseB
        geom = self.centralWidget().geometry()

        hiddenGeom = Qt.QPoint(0 - optionsBaseB.width(),
                               geom.height() - optionsBaseB.height())
        visibleGeom = Qt.QPoint(0,
                                geom.height() - optionsBaseB.height())

        if self.__hidden:
            icon = Qt.QStyle.SP_ArrowLeft
            startVal = hiddenGeom
            endVal = visibleGeom
        else:
            icon = Qt.QStyle.SP_ArrowRight
            startVal = visibleGeom
            endVal = hiddenGeom

        icon = style.standardIcon(icon)
        self.sender().setIcon(icon)

        self.__hidden = not self.__hidden

        animation = Qt.QPropertyAnimation(optionsBase, 'pos', self)
        animation.setDuration(300)
        animation.setStartValue(startVal)
        animation.setEndValue(endVal)
        animation.start(Qt.QAbstractAnimation.DeleteWhenStopped)

    def __colormapBnClicked(self):
        dialog = self.__cmapDialog
        if dialog is None:
            self.__cmapDialog = dialog =\
                XsocsPlot2DColorDialog(self, self.__values.keys()[0])
            self.setAttribute(Qt.Qt.WA_DeleteOnClose)
            self.__cmapDialog.accepted.connect(self.__cmapDialogClosed)
            dialog.show()
        else:
            dialog.raise_()

    def __cmapDialogClosed(self):
        self.__cmapDialog = None

    def removeCurve(self, *args, **kwargs):
        super(XsocsPlot2D, self).removeCurve(*args, **kwargs)
        curves = set(self.getAllCurves())
        thisCurves = set(self.__values.keys())
        diff = thisCurves - curves

        for curve in diff:
            del self.__values[curve]

        if len(self.__values) > 0:
            self.__colormapBn.setDisabled(True)
            self.__save2DAction.setDisabled(True)

    def getPlotValues(self, curve, copy=True):
        values = self.__values.get(curve)
        if values is None:
            return None
        values = values[0]
        if copy:
            values = values[:]
        return values

    def getPlotColormap(self, curve):
        values = self.__values.get(curve)
        if values is None:
            return
        return values[1]

    def setPlotColormap(self, curve, colormap):
        values = self.__values.get(curve)
        if values is None:
            return
        zValues = values[0]

        curveData = self.getCurve(curve)
        if curveData is None:
            return

        x, y = curveData[0:2]

        colors = _applyColormap(colormap, zValues)

        self.__plotData(x, y,
                        legend=curve,
                        color=colors,
                        resetzoom=False)

        values[1] = colormap
        self.__values[curve] = values

    def __plotData(self, x, y, **kwargs):
        legend = self.addCurve(x,
                               y,
                               **kwargs)
        return legend

    def setPlotData(self, x, y,
                    values=None,
                    symbol='s',
                    linestyle='',
                    resetzoom=True,
                    colormap=None,
                    **kwargs):

        colors = None

        # TODO
        # if values is not None and self.__values:
        #     raise ValueError('XsocsPlot2D only supports one 2D scatter plot.')

        finite = np.isfinite(values)

        x = x[finite]
        y = y[finite]
        values = values[finite]

        if colormap is None:
            colormap = XsocsPlot2DColormap(colormap=cm.jet,
                                           minVal=None,
                                           maxVal=None,
                                           nColors=_defaultNColors)

        if values is not None:
            colors = _applyColormap(colormap, values)

        if 'color' in kwargs:
            del kwargs['color']
            print('In XsocsPlot2D.setPlotData : keyword color is ignored.')

        legend = self.__plotData(x,
                                 y,
                                 color=colors,
                                 symbol=symbol,
                                 linestyle=linestyle,
                                 resetzoom=resetzoom,
                                 **kwargs)

        if values is not None:
            self.__values[legend] = [values, colormap]

        if len(self.__values) > 0:
            self.__colormapBn.setDisabled(False)
            self.__save2DAction.setDisabled(False)

        return legend


class ColorFilledProfile(Qt.QFrame):
    _pimapHeight = 100
    _histoBorder = 0

    def __init__(self, parent=None):
        super(ColorFilledProfile, self).__init__(parent)

        self.setFrameStyle(Qt.QFrame.Panel | Qt.QFrame.Sunken)

        self.__pixmap = None
        self.__lineMin = None
        self.__lineMax = None

    def setProfile(self, x, y, colormap):
        """

        :param profile: a 1D numpy array
        :param colormap: an XsocsPlot2DColormap instance
        :param nColors: number of colors
        :return:
        """
        assert x.ndim == 1
        assert y.ndim == 1

        self.__colormap = colormap
        self.__pixmap = pixmap = Qt.QPixmap(Qt.QSize(x.size,
                                                     self._pimapHeight))
        pixmap.fill()

        xMin = x.min()
        xMax = x.max()

        colors = _applyColormap(colormap, x)
        profileValues = (y * (1.0 * self._pimapHeight / y.max()))
        points = [Qt.QPointF(0, 0)]
        points.extend([Qt.QPointF(idx, val)
                       for idx, val in enumerate(profileValues)])
        points.extend([Qt.QPointF(colormap.nColors - 1, 0)])
        poly = Qt.QPolygonF(points)

        if colormap.minVal is not None:
            lineMin = ((colormap.minVal - xMin) * (pixmap.width() - 1) /
                       (xMax - xMin))
        else:
            lineMin = None

        if colormap.maxVal is not None:
            lineMax = ((colormap.maxVal - xMin) * (pixmap.width() - 1) /
                       (xMax - xMin))
        else:
            lineMax = None

        self.__lineMin = lineMin
        self.__lineMax = lineMax

        gradient = Qt.QLinearGradient(Qt.QPoint(0, 0),
                                      Qt.QPoint(colormap.nColors - 1, 0))
        for idx, color in enumerate(colors):
            qColor = Qt.QColor.fromRgbF(*color)
            gradient.setColorAt(idx / (1.0 * (colormap.nColors - 1)), qColor)

        painter = Qt.QPainter(pixmap)
        painter.save()
        painter.scale(1, -1.)
        painter.translate(Qt.QPointF(0., -1.0 * self._pimapHeight))
        brush = Qt.QBrush(gradient)
        painter.setBrush(brush)
        painter.setPen(Qt.QPen(Qt.Qt.NoPen))
        painter.drawPolygon(poly)
        painter.restore()
        painter.end()
        self.update()

    def paintEvent(self, event):

        border = self._histoBorder
        pixmap = self.__pixmap

        painter = Qt.QPainter(self)

        rect = self.frameRect().adjusted(border, border, -border, -border)

        if not pixmap:
            painter.save()
            painter.setPen(Qt.QColor(Qt.Qt.gray))
            painter.setBrush(Qt.QColor(Qt.Qt.white))
            painter.drawRect(rect)
            painter.restore()
        else:
            painter.drawPixmap(rect, pixmap, pixmap.rect())

            painter.save()
            painter.setBrush(Qt.QColor(Qt.Qt.black))
            lineMin = self.__lineMin
            if lineMin:
                xLine = lineMin * (rect.width() - 1.) / (pixmap.width() - 1)
                painter.drawRect(xLine - 2, 0, 2, rect.height())
            lineMax = self.__lineMax
            if lineMax:
                xLine = lineMax * (rect.width() - 1.) / (pixmap.width() - 1)
                painter.drawRect(xLine + 2, 0, 2, rect.height())
            painter.restore()
