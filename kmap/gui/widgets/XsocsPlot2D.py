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
from collections import namedtuple

import numpy as np
from matplotlib import cm


from silx.gui import qt as Qt
from silx.gui.plot import PlotWindow
from silx.gui.icons import getQIcon
from silx.math.histogram import Histogramnd

from ...gui.icons import getQIcon as getKmapIcon
from ..widgets.RangeSlider import RangeSlider
from ..widgets.Input import StyledLineEdit
from ..widgets.Containers import GroupBox


def _arrayToIndexedPixmap(vector, cmap, nColors=256, range=None):
    """

    :param vector:
    :param cmap:
    :param nColors:
    :param range:
    :return:
    """
    assert vector.ndim <= 2

    colors = cmap(np.arange(nColors))  # .to_rgba(np.arange(255))
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


def _applyColormap(colormap, values, dtype=np.float32):
    """

    :param colormap: An Xsocs2DColormap instance
    :param values: Values to convert to color values
    :return:
    """
    minVal = ((colormap.minVal is not None and colormap.minVal)
              or values.min())
    maxVal = ((colormap.maxVal is not None and colormap.maxVal)
              or values.max())
    colors = colormap.colormap(
        (values.astype(dtype) - minVal) / (maxVal - minVal))
    return colors


class XsocsPlot2DColorDialog(Qt.QDialog):
    """
    Color dialog for the XsocsPlot2D.
    Right now only supports one scatter plot!
    """
    nColors = 256

    def __init__(self, plot, **kwargs):

        super(XsocsPlot2DColorDialog, self).__init__(plot, **kwargs)

        self.__plot = weakref.ref(plot)

        self.__histogram = None

        layout = Qt.QGridLayout(self)

        grpBox = GroupBox('Colormap')
        grpBoxLayout = Qt.QGridLayout(grpBox)
        self.__cmapCBox = cmapCBox = Qt.QComboBox()
        cmapCBox.addItem('jet', userData=cm.jet)
        grpBoxLayout.addWidget(cmapCBox, 0, 0, Qt.Qt.AlignCenter)
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

        self.__minEdit = minEdit = StyledLineEdit(nChar=6)
        self.__maxEdit = maxEdit = StyledLineEdit(nChar=6)
        minEdit.setValidator(Qt.QDoubleValidator())
        maxEdit.setValidator(Qt.QDoubleValidator())
        minEdit.editingFinished.connect(self.__lineEditFinished)
        maxEdit.editingFinished.connect(self.__lineEditFinished)
        grpBoxLayout.addWidget(minEdit, 1, 0)
        grpBoxLayout.addWidget(maxEdit, 1, 1)
        grpBox.setSizePolicy(Qt.QSizePolicy.Fixed, Qt.QSizePolicy.Fixed)
        layout.addWidget(grpBox, 1, 0, Qt.Qt.AlignCenter)

        bnBox = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Close)
        bnBox.button(Qt.QDialogButtonBox.Close).clicked.connect(self.accept)

        layout.addWidget(bnBox, 2, 0)

        self.__setupWidgets()

        rngSlider.sigSliderMoved.connect(self.__rngSliderMoved)

    def __rngSliderMoved(self, event):
        blockedMin = self.__minEdit.blockSignals(True)
        blockedMax = self.__maxEdit.blockSignals(True)
        self.__minEdit.setText('{0:6g}'.format(event.left))
        self.__maxEdit.setText('{0:6g}'.format(event.right))
        self.__minEdit.blockSignals(blockedMin)
        self.__maxEdit.blockSignals(blockedMax)
        self.__applyRange()

    def __lineEditFinished(self):
        minVal = float(self.__minEdit.text())
        maxVal = float(self.__maxEdit.text())
        blocked = self.__rngSlider.blockSignals(True)
        self.__rngSlider.setSliderValues(minVal, maxVal)
        self.__rngSlider.blockSignals(blocked)
        self.__applyRange()

    def __applyRange(self):
        plot = self.__plot()
        if plot is None:
            return
        minVal = float(self.__minEdit.text())
        maxVal = float(self.__maxEdit.text())
        cmap = self.getColormap()
        curve = self.getCurve()

        colormap = XsocsPlot2DColormap(cmap, minVal, maxVal)
        plot.setPlotColormap(curve, colormap)

        plot.setPlotColormap(curve, colormap)

    def getCurve(self):
        plot = self.__plot()
        if plot is None:
            return None
        curves = plot.getAllCurves(just_legend=True)
        if curves is None:
            return None
        return curves[0]

    def __setupWidgets(self):
        self.__setColormapPixmap()

        plot = self.__plot()
        if plot is None:
            return

        curve = self.getCurve()
        if curve is None:
            return
        self.__setHistogram(curve)

        colormap = plot.getPlotColormap(curve)
        data = plot.getPlotValues(curve)
        if data is None:
            # problem
            return
        if colormap is None:
            minVal = data.min()
            maxVal = data.max()
        else:
            minVal = colormap.minVal
            maxVal = colormap.maxVal
            if minVal is None:
                minVal = data.min()
            if maxVal is None:
                maxVal = data.max()

        self.__minEdit.setText('{0:6g}'.format(minVal))
        self.__maxEdit.setText('{0:6g}'.format(maxVal))
        self.__rngSlider.setSliderValues(minVal, maxVal)

    def __setHistogram(self, curve):

        rngSlider = self.__rngSlider

        plot = self.__plot()
        if plot is None:
            return

        if self.__histogram is None:
            values = plot.getPlotValues(curve)
            if values is None:
                return

            vMin = values.min()
            vMax = values.max()

            histo = Histogramnd(values,
                                [vMin, vMax],
                                self.nColors,
                                last_bin_closed=True)

            self.__histogram = histo.histo
            rngSlider.setRange((vMin, vMax))
            rngSlider.setSliderValues(vMin, vMax)

        histo = self.__histogram

        cmap = self.getColormap()
        pixmap = _arrayToIndexedPixmap(histo, cmap, self.nColors)
        rngSlider.setSliderPixmap(pixmap)

    def getColormap(self):
        """
        Returns the currently selected colormap.
        :return:
        """
        cmapCBox = self.__cmapCBox
        cmapIndex = cmapCBox.currentIndex()
        return cmapCBox.itemData(cmapIndex, role=Qt.Qt.UserRole)

    def __setColormapPixmap(self):
        """
        Sets the colormap preview label.
        :return:
        """
        style = Qt.QApplication.style()
        size = style.pixelMetric(Qt.QStyle.PM_SmallIconSize)

        colorLabel = self.__colorLabel
        cmap = self.getColormap()

        image = np.tile(np.arange(self.nColors, dtype=np.uint8),
                        (size, 1))

        pixmap = _arrayToIndexedPixmap(image, cmap, nColors=self.nColors)
        colorLabel.setPixmap(pixmap)


XsocsPlot2DColormap = namedtuple('XsocsPlot2DColormap',
                                 ['colormap', 'minVal', 'maxVal'])


class XsocsPlot2D(PlotWindow):
    """
    Base class for the 2D scatter plot.
    The colormap widget only supports one plot at the moment.
    """

    def __init__(self, **kwargs):
        super(XsocsPlot2D, self).__init__(**kwargs)

        self.setActiveCurveHandling(False)
        self.setKeepDataAspectRatio(True)

        self.__logScatter = False
        self.__colormap = cm.jet
        self.__values = {}
        self.__cmapDialog = None

        toolbars = self.findChildren(Qt.QToolBar)
        for toolbar in toolbars:
            toolbar.hide()

        centralWid = self.centralWidget()

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

    def showEvent(self, event):
        super(XsocsPlot2D, self).showEvent(event)
        if self.__firstShow:
            self.__moveOptionBar()
            self.__firstShow = False

    def resizeEvent(self, event):
        super(XsocsPlot2D, self).resizeEvent(event)
        self.__moveOptionBar()

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
            self.__cmapDialog = dialog = XsocsPlot2DColorDialog(self)
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

    def setPlotColormap(self, curve, cmap):
        values = self.__values.get(curve)
        if values is None:
            return
        zValues = values[0]

        curveData = self.getCurve(curve)
        if curveData is None:
            return

        x, y = curveData[0:2]

        colors = _applyColormap(cmap, zValues)

        self.__plotData(x, y,
                        legend=curve,
                        color=colors,
                        resetzoom=False)

        values[1] = cmap
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

        if colormap is None:
            colormap = XsocsPlot2DColormap(colormap=cm.jet,
                                           minVal=None,
                                           maxVal=None)

        if values is not None:
            colors = _applyColormap(colormap, values)
            # minVal = (colormap.minVal is not None and colormap.minVal)\
            #          or values.min()
            # maxVal = (colormap.maxVal is not None and colormap.maxVal) \
            #          or values.max()
            # colors = colormap(
            #     (values.astype(np.float64) - minVal) / (maxVal - minVal))

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

        return legend
