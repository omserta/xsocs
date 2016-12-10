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
from matplotlib import cm, colors as mpl_colors


from silx.gui import qt as Qt
from silx.gui.plot import PlotWindow
from silx.gui.icons import getQIcon
from silx.math.histogram import Histogramnd

from ...gui.icons import getQIcon as getKmapIcon
from ..widgets.RangeSlider import RangeSlider
from ..widgets.Input import StyledLineEdit
from ..widgets.Containers import GroupBox


_defaultNColors = 256

XsocsPlot2DColormap = namedtuple('XsocsPlot2DColormap',
                                 ['colormap', 'minVal', 'maxVal', 'nColors'])


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

    if minVal is None:
        minVal = values.min()

    if maxVal is None:
        maxVal = values.max()

    if clip:
        values = np.clip(values, minVal, maxVal)
    colors = colormap.colormap(
        (values - minVal) / (maxVal - minVal))
    return colors


class XsocsPlot2DColorDialog(Qt.QDialog):
    """
    Color dialog for the XsocsPlot2D.
    Right now only supports one scatter plot!
    """

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
            nColors = colormap.nColors
            if minVal is None:
                minVal = histo.edges[0][0]
            if maxVal is None:
                maxVal = histo.edges[0][-1]

        self.__colormap = XsocsPlot2DColormap(colormap=cmap,
                                              minVal=minVal,
                                              maxVal=maxVal,
                                              nColors=nColors)

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
        colormap = self.__colormap
        curve = self.getCurve()

        colormap = XsocsPlot2DColormap(colormap=colormap.colormap,
                                       minVal=minVal,
                                       maxVal=maxVal,
                                       nColors=colormap.nColors)
        self.__colormap = colormap
        plot.setPlotColormap(curve, colormap)
        self.__drawHistogram()

    def getCurve(self):
        return self.__curve

    def __setupWidgets(self):
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

        pixmap = _arrayToIndexedPixmap(histo.histo,
                                       colormap.colormap,
                                       colormap.nColors)
        rngSlider.setSliderPixmap(pixmap)

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
        style = Qt.QApplication.style()
        size = style.pixelMetric(Qt.QStyle.PM_SmallIconSize)

        colorLabel = self.__colorLabel
        colormap = self.__colormap

        image = np.tile(np.arange(colormap.nColors,
                                  dtype=np.uint8),
                        (size, 1))

        pixmap = _arrayToIndexedPixmap(image,
                                       colormap.colormap,
                                       nColors=colormap.nColors)
        colorLabel.setPixmap(pixmap)


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
