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

import numpy as np
from matplotlib import cm


from silx.gui import qt as Qt
from silx.gui.plot import PlotWindow
from silx.gui.icons import getQIcon

from ...gui.icons import getQIcon as getKmapIcon
from ..widgets.RangeSlider import RangeSlider
from ..widgets.Input import StyledLineEdit
from ..widgets.Containers import GroupBox


class XsocsPlot2DColorDialog(Qt.QDialog):
    """
    Color dialog for the XsocsPlot2D.
    """
    def __init__(self, plot, **kwargs):

        super(XsocsPlot2DColorDialog, self).__init__(plot, **kwargs)

        self.__plot = weakref.ref(plot)

        layout = Qt.QVBoxLayout(self)

        grpBox = GroupBox('Colormap')
        grpBoxLayout = Qt.QHBoxLayout(grpBox)
        cmapCBox = Qt.QComboBox()
        cmapCBox.addItems(['jet'])
        grpBoxLayout.addWidget(cmapCBox)
        layout.addWidget(grpBox)

        grpBox = GroupBox('Range')
        grpBoxLayout = Qt.QGridLayout(grpBox)
        rngSlider = RangeSlider()
        grpBoxLayout.addWidget(rngSlider, 0, 0, 1, 2)

        minLabel = StyledLineEdit(nChar=6)
        maxLabel = StyledLineEdit(nChar=6)
        grpBoxLayout.addWidget(minLabel, 1, 0)
        grpBoxLayout.addWidget(maxLabel, 1, 1)
        layout.addWidget(grpBox)

        bnBox = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Close)
        bnBox.button(Qt.QDialogButtonBox.Close).clicked.connect(self.accept)

        layout.addWidget(bnBox)


class XsocsPlot2D(PlotWindow):

    def __init__(self, **kwargs):
        super(XsocsPlot2D, self).__init__(**kwargs)

        self.setActiveCurveHandling(False)
        self.setKeepDataAspectRatio(True)

        self.__logScatter = False
        self.__colormap = cm.jet
        self.__values = {}

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
        dialog = XsocsPlot2DColorDialog(self)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)
        dialog.exec_()

    def removeCurve(self, *args, **kwargs):
        super(XsocsPlot2D, self).removeCurve(*args, **kwargs)
        curves = set(self.getAllCurves())
        thisCurves = set(self.__values.keys())
        diff = thisCurves - curves

        for curve in diff:
            del self.__values[curve]

        if len(self.__values) > 0:
            self.__colormapBn.setDisabled(True)

    def setPlotData(self, x, y,
                    values=None,
                    symbol='s',
                    linestyle='',
                    resetzoom=True,
                    **kwargs):

        colors = None

        if values is not None:
            min_, max_ = values.min(), values.max()
            colormap = cm.jet
            colors = colormap(
                (values.astype(np.float64) - min_) / (max_ - min_))

        if 'color' in kwargs:
            del kwargs['color']
            print('In XsocsPlot2D.setPlotData : keyword color is ignored.')

        legend = self.addCurve(x,
                               y,
                               color=colors,
                               symbol=symbol,
                               linestyle=linestyle,
                               resetzoom=resetzoom,
                               **kwargs)

        if values is not None:
            self.__values[legend] = values

        if len(self.__values) > 0:
            self.__colormapBn.setDisabled(False)

        return legend
