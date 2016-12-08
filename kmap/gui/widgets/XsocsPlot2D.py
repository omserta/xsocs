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
from ...gui.icons import getQIcon as getKmapIcon


class XsocsPlot2D(PlotWindow):

    def __init__(self, *args, **kwargs):
        super(XsocsPlot2D, self).__init__(*args, **kwargs)

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

        icon = getKmapIcon('gears')
        options = Qt.QToolButton()

        options.setToolTip('Options')
        options.setIcon(icon)
        options.setPopupMode(Qt.QToolButton.InstantPopup)

        menu = Qt.QMenu()

        # apply log values
        # self.__logAction = logAction = menu.addAction('Log')
        self.__logAction = logAction = Qt.QAction('Log', self)
        logAction.setIconVisibleInMenu(True)
        logAction.setDisabled(True)
        logAction.setCheckable(True)
        logAction.triggered.connect(self.__logActionTriggered)
        # menu.addAction(logAction)

        # save to file action
        self.saveAction.setIconVisibleInMenu(True)
        menu.addAction(self.saveAction)

        # print action
        self.printAction.setIconVisibleInMenu(True)
        menu.addAction(self.printAction)

        # screenshot action
        self.copyAction.setIconVisibleInMenu(True)
        menu.addAction(self.copyAction)

        options.setMenu(menu)

        rstZoomAction = self.resetZoomAction
        rstToolBn = Qt.QToolButton()
        rstToolBn.setDefaultAction(rstZoomAction)

        optionsLayout.addWidget(options)
        optionsLayout.addWidget(rstToolBn)

    def resizeEvent(self, event):
        super(XsocsPlot2D, self).resizeEvent(event)
        optionsBase = self.__optionsBase
        geom = self.centralWidget().geometry()
        newPos = Qt.QPoint(0, geom.height() - optionsBase.height())
        optionsBase.move(newPos)

    def __logActionTriggered(self, checked):
        for curve, values in self.__values.items():
            info = self.getCurve(curve)
            if info is None:
                continue
            if checked:
                values = np.log(values - values.min() + 1)
            min_, max_ = values.min(), values.max()
            colormap = cm.jet
            colors = colormap(
                (values.astype(np.float64) - min_) / (max_ - min_))
            # [x, y, legend, info, parameters]
            self.addCurve(info[0], info[1], color=colors, legend=curve,
                          resetzoom=False)

    def removeCurve(self, *args, **kwargs):
        super(XsocsPlot2D, self).removeCurve(*args, **kwargs)
        curves = set(self.getAllCurves())
        thisCurves = set(self.__values.keys())
        diff = thisCurves - curves

        for curve in diff:
            del self.__values[curve]

        if len(self.__values) > 0:
            self.__logAction.setDisabled(True)

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
            self.__logAction.setDisabled(False)

        return legend
