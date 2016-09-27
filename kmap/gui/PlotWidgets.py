import os
import numpy as np
from silx.gui import qt as Qt

from silx.gui.plot import PlotWindow

try:
    from silx.gui.plot.ImageRois import ImageRoiManager
except:
    # TODO remove this import once the ROIs are added to the silx release.
    from .silx_imports.ImageRois import ImageRoiManager

class RectRoiWidget(Qt.QWidget):
    sigRoiApplied = Qt.Signal(object)

    def __init__(self, roiManager, parent=None):
        # TODO :
        # support multiple ROIs then batch them
        super(RectRoiWidget, self).__init__(parent=parent)
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        style = Qt.QApplication.style()
        icon = style.standardIcon(Qt.QStyle.SP_DialogCloseButton)
        self.__discardBn = discardBn = Qt.QToolButton()
        discardBn.setToolTip('Discard ROI')
        discardBn.setStatusTip('Discard ROI')
        discardBn.setIcon(icon)
        discardBn.setEnabled(False)
        layout.addWidget(discardBn)
        discardBn.clicked.connect(self.__discardRoi)

        icon = style.standardIcon(Qt.QStyle.SP_DialogApplyButton)
        self.__applyBn = applyBn = Qt.QToolButton()
        applyBn.setToolTip('Apply ROI')
        applyBn.setStatusTip('Apply ROI')
        applyBn.setIcon(icon)
        applyBn.setEnabled(False)
        layout.addWidget(applyBn)
        applyBn.clicked.connect(self.__applyRoi)

        layout.addWidget(Qt.QLabel('x='))
        self._xEdit = edit = Qt.QLineEdit()
        layout.addWidget(edit, stretch=0, alignment=Qt.Qt.AlignLeft)
        layout.addWidget(Qt.QLabel('y='))
        self._yEdit = edit = Qt.QLineEdit()
        layout.addWidget(edit, stretch=0, alignment=Qt.Qt.AlignLeft)
        self._wEdit = edit = Qt.QLineEdit()
        layout.addWidget(Qt.QLabel('w='))
        layout.addWidget(edit, stretch=0, alignment=Qt.Qt.AlignLeft)
        self._hEdit = edit = Qt.QLineEdit()
        layout.addWidget(Qt.QLabel('h='))
        layout.addWidget(edit, stretch=0, alignment=Qt.Qt.AlignLeft)

        fm = edit.fontMetrics()
        padding = 2 * fm.width('0')
        text = '0' * 8
        width = fm.width(text) + padding
        self._xEdit.setFixedWidth(width)
        self._yEdit.setFixedWidth(width)
        self._wEdit.setFixedWidth(width)
        self._hEdit.setFixedWidth(width)
        layout.setSizeConstraint(Qt.QLayout.SetMinimumSize)

        # TODO : weakref
        self.__roiManager = roiManager
        roiManager.sigRoiDrawingFinished.connect(self.__roiDrawingFinished,
                                       Qt.Qt.QueuedConnection)
        roiManager.sigRoiRemoved.connect(self.__roiRemoved,
                                         Qt.Qt.QueuedConnection)
        roiManager.sigRoiMoved.connect(self.__roiMoved,
                                        Qt.Qt.QueuedConnection)

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
        if xMax < xMin: xMin, xMax = xMax, xMin
        yMin, yMax = yData[0], yData[1]
        if yMax < yMin : yMin, yMax = yMax, yMin 
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


class IntensityWindow(Qt.QMainWindow):
    sigRoiApplied = Qt.Signal(object)

    plot = property(lambda self: self.__plotWindow)

    def __init__(self, parent=None, **kwargs):
        super(IntensityWindow, self).__init__(parent=parent)

        self.__plotWindow = plotWindow = PlotWindow(**kwargs)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)

        self.__roiManager = roiManager = ImageRoiManager(plotWindow)
        roiToolBar = roiManager.toolBar(rois=['rectangle'],
                                        options=['show'])
        roiToolBar.addSeparator()

        rectRoiWidget = RectRoiWidget(roiManager)
        roiToolBar.addWidget(rectRoiWidget)
        rectRoiWidget.sigRoiApplied.connect(self.sigRoiApplied)

        plotWindow.addToolBarBreak()
        plotWindow.addToolBar(roiToolBar)

        self.setCentralWidget(plotWindow)


if __name__ == '__main__':
    pass
