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
from plot3d.ScalarFieldView import ScalarFieldView

from ...io.QSpaceH5 import QSpaceH5
from ..model.ModelDef import ModelRoles
from ..project.HybridItem import HybridItem
from ..project.QSpaceItem import QSpaceItem

from .DataViewWidget import DataViewWidget, DataViewEvent


class QSpaceViewWidgetEvent(DataViewEvent):
    pass


class QSpaceViewWidget(DataViewWidget):

    plot = property(lambda self: self.__plotWindow)

    def __init__(self, index, parent=None, **kwargs):
        super(QSpaceViewWidget, self).__init__(index, parent=parent)

        self.__plotWindow = plotWindow = PlotWindow(aspectRatio=True,
                                                    curveStyle=False,
                                                    mask=False,
                                                    roi=False,
                                                    **kwargs)
        plotWindow.sigPlotSignal.connect(self.__plotSignal)
        plotWindow.setKeepDataAspectRatio(True)
        plotWindow.setActiveCurveHandling(False)
        self.setCentralWidget(plotWindow)
        self.__isoView = None
        self.__isoPosition = None
        self.__plotType = None

    # TODO : refactor this in a common base with RealSpaceViewWidget
    def setPlotData(self, x, y, data):
        plot = self.__plotWindow
        if data.ndim == 1:
            # scatter
            min_, max_ = data.min(), data.max()
            colormap = cm.jet
            colors = colormap((data.astype(np.float64) - min_) / (max_ - min_))
            plot.addCurve(x, y,
                          color=colors,
                          symbol='s',
                          linestyle='')
            self.__plotType = 'scatter'
        elif data.ndim == 2:
            # image
            min_, max_ = data.min(), data.max()
            colormap = {'name': 'temperature',
                        'normalization': 'linear',
                        'autoscale': True,
                        'vmin': min_,
                        'vmax': max_}
            origin = x[0], y[0]
            scale = (x[-1] - x[0]) / len(x), (y[-1] - y[0]) / len(y)
            plot.addImage(data,
                          origin=origin,
                          scale=scale,
                          colormap=colormap)
            self.__plotType = 'image'
        else:
            raise ValueError('data has {0} dimensions, expected 1 or 2.'
                             ''.format(data.ndim))

    def __plotSignal(self, event):
        if event['event'] not in ('curveClicked',): # , 'mouseClicked'):
            return
        x, y = event['xdata'], event['ydata']

        self.__showIsoView(x, y)

    def __showIsoView(self, x, y):
        if self.__isoPosition is not None:
            if self.__isoPosition[0] == x and self.__isoPosition[1] == y:
                return
        isoView = self.__isoView
        if isoView is None:
            isoView = IsoViewMainWindow(parent=self)
            if isinstance(self.parent(), Qt.QMdiSubWindow):
                self.parent().mdiArea().addSubWindow(isoView)

        node = self.index.data(ModelRoles.InternalDataRole)

        item = HybridItem(node.projectFile, node.path)

        if item.hasScatter():
            xPos, yPos, _ = item.getScatter()
        elif item.hasImage():
            xPos, yPos, _ = item.getImage()
        else:
            return None

        # TODO : this wont work with images
        try:
            xIdx = (np.abs(xPos - x)).argmin()
        except:
            print x
            xIdx = (np.abs(xPos - x[0])).argmin()

        # TODO : this is not robust at all
        qspaceNode = node.parent()
        qspaceItem = QSpaceItem(qspaceNode.projectFile, qspaceNode.path)
        qspaceH5 = QSpaceH5(qspaceItem.qspaceFile)

        qspace = qspaceH5.qspace_slice(xIdx)

        isoView.setData(qspace)
        self.__isoView = isoView
        isoView.show()


class _FloatEdit(Qt.QLineEdit):
    """Field to edit a float value.

    :param parent: See :class:`QLineEdit`
    :param float value: The value to set the QLineEdit to.
    """
    def __init__(self, parent=None, value=None):
        Qt.QLineEdit.__init__(self, parent)
        self.setValidator(Qt.QDoubleValidator())
        self.setAlignment(Qt.Qt.AlignRight)
        self.setValue(value)

    def value(self):
        """Return the QLineEdit current value as a float."""
        text = self.text()
        return float(text) if text else None

    def setValue(self, value):
        """Set the current value of the LineEdit

        :param float value: The value to set the QLineEdit to.
        """
        self.setText('' if value is None else '%g' % value)


class _Controller(Qt.QDockWidget):
    """Widgets controlling an isosurface display.

    :param parent: See :class:`QDockWidget`
    :param viewer: ScalarFieldView widget to control
    """

    def __init__(self, parent=None, viewer=None):
        super(_Controller, self).__init__(parent)
        self._range = 0., 1.

        assert viewer is not None
        self.viewer = viewer

        # form
        form = Qt.QFormLayout()
        form.setFieldGrowthPolicy(Qt.QFormLayout.FieldsStayAtSizeHint)

        # Iso value
        self._isoEdit = Qt.QLineEdit()
        self._isoEdit.setValidator(Qt.QDoubleValidator())
        self._isoEdit.editingFinished.connect(self._isoEditingFinished)
        form.addRow('Level:', self._isoEdit)

        self._isoSlider = Qt.QSlider(Qt.Qt.Horizontal)
        self._isoSlider.sliderReleased.connect(self._isoSliderReleased)
        self._isoSlider.actionTriggered.connect(self._isoSliderActionTriggered)
        form.addRow(self._isoSlider)

        # Color chooser remove and use colormap?
        self._colorBtn = Qt.QPushButton()
        pixmap = Qt.QPixmap(32, 32)
        color = np.array(np.array(self.viewer.getIsoColor()) * 255,
                            dtype=np.int)
        pixmap.fill(Qt.QColor(color[0], color[1], color[2]))
        self._colorBtn.setIcon(Qt.QIcon(pixmap))
        self._colorBtn.clicked.connect(self._colorBtnClicked)
        form.addRow('Color:', self._colorBtn)

        self.viewer.valueChanged.connect(self._viewerValueChanged)

        # Colormap
        self._comboBoxColormap = Qt.QComboBox()
        for name in ('gray', 'temperature', 'reversed gray', 'green'):
            self._comboBoxColormap.addItem(name)
        self._comboBoxColormap.setCurrentIndex(0)
        self._comboBoxColormap.currentIndexChanged[int].connect(
            self._cmapChanged)
        form.addRow('Colormap:', self._comboBoxColormap)

        self._normButtonLinear = Qt.QRadioButton('Linear')
        self._normButtonLinear.setChecked(True)
        self._normButtonLog = Qt.QRadioButton('Log')

        # Normalization
        normButtonGroup = Qt.QButtonGroup(self)
        normButtonGroup.setExclusive(True)
        normButtonGroup.addButton(self._normButtonLinear)
        normButtonGroup.addButton(self._normButtonLog)
        normButtonGroup.buttonClicked[int].connect(self._cmapChanged)

        normLayout = Qt.QHBoxLayout()
        normLayout.setContentsMargins(0, 0, 0, 0)
        normLayout.setSpacing(10)
        normLayout.addWidget(self._normButtonLinear)
        normLayout.addWidget(self._normButtonLog)

        form.addRow('Normalization:', normLayout)

        # Range row
        self._rangeAutoscaleButton = Qt.QCheckBox('Autoscale')
        self._rangeAutoscaleButton.setChecked(True)
        self._rangeAutoscaleButton.clicked.connect(self._cmapChanged)
        self._rangeAutoscaleButton.toggled.connect(
            self._rangeAutoscaleButtonToggled)
        form.addRow('Range:', self._rangeAutoscaleButton)

        # Min row
        self._minValue = _FloatEdit()
        self._minValue.setEnabled(False)
        self._minValue.editingFinished.connect(self._cmapChanged)
        form.addRow('\tMin:', self._minValue)

        # Max row
        self._maxValue = _FloatEdit()
        self._maxValue.setEnabled(False)
        self._maxValue.editingFinished.connect(self._cmapChanged)
        form.addRow('\tMax:', self._maxValue)

        # ROI
        self._roiButton = Qt.QCheckBox('Enabled')
        self._roiButton.setChecked(False)
        self._roiButton.clicked.connect(self._roiUpdated)
        form.addRow('Selection:', self._roiButton)
        self._xmin = _FloatEdit(value=0.)
        self._xmin.setEnabled(False)
        self._xmin.editingFinished.connect(self._roiUpdated)
        self._roiButton.toggled.connect(self._xmin.setEnabled)
        form.addRow('\t X Min:', self._xmin)
        self._xmax = _FloatEdit(value=0.)
        self._xmax.setEnabled(False)
        self._xmax.editingFinished.connect(self._roiUpdated)
        self._roiButton.toggled.connect(self._xmax.setEnabled)
        form.addRow('\t X Max:', self._xmax)
        self._ymin = _FloatEdit(value=0.)
        self._ymin.setEnabled(False)
        self._ymin.editingFinished.connect(self._roiUpdated)
        self._roiButton.toggled.connect(self._ymin.setEnabled)
        form.addRow('\t Y Min:', self._ymin)
        self._ymax = _FloatEdit(value=0.)
        self._ymax.setEnabled(False)
        self._ymax.editingFinished.connect(self._roiUpdated)
        self._roiButton.toggled.connect(self._ymax.setEnabled)
        form.addRow('\t Y Max:', self._ymax)
        self._zmin = _FloatEdit(value=0.)
        self._zmin.setEnabled(False)
        self._zmin.editingFinished.connect(self._roiUpdated)
        self._roiButton.toggled.connect(self._zmin.setEnabled)
        form.addRow('\t Z Min:', self._zmin)
        self._zmax = _FloatEdit(value=0.)
        self._zmax.setEnabled(False)
        self._zmax.editingFinished.connect(self._roiUpdated)
        self._roiButton.toggled.connect(self._zmax.setEnabled)
        form.addRow('\t Z Max:', self._zmax)

        widget = Qt.QWidget()
        widget.setLayout(form)

        self.setWidget(widget)

    def _roiUpdated(self, *args):
        if self._roiButton.isChecked():
            xrange_ = self._xmin.value() or 0, self._xmax.value() or 0
            yrange = self._ymin.value() or 0, self._ymax.value() or 0
            zrange = self._zmin.value() or 0, self._zmax.value() or 0
        else:
            xrange_, yrange, zrange = None, None, None
        self.viewer.setSelectedRegion(
            xrange_=xrange_, yrange=yrange, zrange=zrange)

    def _rangeAutoscaleButtonToggled(self, checked):
        self._minValue.setDisabled(checked)
        self._maxValue.setDisabled(checked)
        if checked:
            self._minValue.setValue(None)
            self._maxValue.setValue(None)
        else:
            data = self.viewer.getData(copy=False)
            if data is not None:
                vmin, vmax = data.min(), data.max()
                self._minValue.setValue(vmin)
                self._maxValue.setValue(vmax)

    def _cmapChanged(self, *args):
        norm = 'linear' if self._normButtonLinear.isChecked() else 'log'
        if self._rangeAutoscaleButton.isChecked():
            vmin, vmax = None, None
        else:
            vmin, vmax = self._minValue.value(), self._maxValue.value()
        self.viewer.setColormap(
            name=str(self._comboBoxColormap.currentText()),
            norm=norm,
            vmin=vmin,
            vmax=vmax)

    def _colorBtnClicked(self, checked=False):
        color = np.array(np.array(self.viewer.getIsoColor()) * 255,
                            dtype=np.int)
        Qt.QColor(color[0], color[1], color[2])
        color = Qt.QColorDialog.getColor(
                initial=Qt.QColor(color[0], color[1], color[2]),
                parent=self)

        self.viewer.setIsoColor(
            (color.red() / 255., color.green() / 255., color.blue() / 255.))
        pixmap = Qt.QPixmap(32, 32)
        pixmap.fill(color)
        self._colorBtn.setIcon(Qt.QIcon(pixmap))

    def _isoEditingFinished(self):
        self.viewer.setIsoLevel(float(self._isoEdit.text()))

    def _isoSliderActionTriggered(self, action):
        if action != Qt.QAbstractSlider.SliderMove:
            self._isoSliderReleased()

    def _isoSliderReleased(self):
        offset = ((self._isoSlider.value() - self._isoSlider.minimum()) /
                  (self._isoSlider.maximum() - self._isoSlider.minimum()))
        isolevel = offset * self._range[1] + (1. - offset) * self._range[0]
        self.viewer.setIsoLevel(isolevel)

    def _viewerValueChanged(self, value):
        self._isoEdit.setText(str(float(value)))

        offset = (value - self._range[0]) / (self._range[1] - self._range[0])
        self._isoSlider.setValue(int(
            offset * self._isoSlider.maximum() +
            (1. - offset) * self._isoSlider.minimum()))

    def setRange(self, min_, max_):
        """Set range of accepted iso-level values"""
        self._range = min_, max_


class IsoViewMainWindow(ScalarFieldView):
    """Window displaying an isosurface and some controls."""

    def __init__(self, *args, **kwargs):
        super(IsoViewMainWindow, self).__init__(*args, **kwargs)
        self.setAttribute(Qt.Qt.WA_DeleteOnClose)

        # Adjust lighting
        self.plot3D.viewport.light.direction = 0., 0., -1.
        self.plot3D.viewport.light.shininess = 32
        self.plot3D.viewport.bgColor = 0.2, 0.2, 0.2, 1.

        # Add controller dock widget
        self._control = _Controller(viewer=self)
        self._control.setWindowTitle('Isosurface')
        self.addDockWidget(Qt.Qt.RightDockWidgetArea, self._control)

    def setData(self, data, copy=True):
        data = np.asarray(data)
        self._control.setRange(data.min(), data.max())
        super(IsoViewMainWindow, self).setData(data, copy)