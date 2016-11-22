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
__date__ = "01/11/2016"

from silx.gui import qt as Qt

from .Node import Node
from .NodeEditor import EditorMixin

#
# class SliderEditor(NodeEditor):
#     persistent = True
#
#     def __init__(self, parent, option, index):
#         super(SliderEditor, self).__init__(parent, option, index)
#         slider = Qt.QSlider()
#         slider.setOrientation(Qt.Qt.Horizontal)
#         slider.valueChanged.connect(self.valueChanged)
#         self.setWidget(slider)
#
#     def getEditorData(self):
#         return self.widget().value()
#
#     def updateFromNode(self, node):
#         if isinstance(node, ValMinMaxNode):
#             slider = self.widget()
#             slider.setMinimum(node.minValue)
#             slider.setMaximum(node.maxValue)
#             value = node.data(self.column, role=Qt.Qt.EditRole)
#             if value is not None:
#                 slider.setValue(value)
#             else:
#                 # TODO : ERROR
#                 pass
#             return True
#         return False
#
#
# class ValMinMaxNode(Node):
#     editors = SliderEditor
#
#     minValue = 0
#     maxValue = 100


class ProgressBarEditor(EditorMixin):
    editable = False

    @classmethod
    def paint(cls, painter, option, index):
        progress = index.data(Qt.Qt.EditRole)

        progressBarOption = Qt.QStyleOptionProgressBar()
        progressBarOption.rect = option.rect
        progressBarOption.minimum = 0
        progressBarOption.maximum = 100
        progressBarOption.progress = progress
        progressBarOption.text = '{0}%'.format(progress)
        progressBarOption.textVisible = True

        Qt.QApplication.style().drawControl(Qt.QStyle.CE_ProgressBar,
                                            progressBarOption,
                                            painter)
        return True
#
#
# class DoubleEditor(NodeEditor):
#     persistent = False
#
#     def __init__(self, *args, **kwargs):
#         super(DoubleEditor, self).__init__(*args, **kwargs)
#         edit = Qt.QLineEdit()
#         edit.setValidator(Qt.QDoubleValidator())
#         edit.editingFinished.connect(self.valueChanged)
#         self.setWidget(edit)
#
#     def getEditorData(self):
#         return float(self.widget().text())
#
#     def updateFromNode(self, node):
#         # TODO : try/catch/log
#         edit = self.widget()
#         edit.setText(str(node.value(self.column)))
#         return True
#
#
# class QColorEditor(NodeEditor):
#     persistent = True
#
#     def __init__(self, *args, **kwargs):
#         super(QColorEditor, self).__init__(*args, **kwargs)
#         base = Qt.QWidget()
#         layout = Qt.QHBoxLayout(base)
#         layout.setContentsMargins(0, 0, 0, 0)
#         button = Qt.QToolButton()
#         icon = Qt.QIcon(Qt.QPixmap(32, 32))
#         button.setIcon(icon)
#         layout.addWidget(button)
#         self.setWidget(base)
#         button.clicked.connect(self.__showColorDialog)
#         layout.addStretch(1)
#
#         self.__color = None
#         self.__dialog = None
#         self.__previousColor = None
#
#     def getEditorData(self):
#         return self.__color
#
#     def updateFromNode(self, node):
#         if isinstance(node, QColorNode):
#             qColor = node.data(self.column, Qt.Qt.EditRole)
#             if qColor is not None:
#                 self._setColor(qColor)
#             else:
#                 # TODO : error
#                 pass
#             return True
#
#         return False
#
#     def _setColor(self, qColor):
#         widget = self.widget()
#         button = widget.findChild(Qt.QToolButton)
#         pixmap = Qt.QPixmap(32, 32)
#         pixmap.fill(qColor)
#         button.setIcon(Qt.QIcon(pixmap))
#         self.__currentColor = qColor
#
#     def __showColorDialog(self):
#         if self.__dialog is not None:
#             self.__dialog.reject()
#             return
#         self.__dialog = dialog = Qt.QColorDialog()
#         dialog.setOption(Qt.QColorDialog.ShowAlphaChannel, True)
#         self.__previousColor = self.__currentColor
#         dialog.setAttribute(Qt.Qt.WA_DeleteOnClose)
#         dialog.currentColorChanged.connect(self.__colorChanged)
#         dialog.finished.connect(self.__dialogClosed)
#         dialog.show()
#
#     def __colorChanged(self, color):
#         self.__color = color
#         self._setColor(color)
#         self.valueChanged(color)
#
#     def __dialogClosed(self, result):
#         if result == Qt.QDialog.Rejected:
#             self.__colorChanged(self.__previousColor)
#         self.__dialog = None
#         self.__previousColor = None
#
#
# class QColorNode(Node):
#     editors = QColorEditor