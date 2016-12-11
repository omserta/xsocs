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

import os

from silx.gui import qt as Qt

from ...io.XsocsH5 import XsocsH5
from .FileChooser import FileChooser
from ..widgets.Containers import GroupBox
from ..project.XsocsProject import XsocsProject


class ProjectChooserDialog(Qt.QDialog):
    projectFile = property(lambda self: self.__projectFile)

    def __init__(self, parent=None):
        super(ProjectChooserDialog, self).__init__(parent)

        self.__projectFile = None

        layout = Qt.QVBoxLayout(self)

        prjChooser = ProjectChooser()
        layout.addWidget(prjChooser)
        prjChooser.sigProjectPicked.connect(self.__prjPicked)

        self.__bnBox = bnBox = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Open |
                                                   Qt.QDialogButtonBox.Cancel)

        bnBox.rejected.connect(self.reject)
        bnBox.accepted.connect(self.accept)

        bnBox.button(Qt.QDialogButtonBox.Open).setEnabled(False)

        layout.addWidget(bnBox)

    def __prjPicked(self, filename):
        if filename:
            self.__projectFile = filename
            enabled = True
        else:
            self.__projectFile = None
            enabled = False
        self.__bnBox.button(Qt.QDialogButtonBox.Open).setEnabled(enabled)


class ProjectChooser(Qt.QWidget):
    sigProjectPicked = Qt.Signal(object)

    projectSummary = property(lambda self:
                              self.findChild(ProjectSummaryWidget))

    def __init__(self, parent=None):
        super(ProjectChooser, self).__init__(parent)

        layout = Qt.QVBoxLayout(self)

        self.__isValid = False
        self.__selectedPath = None

        group = GroupBox('Please select the project file to open.')
        layout.addWidget(group)

        grpLayout = Qt.QHBoxLayout(group)
        filePicker = FileChooser(fileMode=Qt.QFileDialog.ExistingFile)
        filePicker.setObjectName('PROJ_FILEPICKER')
        grpLayout.addWidget(filePicker)

        filePicker.sigSelectionChanged.connect(self.__filePicked)

        fileDialog = filePicker.fileDialog
        fileDialog.setNameFilters(['Xsocs project files (*.prj)',
                                   'Any files (*)'])

        group = GroupBox('Project Summary')
        layout.addWidget(group)
        grpLayout = Qt.QVBoxLayout(group)
        view = ProjectSummaryWidget()
        grpLayout.addWidget(view)

    def __filePicked(self, selectedPath):
        self.__selectedPath = selectedPath

        view = self.projectSummary
        view.setProjectFile(selectedPath)

        valid = view.isValidProject()

        self.__isValid = valid

        if valid:
            self.sigProjectPicked.emit(selectedPath)
        else:
            self.sigProjectPicked.emit('')

    def isValid(self):
        return self.__isValid


class ProjectSummaryWidget(Qt.QWidget):
    def __init__(self, projectFile=None, parent=None):
        super(ProjectSummaryWidget, self).__init__(parent)

        self.__valid = False

        layout = Qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        view = Qt.QTreeWidget()
        view.setColumnCount(2)
        view.setHeaderLabels(['Name', 'Value'])
        view.header().setResizeMode(Qt.QHeaderView.ResizeToContents)
        layout.addWidget(view)

        self.setProjectFile(projectFile)

    def isValidProject(self):
        return self.__valid

    def setProjectFile(self, projectFile):
        view = self.findChild(Qt.QTreeWidget)

        view.clear()

        self.__valid = False

        if projectFile is None:
            return

        errMsg = ''

        try:
            # reading project file
            errMsg = 'Failed to open X-Socs project file.'
            projectH5 = XsocsProject(projectFile, mode='r')

            # reading XSOCS data file
            errMsg = 'Failed to open X-Socs data file.'
            xsocsFile = projectH5.xsocsFile
            xsocsH5 = XsocsH5(xsocsFile, mode='r')

            # getting entries
            errMsg = 'Failed to read entries from data file.'
            entries = xsocsH5.entries()

            # getting entries
            errMsg = 'Failed to read scan parameters.'
            params = xsocsH5.scan_params(entries[0])

            inputItem = Qt.QTreeWidgetItem(
                ['Data file', os.path.basename(xsocsFile)])
            inputItem.setToolTip(0, xsocsFile)
            inputItem.setToolTip(1, xsocsFile)
            inputItem.addChild(Qt.QTreeWidgetItem(['Full path', xsocsFile]))
            view.addTopLevelItem(inputItem)

            # getting scan angles
            errMsg = 'Failed to read scan angles.'
            # TODO : check that there are at least 2 angles
            text = '{0} [{1} -> {2}]'.format(
                str(len(entries)),
                str(xsocsH5.scan_angle(entries[0])),
                str(xsocsH5.scan_angle(entries[-1])))
            entriesItem = Qt.QTreeWidgetItem(['Angles', text])
            for entryIdx, entry in enumerate(entries):
                text = 'eta = {0}'.format(str(xsocsH5.scan_angle(entry)))
                entryItem = Qt.QTreeWidgetItem([str(entryIdx), text])
                entriesItem.addChild(entryItem)
            view.addTopLevelItem(entriesItem)

            # getting acquisition params
            errMsg = 'Failed to read Acquisition parameters.'
            title = ' '.join(str(value) for value in params.values())
            commandItem = Qt.QTreeWidgetItem(['Scan', title])
            commandItem.setToolTip(0, title)
            commandItem.setToolTip(1, title)
            for key, value in params.items():
                commandItem.addChild(Qt.QTreeWidgetItem([key, str(value)]))
            view.addTopLevelItem(commandItem)

            for key, value in xsocsH5.acquisition_params(entries[0]).items():
                view.addTopLevelItem(Qt.QTreeWidgetItem([key, str(value)]))

        except Exception as ex:
            style = Qt.QApplication.style()
            errorItem = Qt.QTreeWidgetItem(['', errMsg])
            icon = style.standardIcon(Qt.QStyle.SP_MessageBoxCritical)
            errorItem.setIcon(0, icon)
            errorItem.setBackground(1, Qt.QBrush(Qt.Qt.red))
            exItem = Qt.QTreeWidgetItem([ex.__class__.__name__, str(ex)])
            errorItem.addChild(exItem)
            view.addTopLevelItem(errorItem)
            errorItem.setExpanded(True)
            return

        self.__valid = True