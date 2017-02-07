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
from functools import partial

from silx.gui import qt as Qt

from .. import icons as XsocsIcons
from .FileChooser import FileChooser
from ..widgets.Containers import GroupBox
from ..process.MergeWidget import MergeWidget
from ..project.XsocsProject import XsocsProject
from .ProjectChooser import ProjectSummaryWidget


class XsocsWizard(Qt.QWizard):
    (IntroId, OpenId, CreateId, SelectDataId,
     LoadXsocsId, ReviewId, MaxId) = range(7)

    def __init__(self, parent=None):
        super(XsocsWizard, self).__init__(parent)

        self.__projectFile = None

        self.setPage(XsocsWizard.IntroId, IntroPage())
        self.setPage(XsocsWizard.OpenId, OpenProjectPage())
        self.setPage(XsocsWizard.CreateId, NewProjectPage())
        self.setPage(XsocsWizard.SelectDataId, SelectDataPage())
        self.setPage(XsocsWizard.LoadXsocsId, LoadXsocsDataPage())
        self.setPage(XsocsWizard.ReviewId, ReviewProjectPage())

    projectFile = property(lambda self: self.__projectFile)

    @projectFile.setter
    def projectFile(self, projectFile):
        self.__projectFile = projectFile


class LoadXsocsDataPage(Qt.QWizardPage):

    def __init__(self, parent=None):
        super(LoadXsocsDataPage, self).__init__(parent)

        self.setTitle('X-Socs')
        self.setSubTitle('New project : load X-SOCS data (HDF5).')

        self.__nextId = XsocsWizard.ReviewId

        icon = XsocsIcons.getQPixmap('xsocs')
        self.setPixmap(Qt.QWizard.WatermarkPixmap, icon)
        icon = XsocsIcons.getQPixmap('logo')
        self.setPixmap(Qt.QWizard.LogoPixmap, icon)

        layout = Qt.QVBoxLayout(self)

        group = GroupBox('Please select the XSOCS data file to load.')
        layout.addWidget(group)
        grpLayout = Qt.QHBoxLayout(group)
        filePicker = FileChooser(fileMode=Qt.QFileDialog.ExistingFile)
        grpLayout.addWidget(filePicker)

        self.registerField('XsocsDataFile*', filePicker.lineEdit)

    def nextId(self):
        return self.__nextId

    def validatePage(self):
        projectFile = self.wizard().projectFile
        xsocsFile = self.wizard().field('XsocsDataFile')

        try:
            projectH5 = XsocsProject(projectFile, mode='a', gui=self)
        except Exception as ex:
            Qt.QMessageBox.critical(self,
                                    'Failed to open project file.',
                                    str(ex))
            return

        try:
            projectH5.xsocsFile = xsocsFile
        except Exception as ex:
            Qt.QMessageBox.critical(self,
                                    'Failed to set data file.',
                                    str(ex))
            return
        self.setCommitPage(True)
        return True


class SelectDataPage(Qt.QWizardPage):

    def __init__(self, parent=None):
        super(SelectDataPage, self).__init__(parent)

        self.setTitle('X-Socs')
        self.setSubTitle('New project : select data to load/import.')

        icon = XsocsIcons.getQPixmap('xsocs')
        self.setPixmap(Qt.QWizard.WatermarkPixmap, icon)
        icon = XsocsIcons.getQPixmap('logo')
        self.setPixmap(Qt.QWizard.LogoPixmap, icon)

        self.setTitle('Select input data.')

        self.__nextId = -1

        layout = Qt.QGridLayout(self)
        icon = XsocsIcons.getQIcon('logo')
        xsocsBn = Qt.QToolButton()
        xsocsBn.setIcon(icon)
        layout.addWidget(xsocsBn, 1, 1)
        layout.addWidget(Qt.QLabel('Load X-Socs Data (HDF5).'), 1, 2)

        icon = XsocsIcons.getQIcon('spec')
        specBn = Qt.QToolButton()
        specBn.setIcon(icon)
        layout.addWidget(specBn, 2, 1)
        layout.addWidget(Qt.QLabel('Import SPEC data.'), 2, 2)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(3, 1)
        layout.setColumnStretch(3, 1)

        xsocsBn.clicked.connect(partial(self.__buttonClicked, source='XSOCS'))
        specBn.clicked.connect(partial(self.__buttonClicked, source='SPEC'))

    def nextId(self):
        return self.__nextId

    def isComplete(self):
        return False

    def initializePage(self):
        self.setCommitPage(False)

    def __buttonClicked(self, source=None):
        self.__nextId = -1
        if source == 'XSOCS':
            self.__nextId = XsocsWizard.LoadXsocsId
        if source == 'SPEC':
            outputDir = os.path.dirname(self.wizard().projectFile)
            mergeWid = MergeWidget(parent=self, output_dir=outputDir)
            if mergeWid.exec_() == Qt.QDialog.Accepted:
                xsocsH5 = mergeWid.xsocsH5
                mergeWid.deleteLater()

                if xsocsH5 is not None:
                    projectFile = self.wizard().projectFile
                    try:
                        projectH5 = XsocsProject(projectFile, mode='a')
                    except Exception as ex:
                        Qt.QMessageBox.critical(self,
                                                'Failed to open project file.',
                                                str(ex))
                        return

                    try:
                        projectH5.xsocsFile = xsocsH5
                    except Exception as ex:
                        Qt.QMessageBox.critical(self,
                                                'Failed to set data file.',
                                                str(ex))
                        return

                    self.__nextId = XsocsWizard.ReviewId
                    self.setCommitPage(True)

        if self.__nextId != -1:
            self.wizard().next()


class ReviewProjectPage(Qt.QWizardPage):

    def __init__(self, parent=None):
        super(ReviewProjectPage, self).__init__(parent)

        self.setTitle('X-Socs')
        self.setSubTitle('New project created.')

        icon = XsocsIcons.getQPixmap('xsocs')
        self.setPixmap(Qt.QWizard.WatermarkPixmap, icon)
        icon = XsocsIcons.getQPixmap('logo')
        self.setPixmap(Qt.QWizard.LogoPixmap, icon)

        layout = Qt.QVBoxLayout(self)
        group = GroupBox('Project Summary')
        layout.addWidget(group)
        grpLayout = Qt.QVBoxLayout(group)
        view = ProjectSummaryWidget()
        grpLayout.addWidget(view)

    def initializePage(self):
        """
       Fills the AcqParamWidget with
       info found in the input file
       """
        view = self.findChild(ProjectSummaryWidget)
        view.setProjectFile(self.wizard().projectFile)

    def nextId(self):
        return -1


class IntroPage(Qt.QWizardPage):
    def __init__(self, parent=None):
        super(IntroPage, self).__init__(parent)

        self.__nextId = XsocsWizard.OpenId

        self.setTitle('Welcome to X-SOCS.')
        self.setSubTitle('X-ray Strain Orientation Calculation Software')

        style = Qt.QApplication.style()
        icon = XsocsIcons.getQPixmap('xsocs')
        self.setPixmap(Qt.QWizard.WatermarkPixmap, icon)
        icon = XsocsIcons.getQPixmap('logo')
        self.setPixmap(Qt.QWizard.LogoPixmap, icon)

        layout = Qt.QGridLayout(self)
        icon = style.standardIcon(Qt.QStyle.SP_DialogOpenButton)
        openBn = Qt.QToolButton()
        openBn.setIcon(icon)
        layout.addWidget(openBn, 1, 1)
        layout.addWidget(Qt.QLabel('Open an existing project.'), 1, 2)

        icon = style.standardIcon(Qt.QStyle.SP_FileDialogNewFolder)
        newBn = Qt.QToolButton()
        newBn.setIcon(icon)
        layout.addWidget(newBn, 2, 1)
        layout.addWidget(Qt.QLabel('Create new project.'), 2, 2)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(3, 1)
        layout.setColumnStretch(3, 1)

        openBn.clicked.connect(partial(self.__buttonClicked,
                                       nextId=XsocsWizard.OpenId))
        newBn.clicked.connect(partial(self.__buttonClicked,
                                      nextId=XsocsWizard.CreateId))

    def isComplete(self):
        return False

    def __buttonClicked(self, nextId=-1):
        self.__nextId = nextId
        self.wizard().next()

    def nextId(self):
        return self.__nextId


class NewProjectPage(Qt.QWizardPage):
    # TODO : there is some duplicated code in this class and
    # the OpenProjectPage class

    def __init__(self, parent=None):
        super(NewProjectPage, self).__init__(parent)
        layout = Qt.QVBoxLayout(self)

        self.setTitle('X-Socs')
        self.setSubTitle('New project : select a project directory.')

        icon = XsocsIcons.getQPixmap('xsocs')
        self.setPixmap(Qt.QWizard.WatermarkPixmap, icon)
        icon = XsocsIcons.getQPixmap('logo')
        self.setPixmap(Qt.QWizard.LogoPixmap, icon)

        self.__nextId = XsocsWizard.SelectDataId
        self.__isComplete = False
        self.__selectedPath = None

        group = GroupBox('Create new project into...')
        layout.addWidget(group)

        grpLayout = Qt.QHBoxLayout(group)
        filePicker = FileChooser(fileMode=Qt.QFileDialog.Directory,
                                 appendPath=os.path.sep + 'xsocs.prj',
                                 options=Qt.QFileDialog.ShowDirsOnly)
        filePicker.sigSelectionChanged.connect(self.__filePicked)
        grpLayout.addWidget(filePicker)

    def __filePicked(self, selectedPath):
        self.__selectedPath = selectedPath

        self.__isComplete = len(selectedPath) > 0

        self.completeChanged.emit()

    def isComplete(self):
        return self.__isComplete

    def validatePage(self):
        selectedPath = self.__selectedPath
        if not selectedPath:
            return False

        if os.path.exists(selectedPath):
            buttons = Qt.QMessageBox.Yes | Qt.QMessageBox.Cancel
            ans = Qt.QMessageBox.warning(self,
                                         'Overwrite?',
                                         ('This folder already contains a'
                                          ' project.\n'
                                          'Are you sure you want to '
                                          'overwrite it?'),
                                         buttons=buttons)
            if ans == Qt.QMessageBox.Cancel:
                return False
        try:
            XsocsProject(selectedPath, mode='w')
        except Exception as ex:
            Qt.QMessageBox.critical(self, 'Failed to create file.', str(ex))
            return False
        self.wizard().projectFile = selectedPath
        return True

    filePicker = property(lambda self: self.findChild(FileChooser))

    def nextId(self):
        return self.__nextId


class OpenProjectPage(Qt.QWizardPage):
    # TODO : there is some duplicated code in this class and
    # the NewProjectPage class

    def __init__(self, parent=None):
        super(OpenProjectPage, self).__init__(parent)

        self.setTitle('X-Socs')
        self.setSubTitle('Open project : select the project file.')

        icon = XsocsIcons.getQPixmap('xsocs')
        self.setPixmap(Qt.QWizard.WatermarkPixmap, icon)
        icon = XsocsIcons.getQPixmap('logo')
        self.setPixmap(Qt.QWizard.LogoPixmap, icon)

        layout = Qt.QVBoxLayout(self)

        self.__isComplete = False
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

        self.setCommitPage(True)

    def __filePicked(self, selectedPath):
        self.__selectedPath = selectedPath

        view = self.projectSummary
        view.setProjectFile(selectedPath)

        self.__isComplete = view.isValidProject()

        self.completeChanged.emit()

    def isComplete(self):
        return self.__isComplete

    def validatePage(self):
        self.wizard().projectFile = None
        if self.projectSummary.isValidProject():
            self.wizard().projectFile = self.__selectedPath
            return True
        return False

    def nextId(self):
        return -1

    projectSummary = property(lambda self:
                              self.findChild(ProjectSummaryWidget))
