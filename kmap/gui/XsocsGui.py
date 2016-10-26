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

import os


from ..io import XsocsH5
from silx.gui import qt as Qt

from .widgets.Wizard import XsocsWizard
from .Widgets import (AcqParamsWidget,
                      AdjustedLineEdit,
                      AdjustedPushButton)
from .process.MergeWidget import MergeWidget
from .Utils import (viewWidgetFromProjectEvent,
                    processWidgetFromViewEvent,
                    processDoneEvent)
from .project.XsocsProject import XsocsProject


_COMPANY_NAME = 'ESRF'
_APP_NAME = 'XSOCS'


class XsocsGui(Qt.QMainWindow):
    __firstInitSig = Qt.Signal()

    def __init__(self,
                 parent=None,
                 workspaceH5File=None):
        super(XsocsGui, self).__init__(parent)

        mdiArea = Qt.QMdiArea()
        self.setCentralWidget(mdiArea)
        self.__createDocks()
        self.__createActions()
        self.__createMenus()
        self.__createToolBars()

        self.__startupWorkspaceH5File = workspaceH5File
        self.__widget_setup = False

        self.__firstInitSig.connect(self.__showWizard, Qt.Qt.QueuedConnection)
        self.__readSettings()

    def showEvent(self, event):
        super(XsocsGui, self).showEvent(event)
        if not self.__widget_setup:
            self.__firstInitSig.emit()
            self.__widget_setup = True

    def closeEvent(self, event):
        self.__writeSettings()
        super(XsocsGui, self).closeEvent(event)

    def __writeSettings(self):
        settings = Qt.QSettings(_COMPANY_NAME, _APP_NAME)
        settings.beginGroup("GUI")
        settings.setValue("MainWindow/size", self.size())
        settings.setValue("MainWindow/pos", self.pos())
        settings.setValue('MainWindow/state', self.saveState())
        settings.endGroup()

    def __readSettings(self):
        settings = Qt.QSettings(_COMPANY_NAME, _APP_NAME)
        settings.beginGroup("gui")
        self.resize(settings.value("MainWindow/size", Qt.QSize(400, 400)))
        self.move(settings.value("MainWindow/pos", Qt.QPoint(200, 200)))
        self.restoreState(settings.value("MainWindow/state", Qt.QByteArray()))
        settings.endGroup()

    def __showWizard(self):
        projectFile = self.__startupWorkspaceH5File
        if projectFile is None:
            wizard = XsocsWizard(parent=self)
            if wizard.exec_() == Qt.QDialog.Accepted:
                # TODO : we suppose that we get a valid file... maybe we should
                # perform some checks...
                projectFile = wizard.projectFile
                wizard.deleteLater()
            else:
                self.close()
                return
        self.__setupWorkspace(ws_file=projectFile)

    def __projectViewSlot(self, event):
        # TODO : store the plot window in a dictionary + weakref w delete
        #   callback when object is destroyed
        mdi = self.centralWidget()
        widget = viewWidgetFromProjectEvent(self.__project, event)
        if widget is None:
            print('UNKNOWN VIEW EVENT')
            return
        widget.setAttribute(Qt.Qt.WA_DeleteOnClose)
        try:
            widget.sigProcessApplied.connect(self.__processApplied)
        except AttributeError:
            pass
        mdi.addSubWindow(widget)
        widget.show()

    def __processApplied(self, event):
        widget = processWidgetFromViewEvent(self.__project, event, parent=self)
        if widget is None:
            print('UNKNOWN PROCESS EVENT')
            return
        widget.setWindowFlags(Qt.Qt.Dialog)
        widget.setWindowModality(Qt.Qt.WindowModal)
        widget.sigProcessDone.connect(self.__processDone)
        widget.setAttribute(Qt.Qt.WA_DeleteOnClose)
        widget.show()

    def __processDone(self, event):
        processDoneEvent(self.__project, event)

    def __setupWorkspace(self, ws_file=None):
        mode = 'a'
        wkSpace = XsocsProject(ws_file, mode=mode)
        self.__sessionDock.widget().setXsocsWorkspace(wkSpace)
        self.__dataDock.widget().setXsocsWorkspace(wkSpace)
        self.__project = wkSpace

        return True

    def __createActions(self):
        style = Qt.QApplication.style()
        self.__actions = actions = {}

        # # open an existing session
        # icon = style.standardIcon(Qt.QStyle.SP_DialogOpenButton)
        # openAct = Qt.QAction(icon, '&Open', self)
        # openAct.setShortcuts(Qt.QKeySequence.Open)
        # openAct.setStatusTip('Open session')
        # openAct.triggered.connect(self.__loadProject)
        # actions['open'] = openAct
        #
        # # new session
        # icon = style.standardIcon(Qt.QStyle.SP_FileDialogNewFolder)
        # newAct = Qt.QAction(icon, '&Load', self)
        # newAct.setStatusTip('Load data')
        # newAct.setShortcuts(Qt.QKeySequence.New)
        # newAct.triggered.connect(self.__loadData)
        # actions['new'] = newAct
        #
        # # import data from spec
        # icon = style.standardIcon(Qt.QStyle.SP_DialogOkButton)
        # importAct = Qt.QAction(icon, '&Import', self)
        # importAct.setStatusTip('Import SPEC data')
        # importAct.triggered.connect(self.__importData)
        # actions['import'] = importAct

        # exit the application
        icon = style.standardIcon(Qt.QStyle.SP_DialogCancelButton)
        quitAct = Qt.QAction(icon, 'E&xit', self)
        quitAct.setShortcuts(Qt.QKeySequence.Quit)
        quitAct.setStatusTip('Exit the application')
        quitAct.triggered.connect(Qt.qApp.closeAllWindows)
        actions['quit'] = quitAct

        # "about" action
        aboutAct = Qt.QAction('&About', self)
        aboutAct.setStatusTip('Show the application\'s About box')
        actions['about'] = aboutAct

        # toggle session dockwidget visibility
        sessionAct = self.__sessionDock.toggleViewAction()
        actions['sessionDock'] = sessionAct

        # toggle data dockwidget visibility
        dataAct = self.__dataDock.toggleViewAction()
        actions['dataDock'] = dataAct

    def __createMenus(self):

        self.__menus = menus = {}
        actions = self.__actions
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu('&File')
        # fileMenu.addAction(actions['open'])
        # fileMenu.addAction(actions['new'])
        # fileMenu.addAction(actions['import'])
        fileMenu.addSeparator()
        fileMenu.addAction(actions['quit'])

        viewMenu = menuBar.addMenu('&View')
        viewMenu.addAction(actions['sessionDock'])
        viewMenu.addAction(actions['dataDock'])

        menuBar.addSeparator()

        helpMenu = menuBar.addMenu('&Help')
        helpMenu.addAction(actions['about'])

        menus['file'] = fileMenu
        menus['help'] = helpMenu

    def __createToolBars(self):
        self.__toolBars = toolBars = {}
        actions = self.__actions

        # fileToolBar = self.addToolBar('File')
        # fileToolBar.setObjectName('fileToolBar')
        # fileToolBar.addAction(actions['open'])
        # fileToolBar.addAction(actions['new'])
        # fileToolBar.addAction(actions['import'])
        # toolBars[fileToolBar.windowTitle()] = fileToolBar
        #
        # viewToolBar = self.addToolBar('View')
        # viewToolBar.setObjectName('viewToolBar')
        # viewToolBar.addAction(actions['sessionDock'])
        # viewToolBar.addAction(actions['dataDock'])
        # toolBars[viewToolBar.windowTitle()] = viewToolBar

    def __createDocks(self):
        self.__sessionDock = sessionDock = Qt.QDockWidget('Infos')
        sessionDock.setWidget(SessionWidget())
        sessionDock.setObjectName('SessionDock')
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, sessionDock)

        self.__dataDock = plotDataDock = Qt.QDockWidget('Data')
        plotDataDock.setWidget(PlotDataWidget())
        plotDataDock.setObjectName('DataDock')
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, plotDataDock)
        plotDataDock.widget().sigViewEvent.connect(self.__projectViewSlot,
                                                   Qt.Qt.QueuedConnection)


# ####################################################################
# ####################################################################
# ####################################################################


class PlotDataWidget(Qt.QWidget):
    sigViewEvent = Qt.Signal(object)

    def __init__(self, parent=None):
        super(PlotDataWidget, self).__init__(parent)
        layout = Qt.QGridLayout(self)
        self.__xsocsProject = None

        self.__treeView = treeView = Qt.QTreeView()
        layout.addWidget(treeView, 0, 0)

    def setXsocsWorkspace(self, xsocsWorkspace):
        self.__xsocsProject = xsocsWorkspace
        self.__setupWidget()

    def __setupWidget(self):
        # TODO : better
        if self.__xsocsProject is not None:
            if self.__treeView:
                self.layout().takeAt(0)
                self.__treeView.deleteLater()
                self.__treeView = None
            view = self.__xsocsProject.view(parent=self)
            view.sigItemEvent.connect(self.sigViewEvent)
            self.layout().addWidget(view)
            self.__treeView = view


# ####################################################################
# ####################################################################
# ####################################################################


class SessionWidget(Qt.QWidget):
    def __init__(self, parent=None, h5_f=None):
        super(SessionWidget, self).__init__(parent)

        layout = Qt.QGridLayout(self)
        self.__xsocsH5 = None
        self.__xsocsProject = None

        # #########
        # file name
        # #########
        gbox = Qt.QGroupBox('XSocs file')
        boxLayout = Qt.QVBoxLayout(gbox)
        self.__fileLabel = fileLabel = AdjustedLineEdit(read_only=True)
        fileLabel.setAlignment(Qt.Qt.AlignLeft)
        boxLayout.addWidget(fileLabel)

        h_line = Qt.QFrame()
        h_line.setFrameShape(Qt.QFrame.HLine)
        h_line.setFrameShadow(Qt.QFrame.Sunken)
        boxLayout.addWidget(h_line)

        # misc. info
        gridLayout = Qt.QGridLayout()

        # number of angles
        rowIdx = 0
        self.__anglesText = anglesText = AdjustedLineEdit(3, read_only=True)
        gridLayout.addWidget(Qt.QLabel('# angles :'),
                             rowIdx, 0, alignment=Qt.Qt.AlignLeft)
        gridLayout.addWidget(anglesText,
                             rowIdx, 1, alignment=Qt.Qt.AlignLeft)

        rowIdx += 1
        self.__nImgText = nImgText = AdjustedLineEdit(5, read_only=True)
        gridLayout.addWidget(Qt.QLabel('Images :'),
                             rowIdx, 0, alignment=Qt.Qt.AlignLeft)
        gridLayout.addWidget(nImgText, rowIdx, 1, alignment=Qt.Qt.AlignLeft)
        self.__imgSizeText = imgSizeText = AdjustedLineEdit(10, read_only=True)
        gridLayout.addWidget(Qt.QLabel('Size :'),
                             rowIdx, 2, alignment=Qt.Qt.AlignLeft)
        gridLayout.addWidget(imgSizeText, rowIdx, 3, alignment=Qt.Qt.AlignLeft)

        gridLayout.setColumnStretch(gridLayout.columnCount(), 1)
        boxLayout.addLayout(gridLayout)

        layout.addWidget(gbox, 0, 0,
                         alignment=Qt.Qt.AlignTop)

        # ######################
        # acquisition parameters
        # ######################
        gbox = Qt.QGroupBox('Acquisition parameters')
        boxLayout = Qt.QVBoxLayout(gbox)
        self.__acqParamsWid = acqParamsWid = AcqParamsWidget(read_only=True)
        boxLayout.addWidget(acqParamsWid)
        layout.addWidget(gbox, 1, 0,
                         alignment=Qt.Qt.AlignLeft | Qt.Qt.AlignTop)

        layout.setRowStretch(layout.rowCount(), 1)

    def setXsocsWorkspace(self, xsocsWorkspace):
        self.__xsocsProject = xsocsWorkspace
        self.__setupWidget()

    def __setupWidget(self):
        acqParamsWid = self.__acqParamsWid

        xsocsH5 = self.__xsocsProject.xsocsH5

        acqParamsWid.clear()
        filename = ''
        n_entries = '0'
        n_images = '0'
        image_size = ''

        # TODO : check if file is not empty, else display popup
        if xsocsH5 is not None:
            try:
                entry = xsocsH5.get_entry_name(0)
                acqParamsWid.beam_energy = xsocsH5.beam_energy(entry)
                acqParamsWid.detector_orient = xsocsH5.detector_orient(entry)
                (acqParamsWid.direct_beam_h,
                 acqParamsWid.direct_beam_v) = xsocsH5.direct_beam(entry)
                (acqParamsWid.chperdeg_h,
                 acqParamsWid.chperdeg_v) = xsocsH5.chan_per_deg(entry)
                (acqParamsWid.pixelsize_h,
                 acqParamsWid.pixelsize_v) = xsocsH5.pixel_size(entry)
                filename = xsocsH5.filename
                n_entries = '{0}'.format(len(xsocsH5.entries()))
                n_images = '{0}'.format(xsocsH5.n_images(entry))
                image_size = '{0}'.format(xsocsH5.image_size(entry))
            except XsocsH5.InvalidEntryError:
                filename = xsocsH5.filename
                n_entries = '0'
                n_images = '0'
                image_size = ''
                
        self.__fileLabel.setText(filename)
        self.__anglesText.setText(n_entries)
        self.__nImgText.setText(n_images)
        self.__imgSizeText.setText(image_size)


# ####################################################################
# ####################################################################
# ####################################################################


class _GreeterDialog(Qt.QDialog):
    def __init__(self,
                 parent,
                 openAction,
                 loadAction,
                 importAction):
        super(_GreeterDialog, self).__init__(parent)
        self.setModal(True)
        layout = Qt.QGridLayout(self)

        openToolBn = Qt.QToolButton()
        openToolBn.setDefaultAction(openAction)
        layout.addWidget(openToolBn, 0, 0)
        layout.addWidget(Qt.QLabel('Open existing project'), 0, 1)

        loadToolBn = Qt.QToolButton()
        loadToolBn.setDefaultAction(loadAction)
        layout.addWidget(loadToolBn, 1, 0)
        layout.addWidget(Qt.QLabel('Load XSOCS data'), 1, 1)

        importToolBn = Qt.QToolButton()
        importToolBn.setDefaultAction(importAction)
        layout.addWidget(importToolBn, 2, 0)
        layout.addWidget(Qt.QLabel('Import SPEC data'), 2, 1)

        bnBox = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Cancel)
        layout.addWidget(bnBox, 3, 0, 1, 2)
        bnBox.rejected.connect(self.reject)


class _WorkspaceDirDialog(Qt.QDialog):
    def __init__(self,
                 parent=None,
                 nameHint=None,
                 **kwargs):
        super(_WorkspaceDirDialog, self).__init__(parent, **kwargs)
        self.setModal(True)
        layout = Qt.QGridLayout(self)

        label = Qt.QLabel('Workspace :')
        lineEdit = AdjustedLineEdit(40)
        lineEdit.setAlignment(Qt.Qt.AlignLeft)
        pickButton = AdjustedPushButton('...')
        layout.addWidget(label,
                         0, 0,
                         Qt.Qt.AlignLeft)
        layout.addWidget(lineEdit,
                         0, 1,
                         Qt.Qt.AlignLeft)
        layout.addWidget(pickButton,
                         0, 2,
                         Qt.Qt.AlignLeft)
        pickButton.clicked.connect(self.__pickFile)
        if nameHint is not None:
            wsPath = os.path.join(Qt.QDir.homePath(), 'xsocs', nameHint)
            lineEdit.setText(wsPath)

        bnBox = Qt.QDialogButtonBox(Qt.QDialogButtonBox.Ok |
                                     Qt.QDialogButtonBox.Cancel)
        layout.addWidget(bnBox, 1, 0, 1, 3)
        bnBox.rejected.connect(self.reject)
        bnBox.accepted.connect(self.accept)

        lineEdit.textChanged.connect(self.__textChanged)
        if len(lineEdit.text()) == 0:
            bnBox.button(Qt.QDialogButtonBox.Ok).setEnabled(False)

        self.__lineEdit = lineEdit
        self.__buttonBox = bnBox

    workspaceFile = property(lambda self: self.__lineEdit.text())

    def __pickFile(self):
        dialog = Qt.QFileDialog(self,
                                'Select workspace folder')
        dialog.setFileMode(Qt.QFileDialog.Directory)
        if dialog.exec_():
            dir_name = dialog.selectedFiles()[0]
            self.__lineEdit.setText(dir_name)

    def __textChanged(self):
        if len(self.__lineEdit.text()) == 0:
            enabled = False
        else:
            enabled = True
        self.__buttonBox.button(Qt.QDialogButtonBox.Ok).setEnabled(enabled)


if __name__ == '__main__':
    pass
