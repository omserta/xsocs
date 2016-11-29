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

from silx.gui import qt as Qt

from .widgets.Wizard import XsocsWizard
from .Widgets import (AcqParamsWidget,
                      AdjustedLineEdit)
# from .Utils import (viewWidgetFromProjectEvent,
#                     processWidgetFromViewEvent,
#                     processDoneEvent)
from .process.RecipSpaceWidget import RecipSpaceWidget
from .process.FitWidget import FitWidget
from .view.IntensityView import IntensityView
from .view.QspaceView import QSpaceView
from .view.FitView import FitView
from .project.XsocsProject import XsocsProject
from .model.TreeView import TreeView
from .model.Model import Model
from .project.IntensityGroup import IntensityGroup
from .project.QSpaceGroup import QSpaceItem
from .project.FitGroup import FitItem
from .project.XsocsH5Factory import XsocsH5Factory, h5NodeToProjectItem
from .project.Hdf5Nodes import setH5NodeFactory, H5File
from .Utils import nextFileName
from .model.ModelDef import ModelRoles


_COMPANY_NAME = 'ESRF'
_APP_NAME = 'XSOCS'


class ProjectTree(TreeView):
    sigDelegateEvent = Qt.Signal(object, object)

    def delegateEvent(self, column, node, *args, **kwargs):
        # TODO : proper event
        event = (args and args[0]) or None
        self.sigDelegateEvent.emit(node, event)


class XsocsGui(Qt.QMainWindow):
    __firstInitSig = Qt.Signal()

    def __init__(self,
                 parent=None,
                 projectH5File=None):
        super(XsocsGui, self).__init__(parent)

        setH5NodeFactory(XsocsH5Factory)
        self.move(0, 0)
        self.resize(300, 600)

        self.__intensityView = None
        self.__qspaceViews = {}
        self.__fitViews = {}

        self.__createViews()
        self.__createActions()
        self.__createMenus()
        # self.__createToolBars()

        self.__startupprojectH5File = projectH5File
        self.__widget_setup = False

        self.__firstInitSig.connect(self.__showWizard, Qt.Qt.QueuedConnection)
        self.__readSettings()

    def __createViews(self):
        tree = ProjectTree()
        tree.setShowUniqueGroup(False)
        tree.sigDelegateEvent.connect(self.__viewEvent)
        model = Model(parent=tree)
        model.startModel()
        tree.setModel(model)
        self.setCentralWidget(tree)

    def __viewEvent(self, node, event):
        projectItem = h5NodeToProjectItem(node)
        if projectItem is None:
            raise ValueError('Unknwown event for node {0} : {1}.'
                             ''.format(node, event))

        if isinstance(projectItem, IntensityGroup)\
                and event['event'] == 'scatter':
            self.__showIntensity(node)
        elif isinstance(projectItem, QSpaceItem)\
                and event['event'] == 'qspace':
            self.__showQSpace(node)
        elif isinstance(projectItem, FitItem)\
                and event['event'] == 'fit':
            self.__showFit(node)
        else:
            ValueError('Unknwown event for item {0} : {1}.'
                       ''.format(projectItem, event))

    def __showIntensity(self, node):
        view = self.__intensityView
        if not view:
            self.__intensityView = view = IntensityView(self,
                                                        model=node.model,
                                                        node=node)
            screen = Qt.QApplication.desktop()
            size = screen.availableGeometry(view).size()
            size.scale(size.width() * 0.6,
                       size.height() * 0.6,
                       Qt.Qt.IgnoreAspectRatio)
            view.resize(size)
            view.sigProcessApplied.connect(self.__intensityRoiApplied)
        view.show()
        view.raise_()

    def __intensityRoiApplied(self, event):
        xsocsFile = os.path.basename(self.__project.xsocsFile)
        xsocsPrefix = xsocsFile.rpartition('.')[0]
        template = '{0}_qspace_{{0:>04}}.h5'.format(xsocsPrefix)
        output_f = nextFileName(self.__project.workdir, template)
        widget = RecipSpaceWidget(parent=self.sender(),
                                  data_h5f=self.__project.xsocsFile,
                                  output_f=output_f,
                                  qspace_size=None,
                                  image_binning=None,
                                  rect_roi=event)
        widget.exec_()
        if widget.status == RecipSpaceWidget.StatusCompleted:
            qspaceF = widget.qspaceH5
            qspaceGroup = self.__project.qspaceGroup()
            qspaceItem = qspaceGroup.addQSpace(qspaceF)
            self.model().refresh()
            index = self.tree.pathToIndex(qspaceItem.path)
            if index.isValid():
                self.tree.setCurrentIndex(index)
                self.__showQSpace(index.data(ModelRoles.InternalDataRole))
        widget.deleteLater()

    tree = property(lambda self: self.centralWidget())

    def __showQSpace(self, node):
        view = self.__qspaceViews.get(node)
        if not view:
            view = QSpaceView(self, model=node.model, node=node)
            self.__qspaceViews[node] = view
            screen = Qt.QApplication.desktop()
            size = screen.availableGeometry(view).size()
            size.scale(size.width() * 0.6,
                       size.height() * 0.6,
                       Qt.Qt.IgnoreAspectRatio)
            view.resize(size)
            view.sigProcessApplied.connect(self.__qspaceRoiApplied)
        view.show()
        view.raise_()

    def __qspaceRoiApplied(self, node, roi):
        item = h5NodeToProjectItem(node)
        xsocsFile = os.path.basename(self.__project.xsocsFile)
        xsocsPrefix = xsocsFile.rpartition('.')[0]
        template = '{0}_fit_{{0:>04}}.h5'.format(xsocsPrefix)
        output_f = nextFileName(self.__project.workdir, template)
        fitWidget = FitWidget(item.qspaceFile,
                              output_f,
                              roi,
                              parent=self.sender())
        fitWidget.exec_()
        if fitWidget.status == FitWidget.StatusCompleted:
            fitFile = fitWidget.fitFile
            fitGroup = item.fitGroup()
            fitItem = fitGroup.addFitFile(fitFile)
            self.model().refresh()
            index = self.tree.pathToIndex(fitItem.path)
            if index.isValid():
                self.tree.setCurrentIndex(index)
                self.__showFit(index.data(ModelRoles.InternalDataRole))
        fitWidget.deleteLater()

    def __showFit(self, node):
        view = self.__fitViews.get(node)
        if not view:
            view = FitView(self, model=node.model, node=node)
            self.__fitViews[node] = view
        view.show()

    def model(self):
        return self.tree.model()

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
        projectFile = self.__startupprojectH5File
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
        self.__setupProject(projectFile=projectFile)

    def __setupProject(self, projectFile=None):
        mode = 'r+'
        project = XsocsProject(projectFile, mode=mode)
        rootNode = H5File(h5File=projectFile)
        self.tree.model().appendGroup(rootNode)
        self.__project = project

        return True

    def __createActions(self):
        style = Qt.QApplication.style()
        self.__actions = actions = {}

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

    def __createMenus(self):

        self.__menus = menus = {}
        actions = self.__actions
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu('&File')
        fileMenu.addSeparator()
        fileMenu.addAction(actions['quit'])

        menuBar.addSeparator()

        helpMenu = menuBar.addMenu('&Help')
        helpMenu.addAction(actions['about'])

        menus['file'] = fileMenu
        menus['help'] = helpMenu


# ####################################################################
# ####################################################################
# ####################################################################


if __name__ == '__main__':
    pass
