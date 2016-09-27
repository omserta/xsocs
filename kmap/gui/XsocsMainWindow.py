#!/users/naudet/bin/python_id01
import os
import time

import h5py
import numpy as np

from silx.gui import qt as Qt
from silx.gui import plot

from matplotlib import cm

from kmap.io import XsocsH5
from kmap.gui.MergeWidget import MergeWidget
from kmap.gui.RecipSpaceWidget import RecipSpaceWidget
from kmap.gui.Widgets import (AcqParamsWidget,
                              _AdjustedLabel,
                              _AdjustedLineEdit,
                              _AdjustedPushButton)
from kmap.gui.XsocsWorkspace import XsocsWorkspace, XsocsWorkspaceView

from kmap.gui.PlotWidgets import IntensityWindow


_COMPANY_NAME = 'ESRF'
_APP_NAME = 'XSOCS'


class XsocsMainWindow(Qt.QMainWindow):
    
    __firstInitSig = Qt.Signal()

    def __init__(self,
                 parent=None,
                 workspaceH5File=None):
        super(XsocsMainWindow, self).__init__(parent)
        self.__pw = None

        mdiArea = Qt.QMdiArea()
        self.setCentralWidget(mdiArea)
        self.__createDocks()
        self.__createActions()
        self.__createMenus()
        self.__createToolBars()

        self.__startupWorkspaceH5File = workspaceH5File
        self.__widget_setup = False
        
        self.__firstInitSig.connect(self.__showGreeter, Qt.Qt.QueuedConnection)
        self.__greeterDiag = None
        self.__readSettings()

    def __plotSlot(self, plotType, index):
        # TODO : store the plot window in a dictionary + weakref w delete
        # callback when object is destroyed
        plotData = self.__workspace.model().plotData(plotType, index)
        if plotData['type'] == 'scatter':
            plotWin = self.showScatterPlot(*plotData['data'])
        elif plotData['type'] == 'image':
            plotWin = self.showImage(*plotData['data'])
        else:
            print 'Unknown plotType : {0}'.format(plotType)

    def showScatterPlot(self, x, y, data, title=''):
        mdi = self.centralWidget()
        min_, max_ = data.min(), data.max()
        colormap = cm.jet
        colors = colormap((data.astype(np.float64) - min_) / (max_ - min_))
        plotWin = IntensityWindow(aspectRatio=True)
        plotWin.setAttribute(Qt.Qt.WA_DeleteOnClose)
        plotWin.plot.addCurve(x, y, color=colors, symbol='s', linestyle='')
        plotWin.sigRoiApplied.connect(self.__roiApplied)
        mdi.addSubWindow(plotWin)
        plotWin.show()
        return plotWin

    def showImage(self, x, y, data, title=''):
        mdi = self.centralWidget()
        plotWin = IntensityWindow(aspectRatio=True)
        plotWin.setAttribute(Qt.Qt.WA_DeleteOnClose)
        min_, max_ = data.min(), data.max()
        colormap = {'name': 'temperature',
                    'normalization': 'linear',
                    'autoscale':True,
                    'vmin':min_,
                    'vmax':max_}
        origin = x[0], y[0]
        scale = (x[-1] - x[0]) / len(x), (y[-1] - y[0]) / len(y)
        plotWin.plot.addImage(data, origin=origin, scale=scale,
                              colormap=colormap)
        plotWin.sigRoiApplied.connect(self.__roiApplied)
        mdi.addSubWindow(plotWin)
        plotWin.show()
        return plotWin

    def __roiApplied(self, roiData):
        widget = RecipSpaceWidget(data_h5f=self.__workspace.xsocsFile,
                                  output_f=None,
                                  qspace_size=None,
                                  image_binning=None,
                                  rect_roi=roiData)
        widget.exec_()
        if widget.result() == Qt.QDialog.Accepted:
            self.__workspace.addQSpaceH5(widget.qspaceH5)

    def showEvent(self, event):
        super(XsocsMainWindow, self).showEvent(event)
        if not self.__widget_setup:
            self.__firstInitSig.emit()
            self.__widget_setup = True

    def closeEvent(self, event):
        self.__writeSettings()
        super(XsocsMainWindow, self).closeEvent(event)

    def __showGreeter(self):
        if self.__startupWorkspaceH5File is None:
            greeterDiag = _GreeterDialog(self.centralWidget(),
                                         self.__actions['open'],
                                         self.__actions['new'],
                                         self.__actions['import'])
            greeterDiag.setAttribute(Qt.Qt.WA_DeleteOnClose)
            #greeterDiag.setWindowFlags(Qt.Qt.WindowStaysOnTopHint |
                                       #greeterDiag.windowFlags())
            greeterDiag.rejected.connect(self.close)
            greeterDiag.show()
            self.__greeterDiag = greeterDiag
        else:
            self.__setupWorkspace(ws_file=self.__startupWorkspaceH5File)

    def __closeGreeter(self):
        if self.__greeterDiag is not None:
            self.__greeterDiag.accept()
            self.__greeterDiag = None

    def __setupWorkspace(self, ws_file=None, xsocsH5=None):
        mode = 'a'
        if ws_file is None:
            mode = 'w'
            if xsocsH5 is not None:
                nameHint = os.path.basename(xsocsH5).rsplit('.')[0]
            else:
                nameHint = None
            dialog = _WorkspaceDirDialog(self, nameHint=nameHint)
            ans = dialog.exec_()
            ws_dir = dialog.workspaceFile
            dialog.deleteLater()
            if not ans == Qt.QDialog.Accepted:
                return False
            if not os.path.exists(ws_dir):
                os.makedirs(ws_dir)
            ws_file = os.path.join(ws_dir, 'xsocs.ws')
            if os.path.exists(ws_file):
                ans = Qt.QMessageBox.warning(self,
                                             'Overwrite?',
                                             ('Workspace file already exists.'
                                              '\nDo you want to overwrite '
                                              'it?'),
                                             buttons=Qt.QMessageBox.Yes |
                                             Qt.QMessageBox.No)
            if ans == Qt.QMessageBox.No:
                return False

        wkSpace = XsocsWorkspace(ws_file, mode=mode)
        if xsocsH5 is not None:
            wkSpace.xsocsFile = xsocsH5
        self.__sessionDock.widget().setXsocsWorkspace(wkSpace)
        self.__dataDock.widget().setXsocsWorkspace(wkSpace)
        self.__workspace = wkSpace

        self.__closeGreeter()

        return True

    def __createActions(self):
        style = Qt.QApplication.style()
        self.__actions = actions = {}

        # open an existing session
        icon = style.standardIcon(Qt.QStyle.SP_DialogOpenButton)
        openAct = Qt.QAction(icon, '&Open', self)
        openAct.setShortcuts(Qt.QKeySequence.Open)
        openAct.setStatusTip('Open session')
        actions['open'] = openAct

        # new session
        icon = style.standardIcon(Qt.QStyle.SP_FileDialogNewFolder)
        newAct = Qt.QAction(icon, '&Load', self)
        newAct.setStatusTip('Load data')
        newAct.setShortcuts(Qt.QKeySequence.New)
        newAct.triggered.connect(self.__loadData)
        actions['new'] = newAct

        # import data from spec
        icon = style.standardIcon(Qt.QStyle.SP_DialogOkButton)
        importAct = Qt.QAction(icon, '&Import', self)
        importAct.setStatusTip('Import SPEC data')
        importAct.triggered.connect(self.__importData)
        actions['import'] = importAct

        # exit the application
        icon = style.standardIcon(Qt.QStyle.SP_DialogCancelButton)
        quitAct = Qt.QAction(icon, 'E&xit', self)
        quitAct.setShortcuts(Qt.QKeySequence.Quit)
        quitAct.setStatusTip('Exit the application')
        quitAct.triggered.connect(Qt.qApp.closeAllWindows)
        #connect(exitAct, SIGNAL(triggered()), qApp, SLOT(closeAllWindows()))
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
        fileMenu.addAction(actions['open'])
        fileMenu.addAction(actions['new'])
        fileMenu.addAction(actions['import'])
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

        fileToolBar = self.addToolBar('File')
        fileToolBar.setObjectName('fileToolBar')
        fileToolBar.addAction(actions['open'])
        fileToolBar.addAction(actions['new'])
        fileToolBar.addAction(actions['import'])

        viewToolBar = self.addToolBar('View')
        viewToolBar.setObjectName('viewToolBar')
        viewToolBar.addAction(actions['sessionDock'])
        viewToolBar.addAction(actions['dataDock'])

    def __createDocks(self):
        self.__sessionDock = sessionDock = Qt.QDockWidget('Infos')
        sessionDock.setWidget(SessionWidget())
        sessionDock.setObjectName('SessionDock')
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, sessionDock)

        self.__dataDock = plotDataDock = Qt.QDockWidget('Data')
        plotDataDock.setWidget(PlotDataWidget())
        plotDataDock.setObjectName('DataDock')
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, plotDataDock)
        plotDataDock.widget().plotSig.connect(self.__plotSlot,
                                              Qt.Qt.QueuedConnection)

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

    def __loadData(self, checked=None, xsocsH5=None):
        if xsocsH5 is None:
            # for some reason the exec doesnt show a modal dialog
            # nor the Qt.QFileDialog.getOpenFileName method
            dialog = Qt.QFileDialog(parent=self,
                                    caption='Select an XSOCS data file',
                                    filter=('Scan files (*.h5);;'
                                            'Any files (*)'))
            dialog.setFileMode(Qt.QFileDialog.ExistingFile)
            dialog.setModal(True)
            ans = dialog.exec_()
            selectedFiles = dialog.selectedFiles()
            dialog.deleteLater()
            if not ans:
                return False
            xsocsH5 = selectedFiles[0]

            return self.__setupWorkspace(xsocsH5=xsocsH5)
        else:
            return self.__setupWorkspace(xsocsH5=xsocsH5)

    def __importData(self):
        mw = MergeWidget(parent=self)
        result = mw.exec_()
        if result == Qt.QDialog.Accepted:
            xsocsH5 = mw.xsocsH5
            mw.deleteLater()
            self.__loadData(xsocsH5=xsocsH5)


# ####################################################################
# ####################################################################
# ####################################################################


class PlotDataWidget(Qt.QWidget):
    plotSig = Qt.Signal(str, object)

    def __init__(self, parent=None):
        super(PlotDataWidget, self).__init__(parent=parent)
        layout = Qt.QGridLayout(self)

        self.__treeView = treeView = XsocsWorkspaceView()
        layout.addWidget(treeView, 0, 0)
        treeView.plotSig.connect(self.plotSig)

    def setXsocsWorkspace(self, xsocsWorkspace):
        self.__xsocsWorkspace = xsocsWorkspace
        self.__setupWidget()

    def __setupWidget(self):
        if self.__xsocsWorkspace is not None:
            treeView = self.__treeView
            treeView.setModel(self.__xsocsWorkspace.model())
            treeView.expand(treeView.model().index(0, 0))


# ####################################################################
# ####################################################################
# ####################################################################


class SessionWidget(Qt.QWidget):
    def __init__(self, parent=None, h5_f=None):
        super(SessionWidget, self).__init__(parent=parent)

        layout = Qt.QGridLayout(self)
        self.__xsocsH5 = None

        # #########
        # file name
        # #########
        gbox = Qt.QGroupBox('XSocs file')
        boxLayout = Qt.QVBoxLayout(gbox)
        self.__fileLabel = fileLabel = _AdjustedLineEdit(read_only=True)
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
        self.__anglesText = anglesText = _AdjustedLineEdit(3, read_only=True)
        gridLayout.addWidget(Qt.QLabel('# angles :'), rowIdx, 0, alignment=Qt.Qt.AlignLeft)
        gridLayout.addWidget(anglesText, rowIdx, 1, alignment=Qt.Qt.AlignLeft)

        rowIdx +=  1
        self.__nImgText = nImgText = _AdjustedLineEdit(5, read_only=True)
        gridLayout.addWidget(Qt.QLabel('Images :'), rowIdx, 0, alignment=Qt.Qt.AlignLeft)
        gridLayout.addWidget(nImgText, rowIdx, 1, alignment=Qt.Qt.AlignLeft)
        self.__imgSizeText = imgSizeText = _AdjustedLineEdit(10, read_only=True)
        gridLayout.addWidget(Qt.QLabel('Size :'), rowIdx, 2, alignment=Qt.Qt.AlignLeft)
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
        self.__xsocsWorkspace = xsocsWorkspace
        self.__setupWidget()

    def __setupWidget(self):
        acqParamsWid = self.__acqParamsWid

        xsocsH5 = self.__xsocsWorkspace.xsocsH5

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
        super(_GreeterDialog, self).__init__(parent=parent)
        self.setModal(True)
        layout = Qt.QGridLayout(self)

        openToolBn = Qt.QToolButton()
        openToolBn.setDefaultAction(openAction)
        layout.addWidget(openToolBn, 0, 0)
        layout.addWidget(Qt.QLabel('Open existing workspace'), 0, 1)

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
        super(_WorkspaceDirDialog, self).__init__(parent=parent, **kwargs)
        self.setModal(True)
        layout = Qt.QGridLayout(self)

        label = Qt.QLabel('Workspace :')
        lineEdit = _AdjustedLineEdit(40)
        lineEdit.setAlignment(Qt.Qt.AlignLeft)
        pickButton = _AdjustedPushButton('...')
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
