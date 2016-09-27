import os

import h5py as _h5py
import numpy as _np

from silx.gui import qt as Qt

from ..io import XsocsH5 as _XsocsH5

class XsocsWorkspace(_XsocsH5.XsocsH5Base):
    XSOCS_H5_F = '/input/xsocs_h5_f'
    GLOBAL_ENTRY = '_global'

    def __init__(self, *args, **kwargs):
        super(XsocsWorkspace, self).__init__(*args, **kwargs)
        self.__xsocs_file = None
        self.__xsocs_file = self.xsocsFile
        self.__model = None

    xsocsH5 = property(lambda self: _XsocsH5.XsocsH5(self.__xsocs_file)
                       if self.__xsocs_file else None)

    @property
    def xsocsFile(self):
        if self.__xsocs_file is None:
            return self._get_scalar_data(XsocsWorkspace.XSOCS_H5_F)
        return self.__xsocs_file

    @xsocsFile.setter
    def xsocsFile(self, xsocs_f):
        # TODO : make sure file exists and is readable
        if self.__xsocs_file is not None:
            raise ValueError('Xsocs input file is already set.')
        self.__xsocsH5 = h5f = _XsocsH5.XsocsH5(xsocs_f)
        self.__xsocs_file = xsocs_f
        self._set_scalar_data(XsocsWorkspace.XSOCS_H5_F, _np.string_(xsocs_f))
        with self._get_file() as inner_file:
            entries = h5f.entries()
            for entry in entries:
                cumul = h5f.image_cumul(entry)
                # TODO check consistency
                # TODO check if entry already exists
                # TODO compute intensity if it is not found in the file
                self._set_array_data('/input/{0}/intensity'.format(entry),
                                     cumul)
                pos_0, pos_1 = h5f.scan_positions(entry)
                self._set_array_data('/input/{0}/motor_0'.format(entry), pos_0)
                self._set_array_data('/input/{0}/motor_1'.format(entry), pos_1)
                scan_params = h5f.scan_params(entry)
                scan_params_path = '/input/{0}/scan_params/{{0}}'.format(entry)
                for param_name, param_value in scan_params.items():
                    self._set_scalar_data(scan_params_path.format(param_name),
                                          param_value)
                entry_f = h5f.entry_filename(entry)
                self._set_scalar_data('/input/{0}/file'.format(entry),
                                      _np.string_(entry_f))

                # TODO : consistency check should be made earlier
                path = '/input/_global/intensity'
                dset = inner_file.require_dataset(path,
                                                  shape=cumul.shape,
                                                  dtype=cumul.dtype)
                dset[:] += cumul
                del dset

                if not '/input/_global/motor_0' in inner_file:
                    dset = inner_file.require_dataset('/input/_global/motor_0',
                                                      shape=pos_0.shape,
                                                      dtype=pos_0.dtype)
                    dset[:] = pos_0
                    del dset
                    dset = inner_file.require_dataset('/input/_global/motor_1',
                                                      shape=pos_1.shape,
                                                      dtype=pos_1.dtype)
                    dset[:] = pos_1
                    del dset
                    scan_params_path = '/input/_global/scan_params/{0}'
                    for param_name, param_value in scan_params.items():
                        self._set_scalar_data(scan_params_path.format(param_name),
                                              param_value)

    def intensityAsScatter(self, entry):
        if entry is None:
            entry = self.GLOBAL_ENTRY
        data = self._get_array_data('/input/{0}/intensity'.format(entry))
        x = self._get_array_data('/input/{0}/motor_0'.format(entry))
        y = self._get_array_data('/input/{0}/motor_1'.format(entry))
        return (x, y, data)

    def intensityAs2d(self, entry):
        if entry is None:
            entry = self.GLOBAL_ENTRY
        x, y, data = self.intensityAsScatter(entry)
        steps_0 = self._get_scalar_data('/input/{0}/scan_params/'
                                             'motor_0_steps'.format(entry))
        steps_1 = self._get_scalar_data('/input/{0}/scan_params/'
                                             'motor_1_steps'.format(entry))
        data.shape = steps_1, steps_0
        x = x[0:steps_0]
        y = y[0::steps_0]
        return (x, y, data)

    def model(self):
        if self.__model is None:
            self.__model = XsocsQModel(self)
        return self.__model

    def input_entries(self):
        if self.__xsocs_file is not None:
            return self.xsocsH5.entries()
        return []

    def addQSpaceH5(self, qspaceH5):
        #with self._get_file() as inner_file:
            #self.add_file_link('/qspace/test', qspaceH5, '/qspace/data')
        titleItem = Qt.QStandardItem('Qspace')
        titleItem.setEditable(False)
        titleItem.setSelectable(False)
        actionItem = Qt.QStandardItem('fubar')
        actionItem.setEditable(False)
        actionItem.setSelectable(False)
        self.__model.appendRow([titleItem, actionItem])

class _DataSetDelegate(Qt.QStyledItemDelegate):

    BN_W = 20
    BN_H = 20
    V_MARGIN = 2
    H_MARGIN = 2
    BN_SPACING = 2
    N_MAX_BUTTONS = 3

    SCATTER_BN, IMG_BN, CUBE_BN = range(3)

    oneDSignal = Qt.Signal(object)
    twoDSignal = Qt.Signal(object)
    threeDSignal = Qt.Signal(object)

    def __init__(self, view, **kwargs):
        super(_DataSetDelegate, self).__init__(**kwargs)
        self.__view = view
        self.bnPressed = None
        self.mousePos = None
        self.mousePress = False
        self.mousePressIndex = None
        self.mousePressPos = None
        self.mouseReleaseIndex = None
        self.mouseReleasePos = None
        self.mousePressBn = None
        self.currentHover = None
        view.viewport().installEventFilter(self)

    def editorEvent(self, event, model, option, index):
        update = False
        evType = event.type()
        if (evType == Qt.QEvent.MouseButtonPress and
            event.button() == Qt.Qt.LeftButton):
            self.bnPressed = True
            self.mousePressIndex = Qt.QPersistentModelIndex(index)
            self.mousePressPos = event.pos()
            self.mouseReleaseIndex = None
            self.mouseReleasePos = None
            update = True
        elif (self.bnPressed and
              evType == Qt.QEvent.MouseButtonRelease and
              event.button() == Qt.Qt.LeftButton):
            self.bnPressed = False
            self.mouseReleaseIndex = Qt.QPersistentModelIndex(index)
            self.mouseReleasePos = event.pos()
            if index != self.mousePressIndex:
                self.__view.update(Qt.QModelIndex(self.mousePressIndex))
            update = True
        elif evType == Qt.QEvent.MouseMove:
            self.mousePos = event.pos()
            update = True
        return update or super(_DataSetDelegate, self).editorEvent(event,
                                                                   model,
                                                                   option,
                                                                   index)

    def eventFilter(self, obj, event):
        evType = event.type()
        # TODO : factorize both event methods
        # dirty hack to notify the item that the mouse has left the view
        # or to update the previous item that the mouse has left its
        # area
        if evType == Qt.QEvent.Leave:
            self.mousePos = None
            self.mousePressPos = None
            self.mousePressIndex = None
            self.mouseReleasePos = None
            self.mouseReleaseIndex = None
            if self.currentHover is not None:
                self.__view.update(Qt.QModelIndex(self.currentHover))
                self.currentHover = None
        elif (evType == Qt.QEvent.MouseButtonRelease and
              event.button() == Qt.Qt.LeftButton):
            index = self.__view.indexAt(event.pos())
            if not index.isValid() or index.column() != 1:
                self.mousePos = None
                self.mousePressPos = None
                self.mouseReleasePos = None
                self.mouseReleaseIndex = None
                if self.mousePressIndex is not None:
                    self.__view.update(Qt.QModelIndex(self.mousePressIndex))
                    self.mousePressIndex = None
        elif evType == Qt.QEvent.MouseMove:
            index = self.__view.indexAt(event.pos())
            if self.currentHover and self.currentHover != index:
                self.__view.update(Qt.QModelIndex(self.currentHover))
            if index.column() == 1:
                self.currentHover = Qt.QPersistentModelIndex(index)
            else:
                self.currentHover = None
        return super(_DataSetDelegate, self).eventFilter(obj, event)

    def sizeHint(self, option, index):
        plotType = index.data(role=_XsocsRoles.PLOT_TYPE_ROLE)
        if plotType:
            w = (self.N_MAX_BUTTONS * self.BN_W +
                 (self.N_MAX_BUTTONS - 1) * self.BN_SPACING +
                 2 * self.H_MARGIN)
            h = 2 * self.V_MARGIN + self.BN_H
            return Qt.QSize(w, h)
        return super(_DataSetDelegate, self).sizeHint(option, index)

    def __getButton(self, hover, enable, text, rect, index, signal):
        button = Qt.QStyleOptionButton()
        button.rect = rect
        button.text = text
        if enable:
            button.state = Qt.QStyle.State_Enabled

        if self.mouseReleaseIndex == index:
            # the left button has been released
            if (button.rect.contains(self.mousePressPos) and
                button.rect.contains(self.mouseReleasePos)):
                # the press and release was done over the same button :
                # emit the signal
                self.mouseReleasePos = None
                self.mouseReleaseIndex = None
                self.mousePressIndex = None
                self.mousePressPos = None
                signal.emit(Qt.QPersistentModelIndex(index))

            if button.rect.contains(self.mousePos):
                button.state |= Qt.QStyle.State_Raised
            
        elif not hover:
            # the mouse isnt over the cell
            if not self.bnPressed:
                # no button pressed, draw flat
                pass

            elif (self.mousePressIndex == index and
                  button.rect.contains(self.mousePressPos)):
                # a button is pressed, raising it to indicate which one it was
                button.state |= Qt.QStyle.State_Raised
        else:
            # the mouse is over this cell
            if self.mousePressIndex and self.mousePressIndex != index:
                # a button is pressed, but not in this cell
                # => draw flat
                pass
            elif self.mousePressIndex == index:
                # a button in this cell is pressed
                # (we suppose that index is never None since paint is called
                # on an existing cell -> self.mousePressIndex is not None
                if button.rect.contains(self.mousePressPos):
                    # this is the button that was pressed
                    if button.rect.contains(self.mousePos):
                        # mouse is over the button that was pressed
                        button.state |= Qt.QStyle.State_Sunken
                    else:
                        # mouse is outside the button that was pressed
                        button.state |= Qt.QStyle.State_Raised
                else:
                    # not the button that was pressed
                    pass
            else:
                # no button pressed
                if self.mousePos and button.rect.contains(self.mousePos):
                    # mouse is over this button
                    button.state |= Qt.QStyle.State_Raised
                else:
                    # mouse isnt over this button
                    pass
        return button

    def paint(self, painter, option, index):
        plotType = index.data(role=_XsocsRoles.PLOT_TYPE_ROLE)
        hover = option.state & Qt.QStyle.State_MouseOver
        if plotType:
            isScatter = True if plotType & _XsocsPlotType.SCATTER else False
            isImage = True if plotType & _XsocsPlotType.IMAGE else False
            isCube = True if plotType & _XsocsPlotType.CUBE else False

            r = option.rect
            x = r.left() + self.H_MARGIN
            y = r.top() + self.V_MARGIN
            rect = Qt.QRect(x, y, self.BN_W, self.BN_H)

            if isScatter:
                scatterBn = self.__getButton(hover,
                                             isScatter,
                                             u'\u25A9',
                                             rect,
                                             index,
                                             self.oneDSignal)
                Qt.QApplication.style().drawControl(Qt.QStyle.CE_PushButton, scatterBn, painter)
            rect.translate(self.BN_W + self.BN_SPACING, 0)

            if isImage:
                imageBn = self.__getButton(hover,
                                           isImage,
                                           u'\u25A3',
                                           rect,
                                           index,
                                           self.twoDSignal)
                Qt.QApplication.style().drawControl(Qt.QStyle.CE_PushButton, imageBn, painter)
            rect.translate(self.BN_W + self.BN_SPACING, 0)

            if isCube:
                cubeBn = self.__getButton(hover,
                                          isCube,
                                          '3d',
                                          rect,
                                          index,
                                          self.threeDSignal)
                Qt.QApplication.style().drawControl(Qt.QStyle.CE_PushButton, cubeBn, painter)

            if self.mouseReleaseIndex == index:
                self.mouseReleasePos = None
                self.mouseReleaseIndex = None
                self.mousePressIndex = None
                self.mousePressPos = None
        else:
            super(_DataSetDelegate, self).paint(painter, option, index)


class XsocsWorkspaceView(Qt.QTreeView):

    plotSig = Qt.Signal(str, object)

    def __init__(self, *args,  **kwargs):
        super(XsocsWorkspaceView, self).__init__(*args, **kwargs)
        self.header().setResizeMode(Qt.QHeaderView.ResizeToContents)
        dataSetDelegate = _DataSetDelegate(self)
        slotLambda = lambda bnName: lambda index: self.__clicked(bnName, index)
        dataSetDelegate.oneDSignal.connect(slotLambda('scatter'))
        dataSetDelegate.twoDSignal.connect(slotLambda('image'))
        dataSetDelegate.threeDSignal.connect(slotLambda('cube'))
        self.setItemDelegateForColumn(1, dataSetDelegate)
        self.viewport().setAttribute(Qt.Qt.WA_Hover, True)
        self.viewport().setMouseTracking(True)

    def __clicked(self, bnName, index):
        self.plotSig.emit(bnName, index)


class _XsocsRoles(object):
    (PLOT_TYPE_ROLE,
     ONE_D_DATA_ROLE,
     TWO_D_DATA_ROLE,
     THREE_D_DATA_ROLE) = range(Qt.Qt.UserRole + 10, Qt.Qt.UserRole + 14)


class XsocsQModel(Qt.QStandardItemModel):
    def __init__(self, xsocsWS, parent=None):
        super(XsocsQModel, self).__init__(parent=parent)
        self.setColumnCount(2)
        self.__xsocsWS = xsocsWS

        xsocs_prefix = os.path.basename(xsocsWS.xsocsFile).rsplit('.')[0]

        # input data root item
        rootItem = Qt.QStandardItem()
        rootItem.setEditable(False)
        rootItem.setData('Input', Qt.Qt.DisplayRole)
        titleItem = Qt.QStandardItem()
        titleItem.setEditable(False)
        titleItem.setData(xsocs_prefix, Qt.Qt.DisplayRole)
        self.appendRow([rootItem, titleItem])

        # global data : image intensity over all angles
        titleItem = Qt.QStandardItem('Intensity')
        titleItem.setEditable(False)
        titleItem.setSelectable(False)
        actionItem = _XsocsImageIntensityItem(xsocsWS, None)
        actionItem.setEditable(False)
        actionItem.setSelectable(False)
        rootItem.appendRow([titleItem, actionItem])

        # entry items
        entriesItem = Qt.QStandardItem()
        entriesItem.setData('Entries', Qt.Qt.DisplayRole)
        entriesItem.setEditable(False)
        n_entries = len(xsocsWS.input_entries())
        rootItem.appendRow([entriesItem, Qt.QStandardItem(str(n_entries))])

        entries = xsocsWS.xsocsH5.entries()
        for entry in entries:
            titleItem = _XsocsEntryItem(xsocsWS, entry)
            titleItem.setEditable(False)
            titleItem.setSelectable(False)
            actionItem = Qt.QStandardItem()
            actionItem.setSelectable(False)
            actionItem.setEditable(False)
            entriesItem.appendRow([titleItem, actionItem])

    def plotData(self, plotType, index):
        # TODO : check validity of index
        internalId =  Qt.QModelIndex(index).internalId()
        if plotType == 'scatter':
            data = index.data(role=_XsocsRoles.ONE_D_DATA_ROLE)
        if plotType == 'image':
            data = index.data(role=_XsocsRoles.TWO_D_DATA_ROLE)
        if plotType == 'cube':
            data = index.data(role=_XsocsRoles.TWO_D_DATA_ROLE)
        return {'type':plotType, 'data':data, 'id':internalId, 'title':''}


class _XsocsEntryItem(Qt.QStandardItem):
    def __init__(self, xsocsWS, entry):
        super(_XsocsEntryItem, self).__init__()
        self.__xsocsWS = xsocsWS
        angle = xsocsWS.xsocsH5.positioner(entry, 'eta')
        self.setToolTip('{0}'.format(angle))
        self.setData(str(angle), role=Qt.Qt.DisplayRole)
        self.setEditable(False)
        titleItem = Qt.QStandardItem('Intensity')
        titleItem.setEditable(False)
        actionItem = _XsocsImageIntensityItem(xsocsWS, entry)
        self.appendRow([titleItem, actionItem])


class _XsocsPlotType(object):
    NONE = 0
    SCATTER = 1
    IMAGE = 1 << 1
    CUBE = 1 << 2


class _XsocsPlotDataItem(Qt.QStandardItem):
    def __init__(self,
                 isScatter=False,
                 isImage=False,
                 isCube=False):
        super(_XsocsPlotDataItem, self).__init__()
        self.setEditable(False)
        self.setSelectable(False)
        plotFlags = 0
        if isScatter:
            plotFlags |= _XsocsPlotType.SCATTER
        if isImage:
            plotFlags |= _XsocsPlotType.IMAGE
        if isCube:
            plotFlags |= _XsocsPlotType.CUBE
        self.setData(plotFlags, _XsocsRoles.PLOT_TYPE_ROLE)

    def data(self, role):
        if role == _XsocsRoles.ONE_D_DATA_ROLE:
            return self.scatterData()
        if role == _XsocsRoles.TWO_D_DATA_ROLE:
            return self.imageData()
        if role == _XsocsRoles.THREE_D_DATA_ROLE:
            return self.cubeData()
        return super(_XsocsPlotDataItem, self).data(role)

    def scatterData(self):
        return None

    def imageData(self):
        return None

    def cubeData(self):
        return None


class _XsocsImageIntensityItem(_XsocsPlotDataItem):
    def __init__(self, xsocsWS, entry):
        super(_XsocsImageIntensityItem, self).__init__(isScatter=True,
                                                       isImage=True)
        self.__entry = entry
        self.__xsocsWS = xsocsWS

    def scatterData(self):
        return self.__xsocsWS.intensityAsScatter(self.__entry)

    def imageData(self):
        return self.__xsocsWS.intensityAs2d(self.__entry)


#button.initFrom(Qt.QPushButton('S'))
                #print int(button.state)
                #button.toolButtonStyle = Qt.Qt.ToolButtonIconOnly
                #button.subControls = Qt.QStyle.SC_ToolButton
                #button.icon = Qt.QIcon('/users/naudet/workspace/dau/id01/devs/kmap_sandbox/analyze/2d_icon.png')
                #button.iconSize = Qt.QSize(20, 20);
#button.features = Qt.QStyleOptionButton.DefaultButton
                #button.toolButtonStyle = Qt.Qt.ToolButtonIconOnly

                #Qt.QApplication.style().drawComplexControl(Qt.QStyle.CC_ToolButton, button, painter);



#class XsocsWorkspace(object):
    #"""
    #WARNING : no thread safety
    #"""
    ## TODO : check entries consistency, raise warning
    #XSOCS_H5_F = '/input/xsocs_h5_f'
    

    #def __init__(self, ws_h5_f, xsocsGui, mode='a'):
        #super(XsocsWorkspace, self).__init__()
        #self.__ws_h5_f = ws_h5_f
        #self.__xsocsGui = xsocsGui

        #self.__wd_dir = os.path.dirname(ws_h5_f)

        ## making sure we can create/open the files
        #with _h5py.File(ws_h5_f, mode) as ws_h5:
            #pass

        #self.__model = None

        #self.__initXsocsH5()

    #def __initXsocsH5(self, xsocs_h5_f=None):

        #if xsocs_h5_f is None:
            #xsocs_h5_f = self.xsocsH5File

        #if xsocs_h5_f is not None:
            #self.__xsocs_h5 = xsocsH5 = _XsocsH5.XsocsH5(xsocs_h5_f)
            #self.__setScalarData(XsocsWorkspace.XSOCS_H5_F,
                                 #_np.string_(xsocs_h5_f))
            ## TODO : compute the sum for each angle if it s not available
            #fullsum = xsocsH5.image_cumul(_XsocsH5.XsocsH5.TOP_ENTRY)
            #entries = xsocsH5.entries()
            #n_images = xsocsH5.n_images(entries[0])
            #fullsum = _np.data_sum = _np.zeros((n_images,),
                                               #dtype=_np.float64)
            #for entry in entries:
                #fullsum += xsocsH5.image_cumul(entry)
        #else:
            #self.__xsocs_h5 = None
        #self.__xsocs_h5_f = xsocs_h5_f

    #xsocsH5 = property(lambda self: self.__xsocs_h5)

    #@property
    #def workspaceH5(self):
        #return self.__ws_h5_f

    #@property
    #def xsocsH5File(self):
        #try:
            #return self.__getScalarData(XsocsWorkspace.XSOCS_H5_F)
        #except:
            #return None

    #@xsocsH5File.setter
    #def xsocsH5File(self, xsocs_h5_f):
        #self.__init_xsocs_h5(xsocs_h5_f)

    #def __getScalarData(self, path):
        #with _h5py.File(self.__ws_h5_f, 'r') as h5_f:
            #return h5_f.get(path, _np.array(None))[()]

    #def __setScalarData(self, path, value):
        #with _h5py.File(self.__ws_h5_f, 'a') as h5_f:
            #dset = h5_f.require_dataset(path,
                                        #shape=value.shape,
                                        #dtype=value.dtype)
            #dset[()] = value

    #def model(self):
        #if self.__model is None:
            #self.__model = XsocsQModel(self.xsocsH5)
        #return self.__model

    #def treeView(self):
        #treeView = XsocsWorkspaceView()
        #treeView.setModel(self.model())
        #treeView.expand(treeView.model().index(0, 0))
        #treeView.plotSig.connect(self.__plotSlot, Qt.Qt.QueuedConnection)
        #return treeView

    #def __plotSlot(self, plotType, index):
        #if plotType == '1D':
            #x, y, data = index.data(_XsocsRoles.ONE_D_DATA_ROLE)
            #self.__xsocsGui.showScatterPlot(x, y, data, title='')
        #elif plotType == '2D':
            #x, y, data = index.data(_XsocsRoles.TWO_D_DATA_ROLE)
            #print data.shape
            #self.__xsocsGui.showImage(x, y, data, title='')
        #elif plotType == '3D':
            #pass

    #def __getModelData(self, index, role):
        #model = self.model()
        #print model, index.isValid()
        #if model is None:
            #return None
        #if not index.isValid():
            #return None
        #return index.data(role)
