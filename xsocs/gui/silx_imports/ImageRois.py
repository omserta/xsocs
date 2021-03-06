# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""Roi items."""

from collections import OrderedDict, namedtuple

import numpy as np
from silx.gui import qt, icons

__author__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "01/09/2016"


_registeredRoiItems = OrderedDict()


def RoiItemClassDef(roiName, shape,
                    actionToolTip=None,
                    actionIcon=None,
                    actionText=None):
    def inner(cls):
        cls.roiName = roiName
        cls.shape = shape
        cls.actionToolTip = actionToolTip
        cls.actionIcon = actionIcon
        cls.actionText = actionText
        registerRoiItem(cls)
        return cls

    return inner


_RoiData = namedtuple('RoiData', ['x', 'y', 'shape'])
""" Named tuple used to return a ROI's x and y data """


def registerRoiItem(klass):
    global _registeredRoiItems

    roiName = klass.roiName
    if roiName is None:
        raise AttributeError('Failed to register Roi class {0} roiName '
                             'attribute is None.'.format(klass.__name__))

    # TODO : some kind of checks
    if roiName in _registeredRoiItems:
        raise ValueError('Cannot register roi class "{0}" :'
                         ' a ROI with the same roiName already exists.'
                         ''.format(roiName))

    # TODO : some kind of checks on the klass
    _registeredRoiItems[roiName] = klass


class RoiItemBase(qt.QObject):
    sigRoiDrawingStarted = qt.Signal(str)
    sigRoiDrawingFinished = qt.Signal(object)
    sigRoiDrawingCanceled = qt.Signal(str)
    sigRoiMoved = qt.Signal(object)

    roiName = None
    shape = None
    actionIcon = None
    actionText = None
    actionToolTip = None

    plot = property(lambda self: self._plot)
    name = property(lambda self: self._name)

    def __init__(self, plot, parent, name=None):
        super(RoiItemBase, self).__init__(parent=parent)

        self._manager = parent
        self._plot = plot
        self._handles = []
        self._items = []
        self._points = {}
        self._kwargs = []

        self._finished = False
        self._startNotified = False
        self._connected = False
        self._editing = False
        self._visible = True

        self._xData = []
        self._yData = []

        if not name:
            uuid = str(id(self))
            name = '{0}_{1}'.format(self.__class__.__name__, uuid)

        self._name = name

    def setVisible(self, visible):
        changed = self._visible == visible
        self._visible = visible
        if not visible:
            self._disconnect()
            self._remove()
        else:
            self._connect()
            self._draw(drawHandles=self._editing)

        if changed:
            self._visibilityChanged(visible)

    def _remove(self, handles=True, shape=True):
        if handles:
            for item in self._handles:
                self._plot.removeMarker(item)
        if shape:
            self._plot.removeItem(self._name)
            for item in self._items:
                self._plot.removeItem(item)

    def _interactiveModeChanged(self, source):
        if source is not self or source is not self.parent():
            self.stop()

    def _plotSignal(self, event):
        evType = event['event']
        if (evType == 'drawingFinished' and
                event['parameters']['label'] == self._name):
            self._finish(event)
        elif (evType == 'drawingProgress' and
                event['parameters']['label'] == self._name):
            if not self._startNotified:
                self._drawStarted()
                # TODO : this is a temporary workaround
                # until we can get a mouse click event
                self.sigRoiDrawingStarted.emit(self.name)
                self._startNotified = True
            self._drawEvent(event)
            self._emitDataEvent(self.sigRoiMoved)
        elif evType == 'markerMoving':
            label = event['label']
            try:
                idx = self._handles.index(label)
            except ValueError:
                idx = None
            else:
                x = event['x']
                y = event['y']
                self._setHandleData(label, (x, y))
                self._handleMoved(label, x, y, idx)
                self._emitDataEvent(self.sigRoiMoved)
                self._draw()

    def _registerHandle(self, handle, point, idx=-1):
        if handle in self._handles:
            return
        if idx is not None and idx >= 0 and idx < len(self._handles):
            self._handles.insert(handle, idx)
        else:
            self._handles.append(handle)
            idx = len(self._handles)
        self._points[handle] = point
        return idx

    def _unregisterHandle(self, label):
        try:
            self._handles.remove(label)
        except ValueError:
            pass

    def _registerItem(self, legend):
        if legend in self._items:
            raise ValueError('Item {0} is already registered.'
                             ''.format(legend))
        self._items.append(legend)

    def _unregisterItem(self, legend):
        try:
            self._items.remove(legend)
        except ValueError:
            pass

    def _connect(self):
        if self._connected:
            return

        self._plot.sigPlotSignal.connect(self._plotSignal)
        self._plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged)
        self._connected = True

    def _disconnect(self):
        if not self._connected:
            return

        self._plot.sigPlotSignal.disconnect(self._plotSignal)
        self._plot.sigInteractiveModeChanged.disconnect(
            self._interactiveModeChanged)
        self._connected = False

    def _draw(self, drawHandles=True, excludes=()):
        if drawHandles:
            if excludes is not None and len(excludes) > 0:
                draw_legends = set(self._handles) - set(excludes)
            else:
                draw_legends = self._handles
            self._drawHandles(draw_legends)

        self._drawShape()

    def _drawHandles(self, handles):
        for i_handle, handle in enumerate(handles):
            item = self._plot.addMarker(self._points[handle][0],
                                        self._points[handle][1],
                                        legend=handle,
                                        draggable=True,
                                        symbol='x',
                                        color='pink')
            assert item == handle

    def _drawShape(self):
        item = self._plot.addItem(self.xData,
                                  self.yData,
                                  shape=self.shape,
                                  legend=self._name,
                                  overlay=True,
                                  color='pink')
        assert item == self._name

    def _setHandleData(self, name, point):
        self._points[name] = point

    def start(self):
        self.edit(False)
        self._finished = False
        self._startNotified = False
        self._visible = True
        self._plot.setInteractiveMode('draw',
                                      shape=self.shape,
                                      source=self,
                                      label=self._name,
                                      color='pink')
        self._connect()

    def edit(self, enable):
        if not self._finished:
            return

        if self._editing == enable:
            return

        if enable:
            self._connect()
            self._editStarted()
            self._draw()
        else:
            self._disconnect()
            self._remove(shape=False)
            self._editStopped()
            self._draw(drawHandles=False)

        self._editing = enable

    def _finish(self, event):
        self._drawFinished(event)
        self._draw(drawHandles=False)
        self._finished = True
        self._emitDataEvent(self.sigRoiDrawingFinished)
        self._disconnect()

    def _emitDataEvent(self, signal):
        signal.emit({'name': self._name,
                     'shape': self.shape,
                     'xdata': self._xData,
                     'ydata': self._yData})

    def stop(self):
        """
        Stops whatever state the ROI is in.
        draw state : cancel the drawning, emit sigRoiDrawningCanceled
        edit state : ends the editing, emit sigRoiEditingFinished
        """
        if not self._finished:
            self._disconnect()
            self._drawCanceled()
            if self._startNotified:
                self.sigRoiDrawingCanceled.emit(self.name)
            return
        if self._editing:
            self.edit(False)

    xData = property(lambda self: self._xData[:])
    yData = property(lambda self: self._yData[:])

    def _drawEvent(self, event):
        """
        This method updates the _xData and _yData members with data found
        in the event object. The default implementation just uses the
        xdata and ydata fields from the event dictionary.
        This method should be overridden if necessary.
        """
        self._xData = event['xdata'].reshape(-1)
        self._yData = event['ydata'].reshape(-1)

    def _handleMoved(self, label, x, y, idx):
        """
        Called when one of the registered handle has moved.
        To be overridden if necessary
        """
        pass

    def _drawStarted(self):
        """
        To be overridden if necessary
        """
        pass

    def _drawFinished(self, event):
        """
        To be overridden if necessary
        """
        pass

    def _drawCanceled(self):
        """
        To be overridden if necessary
        """
        pass

    def _editStarted(self):
        """
        To be overridden if necessary
        """
        pass

    def _editStopped(self):
        """
        To be overridden if necessary
        """
        pass

    def _visibilityChanged(self, visible):
        """
        To be overridden if necessary
        """
        pass


class ImageRoiManager(qt.QObject):
    """
    Developpers doc : to add a new ROI simply append the necessary values to
    these three members
    """

    sigRoiDrawingStarted = qt.Signal(str)
    sigRoiDrawingFinished = qt.Signal(object)
    sigRoiDrawingCanceled = qt.Signal(str)
    sigRoiMoved = qt.Signal(object)
    sigRoiRemoved = qt.Signal(str)

    def __init__(self, plot, parent=None):
        super(ImageRoiManager, self).__init__(parent=parent)

        self._plot = plot

        self._klassInfos = _registeredRoiItems

        self._multipleSelection = False
        self._roiVisible = True
        self._roiInProgress = None

        self._roiActions = None
        self._optionActions = None
        self._roiActionGroup = None

        self._currentKlass = None

        self._rois = {}

        self._plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged, qt.Qt.QueuedConnection)

    def _createRoiActions(self):

        if self._roiActions:
            return self._roiActions

        # roi shapes
        self._roiActionGroup = roiActionGroup = qt.QActionGroup(self)

        self._roiActions = roiActions = OrderedDict()

        for name, klass in self._klassInfos.items():

            try:
                qIcon = icons.getQIcon(klass.actionIcon)
            except:
                qIcon = qt.QIcon()

            text = klass.actionText
            if text is None:
                text = klass.roiName

            action = qt.QAction(qIcon, text, None)
            action.setCheckable(True)
            toolTip = klass.actionToolTip
            if toolTip is not None:
                action.setToolTip(toolTip)
            roiActions[name] = action
            roiActionGroup.addAction(action)

            if klass.roiName == self._currentKlass:
                action.setChecked(True)
            else:
                action.setChecked(False)

        roiActionGroup.triggered.connect(self._roiActionTriggered,
                                         qt.Qt.QueuedConnection)

        return roiActions

    def _createOptionActions(self):

        if self._optionActions:
            return self._optionActions

        # options
        self._optionActions = optionActions = OrderedDict()

        # temporary Unicode icons until I have time to draw some icons.
        action = qt.QAction(u'\u2200', None)
        action.setCheckable(False)
        action.setToolTip('Select all [WIP]')
        action.triggered.connect(self._selectAll)
        action.setEnabled(False)
        optionActions['selectAll'] = action

        # temporary Unicode icons until I have time to draw some icons.
        action = qt.QAction(u'\u2717', None)
        action.setCheckable(False)
        action.setToolTip('Clear all ROIs')
        action.triggered.connect(self._clearRois)
        optionActions['clearAll'] = action

        action = qt.QAction(u'\u2685', None)
        action.setCheckable(True)
        action.setChecked(self._multipleSelection)
        action.setToolTip('Single/Multiple ROI selection')
        action.setText(u'\u2682' if self._multipleSelection else u'\u2680')
        action.triggered.connect(self.allowMultipleSelections)
        optionActions['multiple'] = action

        action = qt.QAction(u'\U0001F440', None)
        action.setCheckable(True)
        action.setChecked(self._roiVisible)
        action.setToolTip('Show/Hide ROI(s)')
        action.triggered.connect(self.showRois)
        optionActions['show'] = action

        return optionActions

    def _selectAll(self, checked):
        pass

    def _clearRois(self, checked):
        self.clear()

    def clear(self, name=None):
        if name is None:
            for roi in self._rois.values():
                roi.stop()
                roi.setVisible(False)
                try:
                    roi.sigRoiMoved.disconnect(self._roiMoved)
                except:
                    pass
                self.sigRoiRemoved.emit(roi.name)
            self._rois = {}
        else:
            try:
                roi = self._rois.pop(name)
                roi.stop()
                roi.setVisible(False)
                try:
                    roi.sigRoiMoved.disconnect(self._roiMoved)
                except:
                    pass
                self.sigRoiRemoved.emit(roi.name)
            except KeyError:
                # TODO : to raise or not to raise?
                pass

    rois = property(lambda self: list(self._rois.keys()))

    def showRois(self, show):
        # TODO : name param to tell that we only want to toggle
        # one specific ROI
        # TODO : exclusive param to tell that we want to show only
        # one given roi (w/ name param)
        self._roiVisible = show

        if self._optionActions:
            action = self._optionActions['show']
            action.setText(u'\U0001F440' if show else u'\u2012')
            if self.sender() != action:
                action.setChecked(show)

        for roi in self._rois.values():
            roi.setVisible(show)

    def allowMultipleSelections(self, allow):

        self._multipleSelection = allow
        if self._optionActions:
            action = self._optionActions['multiple']
            action.setText(u'\u2682' if allow else u'\u2680')
            if self.sender() != action:
                action.setChecked(allow)

    def _roiActionTriggered(self, action):

        if not action.isChecked():
            return

        name = [k for k, v in self._roiActions.items() if v == action][0]
        self._currentKlass = name
        self._startRoi()

    def _editRois(self):
        # TODO : should we call stop first?
        for item in self._rois.values():
            item.edit(True)

    def _startRoi(self):
        """
        Initialize a new roi, ready to be drawn.
        """
        if self._currentKlass not in self._klassInfos.keys():
            return

        self._stopRoi()

        for item in self._rois.values():
            item.edit(False)
        self.showRois(True)

        klass = self._klassInfos[self._currentKlass]
        item = klass(self._plot, self)

        self._roiInProgress = item

        item.sigRoiDrawingFinished.connect(self._roiDrawingFinished,
                                           qt.Qt.QueuedConnection)
        item.sigRoiDrawingStarted.connect(self._roiDrawingStarted,
                                          qt.Qt.QueuedConnection)
        item.sigRoiDrawingCanceled.connect(self._roiDrawingCanceled,
                                           qt.Qt.QueuedConnection)
        item.sigRoiMoved.connect(self.sigRoiMoved,
                                 qt.Qt.QueuedConnection)
        item.start()

    def _stopRoi(self):
        """
        Stops the roi that was ready to be drawn, if any.
        """
        if self._roiInProgress is None:
            return
        self._roiInProgress.stop()
        self._roiInProgress = None

    def _roiDrawingStarted(self, name):
        if not self._multipleSelection:
            self.clear()
        self.sigRoiDrawingStarted.emit(name)

    def _roiDrawingFinished(self, event):
        # TODO : check if the sender is the same as the roiInProgress
        item = self._roiInProgress
        assert item.name == event['name']

        self._roiInProgress = None

        item.sigRoiDrawingFinished.disconnect(self._roiDrawingFinished)
        item.sigRoiDrawingStarted.disconnect(self._roiDrawingStarted)
        item.sigRoiDrawingCanceled.disconnect(self._roiDrawingCanceled)

        self._rois[item.name] = item

        self._startRoi()

        self.sigRoiDrawingFinished.emit(event)

    def _roiDrawingCanceled(self, name):
        self.sigRoiDrawingCanceled.emit(name)

    def _interactiveModeChanged(self, source):
        """Handle plot interactive mode changed:
        If changed from elsewhere, disable tool.
        """
        if source in self._rois.values() or source == self._roiInProgress:
            pass
        elif source is not self:
            if self._roiActions:
                if self._currentKlass:
                    self._roiActions[self._currentKlass].setChecked(False)

                # specific code needed to update the zoom action
                mode = self._plot.getInteractiveMode()['mode']
                if mode == 'zoom':
                    self._roiActions['zoom'].setChecked(True)

            self._currentKlass = None
            self._stopRoi()

    roiActions = property(lambda self: self._createRoiActions().copy())

    optionActions = property(lambda self: self._createOptionActions())

    def toolBar(self,
                options=None,
                rois=None):
        """
        shapes : list
        options : list
        """
        roiActions = self._createRoiActions()
        if rois is not None:
            # this wont work if shape is a string and not an array
            diff = set(rois) - set(roiActions.keys())
            if len(diff) > 0:
                raise ValueError('Unknown roi(s) {0}.'.format(diff))
        else:
            rois = roiActions.keys()

        try:
            rois.pop('zoom')
        except:
            pass
        try:
            rois.pop('edit')
        except:
            pass

        rois = ['zoom', 'edit'] + rois

        optionActions = self._createOptionActions()
        # TODO : find a better way to ensure that the order of
        # actions returned is always the same
        optionNames = sorted(optionActions.keys())
        if options is not None:
            # this wont work if shape is a string and not an array
            diff = set(options) - set(optionNames)
            if len(diff) > 0:
                raise ValueError('Unknown options(s) {0}.'.format(diff))
            options = [option for option in optionNames
                       if option in options]
        else:
            options = optionNames

        keepRoiActions = [roiActions[roi] for roi in rois]
        keepOptionActions = [optionActions[option] for option in options]

        toolBar = qt.QToolBar('Roi')
        # toolBar.addWidget(qt.QLabel('Roi'))
        for action in keepRoiActions:
            toolBar.addAction(action)

        toolBar.addSeparator()

        for action in keepOptionActions:
            toolBar.addAction(action)

        return toolBar

    def roiData(self, name):
        item = self.roiItem(name)
        return _RoiData(x=item.xData, y=item.yData, shape=item.shape)

    def roiItem(self, name):
        if self._roiInProgress and self._roiInProgress.name == name:
            return self._roiInProgress
        else:
            try:
                return self._rois[name]
            except KeyError:
                raise ValueError('Unknown roi {0}.'.format(name))


# WARNING : if you change the name of this particular item
#  make sure to change it in ImageRoiManager::_interactiveModeChanged
#  as well, otherwise its action button will not be properly set
# if the zoom mode is set elsewhere.
@RoiItemClassDef('zoom', None,
                 actionIcon='normal',
                 actionToolTip='Zoom.')
class _DummyZoomRoiItem(RoiItemBase):
    def start(self):
        self._plot.setInteractiveMode('zoom', source=self)

    def stop(self):
        pass

    def edit(self):
        pass


@RoiItemClassDef('edit', None,
                 actionIcon='crosshair',
                 actionToolTip='Edit ROI(s)')
class _DummyEditRoiItem(RoiItemBase):
    def start(self):
        self._plot.setInteractiveMode('zoom', source=self)
        self._manager._editRois()

    def stop(self):
        pass

    def edit(self):
        pass


@RoiItemClassDef('rectangle', shape='rectangle',
                 actionIcon='shape-rectangle',
                 actionToolTip='Draw a rectangle ROI.')
class RectRoiItem(RoiItemBase):

    pos = property(lambda self: (self._left, self._bottom))
    """
    Coordinates of the lower left point.
    """

    width = property(lambda self: self._right - self._left)
    """
    Roi width.
    """

    height = property(lambda self: self._top - self._bottom)
    """
    Roi height.
    """

    center = property(lambda self:
                      (self._left + (self._right - self._left) / 2.,
                       self._bottom + (self._top - self._bottom) / 2.))
    """
    Center point of the rectangle.
    """

    def _drawFinished(self, event):
        self._drawEvent(event)
        self._updateSides()

        # initial coordinates of the rect, in that order :
        # bottom left, top left, top right, bottom right
        # which means the edges will always be (and the code assumes this) :
        #   vertical edges : [0, 1] and [2, 3]
        #   horizontal edges : [0, 3] and [1, 2]
        uuid = str(id(self))
        corners = ['C{0}_{1}'.format(idx, uuid) for idx in range(4)]
        xcoords = np.array([self._left, self._left, self._right, self._right])
        ycoords = np.array([self._bottom, self._top, self._top, self._bottom])
        opposites = [[3, 1], [2, 0], [1, 3], [0, 2]]
        rubber = 'RUBBER_{0}'.format(uuid)
        center = 'CENTER_{0}'.format(uuid)

        self._corners = corners
        self._rubber = rubber
        self._xData = xcoords
        self._yData = ycoords
        self._opposites = opposites
        self._center = center

        # note that the order of the handles is preserved
        # (this is the index sent to the handleMoved method)
        # since we re registering the corners first,
        # we will be able to use the index given to the handleMoved
        # function to get data in the _xData, _yData, ... arrays.
        # this only works because we re not adding or removing vertices
        # when editing
        for i, corner in enumerate(corners):
            self._registerHandle(corner, (xcoords[i], ycoords[i]))
        self._registerHandle(center, self.center)

    def _handleMoved(self, name, x, y, index):
        if name == self._center:
            # center moved
            c_x, c_y = self.center
            self._xData += x - c_x
            self._yData += y - c_y
            for i, corner in enumerate(self._corners):
                self._setHandleData(corner, (self._xData[i], self._yData[i]))
        else:
            # see the comment about the index value
            # (in the finished method)
            h_op, v_op = self._opposites[index]

            v_op_x = self._xData[v_op]
            h_op_y = self._xData[h_op]

            newLeft = min(x, v_op_x)
            newRight = max(x, v_op_x)
            newBottom = min(y, h_op_y)
            newTop = max(y, h_op_y)

            if newLeft != v_op_x:
                self._xData[v_op] = newLeft
            if newRight != v_op_x:
                self._xData[v_op] = newRight
            if newBottom != h_op_y:
                self._yData[h_op] = newBottom
            if newTop != h_op_y:
                self._yData[h_op] = newTop

            self._xData[index] = x
            self._yData[index] = y

            self._setHandleData(self._center, self.center)
            for i in (v_op, h_op):
                self._setHandleData(self._corners[i],
                                    (self._xData[i], self._yData[i]))

        self._updateSides()

    def _updateSides(self):
        # caching positions
        self._left = min(self._xData[1:3])
        self._right = max(self._xData[1:3])
        self._bottom = min(self._yData[0:2])
        self._top = max(self._yData[0:2])

    def _drawEvent(self, event):
        left = event['x']
        bottom = event['y']
        right = left + event['width']
        top = bottom + event['height']
        self._xData = np.array([left, left, right, right])
        self._yData = np.array([bottom, top, top, bottom])
        self._left, self._bottom, self._right, self._top = \
            left, bottom, right, top


@RoiItemClassDef('polygon', shape='polygon',
                 actionIcon='shape-polygon',
                 actionToolTip='Draw a polygon ROI.')
class PolygonRoiItem(RoiItemBase):

    def _drawFinished(self, event):
        self._xData = event['xdata'].reshape(-1)
        self._yData = event['ydata'].reshape(-1)
        points = event['points']
        uuid = str(id(self))

        # len(points) - 1 because the first and last points are the same!
        vertices = ['V{0}_{1}'.format(idx, uuid)
                    for idx in range(len(points) - 1)]
        map(self._registerHandle, vertices, points[:-1])

    def _handleMoved(self, label, x, y, idx):
        self._xData[idx] = x
        self._yData[idx] = y
        self._xData[-1] = self._xData[0]
        self._yData[-1] = self._yData[0]


if __name__ == '__main__':
    pass
