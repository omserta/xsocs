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


import weakref
from collections import OrderedDict

from silx.gui import icons
from silx.gui import qt as Qt

from .ModelDef import ModelColumns, ModelRoles
from .NodeEditor import EditorMixin


class EventData(object):

    def __init__(self, signalId=None, args=(), kwargs={}):
        if len(args) == 0:
            args = (None,)

        self.signalId = signalId
        self.args = args
        self.kwargs = kwargs


class _SignalHandler(Qt.QObject):

    node = property(lambda self: self.__node())
    sigInternalDataChanged = Qt.Signal(object)

    def __init__(self, node):
        super(_SignalHandler, self).__init__()
        self.__node = weakref.ref(node)

        self.indices = None
        self.sig = None
        self.child = None

    def internalDataChanged(self, data):

        node = self.__node()
        sender = self.sender().node
        if node:
            node._childInternalDataChanged(sender, data)


class ModelDataList(object):
    def __init__(self, value, column=None, role=None):
        self.data = {(column, role): value}

    def addData(self, value, column=None, role=None):
        self.data[(column, role)] = value


class Node(object):
    className = None
    icons = None
    editors = None
    editableColumns = [False, False]
    activeColumns = [ModelColumns.ValueColumn]
    groupClasses = []
    deletable = False
    dragEnabledColumns = [None, None]
    columnCount = ModelColumns.ColumnMax
    checkable = False

    # TODO : count visible references to unload data that isn't
    # displayed anymore

    subject = property(lambda self: self.__subject()
                       if self.__subject else None)

    # TODO pass weakref?
    sigInternalDataChanged = property(lambda self:
                                      self._sigHandler.sigInternalDataChanged)

    branchName = property(lambda self: self.__branchName)

    def __init__(self,
                 subject=None,
                 nodeName=None,
                 branchName=None,
                 model=None,
                 **kwargs):
        super(Node, self).__init__()

        self.__childrenClasses = OrderedDict()

        if self.groupClasses is not None:
            for klass in self.groupClasses:
                self._appendGroupClass(klass[1], klass[0])

        self.__started = False
        self.__connected = False
        self.__model = None

        self.__loaded = False

        editors = kwargs.get('editors', None)
        if editors is not None:
            self.editors = editors

        self._sigHandler = _SignalHandler(self)

        # TODO : store subject as weakref
        if nodeName is None:
            nodeName = self.className or self.__class__.__name__

        if branchName is not None:
            self.__branchName = branchName
        else:
            self.__branchName = None

        self.__subject = None
        self.__parent = None
        self.__children = None
        self.__childCount = None
        self.__hidden = False
        # TODO : get max column from self.model()
        self.__nodeName = None
        self.__enabled = True

        self.__slots = None

        activeColumns = self.activeColumns
        if activeColumns is None:
            activeColumns = []
        elif not isinstance(activeColumns, (list, tuple)):
            activeColumns = [activeColumns]
        elif len(self.activeColumns) == 0:
            activeColumns = []
        # Copying it because it s modified later on, and we dont want to
        # share the ref amongst all instances
        # this is bad practice and has to be refactored
        self.activeColumns = activeColumns[:]

        dragEnabledColumns = self.dragEnabledColumns
        if dragEnabledColumns is None:
            dragEnabledColumns = []
        elif not isinstance(dragEnabledColumns, (list, tuple)):
            dragEnabledColumns = [dragEnabledColumns]
        elif len(self.dragEnabledColumns) == 0:
            dragEnabledColumns = []
        # Copying it because it s modified later on, and we dont want to
        # share the ref amongst all instances
        # this is bad practice and has to be refactored
        self.dragEnabledColumns = dragEnabledColumns[:]

        if not isinstance(self.icons, (list, tuple)):
            self.icons = [self.icons]

        # TODO : simplify
        if self.editableColumns is None:
            self.editableColumns = False

        self.__data = []
        self._setModel(model)

        self.nodeName = nodeName

        if subject is not None:
            self.setSubject(subject)

    hidden = property(lambda self: self.__hidden)

    def _notifyDataChange(self, column=0, role=Qt.Qt.EditRole):
        # TODO : manage role
        self.sigInternalDataChanged.emit([column])

    def blockSignals(self, block):
        # tODO : find out why block doesnt seem to work
        return self._sigHandler.blockSignals(block)

    def _setModel(self, model):
        if model:
            self.__model = weakref.ref(model)
            columnCount = max(model.columnCount(), self.columnCount)
        else:
            columnCount = self.columnCount
            self.__model = None

        currentCount = len(self.__data)
        diff = columnCount - currentCount
        if diff > 0:
            self.__data.extend([{ModelRoles.InternalDataRole: self,
                                 Qt.Qt.CheckStateRole: None}
                                for _ in range(diff)])

        # setting data for new columns
        # setting icons
        if self.icons is not None:
            style = Qt.QApplication.style()
            for iconIdx, icon in enumerate(self.icons):
                if iconIdx < currentCount:
                    # nothing to add
                    continue
                if isinstance(icon, str):
                    icon = icons.getQIcon(icon)
                elif isinstance(icon, Qt.QStyle.StandardPixmap):
                    icon = style.standardIcon(icon)
                else:
                    continue
                self._setData(iconIdx, icon,
                              Qt.Qt.DecorationRole, notify=False)

        if not isinstance(self.editableColumns, (list, tuple)):
            editableColumns = [False] * columnCount
            editableColumns[ModelColumns.ValueColumn] = editableColumns
            self.editableColumns = tuple(editableColumns)
        elif len(self.editableColumns) < columnCount:
            diff = (columnCount - len(self.editableColumns))
            editableColumns = (tuple(self.editableColumns) +
                               ((False,) * diff))
            self.editableColumns = editableColumns
        else:
            # Copying it because it s modified later on, and we dont want to
            # share the ref amongst all instances
            # this is bad practice and has to be refactored
            self.editableColumns = self.editableColumns[:]

        if not isinstance(self.dragEnabledColumns, (list, tuple)):
            dragEnabledColumns = [False] * columnCount
            dragEnabledColumns[ModelColumns.ValueColumn] = dragEnabledColumns
            self.dragEnabledColumns = tuple(dragEnabledColumns)
        elif len(self.dragEnabledColumns) < columnCount:
            diff = (columnCount - len(self.dragEnabledColumns))
            dragEnabledColumns = (tuple(self.dragEnabledColumns) +
                                  ((False,) * diff))
            self.dragEnabledColumns = dragEnabledColumns
        else:
            # Copying it because it s modified later on, and we dont want to
            # share the ref amongst all instances
            # this is bad practice and has to be refactored
            self.dragEnabledColumns = self.dragEnabledColumns[:]

        # TODO : simplify
        editors = self.editors
        if editors is not None:
            if not isinstance(self.editors, (list, tuple)):
                editors = [editors]

            if len(editors) == 1:
                editor = editors[0]
                editors = [None] * columnCount
                editors[ModelColumns.ValueColumn] = editor

            editableColumns = list(self.editableColumns)
            for editorIdx, editor in enumerate(editors):
                if editorIdx < currentCount:
                    # nothing to do
                    continue
                if editorIdx >= columnCount:
                    break
                if editor:
                    # TODO : not true if it s just a "paint" editor
                    editableColumns[editorIdx] = getattr(editor,
                                                         'editable',
                                                         True)
                    if editorIdx not in self.activeColumns:
                        self.activeColumns.append(editorIdx)
                if editor and issubclass(editor, EditorMixin):
                    persistent = editor.persistent
                else:
                    persistent = False
                self._setData(editorIdx,
                              editor,
                              ModelRoles.EditorClassRole,
                              notify=False)
                self._setData(editorIdx,
                              persistent,
                              ModelRoles.PersistentEditorRole,
                              notify=False)
                self.editors = editors[:]
            self.editableColumns = tuple(editableColumns)

    @property
    def model(self):
        return (self.__model and self.__model()) or None

    @hidden.setter
    def hidden(self, hidden):
        self.__hidden = hidden
        # TODO : notify the view!
        # self.sigInternalDataChanged.emit([])

    def _setData(self, column, data, role, notify=True):
        if self.__data[column].get(role) == data:
            return False
        self.__data[column][role] = data

        if notify:
            self.sigInternalDataChanged.emit([column])

        return True

    isStarted = property(lambda self: self.__started)

    def __init(self):
        if self.subject and self.isStarted:
            self._setupNode()
            for col in self.activeColumns:
                self._getModelData(col, force=True)

    def start(self):
        self.__started = True
        self._connect()
        for child in self._children(initialize=False):
            child.start()

    def stop(self):
        self._disconnect()
        self.__started = False
        for child in self._children(initialize=False):
            child.stop()

    @branchName.setter
    def branchName(self, branchName, propagate=True, clear=False):
        """

        :param branchName:
        :param propagate: True to propagate the branchName to the children
        :param clear: forces this node's branche to be set to branchName
        :return:
        """
        # TODO : simplify
        if (clear or branchName is not None) and self.branchName != branchName:
            self._disconnect()
            self.__branchName = branchName
            # TODO : reset the whole branch
            self._connect()
        if propagate:
            for child in self._children(initialize=False):
                child.branchName = branchName

    def setSubject(self, subject=None):
        self._disconnect()
        if isinstance(subject, weakref.ref):
            subject = subject()
        if subject is not None:
            self.__subject = weakref.ref(subject, self.setSubject)
        else:
            self.__subject = None
        # TODO : reset the whole branch
        self._connect()

    def _setupNode(self):
        """
        Called each time the subject or the branchName change, or the node
        is started, only if the subject is set and the node is started.
        :return:
        """
        pass

    def _getDepth(self):

        parent = self.parent()
        row = self.row()

        if row < 0:
            return []
        if parent:
            depth = parent._getDepth()
        else:
            depth = []

        depth.append(row)
        return depth

    def index(self):
        model = self.model
        index = Qt.QModelIndex()
        if model is None:
            return index
        depth = self._getDepth()
        for row in depth:
            index = model.index(row, ModelColumns.NameColumn, index)
        return index

    @property
    def checkState(self):
        return self.data(0, role=Qt.Qt.CheckStateRole)

    def setCheckState(self, state):
        self._setData(0, state, role=Qt.Qt.CheckStateRole)

    enabled = property(lambda self: self.__enabled)

    def setEnabled(self, enabled, update=True):
        self.__enabled = enabled
        for child in self._children(initialize=False):
            child.setEnabled(enabled)
        if update:
            self.sigInternalDataChanged.emit([0])

    nodeName = property(lambda self: self.__nodeName)

    @nodeName.setter
    def nodeName(self, nodeName):
        blocked = self.blockSignals(True)
        self.__nodeName = nodeName
        self._setData(ModelColumns.NameColumn,
                      self.__nodeName,
                      Qt.Qt.EditRole)
        self.blockSignals(blocked)
        self._setData(ModelColumns.NameColumn,
                      self.__nodeName,
                      role=Qt.Qt.DisplayRole)

    def _childInternalDataChanged(self, sender, childIndices):
        if not sender:
            return
        index = self.indexOfChild(sender)
        childIndices.append(index)
        self.sigInternalDataChanged.emit(childIndices)

    def indexOfChild(self, child):
        try:
            return self._children().index(child)
        except ValueError:
            return -1

    def appendChild(self, child):
        childCount = self.childCount()
        model = self.model
        if model:
            model._beginRowAdded(self.index(), childCount, childCount)
        self._appendChild(child)
        if model:
            model._endRowAdded()

    def _appendChild(self, child):
        self._children(append=child)

    def removeChild(self, child):
        children = self._children(initialize=False)
        try:
            childIdx = children.index(child)
        except ValueError:
            return
        model = self.model
        if model:
            model.removeRow(childIdx, self.index())

    def _removeChild(self, child):

        children = self._children(initialize=False)
        try:
            childIdx = children.index(child)
        except ValueError:
            return
        child = children.pop(childIdx)
        self._childDisconnect(child)
        child.stop()
        child._setParent(None)

    def _connect(self):
        def gen_slot(_column, _sigIdx):
            def slotfn(*args, **kwargs):
                self._getModelData(_column,
                                   event=EventData(signalId=_sigIdx,
                                                   args=args,
                                                   kwargs=kwargs))
            return slotfn

        if self.__connected:
            return
        if not self.__started:
            return
        if self.__subject is not None:
            self.__slots = {}
            for column in self.activeColumns:
                signals = self.subjectSignals(column)

                if not signals:
                    continue
                slots = []

                for sigIdx, signal in enumerate(signals):
                    slot = gen_slot(column, sigIdx)
                    signal.connect(slot)
                    slots.append((signal, slot))
                self.__slots[column] = slots
            # TODO : list of booleans
            self.__connected = True
            self.__init()

    def _disconnect(self):
        if not self.__connected:
            return
        if self.__slots:
            for slots in self.__slots.values():
                for (signal, slot) in slots:
                    try:
                        signal.disconnect(slot)
                    except TypeError:
                        pass
        # TODO : list of booleans
        self.__connected = False

    def filterEvent(self, column, event):
        return True, event

    def _getModelData(self, column, force=False, event=None):#, init=False):
        pull = True
        if event:# if not force:
            pull, event = self.filterEvent(column, event or EventData())
        if force or pull:# or init:
            result = self.pullModelData(column,
                                        event,
                                        force=force)

            if result is None:
                return
            elif isinstance(result, ModelDataList):
                notify = False
                columns = set()
                for (col, role), value in result.data.items():
                    if col is None:
                        col = column
                    if role is None:
                        self._setData(col, str(value),
                                      Qt.Qt.DisplayRole, notify=False)
                        notify = notify or self._setData(col,
                                                         value,
                                                         Qt.Qt.EditRole,
                                                         notify=False)
                    else:
                        notify = notify or self._setData(col, value,
                                                         role, notify=False)
                    if notify:
                        columns.add(col)
            else:
                self._setData(column, str(result),
                              Qt.Qt.DisplayRole, notify=False)
                notify = self._setData(column, result,
                                       Qt.Qt.EditRole, notify=False)
                columns = [column]

            if notify:
                # TODO : something better (send all columns)
                for column in columns:
                    self.sigInternalDataChanged.emit([column])

    def flags(self, column):
        flags = Qt.Qt.ItemIsSelectable
        flags = flags | ((self.isColumnEditable(column)
                          and Qt.Qt.ItemIsEditable)
                         or Qt.Qt.NoItemFlags)
        enabled = (self.enabled and Qt.Qt.ItemIsEnabled) or Qt.Qt.NoItemFlags
        draggable = (self.isColumnDragEnabled(column)
                     and Qt.Qt.ItemIsDragEnabled) or Qt.Qt.NoItemFlags
        checkable = ((self.checkable and Qt.Qt.ItemIsUserCheckable)
                     or Qt.Qt.NoItemFlags)
        return flags | enabled | draggable | checkable

    def _childConnect(self, child):
        child.sigInternalDataChanged.connect(
            self._sigHandler.internalDataChanged)

    def _childDisconnect(self, child):
        try:
            child.sigInternalDataChanged.disconnect(
                self._sigHandler.internalDataChanged)
        except TypeError:
            pass

    def clear(self):
        for child in self._children(initialize=False):
            self.removeChild(child)
        self.__children = None

    def refresh(self):
        if self.__loaded:
            self._refreshNode()
        for column in self.activeColumns:
            self._getModelData(column, force=True)
        for child in self._children(initialize=False):
            child.refresh()

    def parent(self):
        return self.__parent

    def data(self, column, role=Qt.Qt.DisplayRole):
        if column < 0 or column > len(self.__data):
            return None
        return self.__data[column].get(role)

    def setData(self, column, data, role=Qt.Qt.DisplayRole):
        # WARNING, stores a ref to the data!
        # TODO : check data type
        # TODO : notify
        if column < 0 or column > len(self.__data):
            return False

        if role == Qt.Qt.EditRole:
            # TODO : something better + in __paramChanged
            if self._setData(column, data, Qt.Qt.EditRole, notify=False):
                self._setData(column, data, Qt.Qt.DisplayRole, notify=False)
            # if data != self.__data[column].get(Qt.Qt.EditRole):
            #     self.__data[column][Qt.Qt.DisplayRole] =\
            #         str(data)
            #     self.__data[column][Qt.Qt.EditRole] = data
                if self.__started:
                    self.commitModelData(column, data)
                    if self.subject and not self.subjectSignals(column):
                        self._getModelData(column)
        else:
            # self.__data[column][role] = data
            self._setData(column, data, role)

        # self.sigInternalDataChanged.emit([column])

        return True

    def _children(self, initialize=True, append=None):
        # WARNING : it is expected that _children returns a reference to
        # the list, not a copy (see removeChild)
        children = []
        if self.__children is None:
            if initialize:
                self.__children = children = self._loadGroupClasses()
                for child in children:
                    child._setParent(self)
                self.__loaded = True
        else:
            children = self.__children

        if self.__children is not None and append is not None:
            children = self.__children
            children.append(append)
            append._setParent(self)

        return children

    def _setParent(self, parent):
        if self.__parent is not None:
            self.__parent._childDisconnect(self)

        self.__parent = parent

        if parent is not None:
            self._setModel(parent.model)

        if parent is None:
            return

        parent._childConnect(self)

        if self.branchName is None:
            self.branchName = parent.branchName

        if self.subject is None:
            self.setSubject(parent.subject)

        self.setEnabled(parent.enabled, update=False)

        if parent.isStarted:
            self.start()

    def _appendGroupClass(self, klass, name=None, subject=None):
        # TODO : check for conflicts
        if name is None:
            name = klass.className

        self.__childrenClasses[name] = (klass, subject)

    def childCount(self):
        # if self.__childrenClasses is None or len(self.__childrenClasses) == 0:
        #     return len(self._children())

        count = len(self._children())
        # if count == 0:
        #     count = len(self.__childrenClasses)
        return count

    def _loadGroupClasses(self):
        children = []
        for childName, (klass, subject) in self.__childrenClasses.items():
            if subject is None:
                subject = self.subject
            children.append(klass(subject=subject,
                                  nodeName=childName))
        others = self._loadChildren()
        if others and len(others) != 0:
            children.extend(others)
        return children

    def row(self):
        if self.__parent:
            try:
                return self.__parent._children().index(self)
            except ValueError:
                pass
        return -1

    def value(self, column):
        return self.__data[column].get(Qt.Qt.EditRole)

    def hasChildren(self):
        return self.childCount() > 0

    def child(self, index):
        children = self._children()
        if index < 0 or index >= len(children):
            return None
        return children[index]

    def pullModelData(self, column, event=None, force=False):
        return None

    def commitModelData(self, column, data):
        pass

    def subjectSignals(self, column):
        return []

    def isColumnEditable(self, column):
        return self.editableColumns[column]

    def isColumnDragEnabled(self, column):
        return self.dragEnabledColumns[column]

    def _loadChildren(self):
        return []

    def canRemove(self, parent=None):
        return False

    def getEditor(self, parent, option, index):
        """
        Returns the editor widget used to edit this item's data. The arguments
        are the one passed to the QStyledItemDelegate.createEditor method.

        :param parent:
        :param option:
        :param index:
        :return:
        """
        klass = self.data(index.column(), ModelRoles.EditorClassRole)
        if klass:
            if issubclass(klass, EditorMixin):
                return klass(parent, option, index)
            else:
                return klass(parent)
        return None

    def setEditorData(self, editor, column):
        """
        This is called by the View's delegate just before the editor is shown,
        its purpose it to setup the editors contents. Return False to use
        the delegate's default behaviour.

        :param editor:
        :return:
        """
        return False

    def _setModelData(self, editor, column):
        """
        This is called by the View's delegate just before the editor is closed,
        it allows this item to update itself with data from the editor.

        :param editor:
        :return:
        """
        return False

    def _openedEditorEvent(self, editor, column, args=None, kwargs=None):
        """
        This is called by custom editors while they're opened in the view.
        See ItemDelegate.__notifyView. Defqult implementation calls
        _setModelData on this node.

        :param editor:
        :param column:
        :param args: event's args
        :param kwargs: event's kwargs
        :return:
        """

        return self._setModelData(editor, column)

    def _refreshNode(self):
        """
        Called when Model.refresh is called, only if the node children
        have been loaded at least once (e.g : node expanded in a view)
        :return:
        """
        pass

    def sizeHint(self, column):
        return Qt.QSize()
