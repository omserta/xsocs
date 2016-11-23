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
    sigChildAdded = Qt.Signal(object, object)
    sigChildRemoved = Qt.Signal(object)

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

    def childAdded(self, indices, child):

        node = self.__node()
        sender = self.sender().node
        if indices is None:
            indices = []
        if node:
            # TODO : more error checking
            # check index validity
            index = node.indexOfChild(sender)
            indices.append(index)
            self.sigChildAdded.emit(indices, child)

    def childRemoved(self, data=None):

        node = self.__node()
        sender = self.sender().node
        if data is None:
            data = []
        if node:
            # TODO : more error checking
            # check index validity
            index = node.indexOfChild(sender)
            data.append(index)
            self.sigChildRemoved.emit(data)


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

    # TODO : count visible references to unload data that isn't
    # displayed anymore

    subject = property(lambda self: self.__subject()
                       if self.__subject else None)

    # TODO pass weakref?
    sigInternalDataChanged = property(lambda self:
                                      self._sigHandler.sigInternalDataChanged)
    sigChildAdded = property(lambda self:
                             self._sigHandler.sigChildAdded)
    sigChildRemoved = property(lambda self:
                               self._sigHandler.sigChildRemoved)

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
        self.setModel(model)

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
        columnCount = ModelColumns.ColumnMax
        self.__data = [{} for _ in range(columnCount)]
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

        if not isinstance(self.icons, (list, tuple)):
            self.icons = [self.icons]
        if self.icons is not None:
            style = Qt.QApplication.style()
            for iconIdx, icon in enumerate(self.icons):
                if isinstance(icon, str):
                    icon = icons.getQIcon(icon)
                elif isinstance(icon, Qt.QStyle.StandardPixmap):
                    icon = style.standardIcon(icon)
                else:
                    continue
                self._setData(iconIdx, icon, Qt.Qt.DecorationRole)

        self.nodeName = nodeName

        self._setData(ModelColumns.NameColumn,
                      self.nodeName,
                      Qt.Qt.EditRole)
        self._setData(ModelColumns.NameColumn,
                      self.nodeName,
                      Qt.Qt.DisplayRole)

        # TODO : simplify
        if self.editableColumns is None:
            self.editableColumns = False

        if not isinstance(self.editableColumns, (list, tuple)):
            editableColumns = [False] * columnCount
            editableColumns[ModelColumns.ValueColumn] = editableColumns
            self.editableColumns = tuple(editableColumns)
        elif len(self.editableColumns) < columnCount:
            diff = (columnCount - len(self.editableColumns))
            editableColumns = (tuple(self.editableColumns) +
                               (False,) * ([False] * diff))
            self.editableColumns = editableColumns
        else:
            # Copying it because it s modified later on, and we dont want to
            # share the ref amongst all instances
            # this is bad practice and has to be refactored
            self.editableColumns = self.editableColumns[:]

        # TODO : simplify
        editors = self.editors
        if editors is not None:
            if not isinstance(self.editors, (list, tuple)):
                editors = [editors]

            if len(editors) == 1:
                editor = editors[0]
                editors = [None] * columnCount
                editors[ModelColumns.ValueColumn] = editor

            for editorIdx, editor in enumerate(editors):
                if editor:
                    # TODO : not true if it s just a "paint" editor
                    self.editableColumns[editorIdx] = getattr(editor,
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
                              ModelRoles.EditorClassRole)
                self._setData(editorIdx,
                              persistent,
                              ModelRoles.PersistentEditorRole)
                self.editors = editors[:]

        for colIdx in range(ModelColumns.ColumnMax):
            self._setData(colIdx,
                          # weakref.proxy(self),
                          self,
                          ModelRoles.InternalDataRole)

        if subject is not None:
            self.setSubject(subject)

    hidden = property(lambda self: self.__hidden)

    def setModel(self, model):
        if model:
            self.__model = weakref.ref(model)
        else:
            self.__model = None

    @property
    def model(self):
        return (self.__model and self.__model()) or None

    @hidden.setter
    def hidden(self, hidden):
        self.__hidden = hidden
        # TODO : notify the view!
        # self.sigInternalDataChanged.emit([])

    def _setData(self, column, data, role):
        if self.__data[column].get(role) == data:
            return False
        self.__data[column][role] = data
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
        self.__init()
        # for column in self.activeColumns:
            # self._getModelData(column, init=True)

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
            self.__branchName = branchName
            # TODO : reset the whole branch
            self.__init()
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

    enabled = property(lambda self: self.__enabled)

    def setEnabled(self, enabled, update=True):
        self.__enabled = enabled
        for child in self._children(initialize=False):
            child.setEnabled(enabled)
        if update:
            self.sigInternalDataChanged.emit([])

    nodeName = property(lambda self: self.__nodeName)

    @nodeName.setter
    def nodeName(self, nodeName):
        self.__nodeName = nodeName
        self.setData(ModelColumns.NameColumn,
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
        # TODO : add the node directly if there is no parent!
        if self.model is None:
            self._appendChild(child)
        else:
            self.sigChildAdded.emit([self.childCount()], child)

    def _appendChild(self, child):
        self._children(append=child)

    def removeChild(self, child):
        children = self._children(initialize=False)
        try:
            childIdx = children.index(child)
        except ValueError:
            return
        self.sigChildRemoved.emit([childIdx])

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
            self._setupNode()

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
                        self._setData(col, str(value), Qt.Qt.DisplayRole)
                        notify = notify or self._setData(col, value, Qt.Qt.EditRole)
                    else:
                        notify = notify or self._setData(col, value, role)
                    if notify:
                        columns.add(col)
            else:
                self._setData(column, str(result), Qt.Qt.DisplayRole)
                notify = self._setData(column, result, Qt.Qt.EditRole)
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
        return flags | enabled

    def _childConnect(self, child):
        child.sigInternalDataChanged.connect(
            self._sigHandler.internalDataChanged)
        child.sigChildAdded.connect(
            self._sigHandler.childAdded)
        child.sigChildRemoved.connect(
            self._sigHandler.childRemoved)

    def _childDisconnect(self, child):
        try:
            child.sigInternalDataChanged.disconnect(
                self._sigHandler.internalDataChanged)
        except TypeError:
            pass
        try:
            child.sigChildAdded.disconnect(
                self._sigHandler.childAdded)
        except TypeError:
            pass

    def clear(self):
        for child in self._children(initialize=False):
            self.removeChild(child)
        self.__children = None

    def refresh(self):
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
            if data != self.__data[column].get(Qt.Qt.EditRole):
                self.__data[column][Qt.Qt.DisplayRole] =\
                    str(data)
                self.__data[column][Qt.Qt.EditRole] = data
                if self.__started:
                    self.commitModelData(column, data)
                    if self.subject and not self.subjectSignals(column):
                        self._getModelData(column)
        else:
            self.__data[column][role] = data
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
        self.setModel(parent.model)

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
        its allows this item to update itself with data from the editor.

        :param editor:
        :return:
        """
        return False