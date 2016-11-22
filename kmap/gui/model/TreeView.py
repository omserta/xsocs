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

from .ModelDef import ModelColumns, ModelRoles


class TreeView(Qt.QTreeView):

    def __init__(self, parent=None):
        super(TreeView, self).__init__(parent)
        delegate = ItemDelegate(self)
        self.setItemDelegateForColumn(ModelColumns.NameColumn, delegate)
        delegate = ItemDelegate(self)
        self.setItemDelegateForColumn(ModelColumns.ValueColumn, delegate)
        # WARNING : had to set this as a queued connection, otherwise
        # there was a crash after the slot was called (conflict with
        # __setHiddenNodes probably)
        # TODO : investigate
        self.expanded.connect(self.__expanded, Qt.Qt.QueuedConnection)
        self.collapsed.connect(self.__collapsed)
        self.header().setResizeMode(Qt.QHeaderView.ResizeToContents)
        self.__showUniqueGroup = False
        self.setSelectionBehavior(Qt.QAbstractItemView.SelectRows)

    showUniqueGroup = property(lambda self: self.__showUniqueGroup)

    def keyReleaseEvent(self, event):
        # TODO : better filtering
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Qt.Key_Delete and modifiers == Qt.Qt.NoModifier:
            model = self.model()
            selected = self.selectedIndexes()
            nodes = []
            for index in selected:
                leftIndex = model.sibling(index.row(), 0, index)
                leftNode = leftIndex.data(ModelRoles.InternalDataRole)
                if leftNode not in nodes:
                    nodes.append(leftNode)

            self.selectionModel().clear()

            for node in nodes:
                if node.canRemove(self):
                    parentNode = node.parent()
                    parentNode.removeChild(node)

        super(TreeView, self).keyReleaseEvent(event)

    def setShowUniqueGroup(self, show):
        self.__showUniqueGroup = show
        self.__openPersistentEditors(Qt.QModelIndex(), openEditor=False)
        self.__updateUniqueGroupVisibility()
        self.__openPersistentEditors(Qt.QModelIndex(), openEditor=True)

    def __updateUniqueGroupVisibility(self):
        model = self.model()
        if model and model.rowCount() == 1 and not self.__showUniqueGroup:
            index = model.index(0, 0)
            self.setRootIndex(index)
            self.__setHiddenNodes(index)
        else:
            self.setRootIndex(Qt.QModelIndex())

    def rowsInserted(self, index, start, end):
        super(TreeView, self).rowsInserted(index, start, end)
        self.__updateUniqueGroupVisibility()
        self.__setHiddenNodes(index)
        self.__openPersistentEditors(index, True)

    def __openPersistentEditors(self, parent, openEditor=True):
        model = self.model()

        if not model:
            return

        if openEditor:
            meth = self.openPersistentEditor
        else:
            meth = self.closePersistentEditor

        columnCount = self.model().columnCount()

        if not parent.isValid():
            parent = self.rootIndex()

        children = []
        if self.isExpanded(parent)\
                or parent == self.rootIndex()\
                or not parent.isValid():
            children = [model.index(row, 0, parent)
                        for row in range(model.rowCount(parent))]

        while len(children) > 0:
            curParent = children.pop(-1)

            if self.isExpanded(curParent):
                children.extend([model.index(row, 0, curParent)
                                 for row in range(model.rowCount(curParent))])

            for colIdx in range(columnCount):
                sibling = model.sibling(curParent.row(), colIdx, curParent)
                node = sibling.data(ModelRoles.InternalDataRole)

                if node and node.hidden:
                    continue
                persistent = sibling.data(ModelRoles.PersistentEditorRole)
                if persistent or not openEditor:
                    meth(sibling)

    def setModel(self, model):
        if self.model():
            try:
                self.model().sigRowsRemoved.disconnect(self.rowsRemoved)
            except TypeError:
                pass
        super(TreeView, self).setModel(model)
        self.__updateUniqueGroupVisibility()
        self.__setHiddenNodes(model.index(0, 0))
        self.__openPersistentEditors(Qt.QModelIndex())

        try:
            self.model().sigRowsRemoved.connect(self.rowsRemoved)
        except TypeError:
            pass

    def dataChanged(self, topLeft, bottomRight, roles=None):
        super(TreeView, self).dataChanged(topLeft, bottomRight)
        # TODO : only way i found to force the view to check the flags
        # (in case the enable/disable state has changed)
        self.viewport().update()

    def __setHiddenNodes(self, index):
        if not index.isValid():
            index = self.rootIndex()
        if not self.isExpanded(index) and index != self.rootIndex():
            return
        model = self.model()
        for row in range(model.rowCount(index)):
            childNode = index.child(row, 0).data(ModelRoles.InternalDataRole)
            if childNode and childNode.hidden:
                self.setRowHidden(row, index, True)

    def __expanded(self, index):
        self.__setHiddenNodes(index)
        self.__openPersistentEditors(index, True)

    def __collapsed(self, index):
        self.__openPersistentEditors(index, False)


class ItemDelegate(Qt.QStyledItemDelegate):
    sigDelegateEvent = Qt.Signal(object)

    def __init__(self, parent=None):
        super(ItemDelegate, self).__init__(parent)

    def createEditor(self, parent, option, index):
        node = index.data(role=ModelRoles.InternalDataRole)
        if node:
            editor = node.getEditor(parent, option, index)
            if editor:
                return editor
        return super(ItemDelegate, self).createEditor(parent, option, index)

    def _editorValueChanged(self, *args, **kwargs):
        self.commitData.emit(self.sender())

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def setEditorData(self, editor, index):
        node = index.data(role=ModelRoles.InternalDataRole)
        if node and node.setEditorData(editor, index.column()):
            return
        super(ItemDelegate, self).setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        node = index.data(role=ModelRoles.InternalDataRole)
        if node:
            if node._setModelData(editor, index.column()):
                return
            node.setData(index.column(),
                         editor.getEditorData(),
                         role=Qt.Qt.EditRole)
        super(ItemDelegate, self).setModelData(editor, model, index)

    def paint(self, painter, option, index):
        klass = index.data(role=ModelRoles.EditorClassRole)
        if klass is not None:
            if (not getattr(klass, 'persistent', False) and
                    klass.paint(painter, option, index)):
                return
        super(ItemDelegate, self).paint(painter, option, index)
