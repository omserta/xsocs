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


from silx.gui import qt as Qt
from .ProjectModel import nodeEditorFromIndex
from .ModelDef import ModelRoles


class ProjectView(Qt.QTreeView):
    sigItemEvent = Qt.Signal(object)

    def __init__(self, parent=None):
        super(ProjectView, self).__init__(parent)
        delegate = ItemDelegate(self)
        self.setItemDelegateForColumn(1, delegate)
        delegate.sigDelegateEvent.connect(self.sigItemEvent)
        self.expanded.connect(self.__expanded)
        self.header().setResizeMode(Qt.QHeaderView.ResizeToContents)
        # self.collapsed.connect(self.__collapsed)

    def __expanded(self, index):
        # TODO : closePersistentEditor when node is hidden
        start = self.model().index(0, 0, parent=index)
        indices = self.model().match(start,
                                     ModelRoles.IsXsocsNodeRole,
                                     True,
                                     hits=-1,
                                     flags=Qt.Qt.MatchExactly)
        for index in indices:
            # had to do this otherwise the openPersistentEditor wouldnt work
            idx = self.model().index(index.row(), 1, start.parent())
            self.openPersistentEditor(idx)

    # def __collapsed(self, index):
    #     start = self.model().index(0, 0, parent=index)
    #     indices = self.model().match(start,
    #                                  33,
    #                                  True,
    #                                  hits=-1,
    #                                  flags=(Qt.Qt.MatchExactly |
    #                                     Qt.Qt.MatchRecursive))
    #     for index in indices:
    #         # had to do this otherwise the openPersistentEditor wouldnt work
    #         idx = self.model().index(index.row(), 1, start.parent())
    #         self.closePersistentEditor(idx)


class ItemDelegate(Qt.QStyledItemDelegate):
    sigDelegateEvent = Qt.Signal(object)

    def __init__(self, parent=None):
        super(ItemDelegate, self).__init__(parent)

    def createEditor(self, parent, option, index):
        editorClass = nodeEditorFromIndex(index)
        if editorClass is not None:
            editor = editorClass(parent, option, index)
            editor.sigEditorEvent.connect(self.sigDelegateEvent)
            return editor
        return super(ItemDelegate, self).createEditor(parent,
                                                      option,
                                                      index)
