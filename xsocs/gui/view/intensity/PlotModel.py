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


from xsocs.gui.model.Node import Node
from xsocs.gui.model.TreeView import TreeView
from xsocs.gui.model.Model import Model, ModelColumns


class CurveGroup(Node):
    def _setupNode(self):
        plot = self.subject

        curveName = self.branchName

        # [x, y, legend, info, parameters]
        curveInfo = plot.getCurve(legend=curveName)[4]
        color = Qt.QColor(curveInfo['color'])
        self.setData(ModelColumns.NameColumn, color, role=Qt.Qt.DecorationRole)


class CurveListGroup(Node):
    # editor = Qt.QLabel
    className = 'Curves'
    activeColumns = [ModelColumns.NameColumn]

    def subjectSignals(self, column):
        return [self.subject.sigContentChanged]

    def filterEvent(self, column, event):
        if event is not None:
            eventArgs = event.args
            try:
                evtData = eventArgs[1]
                if evtData == 'curve':
                    return True, [eventArgs[0], eventArgs[2]]
            except Exception as ex:
                pass
        return False, event

    def pullModelData(self, column, event=None, force=False):
        # TODO : error checking

        if event is not None:
            action, curveName = event
            if action == 'add':
                # TODO : find a way to avoid init all children
                for child in self._children():
                    # TODO : findChild(name) method?
                    if child.nodeName == curveName:
                        break
                else:
                    child = CurveGroup(nodeName=curveName,
                                       branchName=curveName)
                    self.appendChild(child)
            elif action == 'remove':
                # TODO : find a way to avoid init all children
                for child in self._children():
                    # TODO : findChild(name) method?
                    if child.nodeName == curveName:
                        self.removeChild(child)
                        break

        return self.childCount()

    def _loadChildren(self):

        children = super(CurveListGroup, self)._loadChildren()
        plot = self.subject
        if plot is not None:
            legends = plot.getAllCurves(just_legend=True)
            children.extend([CurveGroup(nodeName=legend, branchName=legend)
                             for legend in legends])
        return children


class PlotTree(TreeView):
    def __init__(self, plot, **kwargs):
        super(PlotTree, self).__init__(**kwargs)

        model = Model(parent=self)
        self.setModel(model)
        group = CurveListGroup(subject=plot)
        model.appendGroup(group)
        model.startModel()
        self.setExpanded(model.index(0, 0, model.index(0, 0)), True)


if __name__ == '__main__':
    pass
