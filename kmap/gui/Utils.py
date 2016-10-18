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

import os
import h5py

from silx.gui.hdf5 import Hdf5TreeView

from .model.ModelDef import ModelRoles
from .project.ProjectDef import ProcessId
from .project.HybridItem import HybridItem
from .process.RecipSpaceWidget import RecipSpaceWidget
from .view.RealSpaceWidget import RealSpaceWidget, RealSpaceWidgetEvent


# TODO : something a bit more... flexible
def viewWidgetFromProjectEvent(project, event):
    index = event.index
    processId = index.data(ModelRoles.XsocsProcessId)
    eventData = event.data
    print index, processId, eventData
    widgetCls = None
    xsocsType = index.data(ModelRoles.XsocsNodeType)
    if xsocsType == h5py.ExternalLink:
        widget = Hdf5TreeView()
        widget.findHdf5TreeModel().appendFile(event.data)
        return widget

    if xsocsType == 'HybridItem':
        if eventData.evtType == 'scatter':
            plotData = HybridItem(project.filename,
                                  eventData.path).getScatter()
        elif eventData.evtType == 'image':
            plotData = HybridItem(project.filename,
                                  eventData.path).getImage()
        else:
            plotData = None

    if processId == ProcessId.Input:
        widgetCls = RealSpaceWidget

    if widgetCls is not None and plotData is not None:
        widget = widgetCls(index)
        widget.setPlotData(*plotData)
        return widget

    #     # show raw data
    #     plotData = event.plotData()
    #     x, y, data = plotData
    #     widget = RealSpaceWidget(index)
    #     widget.setPlotData(x, y, data)
    # elif processLevel == XsocsProject.XsocsQSpace:
    #     # show qspace data
    #     pass
    # else:
    print('Nothing to DO')
    return None


# TODO : something better!
def nextFileName(root, template, cntMax=10000):
    template = os.path.join(root, template)
    for fIdx in range(cntMax):
        nextFile = template.format(fIdx)
        if not os.path.exists(nextFile):
            return nextFile
    else:
        raise ValueError('No available file names.')


# TODO : cache the widget to reuse previous parameters?
def processWidgetFromViewEvent(project, event, parent=None):
    widget = None
    index = event.index

    if isinstance(event, RealSpaceWidgetEvent):
        xsocsPrefix = os.path.basename(project.xsocsFile).rpartition('.')[0]
        template = '{0}_qspace_{{0:>04}}.h5'.format(xsocsPrefix)
        output_f = nextFileName(project.workdir, template)
        widget = RecipSpaceWidget(parent=parent,
                                  index=index,
                                  data_h5f=project.xsocsFile,
                                  output_f=output_f,
                                  qspace_size=None,
                                  image_binning=None,
                                  rect_roi=event.data)
    return widget
