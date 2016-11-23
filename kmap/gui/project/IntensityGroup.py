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


from collections import OrderedDict
from multiprocessing import Pool, cpu_count, Manager, queues

from silx.gui import qt as Qt, icons

from ..model.Node import Node
from ..model.ModelDef import ModelColumns

from ..model.Model import Model
from ..model.Node import ModelDataList
from ..model.Nodes import ProgressBarEditor
from ..model.TreeView import TreeView
from ...io.XsocsH5 import XsocsH5

from .ProjectItem import ProjectItem
from .ProjectDef import ItemClassDef


@ItemClassDef('IntensityItem')
class IntensityItem(ProjectItem):

    @property
    def entry(self):
        return self.path.rsplit('/')[-1]

    def getScatterData(self):
        entry = self.entry
        intensity = self._get_array_data(self.path)
        scanPositions = self.projectRoot().positions(entry)
        return intensity, scanPositions


@ItemClassDef('IntensityGroup')
class IntensityGroup(ProjectItem):
    IntensityPathTpl = '{0}/{1}'

    def _createItem(self):
        path_tpl = self.IntensityPathTpl.format(self.path, '{0}')
        getIntensity(self.filename, path_tpl, self.gui)

        with self:
            entries = self.xsocsH5.entries()
            intensity = self._get_array_data(path_tpl.format(entries[0]))
            for entry in entries[1:]:
                intensity += self._get_array_data(path_tpl.format(entry))
            itemPath = self.path + '/Total'
            IntensityItem(self.filename,
                          itemPath,
                          mode=self.mode,
                          data=intensity)

    def getScatterData(self):
        entry = self.xsocsH5.entries()[0]
        entryPath = self.IntensityPathTpl.format(self.path, entry)
        intensity = self._get_array_data(entryPath)
        scanPositions = self.xsocsH5.scan_positions(entry)
        return intensity, scanPositions


def _getIntensity(entry, entry_f, projectLock, projectFile, pathTpl, queue):
    queue.put({'id': entry,
               'state': 'started',
               'done':False})
    try:
        # TODO : this works because each entry has its own separate file. Watch
        # out for errors (maybe?) if one day there is only one file for all
        # entries
        with XsocsH5(entry_f) as entryH5:
            cumul = entryH5.image_cumul(entry)
            angle = entryH5.scan_angle(entry)
        dsetPath = pathTpl.format(str(entry))

        projectLock.acquire()
        IntensityItem(projectFile, dsetPath, mode='r+', data=cumul)
        projectLock.release()
    except:
        state = 'error'
    else:
        state = 'done'

    queue.put({'id': entry,
               'state': state,
               'done':True})


def getIntensity(projectFile, pathTpl, view=None):
    xsocsH5 = ProjectItem(projectFile).xsocsH5

    with xsocsH5:
        entries = xsocsH5.entries()

    subject = ProgressSubject()
    tree = TreeView(view)
    tree.setShowUniqueGroup(True)
    model = Model()

    progressGroup = ProgressGroup(subject=subject, nodeName='Intensity')
    progressGroup.start()
    progressGroup.setEntries(entries)
    model.appendGroup(progressGroup)

    app = Qt.qApp

    mw = Qt.QDialog(view)
    mw.setModal(True)
    mw.setWindowTitle('Setting up data.')
    layout = Qt.QVBoxLayout(mw)
    tree.setModel(model)
    layout.addWidget(tree)
    mw.show()
    app.processEvents()

    manager = Manager()
    projectLock = manager.Lock()
    queue = manager.Queue()

    n_proc = cpu_count()

    pool = Pool(n_proc,
                maxtasksperchild=2)
    results = OrderedDict()

    for entry in entries:

        entry_f = xsocsH5.object_filename(entry)

        args = (entry,
                entry_f,
                projectLock,
                projectFile,
                pathTpl,
                queue,)

        results[entry] = pool.apply_async(_getIntensity,
                                          args)
    pool.close()

    while results:
        try:
            msg = queue.get(True, 0.01)
            if msg['done']:
                del results[msg['id']]
            subject.sigStateChanged.emit(msg)
        except queues.Empty:
            pass
        app.processEvents()

    pool.join()

    mw.close()
    mw.deleteLater()


class ProgressSubject(Qt.QObject):
    sigStateChanged = Qt.Signal(object)

    def __init__(self, *args, **kwargs):
        super(ProgressSubject, self).__init__(*args, **kwargs)


class ProgressGroup(Node):
    nodeName = 'Intensity'
    editors = [ProgressBarEditor]

    def subjectSignals(self, column):
        subject = self.subject
        if subject:
            return [self.subject.sigStateChanged]
        return None

    def _setupNode(self):
        self.__completed = 0

    def pullModelData(self, column, event=None, force=False):
        if event is not None:
            self.__completed += 1
        childCount = self.childCount()
        if childCount > 0:
            return int(round(100 * self.__completed / childCount))
        else:
            return 0

    def filterEvent(self, column, event):
        args = (event and event.args and event.args[0]) or None
        if (args is not None
                and args.get('state') == 'done'):
            return True, event
        return False, event

    def setEntries(self, entries):
        for entry in entries:
            self.appendChild(ProgressNode(nodeName=str(entry)))


class ProgressNode(Node):
    activeColumns = [ModelColumns.NameColumn, ModelColumns.ValueColumn]

    def filterEvent(self, column, event):
        args = (event and event.args and event.args[0]) or None
        if (args is not None
                and args.get('id') == self.nodeName):
            return True, event
        return False, event

    def subjectSignals(self, column):
        subject = self.subject
        if subject:
            return [self.subject.sigStateChanged]
        return None

    def _setupNode(self):
        style = Qt.QApplication.style()
        icon = style.standardIcon(Qt.QStyle.SP_MediaPause)
        self.setData(ModelColumns.NameColumn,
                     icon,
                     Qt.Qt.DecorationRole)
        self.setData(ModelColumns.ValueColumn,
                     'Queued',
                     Qt.Qt.DisplayRole)

    def pullModelData(self, column, event=None, force=False):
        args = (event and event.args and event.args[0]) or None

        if args is not None:
            if column == ModelColumns.NameColumn:
                return self._setProgressIcon(args['state'])
            if column == ModelColumns.ValueColumn:
                return self._setProgressText(args['state'])

        return None

    def _setProgressIcon(self, state):
        style = Qt.QApplication.style()
        if state == 'done':
            icon = style.standardIcon(Qt.QStyle.SP_DialogApplyButton)
        elif state == 'started':
            icon = style.standardIcon(Qt.QStyle.SP_BrowserReload)
        elif state == 'queued':
            icon = style.standardIcon(Qt.QStyle.SP_MediaPause)
        else:
            icon = style.standardIcon(Qt.QStyle.SP_TitleBarContextHelpButton)

        return ModelDataList(icon, None, Qt.Qt.DecorationRole)

    def _setProgressText(self, state):
        if state in ('done', 'started', 'queued'):
            text = state
        else:
            text = '?'
        return ModelDataList(text, None, Qt.Qt.DisplayRole)




        # node = index.data(ModelRoles.InternalDataRole)
        # item = HybridItem(node.projectFile, node.path)
        #
        # if item.hasScatter():
        #     icon = icons.getQIcon('item-1dim')
        #     bn = Qt.QToolButton()
        #     bn.setIcon(icon)
        #     bn.clicked.connect(
        #         partial(self.__onClicked, eventType='scatter'))
        #     layout.addWidget(bn, Qt.Qt.AlignLeft)
        # if item.hasImage():
        #     icon = icons.getQIcon('item-2dim')
        #     bn = Qt.QToolButton()
        #     bn.setIcon(icon)
        #     bn.clicked.connect(partial(self.__onClicked, eventType='image'))
        #     layout.addWidget(bn, Qt.Qt.AlignLeft)
        # layout.addStretch(1)

    # def __onClicked(self, eventType=None):
    #     persistentIndex = self.index
    #     index = persistentIndex.model().index(persistentIndex.row(),
    #                                           persistentIndex.column(),
    #                                           persistentIndex.parent())
    #     node = index.internalPointer()
    #     data = HybridItemEvent.HybridEventData(evtType=eventType,
    #                                            path=node.path)
    #     event = HybridItemEvent(self.index, data=data)
    #     self.sigEditorEvent.emit(event)

# class IntensityLoaderGroup(ProjectItem):

# path_tpl = '{0}/{1}/{{0}}'.format(self.path, AcqDataItem.DataPath)
# globalIntensityGrp = HybridItem(self.filename,
#                                 path_tpl.format('intensity'),
#                                 processLevel=ProcessId.Input)
#
# gIntensity = None
# gPos_0 = None
# gPos_1 = None
# gParams = None
# gSteps_0 = None
# gSteps_1 = None
#
# xsocs_f_prefix = os.path.basename(xsocs_f).rsplit('.')[0]
# # adding misc. data
# path_tpl = '{0}/{1}//{2}/{{0}}/{{1}}'.format(self.path,
#                                              AcqDataItem.DataPath,
#                                              AcqDataItem.EntriesPath)
# for entry in entries:
#     idx = entry.find(xsocs_f_prefix)
#     if idx == 0:
#         entry_stripped = entry[len(xsocs_f_prefix):]
#     else:
#         entry_stripped = entry
#     dataGrp = HybridItem(self.filename,
#                          path_tpl.format(entry_stripped,
#                                          'intensity'),
#                          processLevel=ProcessId.Input)
#     data = xsocsH5.image_cumul(entry)
#     pos_0, pos_1 = xsocsH5.scan_positions(entry)
#
#     # intensity as a scatter plot
#     dataGrp.setScatter(pos_0, pos_1, data)
#
#     # intensity as an image
#     scan_params = xsocsH5.scan_params(entry)
#     # xSlice = np.s_[0:scan_params['motor_0_steps']:1]
#     # ySlice = np.s_[0::scan_params['motor_0_steps']]
#     # dataGrp.setImageFromScatter(xSlice, ySlice)
#     steps_0 = scan_params['motor_0_steps']
#     steps_1 = scan_params['motor_1_steps']
#     x = np.linspace(scan_params['motor_0_start'],
#                     scan_params['motor_0_end'], steps_0, endpoint=False)
#     y = np.linspace(scan_params['motor_1_start'],
#                     scan_params['motor_1_end'], steps_1, endpoint=False)
#
#     # TODO : check overflow
#     if gIntensity is None:
#         # TODO : see if we want to keep the first entry positions,
#         # or an avg of all of them or ...
#         gIntensity = data.copy()
#         gPos_0 = pos_0
#         gPos_1 = pos_1
#         gParams = scan_params
#         gSteps_0 = steps_0
#         gSteps_1 = steps_1
#     else:
#         gIntensity += data
#
#     data = data.reshape(steps_1, steps_0)
#     dataGrp.setImage(x, y, data)
#     del dataGrp
#
# globalIntensityGrp.setScatter(gPos_0, gPos_1, gIntensity)
# x = np.linspace(gParams['motor_0_start'],
#                 gParams['motor_0_end'], gSteps_0, endpoint=False)
# y = np.linspace(gParams['motor_1_start'],
#                 gParams['motor_1_end'], gSteps_1, endpoint=False)
# gIntensity = gIntensity.reshape(gSteps_1, gSteps_0)
# globalIntensityGrp.setImage(x, y, gIntensity)
# del globalIntensityGrp
