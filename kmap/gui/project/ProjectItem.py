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

import weakref

import h5py
import numpy as np


from .ProjectDef import getItemClass
from ...io.XsocsH5Base import XsocsH5Base
from ...io.XsocsH5 import XsocsH5


def _getIInfo(h5File, h5Path):
    attrs = dict([(key, value) for key, value in h5File[h5Path].attrs.items()])
    itemClass = h5File.get(h5Path, getclass=True)
    itemLink = h5File.get(h5Path, getclass=True, getlink=True)
    return itemClass, itemLink, attrs


def setProjectItemFactory(factory):
    global _projectItemFactory
    if factory is None:
        _projectItemFactory = projectItemFactory
    else:
        _projectItemFactory= factory


def projectItemFactory(h5File, h5Path, mode=None):
    klass = None
    with h5py.File(h5File, mode='r') as h5f:
        attrs = h5f[h5Path].attrs
        # For some reason attrs.get sometimes fails,
        # using "in" seems a bit more robust.
        if 'XsocsClass' in attrs:
            xsocsClass = attrs['XsocsClass']
            klass = getItemClass(xsocsClass)
        del attrs

    if not klass:
        klass = ProjectItem
    return klass(h5File, h5Path, mode=mode)


_projectItemFactory = projectItemFactory


class ProjectItem(XsocsH5Base):

    InputPath = '/_input'
    XsocsH5FilePath = '/_input/XsocsH5File'
    XsocsClass = None

    gui = property(lambda self: self.__gui()
                                if self.__gui is not None else None)

    def __init__(self,
                 h5File,
                 nodePath='/',
                 mode='r',
                 processLevel=None,
                 data=None,
                 gui=None):
        # TODO : check if parent already has a child with the same name
        super(ProjectItem, self).__init__(h5File, mode=mode)
        self.__nodePath = nodePath
        self.__xsocsFile = None

        self.__gui = weakref.ref(gui) if gui is not None else None

        # TODO : improve (maybe check if attributes are already present
        if nodePath == '/':
            self.__setXsocsAttributes(self.XsocsClass, processLevel)

        if self._path_exists(nodePath):
            self._loadItem()
        else:
            with self._get_file() as h5f:
                if data is not None:
                    h5f[nodePath] = data
                else:
                    h5f.require_group(nodePath)
                self.__setXsocsAttributes(self.XsocsClass,
                                          processLevel)
            self._createItem()

    @staticmethod
    def factory(h5File, h5Path):
        return _projectItemFactory(h5File, h5Path)

    def __setXsocsAttributes(self, xsocsClass, processLevel):
        if self.mode not in ('r',):
            with self._get_file() as h5f:
                item = h5f[self.path]
                if self.XsocsClass is not None\
                        and 'XsocsClass' not in item.attrs:
                    # for some reason there was an error when using
                    # attrs.get(), so I had to use "not in" instead
                    item.attrs['XsocsClass'] = np.string_(xsocsClass)
                if processLevel is not None \
                        and item.attrs.get('XsocsLevel') is None:
                    item.attrs['XsocsLevel'] = processLevel
                del item

    def children(self):
        """
        Returns this items direct children.
        :return:
        """
        children = []
        with self._get_file() as h5f:
            try:
                keys = h5f[self.path].keys()
            except AttributeError:
                return []
        pathTpl = self.path.rstrip('/') + '/{0}'
        for key in keys:
            child = self.factory(self.filename,
                                 pathTpl.format(key.lstrip('/')))
            children.append(child)
        return children

    def cast(self):
        return ProjectItem.factory(self.filename, self.path)

    def projectRoot(self):
        return ProjectItem(self.filename, '/', mode=self.mode).cast()

    def setHidden(self, hidden, path=None):
        if path is None:
            path = self.__nodePath
        self.set_attribute(path, 'XsocsHidden', hidden)

    def isHidden(self, path=None):
        if path is None:
            path = self.__nodePath
        hidden = self.attribute(path, 'XsocsHidden')
        return (hidden is not None and hidden) or False

    path = property(lambda self: self.__nodePath)

    @property
    def xsocsFile(self):
        """ The name of the input data file. """
        if self.__xsocsFile is None:
            with self._get_file() as h5f:
                path = '/' + ProjectItem.XsocsH5FilePath
                group = h5f.get(path)
                if group:
                    self.__xsocsFile = group.file.filename
                del group
        return self.__xsocsFile

    @xsocsFile.setter
    def xsocsFile(self, xsocs_f):
        """ Set the input data file for this Xsocs workspace. The input data
         file can only be set once. To use a different data file you have to
         create a new project. """
        # TODO : make sure file exists and is readable
        if self.xsocsFile is not None:
            raise ValueError('Xsocs input file is already set.')

        # Adding the external link to the file
        self.__xsocsFile = xsocs_f
        path = '/' + ProjectItem.XsocsH5FilePath
        self.add_file_link(path, xsocs_f, '/')
        self.setHidden(True, path=ProjectItem.InputPath)

        self._createItem()

    xsocsH5 = property(lambda self: XsocsH5(self.xsocsFile)
                       if self.xsocsFile else None)

    @classmethod
    def load(cls, h5File, groupPath):
        with h5py.File(h5File, 'r') as h5f:
            grp = h5f[groupPath]
            xsocsType = grp.attrs.get('XsocsType')
            del grp
            if xsocsType is None:
                return None
            klass = getItemClass(xsocsType)
            if klass is None:
                return None
        instance = klass(h5File, groupPath)
        return instance

    @property
    def processLevel(self):
        with self._get_file() as h5f:
            return h5f[self.__nodePath].attrs.get('XsocsLevel')

    @property
    def xsocsClass(self):
        return self.attribute(self.path, 'XsocsClass')

    def _createItem(self):
        """
        Called when the xsocsh5 file is succesfuly called. This should be used
        to create the hdf5 file contents.
        :return:
        """
        pass

    def _loadItem(self):
        """
        Called when the xsocsh5 file is succesfuly called. This should be used
        to load the hdf5 file contents.
        :return:
        """
        pass
