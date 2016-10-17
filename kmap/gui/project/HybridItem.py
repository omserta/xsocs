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

import h5py

from .ProjectItem import ProjectItem
from .ProjectDef import ItemClassDef


@ItemClassDef('HybridItem')
class HybridItem(ProjectItem):

    def setScatter(self, x, y, data=None):
        self._commit(scatter=(x, y, data))

    def setImage(self, x, y, data):
        self._commit(image=(x, y, data))

    # def setImageFromScatter(self, xSlice, ySlice):
    #     self._commit(imageSlice=(xSlice, ySlice))

    def getScatter(self):
        with self._get_file() as h5f:
            grp = h5f.get(self.path)
            if grp is None:
                return None
            scatterPath = grp.attrs.get('XsocsScatter')
            if scatterPath is None:
                return None
            scatterGrp = grp[scatterPath]

            # TODO : checks
            x = scatterGrp.get('x')
            y = scatterGrp.get('y')
            data = scatterGrp.get('data')

            x = x and x[:]
            y = y and y[:]
            data = data and data[:]

        return x, y, data

    def getImage(self):
        with self._get_file() as h5f:
            grp = h5f.get(self.path)
            if grp is None:
                return None
            imagePath = grp.attrs.get('XsocsImage')
            if imagePath is None:
                return None
            imageGrp = grp[imagePath]

            # TODO : checks
            x = imageGrp.get('x')
            y = imageGrp.get('y')
            data = imageGrp.get('data')

            if x:
                if x.shape == ():
                    x = x[()]
                    if isinstance(x, h5py.RegionReference):
                        x = h5f[x][x]
                else:
                    x = x[:]

            if y.shape == ():
                y = y[()]
                if isinstance(y, h5py.RegionReference):
                    y = h5f[y][y]
            else:
                y = y[:]

            reshape = imageGrp.attrs.get('XsocsShape')
            data = data and data[:]
            if reshape is not None:
                data.shape = reshape
        return x, y, data

    def _commit(self,
                scatter=None,
                image=None,
                cube=None,
                imageSlice=None):
        super(HybridItem, self)._commit()
        # TODO : check if data already exists in file.
        with self._get_file(mode='r+') as h5f:
            grp = h5f.require_group(self.path)
            if scatter:
                x, y, data = scatter
                scatterGrp = grp.require_group('scatter')
                scatterGrp['x'] = x
                scatterGrp['y'] = y
                if data is not None:
                    grp['scatter/data'] = data
                grp.attrs.update({'XsocsScatter': 'scatter'})

            if image:
                x, y, data = image
                grp['image/data'] = data
                grp['image/x'] = x
                grp['image/y'] = y
                grp.attrs.update({'XsocsImage': 'image'})
            elif imageSlice:
                xSlice, ySlice = imageSlice
                dataSet = grp.get('scatter/data')
                if dataSet is None:
                    raise ValueError('Cant convert scatter to image : '
                                     'no data.')
                xSet = grp.get('scatter/x')
                ySet = grp.get('scatter/y')
                x = xSet.regionref[xSlice]
                y = ySet.regionref[ySlice]
                xShape = xSet[x].shape
                yShape = ySet[y].shape
                if len(xShape) != 1 or len(yShape) != 1:
                    raise ValueError('Invalid slice shapes x:{0}, y:{1}.'
                                     ''.format(xShape, yShape))
                imageGrp = grp.require_group('image')
                imageGrp['x'] = x
                imageGrp['y'] = y
                imageGrp['data'] = dataSet
                print imageGrp['x'], x
                imageGrp.attrs.update({'XsocsShape': [yShape[0],
                                                      xShape[0]]})

                grp.attrs.update({'XsocsImage': 'image'})

            if cube:
                grp.attrs.update({'XsocsCube': 'cube'})
