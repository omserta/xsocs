#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""X-SOCS GUI launcher"""

__authors__ = ["Thomas Vincent"]
__date__ = "25/10/2016"
__license__ = "MIT"


import argparse
import distutils.util
import glob
import logging
import os
import os.path
import shutil
import site
import subprocess
import sys
import tempfile


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("run_tests")
logger.setLevel(logging.WARNING)

logger.info("Python %s %s", sys.version, tuple.__itemsize__ * 8)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--upgrade", dest="upgrade", action="store_true", default=False,
    help=argparse.SUPPRESS)
parser.add_argument('--prj',
                    metavar='project',
                    dest='project',
                    default=None,
                    type=str,
                    help='opens the project file')
args = parser.parse_args()


def upgrade_project(name, repo):
    """Upgrade a project.

    It runs the following in the project directory:
    - Clone the repo
    - Reinstall in lib/
    - chmod g+w lib/

    :param str name: Python package name of the project
    :param str repo: git project repository
    """
    logger.warning("upgrading %s from %s", name, repo)
    tmp_dir = tempfile.mkdtemp()
    try:
        # TODO do not work when changing version
        install_paths = glob.glob('/scisoft/xsocs/lib/python2.7/site-packages/' + name + "*")
        commands = [  # command, environment variables
            ['rm', '-rf'] + install_paths,
            ['git', 'clone', repo, tmp_dir],
            [sys.executable, "setup.py", "install", "--prefix=/scisoft/xsocs"],
            ["chmod", "-R", "g+wx"] + install_paths,
        ]

        pyenv = os.environ.copy()
        pyenv['PYTHONPATH'] = '/scisoft/xsocs/lib/python2.7/site-packages'

        for cmd in commands:
            logger.warning("> %s", ' '.join(cmd))
            p = subprocess.Popen(cmd, shell=False, cwd=tmp_dir, env=pyenv)
            rc = p.wait()
            if rc != 0:
                logger.error('Return code=%s', rc)
                raise RuntimeError('Failed to run %s' % ' '.join(cmd))

    finally:
        shutil.rmtree(tmp_dir)


if args.upgrade:
    upgrade_project("kmap", "git://gitlab.esrf.fr/kmap/kmap.git")
    upgrade_project("plot3d", "git://gitlab.esrf.fr/tvincent/plot3d.git")
    sys.exit()


# Add local install of xrayutilities, kmap and plot3d
# Build command for xrayutilities:
# PYTHONPATH=/scisoft/xsocs/lib/python2.7/site-packages ; python setup.py --without-openmp install --prefix=/scisoft/xsocs/

logger.info("Add local install path")
script_dir = os.path.dirname(__file__)
site.addsitedir(os.path.join(script_dir, 'lib', 'python2.7', 'site-packages'))

# Start application
from kmap.gui import xsocs_main

xsocs_main(projectH5File=args.project)
