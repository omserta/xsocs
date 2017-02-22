xsocs
====

TBD

Installation
------------

The latest development version can be obtained from the git repository::

    git clone https://gitlab.esrf.fr/kmap/xsocs.git
    cd xsocs

And then install::
    pip install . [--user]

Or, if pip is not available (not recommended):
    python setup.py install [--user]
    
Starting XSOCS
--------------
At the moment the only way to run XSOCS is from the python interpreter :

**>> python -c 'from xsocs.gui import xsocs_main; xsocs_main()'**

Dependencies
------------

* `Python <https://www.python.org/>`_ 2.7, 3.4 or 3.5.
* `numpy <http://www.numpy.org>`_
* `h5py <http://www.h5py.org/>`_
* `silx <https://pypi.python.org/pypi/silx>`_
* A Qt binding: `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ or `PySide <https://pypi.python.org/pypi/PySide/>`_

Supported platforms
-------------------
* Linux

Documentation
-------------
TBD ...

License
-------

The source code of xsocs is licensed under the MIT and LGPL licenses.