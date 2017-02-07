
xsocs
====

TBD

Installation
------------

The latest development version can be obtained from the git repository::

    git clone git://gitlab.esrf.fr:xsocs/xsocs.git
    cd xsocs

And then install::
    pip install . [--user]

Or, if pip is not available (not recommended):
    python setup.py install [--user]

Dependencies
------------

* `Python <https://www.python.org/>`_ 2.7, 3.4 and 3.5.
* `numpy <http://www.numpy.org>`_
* `h5py <http://www.h5py.org/>`_
* `silx <https://pypi.python.org/pypi/silx>`_
.. * A Qt binding: `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ or `PySide <https://pypi.python.org/pypi/PySide/>`_

Supported platforms: Linux
.. , Windows, Mac OS X.

Documentation
-------------

..
    To build the documentation from the source (requires `Sphinx <http://www.sphinx-doc.org>`_), run::

    python setup.py build build_doc
..
Testing
-------
..
    |Travis Status| |Appveyor Status|

    To run the tests, from the source directory, run::

        python run_tests.py
..
License
-------

The source code of xsocs is licensed under the MIT and LGPL licenses.
See the `copyright file TBD`_ for details.

.. |Travis Status| image:: TBD
.. |Appveyor Status| image:: TBD
