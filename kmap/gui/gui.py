import sys

from silx.gui import qt as Qt
print('Using Qt {0}'.format(Qt.qVersion()))

from .MergeWidget import MergeWidget
from .RecipSpaceWidget import RecipSpaceWidget

def merge_window(*args, **kwargs):
    app = Qt.QApplication(sys.argv)
    mw = MergeWidget(*args, **kwargs)
    mw.show()
    app.exec_()

def conversion_window(*args, **kwargs):
    app = Qt.QApplication(sys.argv)
    mw = RecipSpaceWidget(*args, **kwargs)
    mw.show()
    app.exec_()
