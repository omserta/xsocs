import sys

from silx.gui import qt as Qt

from .MergeWidget import MergeWidget

def merge_window(*args, **kwargs):
    app = Qt.QApplication(sys.argv)
    mw = MergeWidget(*args, **kwargs)
    mw.show()
    app.exec_()
    
