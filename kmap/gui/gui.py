import sys

from silx.gui import qt as Qt

from .MergeWidget import MergeWidget

def merge_window():
    app = Qt.QApplication(sys.argv)
    mw = MergeWidget()
    mw.show()
    app.exec_()
    
