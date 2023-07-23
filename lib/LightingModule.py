# PyQt5 imports
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc

class BusyLight(qtw.QWidget):
    """
    Status indicator light when the GUI is busy
    """

    def __init__(self):
        super().__init__()
        self.setFixedSize(12, 12)
        self.color = qtg.QColor('darkorange')
        self.timer = qtc.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(500)

    def paintEvent(self, event):
        painter = qtg.QPainter(self)
        painter.setRenderHint(qtg.QPainter.Antialiasing)
        painter.setPen(qtc.Qt.NoPen)
        painter.setBrush(self.color)
        painter.drawEllipse(self.rect())

    def update(self):
        if self.color == qtc.Qt.yellow:
            self.color = qtc.Qt.transparent
        else:
            self.color = qtc.Qt.yellow
        super().update()


class IdleLight(qtw.QWidget):
    """
        Status indicator light when the GUI is Idle
        """

    def __init__(self):
        super().__init__()
        self.setFixedSize(12, 12)

    def paintEvent(self, event):
        painter = qtg.QPainter(self)
        painter.setRenderHint(qtg.QPainter.Antialiasing)
        painter.setPen(qtc.Qt.NoPen)
        painter.setBrush(qtg.QColor('springgreen'))
        painter.drawEllipse(self.rect())