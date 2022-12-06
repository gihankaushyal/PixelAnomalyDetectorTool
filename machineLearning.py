import pandas as pd
import numpy as np
import sys

from PyQt5 import uic
from PyQt5 import QtWidgets as qtw

from utilities import sortTools
from pathlib import Path

class ML(qtw.QFrame):
    def __int__(self, *args,  **kwargs):
        super(ML, self).__int__( *args,  **kwargs)

        # self.st = sortTools()
        # self.folder = '/Users/gketawal/PycharmProjects/InternalTool/r0484-snr5pix2'
        # self.files = Path(folder).glob('*cxi')

        uic.loadUi("machineLearningGUI.ui", self)

        # for file in files:
        #     self.st.advanceSortFrames(file,'4')

        # df1 = pd.read_csv("badEvents-advanceSearch-r0484_1.list",delimiter=' ')
        # df1.columns = ['FileName', 'EventNumber', 'InflectionPoint1', 'InflectionPoint2']
        # print(df1)
        self.setWindowTitle('Machine Learning')
        self.show()

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    w = ML()
    #w.show()
    sys.exit(app.exec_())


