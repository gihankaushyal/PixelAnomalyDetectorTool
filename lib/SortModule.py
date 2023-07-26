# PyQt5 imports
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot

# Packages for data access
from pathlib import Path
import h5py
import os.path

# Packages for Parallelization
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class SortData(qtw.QWidget):
    readyToSaveGood = qtc.pyqtSignal(dict, str)
    readyToSaveBad = qtc.pyqtSignal(dict, str)

    def __init__(self, model, inDict):
        """

        :param model: trained model
        :param inDict: dictionary with detector panel information
        """

        super(SortData, self).__init__()

        uic.loadUi('UI/sortDataGUI.ui', self)
        self.setWindowTitle('Sort Data')

        self.model = model
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        self.goodEvents = {}
        self.badEvents = {}

        self.tableWidget = self.findChild(qtw.QTableWidget, 'tableWidget')
        self.tableWidget.setColumnWidth(0, 350)

        self.browseButton.clicked.connect(self.browseFiles)
        self.sortButton.clicked.connect(self.buttonClicked)

        self.setAttribute(qtc.Qt.WA_DeleteOnClose)

    @pyqtSlot()
    def browseFiles(self):
        """
        This method gets triggered when the browse button is Clicked in the GUI
        function: The function is to take in a text field where the value needs to be set and called in a dialog box
                with file structure view starting at the 'root' and lets the user select the file they want and set the
                file path to the test field.
        """

        folderName = qtw.QFileDialog.getExistingDirectory(self, caption='Select Folder', directory=' ')

        if folderName != "":
            self.folderPath.setText(folderName)
            self.showFiles()

    def showFiles(self):
        """

        :return: Displays available files in the selected folder in self.availableFiles (QTextEdit)
        """
        folder = self.folderPath.text()

        files = Path(folder).glob('*.cxi')

        for file in files:
            self.availableFiles.append(str(file).split('/')[-1])

        self.sortButton.setEnabled(True)

    @pyqtSlot()
    def buttonClicked(self):
        """
        Asks a user a Question/ Waring about the model that was trained
        :return: Yes or No
        """

        msg = qtw.QMessageBox()
        msg.setWindowTitle('Question')
        msg.setText("Panel Selected: %s                                             " % self.panelName)
        msg.setInformativeText('Please Note! Machine Learning model was trained based on the data from the %s panel. '
                               'Make sure that you are sorting based on %s panel. If not, train a new model for your '
                               'frame of choice. Would you wish to continue?' % (self.panelName, self.panelName))
        msg.setIcon(qtw.QMessageBox.Question)
        msg.setStandardButtons(qtw.QMessageBox.Yes | qtw.QMessageBox.No)
        msg.setDefaultButton(qtw.QMessageBox.Yes)
        # msg.buttonClicked.connect(self.sort)
        msg.buttonClicked.connect(self.sortParallel)
        msg.exec_()

    @staticmethod
    def processFile(args):
        file, model, min_ss, max_ss, min_fs, max_fs = args
        print(' ')
        print('Processing file: %s' % file)  # Print the file being processed

        goodEvents = {}
        badEvents = {}

        # goodList to store all the events with expected pixel intensities for the file
        goodList = []
        # badList to store all the events with detector artifacts for the file
        badList = []

        with h5py.File(file, "r") as f:
            data = f['entry_1']['data_1']['data'][()]
            # print('Number of frames in file: %s' % data.shape[0])  # Print the number of frames in the file

        for i in range(data.shape[0]):
            frame = data[i][min_ss:max_ss, min_fs + 5:max_fs - 5].flatten()
            predictions = model.predict(frame.reshape(1, -1))

            if predictions:
                goodList.append(i)
            else:
                badList.append(i)

        # Print the number of good and bad events
        print('Finished processing file:  %s' % file)

        goodEvents[str(file)] = goodList
        badEvents[str(file)] = badList

        return goodEvents, badEvents

    def sortParallel(self, i):
        if i.text() == '&Yes':
            self.sortButton.setEnabled(False)
            folder = self.folderPath.text()
            print('Sorting the *.cxi files in folder: %s' % folder)  # Print the folder path

            fileSaveLocation = qtw.QFileDialog.getExistingDirectory(self, caption='Select Where You Want to Save the'
                                                                                  'Sorted Files', directory=' ',
                                                                    options=qtw.QFileDialog.DontUseNativeDialog)
            files = list(Path(folder).glob('*.cxi'))
            row = 0

            # Print the number of files
            print('Number of files to be processed: %s' % len(list(files)))

            self.tableWidget.setRowCount(len(list(files)))

            # prepare the arguments for the function
            args = [(file, self.model, self.min_ss, self.max_ss, self.min_fs, self.max_fs) for file in files]

            # Print the number of available CPUs
            numCpus = os.cpu_count()
            print('Number of CPUs available for processing: %s' % numCpus)

            with ProcessPoolExecutor() as executor:
                results = list(tqdm(executor.map(SortData.processFile, args), total=len(files)))

            for file, result in zip(files, results):
                tag = str(file).split('/')[-1].split('.')[0]
                goodEvents, badEvents = result
                self.readyToSaveGood.emit(goodEvents, fileSaveLocation + '/' + 'goodEvents-modelSort-%s.list' % tag)
                self.readyToSaveBad.emit(badEvents, fileSaveLocation + '/' + 'badEvents-modelSort-%s.list' % tag)

                self.tableWidget.setItem(row, 0, qtw.QTableWidgetItem(str(file).split('/')[-1]))
                self.tableWidget.setItem(row, 1, qtw.QTableWidgetItem(str(len(goodEvents[str(file)]))))
                self.tableWidget.setItem(row, 2, qtw.QTableWidgetItem(str(len(badEvents[str(file)]))))
                row += 1

            self.sortButton.setEnabled(False)

    def sort(self, i):
        """
                Sort *cxi files using the trained model
                :return: two separate *.list files for good and bad events for the cxi files
                """

        if i.text() == '&Yes':

            self.sortButton.setEnabled(False)
            folder = self.folderPath.text()

            fileSaveLocation = qtw.QFileDialog.getExistingDirectory(self, caption='Select Where You Want to Save the'
                                                                                  'Sorted Files', directory=' ',
                                                                    options=qtw.QFileDialog.DontUseNativeDialog)
            files = Path(folder).glob('*.cxi')
            row = 0
            self.tableWidget.setRowCount(len(list(files)))

            files = Path(folder).glob('*.cxi')
            for file in files:

                tag = str(file).split('/')[-1].split('.')[0]

                self.goodEvents = {}
                self.badEvents = {}

                # goodList to store all the events with expected pixel intensities for the file
                goodList = []
                # badList to store all the events with detector artifacts for the file
                badList = []

                with h5py.File(file, "r") as f:
                    data = f['entry_1']['data_1']['data'][()]

                for i in range(data.shape[0]):

                    frame = data[i][self.min_ss:self.max_ss, self.min_fs + 5:self.max_fs - 5].flatten()

                    predictions = self.model.predict(frame.reshape(1, 31675))

                    if predictions:
                        goodList.append(i)
                    else:
                        badList.append(i)

                self.goodEvents[str(file)] = goodList
                self.badEvents[str(file)] = badList

                self.readyToSaveGood.emit(self.goodEvents,
                                          fileSaveLocation + '/' + 'goodEvents-modelSort-%s.list' % tag)
                self.readyToSaveBad.emit(self.badEvents, fileSaveLocation + '/' + 'badEvents-modelSort-%s.list' % tag)

                self.tableWidget.setItem(row, 0, qtw.QTableWidgetItem(str(file).split('/')[-1]))
                self.tableWidget.setItem(row, 1, qtw.QTableWidgetItem(str(len(self.goodEvents[str(file)]))))
                self.tableWidget.setItem(row, 2, qtw.QTableWidgetItem(str(len(self.badEvents[str(file)]))))
                row += 1
                self.sortButton.setEnabled(False)