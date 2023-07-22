from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
from PyQt5 import QtCore as qtc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import pyqtSlot
import h5py
import numpy as np
import pandas as pd
import os.path
from pathlib import Path

# Parallelization stuff
from concurrent.futures import ProcessPoolExecutor
import traceback
from tqdm import tqdm
import psutil

# for saving the model
import pickle

class ML(qtw.QMainWindow):

    def __init__(self, inDict):
        """

        :param inDict: dictionary with detector panel information
        """

        super(ML, self).__init__()

        self.messages = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        uic.loadUi("UI/machineLearningGUI.ui", self)

        self.setWindowTitle('Machine Learning')

        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']
        self.trainSplit.setText('70')
        self.testSplit.setText('30')
        self.browseButton.clicked.connect(self.browseFiles)
        self.trainButton.clicked.connect(self.buttonClicked)
        self.testButton.clicked.connect(self.test)
        self.resetButton.clicked.connect(self.reset)
        self.saveButton.clicked.connect(self.saveModel)
        self.comboBox.activated.connect(self.comboBoxChanged)

        # for displaying the confusion matrix
        self.layoutConfusionMatrix = qtw.QHBoxLayout()
        self.figureConfusionMatrix = plt.figure()
        self.canvasConfusionMatrix = FigureCanvasQTAgg(self.figureConfusionMatrix)
        self.layoutConfusionMatrix.addWidget(self.canvasConfusionMatrix)
        self.confusionMatrix.setLayout(self.layoutConfusionMatrix)

        # for displaying the classification report
        self.layoutClassificationReport = qtw.QHBoxLayout()
        self.figureClassificationReport = plt.figure()
        self.canvasClassificationReport = FigureCanvasQTAgg(self.figureClassificationReport)

        self.layoutClassificationReport.addWidget(self.canvasClassificationReport)
        self.classificationReport.setLayout(self.layoutClassificationReport)

        # adding busy and idle lights
        # self.busyLight = BusyLight()
        # self.idleLight = IdleLight()
        # self.statusbar.addPermanentWidget(self.busyLight)
        # self.statusbar.addPermanentWidget(self.idleLight)
        # self.idleLight.show()
        # self.busyLight.hide()

        # self.statusbar.showMessage("Point to where you have the data for model training", 3000)

        self.setAttribute(qtc.Qt.WA_DeleteOnClose)

    # def setBusy(self):
    #     """
    #
    #     :return: Change the light to busy
    #     """
    #     self.busyLight.show()
    #     self.idleLight.hide()

    # def setIdle(self):
    #     """
    #
    #     :return: Change the light to Idle
    #     """
    #     self.busyLight.hide()
    #     self.idleLight.show()

    # def showNextMessage(self, messageList):
    #     message = messageList.pop(0)
    #     self.statusbar.showMessage(message, 3000)
    #     if messageList:
    #         qtc.QTimer.singleShot(3000, lambda: self.showNextMessage(messageList))

    @pyqtSlot()
    def browseFiles(self):
        self.setBusy()

        """
            This method gets triggered when the browse button is Clicked in the GUI
        function: The function is to take in a text field where the value needs to be set and called in a dialog box
        with file structure view starting at the 'root' and lets the user select the file they want and set the file p
        ath to the test field.
        """

        folderName = qtw.QFileDialog.getExistingDirectory(self, caption='Select Folder', directory=' ')

        if folderName != "":
            self.parentDirectory.setText(folderName)
            self.messages = ["Select a model to be trained", "Enter Train/Test split ",
                             "Click the Train button to train the model"]
            self.showNextMessage(self.messages)

        self.setIdle()

    # model training using multiple runs needs to be implemented
    # @pyqtSlot()
    # def checkBoxClicked(self):
    #     """
    #
    #     :return: Pass ** functionality hasn't been implemented. **
    #     """
    #     if self.checkBox.isChecked():
    #         self.startRun.setEnabled(True)
    #         self.endRun.setEnabled(True)
    #     else:
    #         self.startRun.setEnabled(False)
    #         self.endRun.setEnabled(False)

    @pyqtSlot(dict)
    def readPanelDetails(self, inDict):
        """
         :param inDict: Dictionary with ASIIC/panel information coming from the signal once the user clicked on a panel
        :return: Assigns panel details to class variables from inDict
        """
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

    def modelSelection(self):
        """

        :return: user selected scikit-learn model
        """

        modelSelected = self.comboBox.currentText()
        if modelSelected == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
            return True
        elif modelSelected == 'KNeighborsClassifier':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(n_neighbors=1)
            return True
        elif modelSelected == 'DecisionTreeClassifier':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier()
            return True
        elif modelSelected == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=200)
            return True
        else:
            qtw.QMessageBox.critical(self, 'Caution', 'Please Select a model')
            return False

    def checkTrainTestSplit(self):
        if self.trainSplit.text().isdigit() and self.testSplit.text().isdigit():
            train = int(self.trainSplit.text())
            test = int(self.testSplit.text())

            if train + test > 100 or train + test < 100:
                qtw.QMessageBox.critical(self, 'Alert', 'The Sum of train split + test split = 100%')
                qtw.QMessageBox.information(self, 'Information',
                                            'Setting the train and test split to the default values')
                self.trainSplit.setText('70')
                self.testSplit.setText('30')
                return True
            else:
                return True

        else:
            qtw.QMessageBox.information(self, 'Information', 'Please enter a valid number')
            return False

    @staticmethod
    def readFile(file, flag, min_ss, max_ss, min_fs, max_fs):
        try:
            temp_df = pd.read_csv(str(file), delimiter=" ")
            temp_df.columns = ['FileName', 'EventNumber']

            # reading the panel data from the file
            temp_df['EventNumber'] = temp_df['EventNumber'].apply(lambda x: x.split('/')[2])
            fileName = temp_df['FileName'].iloc[0]

            with h5py.File(fileName, "r") as f:
                # print(f'Started Reading {str(file)}')
                data = f['entry_1']['data_1']['data'][()]
                # print(f'Finished Reading {str(file)}')

            tempList = []
            for i in list(temp_df['EventNumber']):
                frame = data[int(i)][min_ss : max_ss, min_fs + 5 : max_fs - 5]
                tempList.append(frame.flatten())

            temp_df['Data'] = tempList
            temp_df['Flag'] = flag

            return temp_df

        except Exception as e:
            print(f"An error occurred while reading file {str(file)}: {str(e)}")
            print(traceback.format_exc())
            return None

    def dataPrepParalle(self):
        from sklearn.model_selection import train_test_split
        folder = self.parentDirectory.text()

        # Get the number of CPUs
        numCpus = os.cpu_count()
        print(f'Number of CPUs: {numCpus}')

        # Get the total memory
        total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GBs
        print(f'Total Memory: {total_memory:.2f} GB')

        with ProcessPoolExecutor() as executor:
            # Bad Events
            files = list(Path(folder).glob('badEvents-advanceSort-*.list'))
            # dataFrames_bad = list(tqdm(executor.map(self.readFile, files, [0] * len(files))))
            dataFrames_bad = list(tqdm(
                executor.map(ML.readFile, files, [0] * len(files), [self.min_ss] * len(files), [self.max_ss] * len(files),
                             [self.min_fs] * len(files), [self.max_fs] * len(files)), total=len(files)))
            dataFrame_bad = pd.concat([df for df in dataFrames_bad if df is not None])

            # Good Events
            files = list(Path(folder).glob('goodEvents-advanceSort-*.list'))
            # dataFrames_good = list(tqdm(executor.map(self.readFile, files, [1] * len(files))))
            dataFrames_good = list(tqdm(
                executor.map(ML.readFile, files, [0] * len(files), [self.min_ss] * len(files),
                             [self.max_ss] * len(files),
                             [self.min_fs] * len(files), [self.max_fs] * len(files)), total=len(files)))
            dataFrame_good = pd.concat([df for df in dataFrames_good if df is not None])

        dataFrame_good = pd.concat([dataFrame_good['FileName'], dataFrame_good['EventNumber'],
                                    dataFrame_good.pop('Data').apply(pd.Series), dataFrame_good['Flag']], axis=1)
        X_good = dataFrame_good.drop(['FileName', 'EventNumber', 'Flag'], axis=1)
        y_good = dataFrame_good['Flag']
        X_good_train, X_good_test, y_good_train, y_good_test = train_test_split(X_good, y_good,
                                                                                test_size=int(self.testSplit.text()))

        dataFrame_bad = pd.concat([dataFrame_bad['FileName'], dataFrame_bad['EventNumber'],
                                   dataFrame_bad.pop('Data').apply(pd.Series), dataFrame_bad['Flag']], axis=1)
        X_bad = dataFrame_bad.drop(['FileName', 'EventNumber', 'Flag'], axis=1)
        y_bad = dataFrame_bad['Flag']

        X_bad_train, X_bad_test, y_bad_train, y_bad_test = train_test_split(X_bad, y_bad,
                                                                            test_size=int(self.testSplit.text()))

        self.X_train = pd.concat([X_good_train, X_bad_train])
        self.y_train = pd.concat([y_good_train, y_bad_train])
        self.X_test = pd.concat([X_good_test, X_bad_test])
        self.y_test = pd.concat([y_good_test, y_bad_test])

    def dataPrep(self):
        """
        This method look into the folder where the sorted files are stored (by sort() in SortingForMl) and prepare the
        data for training and testing.
        :return: X_train, X_test, y_train, y_test
        """

        from sklearn.model_selection import train_test_split
        folder = self.parentDirectory.text()

        # Bad Events
        files = Path(folder).glob('badEvents-advanceSort-*.list')
        dataFrame_bad = pd.DataFrame(columns=['FileName', 'EventNumber', 'Data'])

        for file in files:

            try:
                temp_df = pd.read_csv(str(file), delimiter=" ")
                temp_df.columns = ['FileName', 'EventNumber']

                # reading the panel data from the file
                temp_df['EventNumber'] = temp_df['EventNumber'].apply(lambda x: x.split('/')[2])
                fileName = temp_df['FileName'].iloc[0]

                with h5py.File(fileName, "r") as f:
                    print('Reading %s' % str(file))
                    data = f['entry_1']['data_1']['data'][()]

                tempList = []
                for i in list(temp_df['EventNumber']):
                    frame = data[int(i)][self.min_ss:self.max_ss, self.min_fs + 5:self.max_fs - 5]
                    tempList.append(frame.flatten())

                temp_df['Data'] = tempList

                dataFrame_bad = pd.concat([dataFrame_bad, temp_df])
            except Exception as e:
                msg = qtw.QMessageBox()
                msg.setWindowTitle('Warning')
                msg.setText(
                    "An error occurred while reading bad events file %s                                  " % str(file))
                msg.setInformativeText(str(e) + " dataPrep()")
                msg.setIcon(qtw.QMessageBox.Warning)
                msg.exec_()
                continue
        dataFrame_bad['Flag'] = 0
        print('Done reading Bad events...')
        print('Found %i Good events' % len(dataFrame_bad))
        print(' ')

        # Good Events
        files = Path(folder).glob('goodEvents-advanceSort-*.list')
        dataFrame_good = pd.DataFrame(columns=['FileName', 'EventNumber', 'Data'])

        for file in files:
            try:
                temp_df = pd.read_csv(str(file), delimiter=" ")
                temp_df.columns = ['FileName', 'EventNumber']

                # reading the panel data from the file
                temp_df['EventNumber'] = temp_df['EventNumber'].apply(lambda x: x.split('/')[2])
                fileName = temp_df['FileName'].iloc[0]

                with h5py.File(fileName, "r") as f:
                    print('Reading %s' % str(file))
                    data = f['entry_1']['data_1']['data'][()]

                tempList = list()
                for i in list(temp_df['EventNumber']):
                    frame = data[int(i)][self.min_ss:self.max_ss, self.min_fs + 5:self.max_fs - 5]
                    tempList.append(frame.flatten())

                temp_df['Data'] = tempList

                dataFrame_good = pd.concat([dataFrame_good, temp_df])
            except Exception as e:
                msg = qtw.QMessageBox()
                msg.setWindowTitle('Error')
                msg.setText(
                    "An error occurred while reading good events file %s                                  " % str(file))
                msg.setInformativeText(str(e) + " dataPrep()")
                msg.setIcon(qtw.QMessageBox.Information)
                msg.exec_()
                continue
        dataFrame_good['Flag'] = 1
        print('Done reading Good events...')
        print('Found %i Good events' % len(dataFrame_good))

        dataFrame_good = pd.concat([dataFrame_good['FileName'], dataFrame_good['EventNumber'],
                                    dataFrame_good.pop('Data').apply(pd.Series), dataFrame_good['Flag']], axis=1)
        X_good = dataFrame_good.drop(['FileName', 'EventNumber', 'Flag'], axis=1)
        y_good = dataFrame_good['Flag']
        X_good_train, X_good_test, y_good_train, y_good_test = train_test_split(X_good, y_good,
                                                                                test_size=int(self.testSplit.text()))

        dataFrame_bad = pd.concat([dataFrame_bad['FileName'], dataFrame_bad['EventNumber'],
                                   dataFrame_bad.pop('Data').apply(pd.Series), dataFrame_bad['Flag']], axis=1)
        X_bad = dataFrame_bad.drop(['FileName', 'EventNumber', 'Flag'], axis=1)
        y_bad = dataFrame_bad['Flag']

        X_bad_train, X_bad_test, y_bad_train, y_bad_test = train_test_split(X_bad, y_bad,
                                                                            test_size=int(self.testSplit.text()))

        self.X_train = pd.concat([X_good_train, X_bad_train])
        self.y_train = pd.concat([y_good_train, y_bad_train])
        self.X_test = pd.concat([X_good_test, X_bad_test])
        self.y_test = pd.concat([y_good_test, y_bad_test])

    @pyqtSlot()
    def buttonClicked(self):
        """
        This method gets triggered when the "Train" button is pressed and asks a question from the user. Based on the
        answer it either moves forward to train a model or allow user to go back and select a different
        ASCI for training.
        :return: Yes or No
        """
        msg = qtw.QMessageBox()
        msg.setWindowTitle('Question')
        msg.setText("Panel Selected: %s                                             " % self.panelName)
        msg.setInformativeText('Machine Learning model will be trained on the pixel data associated with the '
                               'selected: %s panel. Would you wish to continue?' % self.panelName)
        msg.setIcon(qtw.QMessageBox.Question)
        msg.setStandardButtons(qtw.QMessageBox.Yes | qtw.QMessageBox.No)
        msg.setDefaultButton(qtw.QMessageBox.Yes)
        msg.buttonClicked.connect(self.train)
        msg.exec_()

    def train(self, i):
        """
        Method to train the user selected model using the data from the selected ASCI
        :param i: QMessageBox output( &Yes or &No)
        :return: Model and Enables the "Test" button
        """

        if i.text() == '&Yes':
            self.setBusy()
            self.trainButton.setEnabled(False)
            if self.modelSelection() and self.checkTrainTestSplit():
                # self.dataPrep()
                self.dataPrepParalle()
                self.model.fit(self.X_train, self.y_train)
                self.testButton.setEnabled(True)
                self.setIdle()
                qtw.QMessageBox.information(self, 'Success', "Done Training")
                self.statusbar.showMessage("Now you can save the model or try training a new model", 3000)

            else:
                self.reset()
        else:
            self.reset()

    @pyqtSlot()
    def test(self):
        """
        Method to test the validity of the trained model
        :return: Confusion matrix and a Classification Report
        """
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
        self.predictions = self.model.predict(self.X_test)

        self.setBusy()

        self.figureConfusionMatrix.clear()
        self.figureClassificationReport.clear()

        # printing a heatmap for Confusion matrix
        # cfm = confusion_matrix(self.y_test, self.predictions)
        # cfm_df = pd.DataFrame(cfm, index=['0', '1'], columns=['0', '1'])
        # ax1 = self.figureConfusionMatrix.add_subplot(111)
        # sns.heatmap(cfm_df, annot=True, cmap='mako', ax=ax1, cbar=False)
        # ax1.set_ylabel("True Label")
        # ax1.set_xlabel("Predicted Label")
        # self.canvasConfusionMatrix.draw()
        ax1 = self.figureConfusionMatrix.add_subplot(111)
        ConfusionMatrixDisplay.from_predictions(self.y_test, self.predictions, ax=ax1, colorbar=False, cmap='mako')
        self.canvasConfusionMatrix.draw()

        # printing a heatmap for Classification report
        cr = classification_report(self.y_test, self.predictions)
        columns = cr.strip().split()[0:3]
        indexes = ['Bad', 'Good', 'Avg', 'Wt. Avg']
        data = np.array(
            [cr.strip().split()[5:8], cr.strip().split()[10:13], cr.strip().split()[19:22],
             cr.strip().split()[25:28]], dtype=float)
        cr_df = pd.DataFrame(data=data, columns=columns, index=indexes)
        ax2 = self.figureClassificationReport.add_subplot(111)
        sns.heatmap(cr_df, annot=True, cmap='mako', ax=ax2, cbar=True, linewidth=1)
        self.canvasClassificationReport.draw()

        self.testButton.setEnabled(False)
        self.saveButton.setEnabled(True)
        self.setIdle()

    @pyqtSlot()
    def reset(self):
        """
        Method to clear out the output from the test()
        :return: clear the self.confusionMatrix and self.classificationReport
        """
        self.figureConfusionMatrix.clear()
        self.figureClassificationReport.clear()
        self.trainButton.setEnabled(True)
        self.testButton.setEnabled(False)
        self.comboBox.setCurrentIndex(0)
        self.trainSplit.setText('70')
        self.testSplit.setText('30')

    @pyqtSlot(int)
    def comboBoxChanged(self, index):
        self.reset()
        self.comboBox.setCurrentIndex(index)

    @pyqtSlot()
    def saveModel(self):
        filename, _ = qtw.QFileDialog.getSaveFileName(self, "Save File", "", "Pickle Files (*.pkl)")
        data = {'panel_name': self.panelName, 'min_ss': self.min_ss, 'min_fs': self.min_fs, 'max_ss': self.max_ss,
                'max_fs':self.max_fs, 'model': self.model}
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)