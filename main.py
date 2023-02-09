#!/usr/bin/env python3

# imports
import random
import warnings
from builtins import Exception
# basic stuff
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
# Gui stuff
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
# from PyQt5 import QtWebEngineWidgets as qtwew # for graphing with plotly
# Graphing stuff
import pyqtgraph as pg
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# import plotly.express as px
from PyQt5.QtCore import pyqtSlot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5 import uic

# import sys

import lib.cfel_filetools as fileTools
import lib.cfel_imgtools as imgTools
from lib.geometry_parser.GeometryFileParser import *


class DisplayImage(qtw.QWidget):
    panelSelected = qtc.pyqtSignal(dict)

    def __init__(self, fileName, geometry):

        """

        :param fileName: name of the *cxi file (with the full path)
        :param geometry: path to the geometry file
        """
        super(DisplayImage, self).__init__()

        # setting the size and location of the window
        self.outgoingDict = None
        self.setGeometry(10, 100, 600, 600)

        # assigning the file name and the geometry
        self.fileName = fileName
        self.geometryName = geometry

        # class variables
        self.eventNumber = None
        self.imageToDraw = None
        self.cxi = None
        self.size = None
        self.panelLocFromGeom = {}
        self.panelFsSs = {}
        self.panelsXYEdges = {}
        self.outgoingDict = {}

        # main window for display the data
        self.mainWidget = pg.ImageView()

        # adding a checkBoxes
        self.foundPeaksCheckBox = qtw.QCheckBox('Found Peaks')

        # connecting the checkBoxes to a method
        self.foundPeaksCheckBox.stateChanged.connect(lambda: self.drawImage(self.eventNumber))

        # adding a layout and add checkbox and the mainwindow to the layout
        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.mainWidget)
        self.layout.addWidget(self.foundPeaksCheckBox)

        # reading the geometry file
        try:
            self.parser = GeometryFileParser(self.geometryName)
            self.geometry = self.parser.pixel_map_for_cxiview()

            for panelName in self.parser.dictionary['panels'].keys():
                self.panelFsSs[panelName] = [self.parser.dictionary['panels'][panelName]['min_fs'],
                                             self.parser.dictionary['panels'][panelName]['max_fs'],
                                             self.parser.dictionary['panels'][panelName]['min_ss'],
                                             self.parser.dictionary['panels'][panelName]['max_ss']]

            for panelName in self.panelFsSs.keys():
                # bottom left conner
                x1 = (self.panelFsSs[panelName][0], self.panelFsSs[panelName][2])
                # bottom right conner
                x2 = (self.panelFsSs[panelName][0], self.panelFsSs[panelName][3])
                # top right conner
                x3 = (self.panelFsSs[panelName][1], self.panelFsSs[panelName][3])
                # top left conner
                x4 = (self.panelFsSs[panelName][1], self.panelFsSs[panelName][2])

                self.panelLocFromGeom[panelName] = [x1, x2, x3, x4]

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', self.geometryName, " was not found -reading geometry __init__")

        # adding an overlapping canvas to the found peaks
        self.foundPeaksCanvas = pg.ScatterPlotItem()
        self.mainWidget.getView().addItem(self.foundPeaksCanvas)

        # adding a canvas for displaying panel edges
        self.panelEdgesCanvas = pg.PlotDataItem()
        self.mainWidget.getView().addItem(self.panelEdgesCanvas)

        # connecting a mouse clicked event to a select panel method
        self.mainWidget.getView().scene().sigMouseClicked.connect(self.selectPanel)

        self.setLayout(self.layout)

    @pyqtSlot(int)
    def drawImage(self, eventNumber):
        """
         reading and displaying data
        :param eventNumber: event number to be displayed
        :return: pixel map from the cxi file
        """

        try:
            # applying the geometry and displaying the image
            self.eventNumber = eventNumber
            print(self.fileName)
            # reading the given eventNumber from the cxi file
            self.cxi = fileTools.read_cxi(self.fileName, frameID=self.eventNumber, data=True, slab_size=True,
                                          peaks=True)
            self.size = self.cxi['stack_shape'][0]
            # reading data
            imgData = self.cxi['data']
            # converting data into a pixel map to display and applying geometry
            self.imageToDraw = imgTools.pixel_remap(imgData, self.geometry['x'], self.geometry['y'])
            # showing the pixel map in the main window
            self.mainWidget.setImage(self.imageToDraw)
            # setting a window title with the eventNumber and the total number of event in the file
            self.setWindowTitle("Showing %i of %i " % (self.eventNumber, self.size - 1))

            if self.eventNumber == 0:
                self.drawInitialPanel()

            self.drawPeaks()

        except IndexError as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Information')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " drawImage()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    def drawPeaks(self):
        """
        :return: draw circles around the found peaks extracted from the cxi file
        """
        try:
            if self.foundPeaksCheckBox.isChecked():
                peaks_x = []
                peaks_y = []

                # temp = fileTools.read_event()
                n_peaks = self.cxi['n_peaks']
                x_data = self.cxi['peakXPosRaw']
                y_data = self.cxi['peakYPosRaw']

                for i in range(0, n_peaks):
                    peak_fs = x_data[i]
                    peak_ss = y_data[i]

                    peak_in_slab = int(round(peak_ss)) * self.cxi['stack_shape'][2] + int(round(peak_fs))
                    peaks_x.append(self.geometry['x'][peak_in_slab] + self.imageToDraw.shape[0] / 2)
                    peaks_y.append(self.geometry['y'][peak_in_slab] + self.imageToDraw.shape[1] / 2)

                ring_pen = pg.mkPen('b', width=2)
                self.foundPeaksCanvas.setData(peaks_x, peaks_y, symbol='o', size=10, pen=ring_pen, brush=(0, 0, 0, 0),
                                              pxMode=False)
            else:
                self.foundPeaksCanvas.clear()
        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " drawPeaks()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    def drawInitialPanel(self):
        """
        draw initial panel (predetermined to be p6a0)
        :return:  a dictionary (self.outgoingDict)
        """

        try:
            for panelName in self.panelLocFromGeom.keys():
                x_edges = []
                y_edges = []

                for i in range(4):
                    edge_fs = self.panelLocFromGeom[panelName][i][0]
                    edge_ss = self.panelLocFromGeom[panelName][i][1]
                    peak_in_slab = int(round(edge_ss)) * self.cxi['stack_shape'][2] + int(round(edge_fs))
                    x_edges.append(self.geometry['x'][peak_in_slab] + self.imageToDraw.shape[0] / 2)
                    y_edges.append(self.geometry['y'][peak_in_slab] + self.imageToDraw.shape[1] / 2)
                x_edges.append(x_edges[0])
                y_edges.append(y_edges[0])

                self.panelsXYEdges[panelName] = [x_edges, y_edges]

            pen = pg.mkPen('r', width=3)
            # plotting a square along the edges of the selected panel
            self.panelEdgesCanvas.setData(self.panelsXYEdges['p6a0'][0],
                                          self.panelsXYEdges['p6a0'][1], pen=pen)
            self.outgoingDict = {'panel_name': 'p6a0',
                                 'min_fs': self.panelFsSs['p6a0'][0], 'max_fs': self.panelFsSs['p6a0'][1],
                                 'min_ss': self.panelFsSs['p6a0'][2], 'max_ss': self.panelFsSs['p6a0'][3]}

            self.panelSelected.emit(self.outgoingDict)
        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " drawInitialPanel()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    def selectPanel(self, event):
        """
        Draw a boarder around the selected ASIIC
        :param event: A mouse clicked event
        :return: Draw a Red boarder around the selected ASIIC
        """

        try:
            # panel locations corrected for displayImage

            pos = event.scenePos()
            if self.mainWidget.getView().sceneBoundingRect().contains(pos):
                mouse_point = self.mainWidget.getView().mapSceneToView(pos)
                x_mouse = int(mouse_point.x())
                y_mouse = int(mouse_point.y())

                for panelName in self.panelsXYEdges.keys():
                    if x_mouse in range(int(min(self.panelsXYEdges[panelName][0])),
                                        int(max(self.panelsXYEdges[panelName][0]))) \
                            and y_mouse in range(int(min(self.panelsXYEdges[panelName][1])),
                                                 int(max(self.panelsXYEdges[panelName][1]))):
                        pen = pg.mkPen('r', width=3)
                        # plotting a square along the edges of the selected panel
                        self.panelEdgesCanvas.setData(self.panelsXYEdges[panelName][0],
                                                      self.panelsXYEdges[panelName][1], pen=pen)
                        # print(self.panelFsSs[panelName])

                        self.outgoingDict = {'panel_name': panelName,
                                             'min_fs': self.panelFsSs[panelName][0],
                                             'max_fs': self.panelFsSs[panelName][1],
                                             'min_ss': self.panelFsSs[panelName][2],
                                             'max_ss': self.panelFsSs[panelName][3]}
                        self.panelSelected.emit(self.outgoingDict)

        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " selectPanel()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()


class SortingForML(qtw.QWidget):
    readyToSaveGood = qtc.pyqtSignal(dict, str)
    readyToSaveBad = qtc.pyqtSignal(dict, str)

    def __init__(self, fileName, oft, inDict):
        """

        :param fileName: name of the file to be sorted
        :param oft: order of the fit for the polynomial
        :param inDict: dictionary with detector panel information
        """

        super(SortingForML, self).__init__()

        self.badEvents = None
        self.goodEvents = None
        self.inflectionPoint1List = None
        self.inflectionPoint2List = None
        self.data = None
        self.setWindowTitle('Sorting for Machine Learning')

        uic.loadUi("UI/sortForMLGUI.ui", self)

        # for plotting with matplotlib
        self.layoutSortingForML = qtw.QHBoxLayout()
        self.figureSortingForML = plt.figure()
        self.canvasForInflectionPoints = FigureCanvasQTAgg(self.figureSortingForML)
        self.layoutSortingForML.addWidget(self.canvasForInflectionPoints)
        self.graphSpace.setLayout(self.layoutSortingForML)

        # for plotting with plotly
        # self.layout = qtw.QHBoxLayout()
        # self.browser = qtwew.QWebEngineView()
        # self.layout.addWidget(self.browser)
        # self.graphSpace.setLayout(self.layout)

        self.fileName = fileName
        self.orderOfFit = oft
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        # setting initial values for spinBoxes (value ranges for inflection points)
        self.doubleSpinBoxIF1.setValue(15)
        self.doubleSpinBoxIF2.setValue(15)
        self.doubleSpinBoxIF1.setSingleStep(0.05)
        self.doubleSpinBoxIF2.setSingleStep(0.05)

        # self.plotInflectionPointsButton.clicked.connect(self.plotInflectionPoints)
        self.plotInflectionPoints()
        self.sortButton.clicked.connect(self.sort)

    @pyqtSlot(dict)
    def readPanelDetails(self, inDict):
        """
        :param inDict: Dictionary with ASIIC/panel information coming from the signal once the user clicked on
                a panel
        :return: Assigns panel detail
        """
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        self.plotInflectionPoints()

    def plotInflectionPoints(self):
        """
        :return: Plot two histograms for inflection point1 and inflection point2 on the self.graphSpace of
         the sortForMlGUI
        """

        self.inflectionPoint1List = []
        self.inflectionPoint2List = []

        try:
            with h5py.File(self.fileName, "r") as f:
                self.data = f['entry_1']['data_1']['data'][()]

            for i in range(len(self.data)):
                frame = self.data[i]

                avgIntensities = []
                for j in range(self.min_fs + 5, self.max_fs - 5):
                    avgIntensities.append(np.average(frame[self.min_ss:self.max_ss, j]))

                fit = np.polyfit(np.arange(self.min_fs + 5, self.max_fs - 5), avgIntensities, deg=int(self.orderOfFit))
                # calculating the inflection points (second derivative of the forth order polynomial)
                # this piece of code would convert a numpy runtime warning to an exception
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    try:

                        x1 = round((-6 * fit[1] + np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]),
                                   2)
                        x2 = round((-6 * fit[1] - np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]),
                                   2)

                        self.inflectionPoint1List.append(x1)
                        self.inflectionPoint2List.append(x2)
                    except IndexError as e:
                        msg = qtw.QMessageBox()
                        msg.setText(str(e).capitalize())
                        msg.setInformativeText('Try entering a different order for the polynomial')
                        msg.setIcon(qtw.QMessageBox.Critical)
                        msg.exec_()
                    except ValueError:
                        qtw.QMessageBox.information(self, 'Skip', 'Calculation Error! \n \n Skipping the frame %i' % i)
                        continue
                    except Warning as e:
                        msg = qtw.QMessageBox()
                        msg.setText(str(e).capitalize())
                        msg.setInformativeText('Error occurred while trying to calculate the sqrt of %i'
                                               % (36 * fit[1] * fit[1] - 96 * fit[0] * fit[2]))
                        msg.setIcon(qtw.QMessageBox.Warning)
                        msg.exec_()

                        continue

        except Exception as e:
            print(e, '-plotInflectionPoint')

        # with plotly
        # df = pd.DataFrame()
        # df['Inflection_poit1'] = self.x1_list
        # df['Inflection_poit2'] = self.x2_list
        # fig = px.histogram(df, nbins=200, opacity=0.5)
        # self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))

        # with seaborn
        self.figureSortingForML.clear()
        df = pd.DataFrame()
        df['Inflection_point1'] = self.inflectionPoint1List
        df['Inflection_point2'] = self.inflectionPoint2List
        colors = ['red', 'green', 'blue', 'violet', 'pink']
        random.shuffle(colors)
        for column in df.columns:
            sns.histplot(data=df[column], color=colors.pop(), binrange=(-300, 300), bins=80, alpha=0.5, label=column)
        plt.title('Distributions of Inflection points 1 and 2')
        plt.ylabel('Count')
        plt.xlabel(' Vertically Average Pixel Intensity')
        plt.xticks()
        plt.legend()

        self.canvasForInflectionPoints.draw()

        # Enabling button and check box after plotting
        self.inflectionPoint1.setEnabled(True)
        self.inflectionPoint1.setText(str(round(np.median(df['Inflection_point1'].dropna()), 2)))
        self.inflectionPoint2.setEnabled(True)
        self.inflectionPoint2.setText(str(round(np.median(df['Inflection_point2'].dropna()), 2)))
        self.sortButton.setEnabled(True)

    @pyqtSlot()
    def sort(self):
        """

        :return: two dictionaries (for both good and bad events) with file names and events sorted out by user
        defined threshold for inflection points and spread of the distribution
        """

        tag = str(self.fileName).split('/')[-1].split('.')[0]

        fileSaveLocation = qtw.QFileDialog.getExistingDirectory(self,  caption='Select Save Location', directory=' ',
                                                                options=qtw.QFileDialog.DontUseNativeDialog)
        if fileSaveLocation != "":
            self.goodEvents = {}
            self.badEvents = {}

            # goodList to store all the events with expected pixel intensities for the file
            goodList = []
            # badList to store all the events with detector artifacts for the file
            badList = []

            try:

                for (i, x1, x2) in zip(range(len(self.data)), self.inflectionPoint1List, self.inflectionPoint2List):

                    if (float(self.inflectionPoint1.text()) - self.doubleSpinBoxIF1.value()) <= x1 <= (
                            float(self.inflectionPoint1.text()) + self.doubleSpinBoxIF1.value()) \
                            and \
                            (float(self.inflectionPoint2.text()) - self.doubleSpinBoxIF2.value()) <= x2 <= (
                            float(self.inflectionPoint2.text()) + self.doubleSpinBoxIF2.value()):

                        goodList.append(i)
                    else:
                        badList.append(i)

                self.goodEvents[str(self.fileName)] = goodList
                self.badEvents[str(self.fileName)] = badList
                qtw.QMessageBox.information(self, "Completed", "Sorted files have being saved.")
                self.readyToSaveGood.emit(self.goodEvents, fileSaveLocation+'/'+'goodEvents-advanceSort-%s.list' % tag)
                self.readyToSaveBad.emit(self.badEvents, fileSaveLocation+'/'+'badEvents-advanceSort-%s.list' % tag)

            except Exception as e:
                msg = qtw.QMessageBox()
                msg.setWindowTitle('Error')
                msg.setText("An error occurred while sorting the file %s                              " % self.fileName)
                msg.setInformativeText(str(e) + " sort()")
                msg.setIcon(qtw.QMessageBox.Information)
                msg.exec_()

        else:
            qtw.QMessageBox.warning(self, 'Warning', 'Please Select a Save Location for sorted files')


class ML(qtw.QWidget):

    def __init__(self, inDict):
        """

        :param inDict: dictionary with detector panel information
        """

        super(ML, self).__init__()

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
        self.checkBox.stateChanged.connect(self.checkBoxClicked)
        self.trainButton.clicked.connect(self.buttonClicked)
        self.testButton.clicked.connect(self.test)
        self.resetButton.clicked.connect(self.reset)
        self.saveButton.clicked.connect(self.save)
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

    @pyqtSlot()
    def browseFiles(self):
        """
            This method gets triggered when the browse button is Clicked in the GUI
        function: The function is to take in a text field where the value needs to be set and called in a dialog box
        with file structure view starting at the 'root' and lets the user select the file they want and set the file p
        ath to the test field.
        """

        folderName = qtw.QFileDialog.getExistingDirectory(self, caption='Select Folder', directory=' ')

        if folderName != "":
            self.parentDirectory.setText(folderName)

    # model training using multiple runs needs to be implemented
    @pyqtSlot()
    def checkBoxClicked(self):
        """

        :return: Pass ** functionality hasn't been implemented. **
        """
        if self.checkBox.isChecked():
            self.startRun.setEnabled(True)
            self.endRun.setEnabled(True)
        else:
            self.startRun.setEnabled(False)
            self.endRun.setEnabled(False)

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

            if train+test > 100 or train+test < 100:
                qtw.QMessageBox.critical(self, 'Alert', 'The Sum of train split + test split = 100%')
                qtw.QMessageBox.information(self, 'Information',
                                            'Setting the train and test split to the default values')
                self.trainSplit.setText('70')
                self.testSplit.setText('30')
                return True
            else:
                return True

        else:
            qtw.QMessageBox.information(self,'Information', 'Please enter a valid number')
            return False

    def dataPrep(self):
        """
        This method look into the folder where the sorted files are stored (by sort() in SortingForMl) and prepare the
        data for training and testing.
        :return: X_train, X_test, y_train, y_test
        """

        from sklearn.model_selection import train_test_split

        try:
            if self.checkBox.isChecked():
                pass
            else:
                folder = self.parentDirectory.text()
        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " dataPrep()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

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
            self.trainButton.setEnabled(False)
            if self.modelSelection() and self.checkTrainTestSplit():
                self.dataPrep()
                self.model.fit(self.X_train, self.y_train)
                self.testButton.setEnabled(True)
                qtw.QMessageBox.information(self, 'Success', "Done Training")

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
        sns.heatmap(cr_df, annot=True, cmap='mako', ax=ax2, cbar=False, linewidth=1)
        self.canvasClassificationReport.draw()

        self.testButton.setEnabled(False)
        self.saveButton.setEnabled(True)

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
    def save(self):
        filename, _ = qtw.QFileDialog.getSaveFileName(self, "Save File", "", "Pickel Files (*.pkl)")

        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)


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
        msg.buttonClicked.connect(self.sort)
        msg.exec_()

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


class BusyLight(qtw.QWidget):
    """
    Status indicator light when the GUI is busy
    """
    def __init__(self):
        super().__init__()
        self.setFixedSize(10, 10)
        self.color = qtc.Qt.yellow
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
    #     self.update()


class IdleLight(qtw.QWidget):
    """
        Status indicator light when the GUI is Idle
        """
    def __init__(self):
        super().__init__()
        self.setFixedSize(10, 10)

    def paintEvent(self, event):
        painter = qtg.QPainter(self)
        painter.setRenderHint(qtg.QPainter.Antialiasing)
        painter.setPen(qtc.Qt.NoPen)
        painter.setBrush(qtc.Qt.green)
        painter.drawEllipse(self.rect())


class MainWindow(qtw.QMainWindow):
    clickedNext = qtc.pyqtSignal(int)
    clickedPrevious = qtc.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi("UI/mainWindow.ui", self)
        self.setGeometry(700, 100, 800, 700)
        # connecting elements to functions
        self.cxiBrowseButton.clicked.connect(self.browseFiles)
        self.geomBrowseButton.clicked.connect(self.browseGeom)
        self.viewFileButton.clicked.connect(self.viewFiles)

        #  First message on status bar
        self.statusbar.showMessage("Browse for CXI file or a list a CXI files ", 5000)

        # # Redirect standard output to custom function
        # sys.stdout = self

        # initializing the popup windows
        self.imageViewer = None
        self.sortForMLGUI = None
        self.mlGUI = None
        self.sortDataGUI = None

        self.fileSize = None
        self.totalEvents = None
        self.plotName = 'plotCurve'

        self.panelDict = None
        self.panelName = None
        self.min_fs = None
        self.max_fs = None
        self.min_ss = None
        self.max_ss = None
        self.detectorLeft = [
                            'p4a0', 'p4a1', 'p4a2', 'p4a3',
                            'p5a0', 'p5a1', 'p5a2', 'p5a3',
                            'p6a0', 'p6a1', 'p6a2', 'p6a3',
                            'p7a0', 'p7a1', 'p7a2', 'p7a3',
                            'p8a0', 'p8a1', 'p8a2', 'p8a3',
                            'p9a0', 'p9a1', 'p9a2', 'p9a3',
                            'p10a0', 'p10a1', 'p10a2', 'p10a3',
                            'p11a0', 'p11a1', 'p11a2', 'p11a3',
                             ]
        # button and input line for calling plotCurves() method to plot vertically average intensity profile for the
        # panel
        self.plotPixelIntensityButton.clicked.connect(self.plotCurve)
        # button for call the fit_curve() method to fit a nth order polynomial for
        # the vertically average intensity profile
        self.poltFitCheckBox.clicked.connect(self.plotFit)
        # button for calling plot_max_pixels() method to plot the pixel with the highest intensity for all
        # the frames of the
        self.plotPeakPixelButton.clicked.connect(lambda: self.plotMaxPixels(self.cxiFilePath.text()))

        self.sortButton.clicked.connect(self.sort)
        self.sortForMLButton.clicked.connect(self.sortForML)
        self.MLButton.clicked.connect(self.machineLearning)

        self.orderOfFit.editingFinished.connect(self.plotFit)
        self.eventNumber.editingFinished.connect(self.curveToPlot)
        self.eventNumber.editingFinished.connect(self.selectDisplay)

        # incrementing through event numbers
        self.nextButton.clicked.connect(lambda: self.nextEvent(self.eventNumber.text()))
        self.previousButton.clicked.connect(lambda: self.previousEvent(self.eventNumber.text()))

        # graphing
        self.graphWidget = pg.PlotWidget()
        self.layout = qtw.QHBoxLayout()
        self.layout.addWidget(self.graphWidget)
        self.graphingSpace.setLayout(self.layout)
        self.graphWidget.setEnabled(False)

        self.setWindowTitle("PixelAnomalyDetector")
        self.show()

        # adding busy and idle lights
        self.busyLight = BusyLight()
        self.idleLight = IdleLight()
        self.statusbar.addPermanentWidget(self.busyLight)
        self.statusbar.addPermanentWidget(self.idleLight)
        self.idleLight.show()
        self.busyLight.hide()

    @pyqtSlot()
    def browseFiles(self):
        """
        This method gets triggered when the browse button is Clicked in the GUI
        function:The function is to take in a text field where the value needs to be set and called in a dialog box with
        file structure view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """

        fileName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'CXI Files (*.cxi)')
        if fileName:
            self.cxiFilePath.setText(fileName)
            self.cxiFileListPath.setEnabled(False)
            self.cxiListBrowseButton.setEnabled(False)
            self.geomBrowseButton.setEnabled(True)
            self.geomFilePath.setEnabled(True)
            self.statusbar.showMessage("Browse for a geometry file ", 5000)

        # resting the main window for the next cxi file
        if self.imageViewer:
            self.imageViewer.close()
            # del self.imageViewer
            # self.imageViewer = None
            self.graphWidget.clear()
            self.eventNumber.setText("0")
            self.eventNumber.setEnabled(False)
            self.plotPixelIntensityButton.setEnabled(False)
            self.poltFitCheckBox.setEnabled(False)
            self.poltFitCheckBox.setChecked(False)
            self.plotPeakPixelButton.setEnabled(False)
            self.sortForMLButton.setEnabled(False)
            self.sortButton.setEnabled(False)
            self.nextButton.setEnabled(False)
            self.previousButton.setEnabled(False)
            self.MLButton.setEnabled(False)
            self.orderOfFit.clear()
            self.orderOfFit.setEnabled(False)
            self.graphWidget.setEnabled(False)

        if self.sortForMLGUI:
            self.sortForMLGUI.close()
            self.sortForMLGUI = None

        if self.mlGUI:
            self.mlGUI.close()
            self.mlGUI = None

        if self.sortForMLGUI:
            self.sortForMLGUI.close()
            self.sortDataGUI = None

    @pyqtSlot()
    def browseGeom(self):
        """
            This method gets triggered when the browse button is Clicked in the GUI
        function:The function is to take in a text field where the value needs to be set and called in a dialog box with
        file structure view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """
        geomName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'geom Files (*.geom)')
        if geomName:
            self.geomFilePath.setText(geomName)
            self.viewFileButton.setEnabled(True)
            self.statusbar.showMessage("Press the View File button to display the cxi file ", 5000)

    @pyqtSlot()
    def curveToPlot(self):
        """
        A method to select the type of curve to be plotted in the self.graphingSpace
        :return: type of plot to display
        """
        if int(self.eventNumber.text()) >= self.totalEvents:
            self.eventNumber.setText(str(self.totalEvents - 1))

        if self.plotName is None:
            # qtw.QMessageBox.information(self, 'Information', 'Plot a curve first!')
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Information')
            msg.setText("Plot a curve first                                                                    ")
            msg.setInformativeText('If you are thinking of changing to a different ASIC, please plot a Pixel '
                                   'intensity curve first')
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()
        elif self.plotName == 'plotCurve':
            self.plotCurve()
        elif self.plotName == 'plotFit':
            self.plotFit()

    @pyqtSlot()
    def selectDisplay(self):
        """
        Based on the conditions this method calls to draw next/previous image from the *cxi file or create a new view to
         display the *cxi file
        :return: Next/Previous event of the *CXI or window for to display the *CXI file.
        """
        if int(self.eventNumber.text()) >= self.totalEvents:
            self.eventNumber.setText(str(self.totalEvents - 1))

        if self.imageViewer:
            print('Im here YES')
            self.imageViewer.drawImage(int(self.eventNumber.text()))
        else:
            self.viewFiles()
            print('im here NO')

    @pyqtSlot(dict)
    def readPanelDetails(self, inDict):
        """
        :param inDict: Dictionary with ASIIC/panel information coming from the signal once the user clicked on a panel
        :return: Assigns panel details to class variables.
        """
        self.panelDict = inDict
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        self.curveToPlot()

    @pyqtSlot()
    def viewFiles(self):
        """
        Spawn an instance of DisplayImage to display the *cxi file
        :return: A gui with the *cxi file open. similar to cxi view. Also, turns ON "Plot Pixel Intensity",
        "Plot Peak Pixel" and "Plot a Fit" checkBox
        """
        self.eventNumber.setEnabled(True)
        if not self.eventNumber.text():
            self.eventNumber.setEnabled(True)
            self.eventNumber.setText("0")

        if self.imageViewer:
            self.imageViewer.close()
            # self.imageViewer = None
            self.imageViewer = DisplayImage(self.cxiFilePath.text(), self.geomFilePath.text())
            self.imageViewer.drawImage(int(self.eventNumber.text()))
            self.totalEvents = self.imageViewer.size
            self.imageViewer.show()
        else:
            self.imageViewer = DisplayImage(self.cxiFilePath.text(), self.geomFilePath.text())
            self.imageViewer.drawImage(int(self.eventNumber.text()))
            self.totalEvents = self.imageViewer.size
            self.imageViewer.show()

        self.messages = ["Click the Plot Pixel Intensity button", "Click Next and Previous "
                                                                  "buttons to navigate through images",
                         "Click the Fit Plot CheckBox to fit a polynomial"]
        self.showNextMessage()

        # initial panel assignment
        if not self.panelDict:
            self.panelDict = self.imageViewer.outgoingDict
            self.panelName = self.imageViewer.outgoingDict['panel_name']
            self.min_fs = self.imageViewer.outgoingDict['min_fs']
            self.max_fs = self.imageViewer.outgoingDict['max_fs']
            self.min_ss = self.imageViewer.outgoingDict['min_ss']
            self.max_ss = self.imageViewer.outgoingDict['max_ss']

        self.imageViewer.panelSelected.connect(self.readPanelDetails)
        self.clickedNext.connect(self.imageViewer.drawImage)
        self.clickedPrevious.connect(self.imageViewer.drawImage)

        if not self.plotPixelIntensityButton.isEnabled():
            self.plotPixelIntensityButton.setEnabled(True)
            self.poltFitCheckBox.setEnabled(True)
            self.plotPeakPixelButton.setEnabled(True)

    def showNextMessage(self):
        message = self.messages.pop(0)
        self.statusbar.showMessage(message, 3000)
        if self.messages:
            qtc.QTimer.singleShot(3000, self.showNextMessage)

    @pyqtSlot(str)
    def nextEvent(self, eventNumber):
        """
        A method to increment an event
        :param eventNumber: Existing event number
        :return: Existing event number +1
        """
        try:
            if int(self.eventNumber.text()) < self.totalEvents - 1:
                self.eventNumber.setText(str(int(eventNumber) + 1))
            elif int(self.eventNumber.text()) == self.totalEvents - 1:
                self.eventNumber.setText(str(0))

            self.curveToPlot()

            self.clickedNext.emit(int(self.eventNumber.text()))
            # print('im next button and here are the values i have:')
            # if self.imageViewer in gc.get_objects(): print('found it')
            # print(self.imageViewer.fileName)
            print(self.eventNumber.text())

        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText(
                "An error occurred while reading bad events file                                  ")
            msg.setInformativeText(str(e) + " nextEvent()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    @pyqtSlot(str)
    def previousEvent(self, eventNumber):
        """
        A method to decrement an event
        :param eventNumber: Existing event number
        :return: Existing event number -1
        """
        try:
            if int(self.eventNumber.text()) > 0:
                self.eventNumber.setText(str(int(eventNumber) - 1))
            elif int(self.eventNumber.text()) == 0:
                self.eventNumber.setText(str(self.totalEvents - 1))

            self.curveToPlot()

            self.clickedPrevious.emit(int(self.eventNumber.text()))

        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText(
                "An error occurred while reading bad events file                                  ")
            msg.setInformativeText(str(e) + " previousEvent()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    @pyqtSlot(dict,str)
    def writeToFile(self, eventsList, fileName):
        """
        A method to save sorted events
        :param eventsList: dictionary with *cxi file path and event numbers
        :param fileName: save file name
        :return: *.list file
        """
        f = open(fileName, 'w')

        for key in eventsList.keys():
            for i in eventsList[key]:
                f.write(key)
                f.write(' ')
                f.write('//')
                f.write(str(i))
                f.write('\n')

        f.close()
        self.statusbar.showMessage("Saving file %s " % fileName, 2000)

        self.sortForMLGUI.close()

    @pyqtSlot()
    def sortForML(self):
        """
        Spawn an instance of the SortingForML
        :return: good and bad lists to be saved. Turns ON "Train a Model" button
        """

        self.sortForMLGUI = SortingForML(self.cxiFilePath.text(), self.orderOfFit.text(), self.panelDict)
        self.sortForMLGUI.show()
        self.imageViewer.panelSelected.connect(self.sortForMLGUI.readPanelDetails)

        self.sortForMLGUI.readyToSaveGood.connect(self.writeToFile)
        self.sortForMLGUI.readyToSaveBad.connect(self.writeToFile)
        self.MLButton.setEnabled(True)

    @pyqtSlot()
    def machineLearning(self):
        """
        Spawn an instance of ML.
        :return: A trained model. Turns ON "Sort" button
        """
        self.mlGUI = ML(self.panelDict)
        self.mlGUI.show()
        self.imageViewer.panelSelected.connect(self.mlGUI.readPanelDetails)

        self.sortButton.setEnabled(True)

    @pyqtSlot()
    def sort(self):
        """
        Spawn an instance of SortData.
        :return: A sorted list of good and bad events to be saved.
        """
        self.sortDataGUI = SortData(self.mlGUI.model, self.panelDict)
        self.sortDataGUI.show()

        self.sortDataGUI.readyToSaveGood.connect(self.writeToFile)
        self.sortDataGUI.readyToSaveBad.connect(self.writeToFile)

    def returnMaxPixel(self, coeff, xRange):
        """
        returns the value where the coeff is maximized
        coeff (object): out put of a numpy curve fit.
        Range (tup) : a tuple with (min, max)
        """

        storingList = []
        for i in range(xRange[0], xRange[1]):
            storingList.append((np.polyval(coeff, i), i))

        storingList.sort()

        return storingList[-1][1]

    def returnMaxPixelsList(self, fileName, deg=1):
        """
        fileName(str) : name of the file to be open
        deg (int) : order of the fit ex: is the fit a straight line (1) or quadratic (2 or more)
        *args(list) : expects a list of events to be considered. **TO BE IMPLEMENTED**
        """
        maxPixels = []

        with h5py.File(fileName, "r") as f:
            data = f['entry_1']['data_1']['data'][()]

        for i in range(len(data)):
            frame = data[i]

            avgIntensities = []
            for j in range(10, 186):
                avgIntensities.append(np.average(frame[2112:2288, j]))

            fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=deg)

            maxPixels.append(self.returnMaxPixel(fit, (10, 186)))

        return maxPixels

    @pyqtSlot()
    def plotCurve(self):
        """
        A method to plot the vertically averaged intensity profile for the selected panel (Default: p6a0)
        :return: A plot in the  self.graphingSpace
        """
        try:
            fileName = self.cxiFilePath.text()
            eventNumber = int(self.eventNumber.text())

            with h5py.File(fileName, "r") as f:
                data = f['entry_1']['data_1']['data'][()]

            frame = data[eventNumber]

            avgIntensities = []

            if self.panelName in self.detectorLeft:
                for i in range(int(self.min_fs) + 5, int(self.max_fs) - 5):
                    avgIntensities.append(np.average(frame[int(self.min_ss):int(self.max_ss), i]))
            else:
                for i in reversed(range(int(self.min_fs) + 5, int(self.max_fs) - 5)):

                    avgIntensities.append(np.average(frame[int(self.min_ss):int(self.max_ss), i]))

            self.graphWidget.clear()
            self.graphWidget.plot(range(int(self.min_fs) + 5, int(self.max_fs) - 5), avgIntensities, name='data')
            self.graphWidget.setTitle("Panel: " + self.panelName)
            self.graphWidget.setLabel('left', "Avg. Pixel intensity")
            self.graphWidget.setLabel('bottom', "Pixel Number")

            self.plotName = 'plotCurve'

            if not self.sortButton.isEnabled():
                self.nextButton.setEnabled(True)
                self.previousButton.setEnabled(True)

            self.poltFitCheckBox.setChecked(False)

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s -plotCurve" % fileName)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path -plotCurve")

        except IndexError:
            qtw.QMessageBox.critical(self, 'Fail',
                                     'Value you ,%s,  entered is out of bound for this cxi file -plotCurve'
                                     % self.eventNumber.text())

    @pyqtSlot()
    def plotFit(self):
        """
        A method plot the polynomial fit
        fileName(str) : name of the file to be open
        eventNumber(int) : event number for the file
        deg (int) : order of the fit ex: is the fit a straight line (1) or quadratic (2 or more)
        """
        if self.poltFitCheckBox.isChecked():
            try:
                if not self.orderOfFit.text():
                    self.orderOfFit.setEnabled(True)
                    self.orderOfFit.setText("4")

                file_name = self.cxiFilePath.text()
                eventNumber = int(self.eventNumber.text())
                avgIntensities = []
                degry = int(self.orderOfFit.text())

                filename = file_name
                with h5py.File(filename, "r") as f:
                    data = f['entry_1']['data_1']['data'][()]

                frame = data[int(eventNumber)]

                if self.panelName in self.detectorLeft:
                    for i in range(int(self.min_fs) + 5, int(self.max_fs) - 5):
                        avgIntensities.append(np.average(frame[int(self.min_ss):int(self.max_ss), i]))
                else:
                    for i in reversed(range(int(self.min_fs) + 5, int(self.max_fs) - 5)):
                        avgIntensities.append(np.average(frame[int(self.min_ss):int(self.max_ss), i]))

                fit = np.polyfit(np.arange(int(self.min_fs) + 5, int(self.max_fs) - 5), avgIntensities, deg=degry)

                self.graphWidget.clear()
                self.graphWidget.plot(range(int(self.min_fs) + 5, int(self.max_fs) - 5), avgIntensities, name='data')
                self.graphWidget.plot(range(int(self.min_fs) + 5, int(self.max_fs) - 5),
                                      np.polyval(fit, range(int(self.min_fs) + 5, int(self.max_fs) - 5)),
                                      name='fit', pen=pg.mkPen(color='r', width=2))
                self.graphWidget.setTitle("Panel: " + self.panelName)
                self.graphWidget.setLabel('left', "Avg. Pixel intensity")
                self.graphWidget.setLabel('bottom', "Pixel Number")
                self.graphWidget.addLegend()

                self.plotName = 'plotFit'

                if not self.sortForMLButton.isEnabled():
                    self.sortForMLButton.setEnabled(True)
                    self.nextButton.setEnabled(True)
                    self.previousButton.setEnabled(True)

                self.orderOfFit.setEnabled(True)

            except FileNotFoundError:
                qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

            except ValueError:
                qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

            except IndexError:
                qtw.QMessageBox.critical(self, 'Fail', 'Value you entered is out of bound -plotFit()')

        else:
            self.plotCurve()

    @pyqtSlot(str)
    def plotMaxPixels(self, fileName):
        """
        :param fileName: path to the *CXI file (file name)
        :return: a plot with pixel with maximum value for the polynomial fit for all the events
        """
        try:
            y = self.returnMaxPixelsList(fileName, deg=int(self.orderOfFit.text()))
            x = range(len(y))
            self.graphWidget.clear()
            self.graphWidget.plot(x, y, pen=None, symbol='o')
            self.graphWidget.setLabel('left', "Pixel Number")
            self.graphWidget.setLabel('bottom', "Frame Number")

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % fileName)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

    # def write(self, message):
    #     print("I got the message:", message)
    #     # Update the status bar with the message
    #     self.statusbar.showMessage(message)

    def closeEvent(self, QCloseEvent):
        """
        :param QCloseEvent: a QCloseEvent
        :return: Closes any of the opened GUIs
        """
        if self.imageViewer:
            self.imageViewer.close()

        if self.sortForMLGUI:
            self.sortForMLGUI.close()

        if self.mlGUI:
            self.mlGUI.close()

        if self.sortDataGUI:
            self.sortDataGUI.close()

    def setBusy(self):
        self.busyLight.show()
        self.idleLight.hide()

    def setIdle(self):
        self.busyLight.hide()
        self.idleLight.show()


# main .
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
