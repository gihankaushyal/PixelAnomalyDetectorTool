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
    """
        This class inherits from the PyQt5 QWidget class and provides a custom widget for displaying images.
        The widget emits a signal when the user selects an image panel, providing a dictionary with panel
        information.
    """

    panelSelected = qtc.pyqtSignal(dict)

    def __init__(self, fileName, geometry):

        """

        :param fileName: name of the *cxi file (with the full path)
        :param geometry: path to the geometry file
        """
        # Call the superclass constructor
        super(DisplayImage, self).__init__()

        # setting the size and location of the window
        self.outgoingDict = None
        self.setGeometry(10, 100, 600, 600)

        # assigning the file name and the geometry
        self.fileName = fileName
        self.geometryName = geometry

        # Initialize class variables
        self.eventNumber = None
        self.imageToDraw = None
        self.cxi = None
        self.size = None
        self.panelLocFromGeom = {}
        self.panelFsSs = {}
        self.panelsXYEdges = {}
        self.outgoingDict = {}

        # Create the main window for display the data
        self.mainWidget = pg.ImageView()

        # Add a checkbox for showing found peaks
        self.foundPeaksCheckBox = qtw.QCheckBox('Found Peaks')

        # Connect the checkbox to the drawImage method
        self.foundPeaksCheckBox.stateChanged.connect(lambda: self.drawImage(self.eventNumber))

        # Create a layout and add the main window and the checkbox to it
        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.mainWidget)
        self.layout.addWidget(self.foundPeaksCheckBox)

        # Read the geometry file
        try:
            self.parser = GeometryFileParser(self.geometryName)
            self.geometry = self.parser.pixel_map_for_cxiview()

            # Create a dictionary for panel boundaries
            for panelName in self.parser.dictionary['panels'].keys():
                self.panelFsSs[panelName] = [self.parser.dictionary['panels'][panelName]['min_fs'],
                                             self.parser.dictionary['panels'][panelName]['max_fs'],
                                             self.parser.dictionary['panels'][panelName]['min_ss'],
                                             self.parser.dictionary['panels'][panelName]['max_ss']]

            # Create a dictionary for panel locations
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

        # Add an overlapping canvas for found peaks
        self.foundPeaksCanvas = pg.ScatterPlotItem()
        self.mainWidget.getView().addItem(self.foundPeaksCanvas)

        # Add a canvas for displaying panel edges
        self.panelEdgesCanvas = pg.PlotDataItem()
        self.mainWidget.getView().addItem(self.panelEdgesCanvas)

        # connecting a mouse clicked event to a select panel method
        self.mainWidget.getView().scene().sigMouseClicked.connect(self.selectPanel)

        # Set the layout for the widget
        self.setLayout(self.layout)

        # Handle what happens after the widget is closed
        self.isClosed = False

        # Set the widget to delete itself on close
        self.setAttribute(qtc.Qt.WA_DeleteOnClose)

    @pyqtSlot(int)
    def drawImage(self, eventNumber):
        """
         Read and display data for a given event number from a CXI file.
        :param eventNumber: event number to be displayed
        :return: pixel map from the cxi file
        """

        try:
            # Set the current event number
            self.eventNumber = eventNumber

            # Read the CXI file for the given event number, with data, slab size, and peaks
            self.cxi = fileTools.read_cxi(self.fileName, frameID=self.eventNumber, data=True, slab_size=True,
                                          peaks=True)
            # Get the size of the CXI file
            self.size = self.cxi['stack_shape'][0]

            # Read the 'data' from the CXI file
            imgData = self.cxi['data']

            # Convert the data into a pixel map and apply the geometry (x and y coordinates)
            self.imageToDraw = imgTools.pixel_remap(imgData, self.geometry['x'], self.geometry['y'])

            # Set the pixel map as the image to be displayed in the main window
            self.mainWidget.setImage(self.imageToDraw)

            # Set the window title to show the current event number and the total number of events in the file
            self.setWindowTitle("Showing %i of %i " % (self.eventNumber, self.size - 1))

            # If the event number is 0, draw the initial panel
            if self.eventNumber == 0:
                self.drawInitialPanel()

            # Draw the peaks on the image
            self.drawPeaks()

        # Handle any exceptions that might occur while processing the data
        except IndexError as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Information')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " drawImage()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    def drawPeaks(self):
        """
        Draw circles around the found peaks extracted from the cxi file.
        :return: None
        """
        try:
            # Check if the "found peaks" checkbox is checked
            if self.foundPeaksCheckBox.isChecked():
                peaks_x = []
                peaks_y = []

                # Get the number of peaks, x and y positions from the cxi file
                n_peaks = self.cxi['n_peaks']
                x_data = self.cxi['peakXPosRaw']
                y_data = self.cxi['peakYPosRaw']

                # Loop through all the peaks
                for i in range(0, n_peaks):
                    peak_fs = x_data[i]
                    peak_ss = y_data[i]

                    # Calculate the peak position in the slab
                    peak_in_slab = int(round(peak_ss)) * self.cxi['stack_shape'][2] + int(round(peak_fs))

                    # Append the x and y positions of the peaks to the lists
                    peaks_x.append(self.geometry['x'][peak_in_slab] + self.imageToDraw.shape[0] / 2)
                    peaks_y.append(self.geometry['y'][peak_in_slab] + self.imageToDraw.shape[1] / 2)

                # Create a blue pen for drawing circles around the peaks
                ring_pen = pg.mkPen('b', width=2)

                # Draw circles around the peaks using the blue pen
                self.foundPeaksCanvas.setData(peaks_x, peaks_y, symbol='o', size=10, pen=ring_pen, brush=(0, 0, 0, 0),
                                              pxMode=False)
            else:
                # If the "found peaks" checkbox is not checked, clear the canvas
                self.foundPeaksCanvas.clear()

        # Handle any exceptions that might occur while processing the data
        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " drawPeaks()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    def drawInitialPanel(self):
        """
        Draw the initial panel (predetermined to be p6a0).
        :return:  a dictionary (self.outgoingDict)
        """

        try:
            # Iterate through all panel names in the panelLocFromGeom dictionary
            for panelName in self.panelLocFromGeom.keys():
                x_edges = []
                y_edges = []

                # Calculate the x and y edges of each panel
                for i in range(4):
                    edge_fs = self.panelLocFromGeom[panelName][i][0]
                    edge_ss = self.panelLocFromGeom[panelName][i][1]
                    peak_in_slab = int(round(edge_ss)) * self.cxi['stack_shape'][2] + int(round(edge_fs))
                    x_edges.append(self.geometry['x'][peak_in_slab] + self.imageToDraw.shape[0] / 2)
                    y_edges.append(self.geometry['y'][peak_in_slab] + self.imageToDraw.shape[1] / 2)

                # Close the square by appending the first edge to the end of the list
                x_edges.append(x_edges[0])
                y_edges.append(y_edges[0])

                # Store the x and y edges for each panel in the panelsXYEdges dictionary
                self.panelsXYEdges[panelName] = [x_edges, y_edges]

            # Create a red pen with width 3 for drawing panel edges
            pen = pg.mkPen('r', width=3)

            # Plot a square along the edges of the selected panel (p6a0)
            self.panelEdgesCanvas.setData(self.panelsXYEdges['p6a0'][0],
                                          self.panelsXYEdges['p6a0'][1], pen=pen)

            # Create a dictionary with panel information
            self.outgoingDict = {'panel_name': 'p6a0',
                                 'min_fs': self.panelFsSs['p6a0'][0], 'max_fs': self.panelFsSs['p6a0'][1],
                                 'min_ss': self.panelFsSs['p6a0'][2], 'max_ss': self.panelFsSs['p6a0'][3]}

            # Emit the panelSelected signal with the outgoingDict
            self.panelSelected.emit(self.outgoingDict)

        # Handle any exceptions that might occur while processing the data
        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " drawInitialPanel()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    def selectPanel(self, event):
        """
        Draw a border around the selected ASIC.
        :param event: A mouse clicked event
        :return: Draw a Red boarder around the selected ASIC
        """

        try:
            # Get the position of the mouse click
            pos = event.scenePos()
            # Check if the mouse click is within the main widget's view
            if self.mainWidget.getView().sceneBoundingRect().contains(pos):
                # Convert the scene position to view position
                mouse_point = self.mainWidget.getView().mapSceneToView(pos)
                x_mouse = int(mouse_point.x())
                y_mouse = int(mouse_point.y())

                # Iterate through all the panel names in the panelsXYEdges dictionary
                for panelName in self.panelsXYEdges.keys():
                    if x_mouse in range(int(min(self.panelsXYEdges[panelName][0])),
                                        int(max(self.panelsXYEdges[panelName][0]))) \
                            and y_mouse in range(int(min(self.panelsXYEdges[panelName][1])),
                                                 int(max(self.panelsXYEdges[panelName][1]))):

                        # Create a red pen with width 3 for drawing panel edges
                        pen = pg.mkPen('r', width=3)

                        # Plot a square along the edges of the selected panel
                        self.panelEdgesCanvas.setData(self.panelsXYEdges[panelName][0],
                                                      self.panelsXYEdges[panelName][1], pen=pen)

                        # Create a dictionary with panel information
                        self.outgoingDict = {'panel_name': panelName,
                                             'min_fs': self.panelFsSs[panelName][0],
                                             'max_fs': self.panelFsSs[panelName][1],
                                             'min_ss': self.panelFsSs[panelName][2],
                                             'max_ss': self.panelFsSs[panelName][3]}

                        # Emit the panelSelected signal with the outgoingDict
                        self.panelSelected.emit(self.outgoingDict)

        # Handle any exceptions that might occur while processing the data
        except Exception as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " selectPanel()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    def closeEvent(self, QCloseEvent):
        self.isClosed = True


class SortingForML(qtw.QWidget):
    """
       This class inherits from the PyQt5 QWidget class and provides a custom widget for sorting images
       to be used for machine learning. The widget emits two signals, one for each category of images
       (good or bad) when the user sorts the images.
    """

    # Define two PyQt signals to indicate when the user is ready to save an image as good or bad
    readyToSaveGood = qtc.pyqtSignal(dict, str)
    readyToSaveBad = qtc.pyqtSignal(dict, str)

    def __init__(self, fileName, oft, inDict):
        """
            Initialize the SortingForML custom widget.

            :param fileName: name of the file to be sorted
            :param oft: order of the fit for the polynomial
            :param inDict: dictionary with detector panel information
        """

        # Call the superclass constructor
        super(SortingForML, self).__init__()

        # Initialize instance variables
        self.badEvents = None
        self.goodEvents = None
        self.inflectionPoint1List = None
        self.inflectionPoint2List = None
        self.data = None
        self.setWindowTitle('Sorting for Machine Learning')

        # Load the user interface layout from a .ui file
        uic.loadUi("UI/sortForMLGUI.ui", self)

        # Set up plotting with matplotlib
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

        # Set the instance variables using the parameters
        self.fileName = fileName
        self.orderOfFit = oft
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        # Set initial values and step size for the spin boxes (value ranges for inflection points)
        self.doubleSpinBoxIF1.setValue(15)
        self.doubleSpinBoxIF2.setValue(15)
        self.doubleSpinBoxIF1.setSingleStep(0.05)
        self.doubleSpinBoxIF2.setSingleStep(0.05)

        # Connect the plotInflectionPointsButton signal to the plotInflectionPoints method
        # self.plotInflectionPointsButton.clicked.connect(self.plotInflectionPoints)
        self.plotInflectionPoints()
        self.sortButton.clicked.connect(self.sort)

        # Set the widget to delete itself on close
        self.setAttribute(qtc.Qt.WA_DeleteOnClose)

    @pyqtSlot(dict)
    def readPanelDetails(self, inDict):
        """
            Read panel details from the provided dictionary and update instance variables accordingly.

            :param inDict: Dictionary with ASIIC/panel information coming from the signal once the user clicked on
                a panel
            :return: Assigns panel detail and calls the plotInflectionPoints method to update the plot
        """

        # Update instance variables using the input dictionary
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        # Call the plotInflectionPoints method to update the plot based on the new panel details
        self.plotInflectionPoints()

    def plotInflectionPoints(self):
        """
        Plot two histograms for inflection point1 and inflection point2 on the self.graphSpace of
            the sortForMlGUI.
        :return: None
        """

        # Initialize lists for inflection points
        self.inflectionPoint1List = []
        self.inflectionPoint2List = []

        try:
            # Read the data from the file
            with h5py.File(self.fileName, "r") as f:
                self.data = f['entry_1']['data_1']['data'][()]

            # Iterate through the data and calculate inflection points
            for i in range(len(self.data)):
                frame = self.data[i]

                # Calculate average intensities for the specified range
                avgIntensities = []
                for j in range(self.min_fs + 5, self.max_fs - 5):
                    avgIntensities.append(np.average(frame[self.min_ss:self.max_ss, j]))

                # Fit a polynomial to the average intensities
                fit = np.polyfit(np.arange(self.min_fs + 5, self.max_fs - 5), avgIntensities, deg=int(self.orderOfFit))

                # Calculate the inflection points (second derivative of the polynomial)
                # Handle various exceptions and warnings during the calculation
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
                        # Show a message box if there is an IndexError
                        msg = qtw.QMessageBox()
                        msg.setText(str(e).capitalize())
                        msg.setInformativeText('Try entering a different order for the polynomial')
                        msg.setIcon(qtw.QMessageBox.Critical)
                        msg.exec_()
                    except ValueError:
                        # Show a message box if there is a ValueError
                        qtw.QMessageBox.information(self, 'Skip', 'Calculation Error! \n \n Skipping the frame %i' % i)
                        continue
                    except Warning as e:
                        # Show a message box if there is a Warning
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

        # Create a DataFrame and plot the histograms using seaborn
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

        # Update the display with inflection point values and enable buttons and checkboxes
        self.inflectionPoint1.setEnabled(True)
        self.inflectionPoint1.setText(str(round(np.median(df['Inflection_point1'].dropna()), 2)))
        self.inflectionPoint2.setEnabled(True)
        self.inflectionPoint2.setText(str(round(np.median(df['Inflection_point2'].dropna()), 2)))
        self.sortButton.setEnabled(True)

    @pyqtSlot()
    def sort(self):
        """

        :return: :return: two dictionaries (for both good and bad events) with file names and events sorted out by user
        defined threshold for inflection points and spread of the distribution

        """
        # Get the postfix for file saving
        tag = str(self.fileName).split('/')[-1].split('.')[0]

        # Get the directory to save the sorted files
        fileSaveLocation = qtw.QFileDialog.getExistingDirectory(self, caption='Select Save Location', directory=' ',
                                                                options=qtw.QFileDialog.DontUseNativeDialog)
        if fileSaveLocation != "":
            # Initialize goodEvents and badEvents dictionaries
            self.goodEvents = {}
            self.badEvents = {}

            # Lists to store good and bad event indices
            goodList = []
            badList = []

            try:
                # Iterate through the data and inflection points (x1, x2)
                for (i, x1, x2) in zip(range(len(self.data)), self.inflectionPoint1List, self.inflectionPoint2List):

                    # Check if the inflection points are within user-defined thresholds
                    if (float(self.inflectionPoint1.text()) - self.doubleSpinBoxIF1.value()) <= x1 <= (
                            float(self.inflectionPoint1.text()) + self.doubleSpinBoxIF1.value()) \
                            and \
                            (float(self.inflectionPoint2.text()) - self.doubleSpinBoxIF2.value()) <= x2 <= (
                            float(self.inflectionPoint2.text()) + self.doubleSpinBoxIF2.value()):

                        # If the inflection points are within thresholds, add the index to the goodList
                        goodList.append(i)
                    else:
                        # If the inflection points are not within thresholds, add the index to the badList
                        badList.append(i)

                # Update the goodEvents and badEvents dictionaries with goodList and badList
                self.goodEvents[str(self.fileName)] = goodList
                self.badEvents[str(self.fileName)] = badList

                # Display a message to inform the user that the sorted files have been saved
                qtw.QMessageBox.information(self, "Completed", "Sorted files have being saved.")

                # Emit the readyToSaveGood and readyToSaveBad signals with the dictionaries and file save location
                self.readyToSaveGood.emit(self.goodEvents,
                                          fileSaveLocation + '/' + 'goodEvents-advanceSort-%s.list' % tag)
                self.readyToSaveBad.emit(self.badEvents, fileSaveLocation + '/' + 'badEvents-advanceSort-%s.list' % tag)

            except Exception as e:
                # If there is an exception, display an error message using qtw.QMessageBox
                msg = qtw.QMessageBox()
                msg.setWindowTitle('Error')
                msg.setText("An error occurred while sorting the file %s                              " % self.fileName)
                msg.setInformativeText(str(e) + " sort()")
                msg.setIcon(qtw.QMessageBox.Information)
                msg.exec_()

        else:
            # If the user does not provide a valid file save location, display a warning message using qtw.QMessageBox
            qtw.QMessageBox.warning(self, 'Warning', 'Please Select a Save Location for sorted files')


class ML(qtw.QMainWindow):

    def __init__(self, inDict):
        """
         Initialize the ML QMainWindow with a dictionary containing detector panel information.
        :param inDict: dictionary with detector panel information
        """
        # Call the superclass constructor
        super(ML, self).__init__()

        # Initialize instance variables
        self.messages = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        uic.loadUi("UI/machineLearningGUI.ui", self)

        # Load the UI file for the Machine Learning GUI
        self.setWindowTitle('Machine Learning')

        # Get the detector panel information from the input dictionary
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        # Set the default values for train and test splits
        self.trainSplit.setText('70')
        self.testSplit.setText('30')

        # Connect buttons to their corresponding slots
        self.browseButton.clicked.connect(self.browseFiles)
        self.trainButton.clicked.connect(self.buttonClicked)
        self.testButton.clicked.connect(self.test)
        self.resetButton.clicked.connect(self.reset)
        self.saveButton.clicked.connect(self.save)
        self.comboBox.activated.connect(self.comboBoxChanged)

        # Setup layouts and canvases for displaying confusion matrix and classification report
        # Confusion Matrix
        self.layoutConfusionMatrix = qtw.QHBoxLayout()
        self.figureConfusionMatrix = plt.figure()
        self.canvasConfusionMatrix = FigureCanvasQTAgg(self.figureConfusionMatrix)
        self.layoutConfusionMatrix.addWidget(self.canvasConfusionMatrix)
        self.confusionMatrix.setLayout(self.layoutConfusionMatrix)

        # Classification report
        self.layoutClassificationReport = qtw.QHBoxLayout()
        self.figureClassificationReport = plt.figure()
        self.canvasClassificationReport = FigureCanvasQTAgg(self.figureClassificationReport)
        self.layoutClassificationReport.addWidget(self.canvasClassificationReport)
        self.classificationReport.setLayout(self.layoutClassificationReport)

        # Add busy and idle lights to the status bar
        self.busyLight = BusyLight()
        self.idleLight = IdleLight()
        self.statusbar.addPermanentWidget(self.busyLight)
        self.statusbar.addPermanentWidget(self.idleLight)
        self.idleLight.show()
        self.busyLight.hide()

        # Show a message on the status bar
        self.statusbar.showMessage("Point to where you have the data for model training", 3000)

        # Set the attribute to delete the window on close
        self.setAttribute(qtc.Qt.WA_DeleteOnClose)

    def setBusy(self):
        """
        Change the light indicator to "busy" status by showing the busy light and hiding the idle light.
        :return: Change the light to busy
        """
        self.busyLight.show()
        self.idleLight.hide()

    def setIdle(self):
        """
        Change the light indicator to "idle" status by hiding the busy light and showing the idle light.
        :return: Change the light to Idle
        """
        self.busyLight.hide()
        self.idleLight.show()

    def showNextMessage(self, messageList):
        """
            Display the next message from the messageList in the status bar for 3 seconds.
            If there are more messages in the messageList, recursively call the function to display the next message.

            :param messageList: List of messages to display in the status bar
            :return: None
        """
        message = messageList.pop(0)
        self.statusbar.showMessage(message, 3000)
        if messageList:
            qtc.QTimer.singleShot(3000, lambda: self.showNextMessage(messageList))

    @pyqtSlot()
    def browseFiles(self):
        """
            This method gets triggered when the browse button is clicked in the GUI.
        Function: The function is to take in a text field where the value needs to be set and called in a dialog box
        with file structure view starting at the 'root' and lets the user select the file they want and set the file path
        to the text field.
        """

        self.setBusy()

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
        This method is used to read the panel details from the provided dictionary (inDict).

        :param inDict: Dictionary with ASIC/panel information coming from the signal once the user clicked on a panel
        :return: Assigns panel details to class variables from inDict
        """
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

    def modelSelection(self):
        """
        This method handles the selection of a scikit-learn model based on the user's choice from the GUI.

        :return: user selected scikit-learn model as a boolean, indicating if the model has been successfully selected
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
        """
            This method checks if the train-test split entered by the user is valid and sums up to 100%.

            :return: boolean, indicating if the train-test split is valid
        """

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

    def dataPrep(self):
        """
        This method looks into the folder where the sorted files are stored (by sort() in SortingForMl) and prepares the
        data for training and testing.

        :return: Sets X_train, X_test, y_train, y_test as class attributes
        """


        from sklearn.model_selection import train_test_split
        folder = self.parentDirectory.text()

        # Processing bad events
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

        # Processing good events
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

        # Preparing the data for training and testing
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
        This method gets triggered when the "Train" button is pressed. It asks the user to confirm the selected panel
        for training the machine learning model. Based on the user's response, it either proceeds with model training or
        allows the user to go back and select a different ASCI for training.

        :return: None
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
        Method to train the user-selected model using the data from the selected ASCI.

        :param i: QMessageBox output (&Yes or &No)
        :return: None
        """


        if i.text() == '&Yes':
            self.setBusy()
            self.trainButton.setEnabled(False)
            if self.modelSelection() and self.checkTrainTestSplit():
                self.dataPrep()
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
        Method to test the validity of the trained model.
        :return: None
        """
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
        self.predictions = self.model.predict(self.X_test)

        self.setBusy()

        self.figureConfusionMatrix.clear()
        self.figureClassificationReport.clear()

        # Displaying the confusion matrix

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

        # Displaying the classification report as a heatmap
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
        Method to clear out the output from the test() method.
        :return: None
        """
        # Clear the confusion matrix and classification report figures
        self.figureConfusionMatrix.clear()
        self.figureClassificationReport.clear()

        # Enable the "Train" button and disable the "Test" button
        self.trainButton.setEnabled(True)
        self.testButton.setEnabled(False)

        # Reset the model selection combo box to its initial state
        self.comboBox.setCurrentIndex(0)

        # Reset the train and test split percentage to default values (70 and 30)
        self.trainSplit.setText('70')
        self.testSplit.setText('30')

    @pyqtSlot(int)
    def comboBoxChanged(self, index):
        """
            Method to handle the change of the model selection in the combo box.
            :param index: The index of the newly selected model in the combo box
            :return: None
        """

        # Reset the interface to its initial state
        self.reset()

        # Set the current index of the combo box to the given index
        self.comboBox.setCurrentIndex(index)

    @pyqtSlot()
    def save(self):
        """
            Method to save the trained machine learning model as a pickle file.
            :return: None
        """

        # Open a file dialog to allow the user to select a file location and name for the saved model
        filename, _ = qtw.QFileDialog.getSaveFileName(self, "Save File", "", "Pickle Files (*.pkl)")

        # If a filename is provided
        if filename:
            # Open the file in write binary mode
            with open(filename, 'wb') as f:
                # Use pickle.dump to save the model to the file
                pickle.dump(self.model, f)


class SortData(qtw.QWidget):
    """
       Custom QWidget subclass that is used to create a custom widget for sorting data in a PyQt5 application.
    """

    # Define a PyQt signal named 'readyToSaveGood' that takes a dictionary and a string as arguments
    readyToSaveGood = qtc.pyqtSignal(dict, str)

    # Define a PyQt signal named 'readyToSaveBad' that takes a dictionary and a string as arguments
    readyToSaveBad = qtc.pyqtSignal(dict, str)

    def __init__(self, model, inDict):
        """
            Initialize the SortData class with a trained model and a dictionary containing detector panel information.

            :param model: trained model
            :param inDict: dictionary with detector panel information
        """

        # Call the __init__ method of the parent class (QWidget)
        super(SortData, self).__init__()

        # Load the UI file (sortDataGUI.ui) to set up the user interface
        uic.loadUi('UI/sortDataGUI.ui', self)
        self.setWindowTitle('Sort Data')

        # Set class attributes with the passed model and detector panel information
        self.model = model
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        # Initialize empty dictionaries for storing good and bad events
        self.goodEvents = {}
        self.badEvents = {}

        # Find and set up the table widget in the UI
        self.tableWidget = self.findChild(qtw.QTableWidget, 'tableWidget')
        self.tableWidget.setColumnWidth(0, 350)

        # Connect signals to slots (methods) for handling user interactions
        self.browseButton.clicked.connect(self.browseFiles)
        self.sortButton.clicked.connect(self.buttonClicked)

        # Set the attribute to delete the widget when it is closed
        self.setAttribute(qtc.Qt.WA_DeleteOnClose)

    @pyqtSlot()
    def browseFiles(self):
        """
        This method gets triggered when the browse button is Clicked in the GUI
        function: The function is to take in a text field where the value needs to be set and called in a dialog box
                with file structure view starting at the 'root' and lets the user select the file they want and set the
                file path to the test field.
        """

        # Open a file dialog to let the user select a folder
        folderName = qtw.QFileDialog.getExistingDirectory(self, caption='Select Folder', directory=' ')

        # If a folder is selected (folderName is not empty)
        if folderName != "":
            # Set the selected folder path to the text field in the user interface
            self.folderPath.setText(folderName)
            # Call the showFiles() method to display the selected files
            self.showFiles()

    def showFiles(self):
        """

        :return: Displays available files in the selected folder in self.availableFiles (QTextEdit)
        """

        # Get the folder path from the folderPath QLineEdit
        folder = self.folderPath.text()

        # Use the pathlib module to find all files with the .cxi extension in the folder
        files = Path(folder).glob('*.cxi')

        # Use the pathlib module to find all files with the .cxi extension in the folder
        for file in files:
            # Append the filename (without the folder path) to the availableFiles QTextEdit
            self.availableFiles.append(str(file).split('/')[-1])

        # Enable the sortButton QPushButton
        self.sortButton.setEnabled(True)

    @pyqtSlot()
    def buttonClicked(self):
        """
            Asks a user a Question/ Waring about the model that was trained
            :return: Yes or No
        """

        # Create a QMessageBox object
        msg = qtw.QMessageBox()
        msg.setWindowTitle('Question')
        msg.setText("Panel Selected: %s                                             " % self.panelName)
        msg.setInformativeText('Please Note! Machine Learning model was trained based on the data from the %s panel. '
                               'Make sure that you are sorting based on %s panel. If not, train a new model for your '
                               'frame of choice. Would you wish to continue?' % (self.panelName, self.panelName))
        msg.setIcon(qtw.QMessageBox.Question)
        msg.setStandardButtons(qtw.QMessageBox.Yes | qtw.QMessageBox.No)
        msg.setDefaultButton(qtw.QMessageBox.Yes)

        # Connect the buttonClicked signal to the sort method
        msg.buttonClicked.connect(self.sort)

        # Show the QMessageBox
        msg.exec_()

    def sort(self, i):
        """
            Sort *cxi files using the trained model
            :return: two separate *.list files for good and bad events for the cxi files
        """

        if i.text() == '&Yes': # Check if the user has confirmed to proceed with the sorting process

            self.sortButton.setEnabled(False)
            folder = self.folderPath.text()

            # Get the location where the sorted files will be saved
            fileSaveLocation = qtw.QFileDialog.getExistingDirectory(self, caption='Select Where You Want to Save the'
                                                                                  'Sorted Files', directory=' ',
                                                                    options=qtw.QFileDialog.DontUseNativeDialog)
            files = Path(folder).glob('*.cxi')
            row = 0
            self.tableWidget.setRowCount(len(list(files)))

            files = Path(folder).glob('*.cxi')
            for file in files: # Iterate through the *cxi files and process them individually

                tag = str(file).split('/')[-1].split('.')[0]

                self.goodEvents = {}
                self.badEvents = {}

                # goodList to store all the events with expected pixel intensities for the file
                goodList = []
                # badList to store all the events with detector artifacts for the file
                badList = []

                with h5py.File(file, "r") as f: # Open the file using the h5py library and read the data
                    data = f['entry_1']['data_1']['data'][()]

                for i in range(data.shape[0]): # Loop through each event in the data

                    # Preprocess the frame and use the trained model to make a prediction
                    frame = data[i][self.min_ss:self.max_ss, self.min_fs + 5:self.max_fs - 5].flatten()
                    predictions = self.model.predict(frame.reshape(1, 31675))

                    # Based on the prediction, append the event index to either the goodList or the badList
                    if predictions:
                        goodList.append(i)
                    else:
                        badList.append(i)

                # Store the good and bad events in their respective dictionaries
                self.goodEvents[str(file)] = goodList
                self.badEvents[str(file)] = badList

                self.readyToSaveGood.emit(self.goodEvents,
                                          fileSaveLocation + '/' + 'goodEvents-modelSort-%s.list' % tag)
                self.readyToSaveBad.emit(self.badEvents, fileSaveLocation + '/' + 'badEvents-modelSort-%s.list' % tag)

                # Update the table widget in the GUI with the results of the sorting process
                self.tableWidget.setItem(row, 0, qtw.QTableWidgetItem(str(file).split('/')[-1]))
                self.tableWidget.setItem(row, 1, qtw.QTableWidgetItem(str(len(self.goodEvents[str(file)]))))
                self.tableWidget.setItem(row, 2, qtw.QTableWidgetItem(str(len(self.badEvents[str(file)]))))
                row += 1
                self.sortButton.setEnabled(False) # Disable the "Sort" button to prevent further sorting


class BusyLight(qtw.QWidget):
    """
        BusyLight is a custom QWidget that serves as a status indicator light when the GUI is busy.
        It alternates its color between yellow and transparent, creating a blinking effect.
    """

    def __init__(self):
        super().__init__()
        # Set the fixed size of the widget
        self.setFixedSize(12, 12)
        # Initialize the default color to darkorange
        self.color = qtg.QColor('darkorange')
        # Create a QTimer and connect it to the update method
        self.timer = qtc.QTimer(self)
        self.timer.timeout.connect(self.update)
        # Start the timer with 500 ms intervals
        self.timer.start(500)

    def paintEvent(self, event):
        """
            This method handles the paint event of the widget, drawing its appearance.
        """
        # Create a QPainter for drawing the widget
        painter = qtg.QPainter(self)
        # Enable antialiasing for smooth rendering
        painter.setRenderHint(qtg.QPainter.Antialiasing)
        # Set the pen to NoPen to prevent any outline from being drawn
        painter.setPen(qtc.Qt.NoPen)
        # Set the brush to the current color of the light
        painter.setBrush(self.color)
        # Draw an ellipse within the widget's rectangular area, creating the circular shape of the light
        painter.drawEllipse(self.rect())

    def update(self):
        """
            This method is called every 500 milliseconds by the QTimer.
            It toggles the color of the light and triggers a repaint of the widget.
        """
        # Toggle the color between yellow and transparent
        if self.color == qtc.Qt.yellow:
            self.color = qtc.Qt.transparent
        else:
            self.color = qtc.Qt.yellow
        # Call the base class's update method to ensure the widget is repainted with the new color
        super().update()


class IdleLight(qtw.QWidget):
    """
    IdleLight is a custom QWidget that serves as a status indicator light when the GUI is idle.
    It displays a static springgreen color to indicate the idle state.
    """

    def __init__(self):
        super().__init__()
        # Set the fixed size of the widget
        self.setFixedSize(12, 12)

    def paintEvent(self, event):
        """
        This method handles the paint event of the widget, drawing its appearance.
        """
        # Create a QPainter for drawing the widget
        painter = qtg.QPainter(self)
        # Enable antialiasing for smooth rendering
        painter.setRenderHint(qtg.QPainter.Antialiasing)
        # Set the pen to NoPen to prevent any outline from being drawn
        painter.setPen(qtc.Qt.NoPen)
        # Set the brush to the springgreen color
        painter.setBrush(qtg.QColor('springgreen'))
        # Draw an ellipse within the widget's rectangular area, creating the circular shape of the light
        painter.drawEllipse(self.rect())


class Hitfinding():
    pass


class MainWindow(qtw.QMainWindow):
    """
        MainWindow is a custom QMainWindow that inherits from the PyQt5 QMainWindow class.
        It contains two custom signals: clickedNext and clickedPrevious.
        These signals are emitted when the user clicks the "Next" or "Previous" buttons within the GUI.
    """

    # Define custom signal clickedNext to emit when the "Next" button is clicked
    clickedNext = qtc.pyqtSignal(int)

    # Define custom signal clickedPrevious to emit when the "Previous" button is clicked
    clickedPrevious = qtc.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        # Call the superclass constructor
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load UI from file
        uic.loadUi("UI/mainwindow-3.ui", self)
        # self.setGeometry(700, 100, 820, 710)

        # Connect UI elements to functions
        self.cxiBrowseButton.clicked.connect(self.browseFiles)
        self.trainingFileBrowseButton.clicked.connect(self.browseFiles)
        self.geomBrowseButton.clicked.connect(self.browseGeom)
        self.geomBrowseButton_2.clicked.connect(self.browseGeom)
        self.viewFileButton.clicked.connect(self.viewFiles)
        self.viewTrainingFilesButton.clicked.connect(self.viewFilesList)
        self.plotPixelIntensityButton.clicked.connect(self.plotCurve)
        self.poltFitCheckBox.clicked.connect(self.plotFit) # the vertically average intensity profile
        self.plotPeakPixelButton.clicked.connect(lambda: self.plotMaxPixels(self.cxiFilePath.text()))
        self.sortButton.clicked.connect(self.sort)
        self.sortForMLButton.clicked.connect(self.sortForML)
        self.MLButton.clicked.connect(self.machineLearning)
        self.orderOfFit.editingFinished.connect(self.plotFit)
        self.eventNumber.editingFinished.connect(self.curveToPlot)
        self.eventNumber.editingFinished.connect(self.selectDisplay)

        # Incrementing through event numbers
        self.nextButton.clicked.connect(lambda: self.nextEvent(self.eventNumber.text()))
        self.nextButton_2.clicked.connect(lambda: self.nextEvent(self.eventNumber.text()))
        self.previousButton.clicked.connect(lambda: self.previousEvent(self.eventNumber.text()))
        self.previousButton_2.clicked.connect(lambda: self.previousEvent(self.eventNumber.text()))

        # Set initial message on the status bar
        self.statusbar.showMessage("Browse for CXI file or a list a CXI files ", 5000)

        # Initialize the popup windows
        self.imageViewer = None
        self.sortForMLGUI = None
        self.mlGUI = None
        self.sortDataGUI = None

        # Initialize variables
        self.fileSize = None
        self.totalEvents = None
        self.plotName = 'plotCurve'
        self.imageViewerClosed = True
        self.messagesViewFile = None
        self.panelDict = None
        self.panelName = None
        self.min_fs = None
        self.max_fs = None
        self.min_ss = None
        self.max_ss = None

        # Initialize list of panel names for the left side of the LCLS epix detector
        self.panelNamesOnLeft = [
            'p4a0', 'p4a1', 'p4a2', 'p4a3',
            'p5a0', 'p5a1', 'p5a2', 'p5a3',
            'p6a0', 'p6a1', 'p6a2', 'p6a3',
            'p7a0', 'p7a1', 'p7a2', 'p7a3',
            'p8a0', 'p8a1', 'p8a2', 'p8a3',
            'p9a0', 'p9a1', 'p9a2', 'p9a3',
            'p10a0', 'p10a1', 'p10a2', 'p10a3',
            'p11a0', 'p11a1', 'p11a2', 'p11a3',
        ]
        # this names could be different for EuXFEL AGIPD

        # Initialize graphing
        self.graphWidget = pg.PlotWidget()
        self.layout = qtw.QHBoxLayout()
        self.layout.addWidget(self.graphWidget)
        self.graphingSpace.setLayout(self.layout)
        self.graphWidget.setEnabled(False)

        # Set window title and show the window
        self.setWindowTitle("PixelAnomalyDetector")
        self.show()

        # Initialize busy and idle lights
        self.busyLight = BusyLight()
        self.idleLight = IdleLight()
        self.statusbar.addPermanentWidget(self.busyLight)
        self.statusbar.addPermanentWidget(self.idleLight)
        self.idleLight.show()
        self.busyLight.hide()

        # Tool tips for buttons
        self.viewFileButton.setToolTip("Click to display the CXI file")
        self.plotPixelIntensityButton.setToolTip("Click to plot the vertically averaged pixel intensity of "
                                                 "the selected panel")
        self.plotPeakPixelButton.setToolTip("Click to display the location of the pixel with the highest vertically "
                                            "averaged intensity for all the images in the CXI file")
        self.sortForMLButton.setToolTip("Click to Plot the distribution of two inflation points")
        self.sortButton.setToolTip("Click to Save data with the trained model")

        #   Accessing Child widget
        self.tabWidget = self.findChild(qtw.QTabWidget, 'tabWidget')

    def setBusy(self):
        """
            Change the light to busy.
            :return: None
        """
        self.busyLight.show()
        self.idleLight.hide()

    def setIdle(self):
        """
            Change the light to idle.
            :return: None
        """
        self.busyLight.hide()
        self.idleLight.show()

    def setImageViewerClosed(self):
        """
            Set the imageViewerClosed attribute to True.
            :return: None
        """
        self.imageViewerClosed = True

    def showNextMessage(self, messageList):
        """
            Display the next message in the messageList in the status bar and remove it from the list.
            If there are more messages, call this function again with a delay.

            :param messageList: list of messages to display in the status bar
            :return: None
        """
        message = messageList.pop(0)
        self.statusbar.showMessage(message, 3000)
        if messageList:
            qtc.QTimer.singleShot(3000, lambda: self.showNextMessage(messageList))

    @pyqtSlot()
    def browseFiles(self):
        """
        This method gets triggered when the browse button is Clicked in the GUI
        function:The function is to take in a text field where the value needs to be set and called in a dialog box with
        file structure view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """
        # Set the status indicator light to busy
        self.setBusy()
        currentTabIndex = self.tabWidget.currentIndex()
        if currentTabIndex == 0:
            # Get the filename of the list of CXI files - from the Hit finding tab
            fileName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'List Files (*.list);; '
                                                                                  'Text Files (*.txt)')
            self.trainingFilesPath.setText(fileName)
            self.viewTrainingFilesButton.setEnabled(True)
            self.nextButton_2.setEnabled(True)
            self.previousButton_2.setEnabled(True)

        else:
            # Open a file dialog to let the user select a CXI file - from the Anomaly Detector tab
            fileName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'CXI Files (*.cxi)')
            if fileName:
                self.cxiFilePath.setText(fileName)
                self.cxiFileListPath.setEnabled(False)
                self.cxiListBrowseButton.setEnabled(False)
                self.geomBrowseButton.setEnabled(True)
                self.geomFilePath.setEnabled(True)
                self.statusbar.showMessage("Browse for a geometry file ", 5000)

            # Reset the main window for the next CXI file
            if self.imageViewer:
                self.imageViewer.close()

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

            # Close any related GUI windows
            if self.sortForMLGUI:
                self.sortForMLGUI.close()
                self.sortForMLGUI = None

            if self.mlGUI:
                self.mlGUI.close()
                self.mlGUI = None

            if self.sortForMLGUI:
                self.sortForMLGUI.close()
                self.sortDataGUI = None

        # Set the status indicator light back to idle
        self.setIdle()

    @pyqtSlot()
    def browseGeom(self):
        """
            This method gets triggered when the browse button is Clicked in the GUI
        function:The function is to take in a text field where the value needs to be set and called in a dialog box with
        file structure view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """

        # Set the status indicator light to busy
        self.setBusy()

        # Open a file dialog to let the user select a geometry file
        geomName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'geom Files (*.geom)')
        if geomName:
            currentTabIndex = self.tabWidget.currentIndex()
            if currentTabIndex == 0:
                self.geomFilePath_2.setText(geomName)
            else:
                self.geomFilePath.setText(geomName)
                self.viewFileButton.setEnabled(True)
                self.statusbar.showMessage("Press the View File button to display the cxi file ", 5000)

        # Set the status indicator light back to idle
        self.setIdle()

    @pyqtSlot()
    def curveToPlot(self):
        """
        A method to select the type of curve to be plotted in the self.graphingSpace
        :return: type of plot to display
        """

        # Check if the event number is greater or equal to the total number of events
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

        # Call the appropriate plotting method based on the value of self.plotName
        elif self.plotName == 'plotCurve':
            self.plotCurve()
        elif self.plotName == 'plotFit':
            self.plotFit()

    @pyqtSlot()
    def selectDisplay(self):
        """
            This method is responsible for updating the displayed event of the *cxi file or creating a new view to display
            the *cxi file based on the current state of the image viewer. It is called when the user wants to navigate through
            events in the *cxi file.
            :return: Next/Previous event of the *cxi file or a new window to display the *cxi file.
        """

        # Check if the event number is greater or equal to the total number of events, and adjust if needed
        if int(self.eventNumber.text()) >= self.totalEvents:
            self.eventNumber.setText(str(self.totalEvents - 1))

        # If the image viewer is open, draw the next/previous image from the *cxi file
        if not self.imageViewerClosed:
            self.imageViewer.drawImage(int(self.eventNumber.text()))

        # If the image viewer is closed, create a new view to display the *cxi file
        else:
            self.viewFiles()

    @pyqtSlot(dict)
    def readPanelDetails(self, inDict):
        """
        This method is called when the user clicks on a panel in the image viewer.
        :param inDict: Dictionary containing ASIC/panel information received from the signal emitted when a panel is clicked
        :return: Assigns panel details to class variables and calls curveToPlot() method.
        """

        # Assign the received panel information to the class variables
        self.panelDict = inDict
        self.panelName = inDict['panel_name']
        self.min_fs = inDict['min_fs']
        self.max_fs = inDict['max_fs']
        self.min_ss = inDict['min_ss']
        self.max_ss = inDict['max_ss']

        # Call the curveToPlot() method to update the curve based on the selected panel
        self.curveToPlot()

    @pyqtSlot()
    def viewFiles(self):
        """
            This method creates an instance of DisplayImage to display the *cxi file.
            :return: A GUI with the *cxi file open, similar to cxi view. It also enables the "Plot Pixel Intensity",
                 "Plot Peak Pixel", and "Plot a Fit" checkBox.
        """

        # Enable the eventNumber field if it's not enabled
        if not self.eventNumber.isEnabled():
            self.eventNumber.setEnabled(True)
        if not self.eventNumber.text():
            self.eventNumber.setText("0")

        # If the image viewer is not closed, close it and create a new instance of DisplayImage
        if not self.imageViewerClosed:
            self.imageViewer.close()
            self.imageViewer = DisplayImage(self.cxiFilePath.text(), self.geomFilePath.text())
            self.imageViewer.drawImage(int(self.eventNumber.text()))
            self.totalEvents = self.imageViewer.size
            self.imageViewerClosed = False
            self.imageViewer.show()
        else:
            self.imageViewer = DisplayImage(self.cxiFilePath.text(), self.geomFilePath.text())
            self.imageViewer.drawImage(int(self.eventNumber.text()))
            self.totalEvents = self.imageViewer.size
            self.imageViewerClosed = False
            self.imageViewer.show()

        # Set up the messages to be shown in the status bar
        self.messagesViewFile = ["Click the Plot Pixel Intensity button", "Click Next and Previous "
                                                                          "buttons to navigate through images",
                                 "Click the Fit Plot CheckBox to fit a polynomial"]
        self.showNextMessage(self.messagesViewFile)

        # Initial panel assignment
        if not self.panelDict:
            self.panelDict = self.imageViewer.outgoingDict
            self.panelName = self.imageViewer.outgoingDict['panel_name']
            self.min_fs = self.imageViewer.outgoingDict['min_fs']
            self.max_fs = self.imageViewer.outgoingDict['max_fs']
            self.min_ss = self.imageViewer.outgoingDict['min_ss']
            self.max_ss = self.imageViewer.outgoingDict['max_ss']

        # Connect signals and slots
        self.imageViewer.panelSelected.connect(self.readPanelDetails)
        self.clickedNext.connect(self.imageViewer.drawImage)
        self.clickedPrevious.connect(self.imageViewer.drawImage)
        self.imageViewer.destroyed.connect(self.setImageViewerClosed)

        # Enable the Plot Pixel Intensity, Plot Peak Pixel, and Plot a Fit CheckBox buttons if they are not enabled
        if not self.plotPixelIntensityButton.isEnabled():
            self.plotPixelIntensityButton.setEnabled(True)
            self.poltFitCheckBox.setEnabled(True)
            self.plotPeakPixelButton.setEnabled(True)

    def viewFilesList(self):
        """
            Read a list of CXI files and display them one by one using the DisplayImage class
            """
        # Get the filename of the list of CXI files
        fileName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'Text Files (*.txt)')

        if fileName:
            # Read the list of CXI files from the text file
            with open(fileName, 'r') as f:
                cxiFileList = f.readlines()

            # Iterate over each CXI file in the list and display its images using DisplayImage class
            for cxiFile in cxiFileList:
                # Remove the trailing newline character
                cxiFile = cxiFile.strip()

                # Create an instance of DisplayImage and display the first image
                imageViewer = DisplayImage(cxiFile, self.geomFilePath.text())
                imageViewer.drawImage(0)
                self.totalEvents = imageViewer.size

                # Connect signals to the appropriate slots
                imageViewer.panelSelected.connect(self.readPanelDetails)
                self.clickedNext.connect(imageViewer.drawImage)
                self.clickedPrevious.connect(imageViewer.drawImage)

    @pyqtSlot(str)
    def nextEvent(self, eventNumber):
        """
        This method increments the event number to navigate to the next event in the *cxi file.
        :param eventNumber: The current event number.
        :return: The updated event number, incremented by 1, or reset to 0 if it reaches the end of the event list.
        """
        try:
            # Increment the event number if it's not the last event
            if int(self.eventNumber.text()) < self.totalEvents - 1:
                self.eventNumber.setText(str(int(eventNumber) + 1))
            # Reset the event number to 0 if it's the last event
            elif int(self.eventNumber.text()) == self.totalEvents - 1:
                self.eventNumber.setText(str(0))

            # Update the plot based on the new event number
            self.curveToPlot()

            # Emit the updated event number
            self.clickedNext.emit(int(self.eventNumber.text()))

        except Exception as e:
            # Display an error message if an exception occurs
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading bad events file")
            msg.setInformativeText(str(e) + " nextEvent()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    @pyqtSlot(str)
    def previousEvent(self, eventNumber):
        """
        This method decrements the event number to navigate to the previous event in the *cxi file.
        :param eventNumber: The current event number.
        :return: The updated event number, decremented by 1, or set to the last event if it reaches the beginning of the event list.
        """
        try:
            # Decrement the event number if it's not the first event
            if int(self.eventNumber.text()) > 0:
                self.eventNumber.setText(str(int(eventNumber) - 1))
            # Set the event number to the last event if it's the first event
            elif int(self.eventNumber.text()) == 0:
                self.eventNumber.setText(str(self.totalEvents - 1))

            # Update the plot based on the new event number
            self.curveToPlot()

            # Emit the updated event number
            self.clickedPrevious.emit(int(self.eventNumber.text()))

        except Exception as e:
            # Display an error message if an exception occurs
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText("An error occurred while reading bad events file")
            msg.setInformativeText(str(e) + " previousEvent()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    @pyqtSlot(dict, str)
    def writeToFile(self, eventsList, fileName):
        """
        This method saves the sorted events in a *.list file.
        :param eventsList: A dictionary containing the *cxi file path and event numbers.
        :param fileName: The file name for the saved file.
        :return: A *.list file containing the sorted events.
        """
        # Open the file in write mode
        f = open(fileName, 'w')

        # Iterate through the keys in the eventsList dictionary
        for key in eventsList.keys():
            # Iterate through the event numbers associated with the key
            for i in eventsList[key]:
                # Write the key, event number, and other necessary characters to the file
                f.write(key)
                f.write(' ')
                f.write('//')
                f.write(str(i))
                f.write('\n')

        # Close the file
        f.close()
        # Display a message in the status bar indicating the file was saved
        self.statusbar.showMessage("Saving file %s " % fileName, 2000)

        # Close the SortForMLGUI if it is open
        if self.sortForMLGUI:
            self.sortForMLGUI.close()
            self.sortForMLGUI = None
            # Display a message in the status bar prompting the user to train a model
            self.statusbar.showMessage("Click on the Train a Model button to get a model trained", 3000)

    @pyqtSlot()
    def sortForML(self):
        """
        Spawn an instance of the SortingForML GUI and connect signals to it.
        Once the SortingForML GUI is closed, emits signals to save good and bad events list to files and enables the "Train
        a Model" button. Also sets the busy cursor until the SortingForML GUI is closed.
        """
        # Spawn an instance of the SortingForML GUI and show it
        self.sortForMLGUI = SortingForML(self.cxiFilePath.text(), self.orderOfFit.text(), self.panelDict)
        self.sortForMLGUI.show()

        # Connect signals from the imageViewer and SortingForML GUIs
        self.imageViewer.panelSelected.connect(self.sortForMLGUI.readPanelDetails)
        self.sortForMLGUI.readyToSaveGood.connect(self.writeToFile)
        self.sortForMLGUI.readyToSaveBad.connect(self.writeToFile)

        # Enable the "Train a Model" button
        self.MLButton.setEnabled(True)

        # Set the busy cursor until the SortingForML GUI is closed
        self.setBusy()

        # Use a QEventLoop to wait for the SortingForML GUI to be closed
        loop = qtc.QEventLoop()
        self.sortForMLGUI.destroyed.connect(loop.quit)
        loop.exec_()

        # Set the idle cursor after the SortingForML GUI is closed
        self.setIdle()

    @pyqtSlot()
    def machineLearning(self):
        """
        This method is triggered when the "Train a Model" button is clicked.
        It spawns an instance of ML to train a machine learning model based on the good and bad lists of events.
        :return: A trained model. Turns ON "Sort" button.
        """
        # create an instance of the ML class and display the window
        self.mlGUI = ML(self.panelDict)
        self.mlGUI.show()
        # connect the panelSelected signal of the imageViewer to the readPanelDetails method of the mlGUI
        self.imageViewer.panelSelected.connect(self.mlGUI.readPanelDetails)

        # enable the Sort button
        self.sortButton.setEnabled(True)

        # set the application status to "busy"
        self.setBusy()

        # create an event loop and connect the destroyed signal of the mlGUI to quit the loop
        loop = qtc.QEventLoop()
        self.mlGUI.destroyed.connect(loop.quit)
        loop.exec_()

        # set the application status to "idle"
        self.setIdle()

    @pyqtSlot()
    def sort(self):
        """
        Spawn an instance of SortData, which sorts events based on the trained model and allows
        the user to save a list of good and bad events to a file.
        :return: None
        """
        # Create a new instance of SortData and show the GUI
        self.sortDataGUI = SortData(self.mlGUI.model, self.panelDict)
        self.sortDataGUI.show()

        # Connect the signals from SortData to this class's writeToFile method, which saves the sorted events to a file
        self.sortDataGUI.readyToSaveGood.connect(self.writeToFile)
        self.sortDataGUI.readyToSaveBad.connect(self.writeToFile)

        # Set the busy cursor while the SortData GUI is open
        self.setBusy()

        # Start a new event loop to keep the SortData GUI running until it is closed by the user
        loop = qtc.QEventLoop()
        self.sortDataGUI.destroyed.connect(loop.quit)
        loop.exec_()

        # Reset the cursor to idle once the SortData GUI is closed
        self.setIdle()

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
        A method to calculate the maximum pixel value for each frame of a CXI file.
        :param fileName: str, name of the file to be opened
        :param deg: int, order of the polynomial fit (1 for a straight line, 2 or more for higher order polynomial)
        :return: a list of maximum pixel values for each frame
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

            # handling the left and right detector pannels
            if self.panelName in self.panelNamesOnLeft:
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

                if self.panelName in self.panelNamesOnLeft:
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
            try:
                self.imageViewer.close()
            except Exception:
                pass

        if self.sortForMLGUI:
            try:
                self.sortForMLGUI.close()
            except Exception:
                pass

        if self.mlGUI:
            try:
                self.mlGUI.close()
            except Exception:
                pass

        if self.sortDataGUI:
            try:
                self.sortDataGUI.close()
            except AttributeError:
                pass


# main .
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
