#!/usr/bin/env python3

# PyQt5 imports
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5.QtCore import pyqtSlot
from PyQt5 import uic

# Builting packages
from builtins import Exception

# Packages for data access
import os.path
import h5py

# Packages for data processing
import numpy as np

# Graphing stuff
import pyqtgraph as pg

# Packages for saving the model
import pickle

# importing custom libraries
from lib.geometry_parser.GeometryFileParser import *
from lib.DisplayingModule import DisplayImage
from lib.LabelingModule import DataLabeler
from lib.MLModule import ML
from lib.SortModule import SortData
from lib.LightingModule import IdleLight, BusyLight


class MainWindow(qtw.QMainWindow):
    clickedNext = qtc.pyqtSignal(int)
    clickedPrevious = qtc.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi("UI/mainWindow.ui", self)
        self.setGeometry(700, 100, 800, 700)

        # connecting elements to functions
        self.cxiBrowseButton.clicked.connect(lambda: self.browseFiles(self.cxiBrowseButton.objectName()))
        self.cxiListBrowseButton.clicked.connect(lambda: self.browseFiles(self.cxiListBrowseButton.objectName()))
        self.geomBrowseButton.clicked.connect(self.browseGeom)
        self.viewFileButton.clicked.connect(self.viewFiles)
        self.plotPixelIntensityButton.clicked.connect(self.plotCurve)
        self.poltFitCheckBox.clicked.connect(self.plotFit)
        self.plotPeakPixelButton.clicked.connect(lambda: self.plotMaxPixels(self.cxiFilePath.text()))
        self.sortButton.clicked.connect(self.sortData)
        self.sortForMLButton.clicked.connect(self.dataLabeling)
        self.MLButton.clicked.connect(self.machineLearning)
        self.loadButton.clicked.connect(self.loadModel)
        self.orderOfFit.editingFinished.connect(self.plotFit)
        self.eventNumber.editingFinished.connect(self.curveToPlot)
        self.eventNumber.editingFinished.connect(self.selectDisplay)

        #  First message on status bar
        self.statusbar.showMessage("Browse for CXI file or a list a CXI files ", 5000)

        # initializing class variables
        # initializing the popup windows
        self.imageViewer = None
        self.sortForMLGUI = None
        self.mlGUI = None
        self.sortDataGUI = None

        self.fileSize = None
        self.totalEvents = None
        self.plotName = 'plotCurve'

        # variables to identify the respective windows are opened or closed.
        self.imageViewerClosed = True

        self.messagesViewFile = None

        self.model = None
        self.panelDict = None
        self.panelName = None
        self.min_fs = None
        self.max_fs = None
        self.min_ss = None
        self.max_ss = None
        self.fileLocation= os.getcwd()
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

        # tool tips for buttons
        self.viewFileButton.setToolTip("Click to display the CXI file")
        self.plotPixelIntensityButton.setToolTip("Click to plot the vertically averaged pixel intensity of "
                                                 "the selected panel")
        self.plotPeakPixelButton.setToolTip("Click to display the location of the pixel with the highest vertically "
                                            "averaged intensity for all the images in the CXI file")
        self.sortForMLButton.setToolTip("Click to Plot the distribution of two inflation points")
        self.sortButton.setToolTip("Click to Save data with the trained model")

        # for debugging and feature adding
        self.MLButton.setEnabled(True)

    def setBusy(self):
        """

        :return: Change the light to busy
        """
        self.busyLight.show()
        self.idleLight.hide()

    def setIdle(self):
        """

        :return: Change the light to Idle
        """
        self.busyLight.hide()
        self.idleLight.show()

    def setImageViewerClosed(self):
        self.imageViewerClosed = True

    def showNextMessage(self, messageList):
        message = messageList.pop(0)
        self.statusbar.showMessage(message, 3000)
        if messageList:
            qtc.QTimer.singleShot(3000, lambda: self.showNextMessage(messageList))

    @pyqtSlot()
    def browseFiles(self, buttonName):
        """
        This method gets triggered when the browse button is Clicked in the GUI
        function:The function is to take in a text field where the value needs to be set and called in a dialog box with
        file structure view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """

        self.setBusy()
        if buttonName == "cxiBrowseButton":
            fileName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'CXI Files (*.cxi)')
            if fileName:
                self.cxiFilePath.setText(fileName)
                self.cxiFileListPath.setEnabled(False)
                self.cxiListBrowseButton.setEnabled(False)
                self.geomBrowseButton.setEnabled(True)
                self.geomFilePath.setEnabled(True)
                self.statusbar.showMessage("Browse for a geometry file ", 5000)
        elif buttonName == "cxiListBrowseButton":
            fileListName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'list Files (*.list)')
            if fileListName:
                self.cxiFileListPath.setText(fileListName)
                self.cxiFilePath.setEnabled(False)
                self.cxiBrowseButton.setEnabled(False)
                self.geomBrowseButton.setEnabled(True)
                self.geomFilePath.setEnabled(True)
                self.statusbar.showMessage("Browse for a geometry file ", 5000)

        # resting the main window for the next cxi file
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
        self.fileLocation = os.getcwd()

        try:
            if self.sortForMLGUI:
                self.sortForMLGUI.close()
                self.sortForMLGUI = None
        except:
            pass

        try:
            if self.mlGUI:
                self.mlGUI.close()
                self.mlGUI = None
        except:
            pass

        self.setIdle()

    @pyqtSlot()
    def browseGeom(self):
        """
            This method gets triggered when the browse button is Clicked in the GUI
        function:The function is to take in a text field where the value needs to be set and called in a dialog box with
        file structure view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """

        self.setBusy()

        geomName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'geom Files (*.geom)')
        if geomName:
            self.geomFilePath.setText(geomName)
            self.viewFileButton.setEnabled(True)
            self.statusbar.showMessage("Press the View File button to display the cxi file ", 5000)

        self.setIdle()

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

        if not self.imageViewerClosed:
            self.imageViewer.drawImage(int(self.eventNumber.text()))
        else:
            self.viewFiles()

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

        if not self.eventNumber.isEnabled():
            self.eventNumber.setEnabled(True)
        if not self.eventNumber.text():
            self.eventNumber.setText("0")

        if not self.imageViewerClosed:
            self.imageViewer.close()
            if self.cxiFilePath.text():
                self.imageViewer = DisplayImage(self.cxiFilePath.text(), self.geomFilePath.text())
                self.imageViewer.drawImage(int(self.eventNumber.text()))
                self.totalEvents = self.imageViewer.size
                self.imageViewerClosed = False
                self.imageViewer.show()
            elif self.cxiFileListPath.text():
                with open(self.cxiFileListPath) as f:
                    for line in f:
                        self.imageViewer = DisplayImage(line,self.geomFilePath.text())
                        self.imageViewer.drawImage(int(self.eventNumber.text()))
                        self.totalEvents = self.imageViewer.size
                        self.imageViewerClosed = False
                        self.imageViewer.show()
        else:
            if self.cxiFilePath.text():
                self.imageViewer = DisplayImage(self.cxiFilePath.text(), self.geomFilePath.text())
                self.imageViewer.drawImage(int(self.eventNumber.text()))
                self.totalEvents = self.imageViewer.size
                self.imageViewerClosed = False
                self.imageViewer.show()
            elif self.cxiFileListPath.text():
                with open(self.cxiFileListPath.text()) as f:
                    for line in f:
                        self.imageViewer = DisplayImage(line.strip(), self.geomFilePath.text())
                        self.imageViewer.drawImage(int(self.eventNumber.text()))
                        self.totalEvents = self.imageViewer.size
                        self.imageViewerClosed = False
                        self.imageViewer.show()

        self.messagesViewFile = ["Click the Plot Pixel Intensity button", "Click Next and Previous "
                                                                          "buttons to navigate through images",
                                 "Click the Fit Plot CheckBox to fit a polynomial"]
        self.showNextMessage(self.messagesViewFile)

        # initial panel assignment
        if not self.panelDict:
            self.panelDict = self.imageViewer.outgoingDict
            self.panelName = self.imageViewer.outgoingDict['panel_name']
            self.min_fs = self.imageViewer.outgoingDict['min_fs']
            self.max_fs = self.imageViewer.outgoingDict['max_fs']
            self.min_ss = self.imageViewer.outgoingDict['min_ss']
            self.max_ss = self.imageViewer.outgoingDict['max_ss']

        # connecting signals from and to, to the imageViewer object
        self.imageViewer.panelSelected.connect(self.readPanelDetails)
        self.clickedNext.connect(self.imageViewer.drawImage)
        self.clickedPrevious.connect(self.imageViewer.drawImage)
        self.imageViewer.goodRadioButton.toggled.connect(self.handleRadioButtons)
        self.imageViewer.badRadioButton.toggled.connect(self.handleRadioButtons)
        self.imageViewer.destroyed.connect(self.setImageViewerClosed)

        # clearing the radio buttons
        if self.imageViewer.goodRadioButton.isChecked():
            print('Good button is clicked')
            self.imageViewer.goodRadioButton.setChecked(False)
        elif self.imageViewer.badRadioButton.isChecked():
            print('bad button is clicked')
            self.imageViewer.badRadioButton.setChecked(False)

        if not self.plotPixelIntensityButton.isEnabled():
            self.plotPixelIntensityButton.setEnabled(True)
            self.poltFitCheckBox.setEnabled(True)
            self.plotPeakPixelButton.setEnabled(True)

    @pyqtSlot()
    def handleRadioButtons(self):
        savingDict = {}
        if self.imageViewer.goodRadioButton.isChecked():
            savingDict[self.cxiFilePath.text()] = [self.eventNumber.text()]
            tag = str(self.cxiFilePath.text()).split('/')[-1].split('.')[0]
            if not os.path.isfile(self.fileLocation + '/'+'goodEvents-advanceSort-%s.list' % tag):
                while True:
                    try:
                        self.fileLocation = qtw.QFileDialog.getExistingDirectory(self, caption='Select Save Location', directory=' ',
                                                                options=qtw.QFileDialog.DontUseNativeDialog)
                        self.writeToFile(savingDict,self.fileLocation + '/' + 'goodEvents-advanceSort-%s.list' % tag)
                        break
                    except OSError:
                        qtw.QMessageBox.information(self,'Information', 'Please select a file save location')
                        continue
            else:
                self.writeToFile(savingDict, self.fileLocation + '/' + 'goodEvents-advanceSort-%s.list' % tag)
        elif self.imageViewer.badRadioButton.isChecked():
            savingDict[self.cxiFilePath.text()] = [self.eventNumber.text()]
            tag = str(self.cxiFilePath.text()).split('/')[-1].split('.')[0]
            if not os.path.isfile(self.fileLocation + '/' + 'badEvents-advanceSort-%s.list' % tag):
                while True:
                    try:
                        self.fileLocation = qtw.QFileDialog.getExistingDirectory(self, caption='Select Save Location',
                                                                            directory=' ',
                                                                            options=qtw.QFileDialog.DontUseNativeDialog)
                        self.writeToFile(savingDict, self.fileLocation + '/' + 'badEvents-advanceSort-%s.list' % tag)
                        break
                    except OSError:
                        qtw.QMessageBox.information(self, 'Information', 'Please select a file save location')
                        continue
            else:
                self.writeToFile(savingDict, self.fileLocation + '/' + 'badEvents-advanceSort-%s.list' % tag)

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

    @pyqtSlot(dict, str)
    def writeToFile(self, eventsDict, fileName):
        """
        A method to save sorted events
        :param eventsDict: dictionary with *cxi file path and event numbers
        :param fileName: save file name
        :return: *.list file
        """
        f = open(fileName, 'a')

        self.statusbar.showMessage("Saving file %s " % fileName, 2000)
        for key in eventsDict.keys():
            for i in eventsDict[key]:
                f.write(key)
                f.write(' ')
                f.write('//')
                f.write(str(i))
                f.write('\n')

        f.close()

        if self.sortForMLGUI:
            self.sortForMLGUI.close()
            self.statusbar.showMessage("Click on the Train a Model button to get a model trained", 3000)

    @pyqtSlot()
    def dataLabeling(self):
        """
        Spawn an instance of the SortingForML
        :return: good and bad lists to be saved. Turns ON "Train a Model" button
        """

        self.sortForMLGUI = DataLabeler(self.cxiFilePath.text(), self.orderOfFit.text(), self.panelDict)
        self.sortForMLGUI.show()
        self.imageViewer.panelSelected.connect(self.sortForMLGUI.readPanelDetails)

        self.sortForMLGUI.readyToSaveGood.connect(self.writeToFile)
        self.sortForMLGUI.readyToSaveBad.connect(self.writeToFile)
        self.MLButton.setEnabled(True)

        self.setBusy()

        loop = qtc.QEventLoop()
        self.sortForMLGUI.destroyed.connect(loop.quit)
        loop.exec_()

        self.setIdle()

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

        self.setBusy()

        loop = qtc.QEventLoop()
        self.mlGUI.destroyed.connect(loop.quit)
        loop.exec_()

        self.setIdle()

    @pyqtSlot()
    def loadModel(self):
        if self.mlGUI is not None:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Question')
            msg.setText("Do you wish to continue?")
            msg.setInformativeText("You are about to replace the model you trained with a saved model!")
            msg.setIcon(qtw.QMessageBox.Question)
            msg.setStandardButtons(qtw.QMessageBox.Yes | qtw.QMessageBox.No)
            msg.setDefaultButton(qtw.QMessageBox.No)
            clickedButton = msg.exec_()

            if clickedButton == qtw.QMessageBox.Yes:
                modelName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'Pickle Files (*.pkl)')
                with open(modelName, 'rb') as f:
                    data = pickle.load(f)
                self.panelDict = {'panel_name': data['panel_name'], 'min_fs': data['min_fs'], 'max_fs': data['max_fs'],
                                  'min_ss': data['min_ss'], 'max_ss': data['max_ss']}
                self.model = data['model']
                qtw.QMessageBox.warning(self, "Warning",
                                        "This models was trained using panel %s please make sure that's the panel you want"
                                        % data['panel_name'])
                self.sortButton.setEnabled(True)

        else:
            modelName, _ = qtw.QFileDialog.getOpenFileName(self, 'Open File', ' ', 'Pickle Files (*.pkl)')
            with open(modelName, 'rb') as f:
                data = pickle.load(f)
            self.panelDict = {'panel_name': data['panel_name'], 'min_fs': data['min_fs'], 'max_fs': data['max_fs'],
                              'min_ss': data['min_ss'], 'max_ss': data['max_ss']}
            self.model = data['model']
            qtw.QMessageBox.warning(self, "Warining",
                                    "This models was trained using panel %s please make sure that's the panel you want"
                                    % data['panel_name'])
            self.sortButton.setEnabled(True)

    @pyqtSlot()
    def sortData(self):
        """
        Spawn an instance of SortData.
        :return: A sorted list of good and bad events to be saved.
        """

        if self.model:
            try:
                self.sortDataGUI = SortData(self.model, self.panelDict)
            except TypeError:
                qtw.QMessageBox.information(self,'Information', 'Please open at least one cxi file to get the geometric information')
        elif self.mlGUI.model:
            self.sortDataGUI = SortData(self.mlGUI.model, self.panelDict)
        self.sortDataGUI.show()

        self.sortDataGUI.readyToSaveGood.connect(self.writeToFile)
        self.sortDataGUI.readyToSaveBad.connect(self.writeToFile)

        self.setBusy()

        loop = qtc.QEventLoop()
        try:
            self.sortForMLGUI.destroyed.connect(loop.quit)
        except:
            pass
        loop.exec_()

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
            if self.cxiFilePath.text():
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
            except Exception:
                pass


# main .
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
