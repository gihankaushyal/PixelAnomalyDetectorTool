# imports
from builtins import Exception

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc

import pyqtgraph as pg
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5 import uic

import sys

import lib.cfel_filetools as fileTools
import lib.cfel_geometry as geomTools
import lib.cfel_imgtools as imgTools


class DisplayImage(qtw.QWidget):
    def __init__(self):
        super(DisplayImage, self).__init__()

        self.setGeometry(10, 100, 600, 600)
        self.mainWidget = pg.ImageView()
        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.mainWidget)
        self.setLayout(self.layout)

        # reading and displaying data

    def drawImage(self, fileName, eventNumber, geometry):
        try:
            self.cxi = fileTools.read_cxi(fileName, frameID=eventNumber, data=True, slab_size=True)
            self.size = self.cxi['stack_shape'][0]
            self.imgData = self.cxi['data']
            try:
                self.geometry = geomTools.read_geometry(geometry)
                self.imageToDraw = imgTools.pixel_remap(self.imgData, self.geometry['x'], self.geometry['y'])
                self.mainWidget.setImage(self.imageToDraw)
                self.setWindowTitle("Showing %i of %i " % (eventNumber, self.size - 1))
            except:
                qtw.QMessageBox.critical(self, 'Fail', "Couldn't read the geometry file, Please Try again!")
        except:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't read the cxi file, Please Try again!")


class ML(qtw.QWidget):

    def __init__(self):

        super(ML, self).__init__()

        self.model = None
        uic.loadUi("machineLearningGUI.ui", self)

        self.setWindowTitle('Machine Learning')

        self.browseButton.clicked.connect(self.browseFiles)
        self.checkBox.stateChanged.connect(self.checkBoxClicked)
        self.trainButton.clicked.connect(self.train)
        self.testButton.clicked.connect(self.test)

    def browseFiles(self):
        """
            This method gets triggered when the browse button is Clicked in the GUI
        function: The function is to take in a text field where the value needs to be set and called in a dialog box with
        file struture view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """

        fname = qtw.QFileDialog.getExistingDirectory(self, caption='Select Folder', directory=' ')

        self.parentDirectory.setText(fname)

    def checkBoxClicked(self):
        if self.checkBox.isChecked():
            self.startRun.setEnabled(True)
            self.endRun.setEnabled(True)
        else:
            self.startRun.setEnabled(False)
            self.endRun.setEnabled(False)

    def modelSelection(self):

        modelSelected = self.comboBox.currentText()
        if modelSelected == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
        elif modelSelected == 'KNeighborsClassifier':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(n_neighbors=1)
        elif modelSelected == 'DecisionTreeClassifier':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier()
        elif modelSelected == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=200)
        else:
            qtw.QMessageBox.critical(self, 'Caution', 'Please Select a model')

    def dataPrep(self):

        from sklearn.model_selection import train_test_split

        try:
            if self.checkBox.isChecked():
                pass
            else:
                folder = self.parentDirectory.text()
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Fail', e)

        files = Path(folder).glob('badEvents-advanceSort-*-ML.list')
        dataFrame_bad = pd.DataFrame(columns=['FileName', 'EventNumber', 'InflectionPoint1', 'InflectionPoint2'])

        for file in files:

            try:
                temp_df = pd.read_csv(str(file), delimiter=" ")
                temp_df.columns = ['FileName', 'EventNumber', 'InflectionPoint1', 'InflectionPoint2']
                dataFrame_bad = pd.concat([dataFrame_bad, temp_df])
            except Exception as e:
                qtw.QMessageBox.information(self, 'information', e)
                continue
        dataFrame_bad['Flag'] = 0

        files = Path(folder).glob('goodEvents-advanceSort-*-ML.list')
        dataFrame_good = pd.DataFrame(columns=['FileName', 'EventNumber', 'InflectionPoint1', 'InflectionPoint2'])

        for file in files:
            try:
                temp_df = pd.read_csv(str(file), delimiter=" ")
                temp_df.columns = ['FileName', 'EventNumber', 'InflectionPoint1', 'InflectionPoint2']
                dataFrame_good = pd.concat([dataFrame_good, temp_df])
            except Exception as e:
                qtw.QMessageBox.information(self, 'information', e)
                continue
        dataFrame_good['Flag'] = 1

        finalData = pd.concat([dataFrame_bad, dataFrame_good])

        X = finalData[['InflectionPoint1', 'InflectionPoint2']]
        y = finalData['Flag']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3)

    def train(self):
        self.modelSelection()
        self.dataPrep()
        self.model.fit(self.X_train, self.y_train)
        self.testButton.setEnabled(True)

    def test(self):
        from sklearn.metrics import classification_report, confusion_matrix
        self.predictions = self.model.predict(self.X_test)
        self.confussionMetrix.setEnabled(True)
        self.classificationReport.setEnabled(True)
        self.confussionMetrix.setText(str(confusion_matrix(self.y_test, self.predictions)))
        self.classificationReport.setText(classification_report(self.y_test, self.predictions))


class AdvanceSorting(qtw.QWidget):

    readyToSaveGood = qtc.pyqtSignal(dict, str,str)
    readyToSaveBad = qtc.pyqtSignal(dict, str,str)

    def __init__(self, fileName, oft):

        super(AdvanceSorting, self).__init__()

        self.badEvents = None
        self.goodEvents = None
        self.setWindowTitle('Advance Sorting')

        uic.loadUi("AdvanceSortGUI.ui", self)

        self.layout = qtw.QHBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)
        self.graphSpace.setLayout(self.layout)

        self.file_name = fileName
        self.orderOfFit = oft
        self.plotInflectionPointsButton.clicked.connect(self.plotInflectionPoints)
        self.sortButton.clicked.connect(self.advanceSort)

    def advanceSort(self):

        tag = str(self.file_name).split('/')[-1].split('.')[0]

        self.goodEvents = {}
        self.badEvents = {}

        # goodList to store all the events with expected pixel intensities for the file
        goodList = []
        # badList to store all the events with detector artifacts for the file
        badList = []
        try:
            with h5py.File(self.file_name, "r") as f:
                data = f['entry_1']['data_1']['data'][()]

            for i in range(len(data)):
                frame = data[i]

                avgIntensities = []
                for j in range(10, 186):
                    avgIntensities.append(np.average(frame[2112:2288, j]))

                fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=int(self.orderOfFit))
                # calculating the inflection points (second derivative of the forth order polynomial)
                # print(fit)
                try:
                    x1 = round((-6 * fit[1] + np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]))
                    x2 = round((-6 * fit[1] - np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]))
                except IndexError:
                    qtw.QMessageBox.information(self, 'Error', 'Please try a different order polynomial')
                except ValueError:
                    qtw.QMessageBox.information(self, 'Skip', 'Calculation Error! \n \n Skipping the frame %i' % i)
                    continue

                if self.checkBox.isChecked():
                    if x1 in np.arange(float(self.inflectionPoint1.text()) - 5, float(self.inflectionPoint1.text()) + 5):
                        goodList.append((i, x1, x2))
                    else:
                        badList.append((i, x1, x2))
                else:
                    if x1 in np.arange(float(self.inflectionPoint1.text()) - 5, float(self.inflectionPoint1.text()) + 5):
                        goodList.append(i)
                    else:
                        badList.append(i)

            self.goodEvents[str(self.file_name)] = goodList
            self.badEvents[str(self.file_name)] = badList

            if self.checkBox.isChecked():
                self.readyToSaveGood.emit(self.goodEvents, 'goodEvents-advanceSort-%s-ML.list' % tag, 'ML')
                self.readyToSaveBad.emit(self.badEvents, 'badEvents-advanceSort-%s-ML.list' % tag, 'ML')
            else:
                self.readyToSaveGood.emit(self.goodEvents, 'goodEvents-advanceSort-%s.list' % tag, 'Not ML')
                self.readyToSaveBad.emit(self.badEvents, 'badEvents-advanceSort-%s.list' % tag, 'Not ML')

            qtw.QMessageBox.information(self, 'Success', "Done Sorting")

        except Exception as e:
            print(e)
            # qtw.QMessageBox.critical(self, 'Fail', e)

    def plotInflectionPoints(self):

        self.x1_list = []
        self.x2_list = []

        with h5py.File(self.file_name, "r") as f:
            data = f['entry_1']['data_1']['data'][()]

        for i in range(len(data)):
            frame = data[i]

            avgIntensities = []
            for j in range(10, 186):
                avgIntensities.append(np.average(frame[2112:2288, j]))

            fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=int(self.orderOfFit))
            # calculating the inflection points (second derivative of the forth order polynomial)
            # print(fit)
            try:
                x1 = round((-6 * fit[1] + np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]))
                x2 = round((-6 * fit[1] - np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]))
                self.x1_list.append(x1)
                self.x2_list.append(x2)
            except IndexError:
                qtw.QMessageBox.information(self, 'Error', 'Please try a different order polynomial')
            except ValueError:
                qtw.QMessageBox.information(self, 'Skip', 'Calculation Error! \n \n Skipping the frame %i' % i)
                continue

        self.figure.clear()
        sns.histplot(self.x1_list, label='x1', kde=True, alpha=0.5)
        sns.histplot(self.x2_list, label='x2',  kde=True, alpha=0.5)

        # plt.hist(self.x1_list,bins=30,label='x1', alpha=0.5)
        # plt.hist(self.x2_list,bins=30,label='x2', alpha=0.5)
        plt.legend()

        self.canvas.draw()




class MainWindow(qtw.QMainWindow):
    clickedNext = qtc.pyqtSignal(str, int, str)
    clickedPrevious = qtc.pyqtSignal(str, int, str)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi("mainWindow.ui", self)
        self.setGeometry(700, 100, 800, 700)
        # connecting elements to functions
        self.browseButton.clicked.connect(self.browseFiles)
        self.browseButton_2.clicked.connect(self.browseGeom)
        self.viewFileButton.clicked.connect(self.viewFiles)

        self.imageViewer = None
        self.fileSize = None
        self.totalEvents = None
        self.buttonClicked = None
        self.sortGUI = None

        # button and input line for calling plotCurves() method to plot vertically average intensity profile for the
        # panel
        self.plotPixelIntensityButton.clicked.connect(self.plotCurve)
        # button for call the fit_curve() method to fit a 4th order polynomial for
        # the vertically average intensity profile
        self.fitPolynormialButton.clicked.connect(self.plotFit)
        # button for calling plot_max_pixels() method to plot the pixel with the highest intensity for all
        # the frames of the
        self.plotPeakPixelButton.clicked.connect(lambda: self.plotMaxPixels(self.fileField.text()))

        self.sortButton.clicked.connect(lambda: self.sortFrames(self.fileField.text()))
        self.advanceSortButton.clicked.connect(self.advanceSortFrames)
        self.MLButton.clicked.connect(self.machineLearning)

        self.orderOfFit.editingFinished.connect(self.plotFit)
        self.eventNumber.editingFinished.connect(self.curveToPlot)
        self.eventNumber.editingFinished.connect(self.viewFiles)

        # incrementing through eventnumbers
        self.nextButton.clicked.connect(lambda: self.nextEvent(self.eventNumber.text()))
        self.previousButton.clicked.connect(lambda: self.previousEvent(self.eventNumber.text()))

        # graphing
        self.graphWidget = pg.PlotWidget()
        self.layout = qtw.QHBoxLayout()
        self.layout.addWidget(self.graphWidget)
        self.graphingSpace.setLayout(self.layout)

        self.setWindowTitle("Detector Analyser")
        self.show()

    def browseFiles(self):
        """
            This method gets triggered when the browse button is Clicked in the GUI
        function: The function is to take in a text field where the value needs to be set and called in a dialog box with
        file struture view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """
        dialog_box = qtw.QDialog()
        fname = qtw.QFileDialog.getOpenFileNames(dialog_box, 'Open File', ' ', 'CXI Files (*.cxi)')
        self.fileField.setText(fname[0][0])
        self.browseButton_2.setEnabled(True)

    def browseGeom(self):
        """
            This method gets triggered when the browse button is Clicked in the GUI
        function: The function is to take in a text field where the value needs to be set and called in a dialog box with
        file struture view starting at the 'root' and lets the user select the file they want and set the file path to
        the test field.
        """
        dialog_box = qtw.QDialog()
        geomName = qtw.QFileDialog.getOpenFileNames(dialog_box, 'Open File', ' ', 'geom Files (*.geom)')
        self.fileField_2.setText(geomName[0][0])
        self.viewFileButton.setEnabled(True)

    def curveToPlot(self):

        if self.buttonClicked is None:
            pass
            # qtw.QMessageBox.information(self,'Info','Plot a curve first!')
        elif self.buttonClicked == 'plotCurve':
            self.plotCurve()
        elif self.buttonClicked == 'plotFit':
            self.plotFit()

    def viewFiles(self):
        if not self.eventNumber.text():
            self.eventNumber.setEnabled(True)
            self.eventNumber.setText("0")

        self.imageViewer = DisplayImage()
        self.imageViewer.drawImage(self.fileField.text(), int(self.eventNumber.text()), self.fileField_2.text())
        self.totalEvents = self.imageViewer.size
        self.clickedNext.connect(self.imageViewer.drawImage)
        self.clickedPrevious.connect(self.imageViewer.drawImage)
        self.imageViewer.show()

        self.plotPixelIntensityButton.setEnabled(True)
        self.fitPolynormialButton.setEnabled(True)
        self.plotPeakPixelButton.setEnabled(True)
        # self.MLButton.setEnabled(True)

    def machineLearning(self):

        self.mlDialog = ML()
        self.mlDialog.show()

    def nextEvent(self, eventNumber):
        try:
            if int(self.eventNumber.text()) < self.totalEvents - 1:
                self.eventNumber.setText(str(int(eventNumber) + 1))
            elif int(self.eventNumber.text()) == self.totalEvents - 1:
                self.eventNumber.setText(str(0))

            self.curveToPlot()

            self.clickedNext.emit(self.fileField.text(), int(self.eventNumber.text()), self.fileField_2.text())
        except:
            qtw.QMessageBox.critical(self, 'Fail', 'Please Enter a valid input')

    def previousEvent(self, eventNumber):
        try:
            if int(self.eventNumber.text()) > 0:
                self.eventNumber.setText(str(int(eventNumber) - 1))
            elif int(self.eventNumber.text()) == 0:
                self.eventNumber.setText(str(self.totalEvents - 1))

            self.curveToPlot()

            self.clickedPrevious.emit(self.fileField.text(), int(self.eventNumber.text()), self.fileField_2.text())
        except:
            qtw.QMessageBox.critical(self, 'Fail', 'Please Enter a valid input')

    def writeToFile(self, eventsList, fileName, sortingForML='Not ML'):
        f = open(fileName, 'w')

        for key in eventsList:

            if sortingForML == 'ML':
                for (i, x1, x2) in eventsList[key]:
                    f.write(key)
                    f.write(' ')
                    f.write('//')
                    f.write(str(i))
                    f.write(' ')
                    f.write(str(x1))
                    f.write(' ')
                    f.write(str(x2))
                    f.write('\n')
            elif sortingForML == 'Not ML':
                for i in eventsList[key]:
                    f.write(key)
                    f.write(' ')
                    f.write('//')
                    f.write(str(i))
                    f.write('\n')

        f.close()

    def sortFrames(self, file_name):
        """
            this method only works on single cxi file. for multiple files code needs to be changed
            function for sorting good frames from the bad frames based on the average intensity ratio of the peak of the
            curve vs crest of the curve
        """

        try:
            tag = str(file_name).split('/')[-1].split('.')[0]
            goodEvents = {}
            badEvents = {}
            # exit()
            # goodList to store all the events with expected pixel intensities for the file
            goodList = []

            # badList to store all the events with detector artifacts for the file
            badList = []

            with h5py.File(file_name, "r") as f:
                data = f['entry_1']['data_1']['data'][()]

            for i in range(0, len(data)):
                frame = data[i]

                peakMeanIntensities = []
                for j in range(60, 80):
                    peakMeanIntensities.append(np.average(frame[2112:2288, j]))

                crestMeanIntensities = []
                for k in range(165, 185):
                    crestMeanIntensities.append(np.average(frame[2112:2288, k]))

                peakMean = np.average(peakMeanIntensities)
                crestMean = np.average(crestMeanIntensities)

                if peakMean / crestMean > 1.3:
                    goodList.append(i)
                else:
                    badList.append(i)

            goodEvents[str(file_name)] = goodList
            badEvents[str(file_name)] = badList

            self.writeToFile(goodEvents, 'goodEvents-simpleSort-%s.list' % tag)
            self.writeToFile(badEvents, 'badEvents-%s.list' % tag)

            qtw.QMessageBox.information(self, 'Success', "Done Sorting")

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

    def advanceSortFrames(self):

        self.sortGUI = AdvanceSorting(self.fileField.text(), self.orderOfFit.text())
        self.sortGUI.show()

        self.sortGUI.readyToSaveGood.connect(self.writeToFile)
        self.sortGUI.readyToSaveBad.connect(self.writeToFile)

    def returnMaxPixel(self, coeff, x_range):
        """
            coeff (object): out put of a numpy curve fit.
            xRange (tup) : a tuple with (min, max)

            returns the value where the coeff is maximized
        """

        storing_list = []
        for i in range(x_range[0], x_range[1]):
            storing_list.append((np.polyval(coeff, i), i))

        storing_list.sort()

        return storing_list[-1][1]

    def returnMaxPixelsList(self, file_name, deg=1, *args):
        """ fileName(str) : name of the file to be open
            deg (int) : order of the fit ex: is the fit a straight line (1) or quadratic (2 or more)
            *args(list) : expects a list of events to be considered. **TO BE IMPLEMENTED**
        """
        filename = file_name
        max_pixels = []

        with h5py.File(filename, "r") as f:
            data = f['entry_1']['data_1']['data'][()]

        for i in range(len(data)):
            frame = data[i]

            avgIntensities = []
            for j in range(10, 186):
                avgIntensities.append(np.average(frame[2112:2288, j]))

            fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=deg)

            max_pixels.append(self.returnMaxPixel(fit, (10, 186)))

        return max_pixels

    def plotCurve(self):
        try:
            file_name = self.fileField.text()
            event_number = int(self.eventNumber.text())

            with h5py.File(file_name, "r") as f:
                data = f['entry_1']['data_1']['data'][()]

            frame = data[event_number]

            avgIntensities = []

            for i in range(10, 185):
                avgIntensities.append(np.average(frame[2112:2288, i]))
            self.graphWidget.clear()
            self.graphWidget.plot(list(np.linspace(10, 185, 175)), avgIntensities)
            self.graphWidget.setTitle('average intensity over the selected panel', size='15pt')
            self.graphWidget.setLabel('left', "Avg. Pixel intensity")
            self.graphWidget.setLabel('bottom', "Pixel Number")

            self.buttonClicked = 'plotCurve'

            self.sortButton.setEnabled(True)
            self.nextButton.setEnabled(True)
            self.previousButton.setEnabled(True)

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

        except IndexError:
            qtw.QMessageBox.critical(self, 'Fail', 'Value you ,%s,  entered is out of bound for this cxi file'
                                     % self.eventNumber.text())

    def plotFit(self):
        """ fileName(str) : name of the file to be open
                eventNumber(int) : event number for the file
                deg (int) : order of the fit ex: is the fit a straight line (1) or quadratic (2 or more)
        """
        try:
            if not self.orderOfFit.text():
                self.orderOfFit.setEnabled(True)
                self.orderOfFit.setText("4")

            file_name = self.fileField.text()
            eventNumber = int(self.eventNumber.text())
            avgIntensities = []
            degry = int(self.orderOfFit.text())

            filename = file_name
            with h5py.File(filename, "r") as f:
                data = f['entry_1']['data_1']['data'][()]

            frame = data[int(eventNumber)]

            for i in range(10, 186):
                avgIntensities.append(np.average(frame[2112:2288, i]))

            fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=degry)

            self.graphWidget.clear()
            self.graphWidget.plot(range(10, 186), avgIntensities, name='data')
            self.graphWidget.plot(range(10, 186), np.polyval(fit, range(10, 186)), name='fit', pen=pg.mkPen(color='r',
                                                                                                            width=2))
            self.graphWidget.setTitle('fitting a polynomial to average intensity over the selected panel', size='15pt')
            self.graphWidget.setLabel('left', "Avg. Pixel intensity")
            self.graphWidget.setLabel('bottom', "Pixel Number")
            self.graphWidget.addLegend()

            self.buttonClicked = 'plotFit'

            self.advanceSortButton.setEnabled(True)
            self.nextButton.setEnabled(True)
            self.previousButton.setEnabled(True)

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

        except IndexError:
            qtw.QMessageBox.critical(self, 'Fail', 'Value you entered is out of bound')

    def plotMaxPixels(self, file_name):
        try:
            y = self.returnMaxPixelsList(file_name, deg=6)
            x = range(len(y))
            self.graphWidget.clear()
            self.graphWidget.plot(x, y, pen=None, symbol='o')
            self.graphWidget.setTitle('change of the pixel with the highest average intensity', size='15pt')
            self.graphWidget.setLabel('left', "Pixel Number")
            self.graphWidget.setLabel('bottom', "Frame Number")

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

    def closeEvent(self, QCloseEvent):
        if self.imageViewer:
            self.imageViewer.close()
        if self.mlDialog:
            self.mlDialog.close()
        if self.sortGUI:
            self.sortGUI.close()


# main .
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
