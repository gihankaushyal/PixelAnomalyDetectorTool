# PyQt5 imports
from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
from PyQt5 import QtCore as qtc
from PyQt5.QtCore import pyqtSlot

# Packages for plotting
from PyQt5 import QtWebEngineWidgets as qtwew # for graphing with plotly
import plotly.express as px

# Packages for data processing
import numpy as np
import pandas as pd

# Packages for data access
import h5py

# Packages for multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Packages for handle warning
import warnings


class DataLabeler(qtw.QWidget):
    readyToSaveGood = qtc.pyqtSignal(dict, str)
    readyToSaveBad = qtc.pyqtSignal(dict, str)

    def __init__(self, fileName, oft, inDict):
        """

        :param fileName: name of the file to be sorted
        :param oft: order of the fit for the polynomial
        :param inDict: dictionary with detector panel information
        """

        super(DataLabeler, self).__init__()

        self.badEvents = None
        self.goodEvents = None
        self.inflectionPoint1List = None
        self.inflectionPoint2List = None
        self.data = None
        self.setWindowTitle('Sorting for Machine Learning')

        uic.loadUi("UI/sortForMLGUI.ui", self)
        self.fileNameLabel.setText("Showing the results for : %s" % fileName.split('/')[-1])

        # for plotting with matplotlib
        # self.layoutSortingForML = qtw.QHBoxLayout()
        # self.figureSortingForML = plt.figure()
        # self.canvasForInflectionPoints = FigureCanvasQTAgg(self.figureSortingForML)
        # self.layoutSortingForML.addWidget(self.canvasForInflectionPoints)
        # self.graphSpace.setLayout(self.layoutSortingForML)

        # for plotting with plotly
        self.layout = qtw.QHBoxLayout()
        self.browser = qtwew.QWebEngineView()
        self.layout.addWidget(self.browser)
        self.graphSpace.setLayout(self.layout)

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
        # self.plotInflectionPoints()
        self.plotInflectionPointsParalle()
        self.sortButton.clicked.connect(self.sort)

        self.setAttribute(qtc.Qt.WA_DeleteOnClose)

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

    @staticmethod
    def processFrame(inTuple):
        """

        :param inTuple: the required arguments as a tuple
        :return: a list of the calculated inflection points
        """
        (frame, min_fs, max_fs, min_ss, max_ss, orderOfFit) = inTuple
        print("Processing frame...")

        avgIntensities = [np.average(frame[min_ss:max_ss, j]) for j in
                          range(min_fs + 5, max_fs - 5)]
        fit = np.polyfit(np.arange(min_fs + 5, max_fs - 5), avgIntensities, deg=int(orderOfFit))

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                x1 = round((-6 * fit[1] + np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]), 2)
                x2 = round((-6 * fit[1] - np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]), 2)

                if x1 > x2:
                    return x1, x2
                else:
                    return x2, x1

            except (IndexError, ValueError, Warning):
                print("Error while processing frame.")
                return None, None  # or some other sentinel value

    def plotInflectionPointsParalle(self):
        print("Plotting inflection points in parallel...")
        self.inflectionPoint1List = []
        self.inflectionPoint2List = []

        try:
            with h5py.File(self.fileName, "r") as f:
                self.data = f['entry_1']['data_1']['data'][()]

            with ProcessPoolExecutor() as executor:
                frames = [(frame, self.min_fs, self.max_fs, self.min_ss, self.max_ss, self.orderOfFit) for frame in
                          self.data]
                results = list(executor.map(DataLabeler.processFrame, frames))

            # After multiprocessing, collect results and handle any errors
            for i, (x1, x2) in enumerate(results):
                if x1 is not None and x2 is not None:  # or whatever your sentinel value was
                    self.inflectionPoint1List.append(x1)
                    self.inflectionPoint2List.append(x2)
                else:
                    print(f"Calculation Error! Skipping the frame {i}")
                    # handle errors - note that this is done in the main process, so it's safe to use GUI functions
                    qtw.QMessageBox.information(self, 'Skip', 'Calculation Error! \n \n Skipping the frame %i' % i)

        except Exception as e:
            print(f"Error in plotInflectionPoint: {e}")

        # with plotly
        df = pd.DataFrame()
        df['Inflection_point1'] = self.inflectionPoint1List
        df['Inflection_point2'] = self.inflectionPoint2List
        fig = px.histogram(df, nbins=200, opacity=0.5)
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))

        # with seaborn
        # self.figureSortingForML.clear()
        # df = pd.DataFrame()
        # df['Inflection_point1'] = self.inflectionPoint1List
        # df['Inflection_point2'] = self.inflectionPoint2List
        # colors = ['red', 'green', 'blue', 'violet', 'pink']
        # random.shuffle(colors)
        # for column in df.columns:
        #     sns.histplot(data=df[column], color=colors.pop(), binrange=(-300, 300), bins=80, alpha=0.5, label=column)
        # plt.title('Distributions of Inflection points 1 and 2')
        # plt.ylabel('Count')
        # plt.xlabel(' Vertically Average Pixel Intensity')
        # plt.xticks()
        # plt.legend()
        # self.canvasForInflectionPoints.draw()

        # Enabling button and check box after plotting
        self.inflectionPoint1.setEnabled(True)
        self.inflectionPoint1.setText(str(round(np.median(df['Inflection_point1'].dropna()), 2)))
        self.inflectionPoint2.setEnabled(True)
        self.inflectionPoint2.setText(str(round(np.median(df['Inflection_point2'].dropna()), 2)))
        self.sortButton.setEnabled(True)
        self.doubleSpinBoxIF1.setValue(round(3*np.std(df['Inflection_point1'].dropna()), 2))
        self.doubleSpinBoxIF2.setValue(round(3*np.std(df['Inflection_point2'].dropna()), 2))
        print("Inflection points plotted successfully.")

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
                        if x1 > x2:
                            self.inflectionPoint1List.append(x1)
                            self.inflectionPoint2List.append(x2)
                        else:
                            self.inflectionPoint2List.append(x1)
                            self.inflectionPoint1List.append(x2)

                        # old method
                        # self.inflectionPoint1List.append(x1)
                        # self.inflectionPoint2List.append(x2)

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
        df = pd.DataFrame()
        df['Inflection_point1'] = self.inflectionPoint1List
        df['Inflection_point2'] = self.inflectionPoint2List
        fig = px.histogram(df, nbins=200, opacity=0.5)
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))

        # with seaborn
        # self.figureSortingForML.clear()
        # df = pd.DataFrame()
        # df['Inflection_point1'] = self.inflectionPoint1List
        # df['Inflection_point2'] = self.inflectionPoint2List
        # colors = ['red', 'green', 'blue', 'violet', 'pink']
        # random.shuffle(colors)
        # for column in df.columns:
        #     sns.histplot(data=df[column], color=colors.pop(), binrange=(-300, 300), bins=80, alpha=0.5, label=column)
        # plt.title('Distributions of Inflection points 1 and 2')
        # plt.ylabel('Count')
        # plt.xlabel(' Vertically Average Pixel Intensity')
        # plt.xticks()
        # plt.legend()
        # self.canvasForInflectionPoints.draw()

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

        fileSaveLocation = qtw.QFileDialog.getExistingDirectory(self, caption='Select Save Location', directory=' ',
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
                self.readyToSaveGood.emit(self.goodEvents,
                                          fileSaveLocation + '/' + 'goodEvents-advanceSort-%s.list' % tag)
                self.readyToSaveBad.emit(self.badEvents, fileSaveLocation + '/' + 'badEvents-advanceSort-%s.list' % tag)

            except Exception as e:
                msg = qtw.QMessageBox()
                msg.setWindowTitle('Error')
                msg.setText("An error occurred while sorting the file %s                              " % self.fileName)
                msg.setInformativeText(str(e) + " sort()")
                msg.setIcon(qtw.QMessageBox.Information)
                msg.exec_()

        else:
            qtw.QMessageBox.warning(self, 'Warning', 'Please Select a Save Location for sorted files')
