# PyQt5 imports
from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
from PyQt5 import QtCore as qtc
from PyQt5.QtCore import pyqtSlot

# importing custom libraries
import lib.cfel_filetools as fileTools
import lib.cfel_imgtools as imgTools
from lib.geometry_parser.GeometryFileParser import *

# Packages for graphing
import pyqtgraph as pg


class DisplayImage(qtw.QWidget):
    panelSelected = qtc.pyqtSignal(dict)

    def __init__(self, fileName, geometry):

        """

        :param fileName: name of the *cxi file (with the full path)
        :param geometry: path to the geometry file
        """
        super(DisplayImage, self).__init__()

        # laoading the UI file
        uic.loadUi("UI/imageViewerGUI.ui", self)
        self.setGeometry(5, 100, 675, 750)

        self.mainLayout = qtw.QHBoxLayout()
        self.imageViewer = pg.ImageView()
        self.mainLayout.addWidget(self.imageViewer)
        self.mainWidget.setLayout(self.mainLayout)

        # assigning the file name and the geometry
        self.fileName = fileName
        self.geometryName = geometry

        # class variables
        self.outgoingDict = None
        self.eventNumber = None
        self.imageToDraw = None
        self.cxi = None
        self.size = None
        self.panelLocFromGeom = {}
        self.panelFsSs = {}
        self.panelsXYEdges = {}
        self.outgoingDict = {}

        # main window for display the data by hand
        # self.mainWidget = pg.ImageView()

        # setting the size and location of the window
        # self.setGeometry(10, 100, 600, 600)
        # adding a checkBoxes
        # self.foundPeaksCheckBox = qtw.QCheckBox('Found Peaks')
        # self.fixHistogramCheckBox = qtw.QCheckBox("Fix Histogram")
        # self.selectFor

        # self.layoutForCheckBoxes = qtw.QHBoxLayout()
        # self.layoutForCheckBoxes.addWidget(self.foundPeaksCheckBox)
        # self.layoutForCheckBoxes.addWidget(self.fixHistogramCheckBox)

        # adding a layout and add checkbox and the mainwindow to the layout
        # self.layout = qtw.QVBoxLayout()
        # self.layout.addWidget(self.mainWidget)
        # self.layout.addLayout(self.layoutForCheckBoxes)
        # self.setLayout(self.layout)

        # self.mainWidget.getView().addItem(self.foundPeaksCanvas)
        # self.mainWidget.getView().addItem(self.panelEdgesCanvas)
        # self.mainWidget.getView().scene().sigMouseClicked.connect(self.selectPanel)

        # showing the pixel map in the main window
        # self.mainWidget.setImage(self.imageToDraw)
        # self.isClosed = False

        # connecting the checkBoxes to a method
        self.foundPeaksCheckBox.stateChanged.connect(lambda: self.drawImage(self.eventNumber))

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
        self.imageViewer.getView().addItem(self.foundPeaksCanvas)

        # adding a canvas for displaying panel edges
        self.panelEdgesCanvas = pg.PlotDataItem()
        self.imageViewer.getView().addItem(self.panelEdgesCanvas)

        # connecting a mouse clicked event to a select panel method
        self.imageViewer.getView().scene().sigMouseClicked.connect(self.selectPanel)

        # handling what happens after the widget is closed
        self.setAttribute(qtc.Qt.WA_DeleteOnClose)

        # connecting signals
        self.imageViewer.getHistogramWidget().item.sigLevelChangeFinished.connect(self.handleHistogram)

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
            # print(self.fileName)
            # reading the given eventNumber from the cxi file
            self.cxi = fileTools.read_cxi(self.fileName, frameID=self.eventNumber, data=True, slab_size=True,
                                          peaks=True)
            self.size = self.cxi['stack_shape'][0]
            # reading data
            imgData = self.cxi['data']
            # converting data into a pixel map to display and applying geometry
            self.imageToDraw = imgTools.pixel_remap(imgData, self.geometry['x'], self.geometry['y'])
            # showing the pixel map in the main window
            # self.mainWidget.setImage(self.imageToDraw)
            self.imageViewer.setImage(self.imageToDraw)

            # setting a window title with the eventNumber and the total number of event in the file
            self.setWindowTitle("%s : Showing %i of %i " % (self.fileName.split('/')[-1], self.eventNumber, self.size - 1))

            if self.eventNumber == 0:
                self.drawInitialPanel()

            self.drawPeaks()

            # resetting the radiobuttons
            if self.goodRadioButton.isChecked():
                self.dummyRadioButton.setChecked(True)
            elif self.badRadioButton.isChecked():
                self.dummyRadioButton.setChecked(True)

        except IndexError as e:
            msg = qtw.QMessageBox()
            msg.setWindowTitle('Information')
            msg.setText("An error occurred while reading %s                                  " % self.fileName)
            msg.setInformativeText(str(e) + " drawImage()")
            msg.setIcon(qtw.QMessageBox.Information)
            msg.exec_()

    def handleHistogram(self):
        # Check if the fixHistogramCheckBox is not checked
        if not self.fixHistogramCheckBox.isChecked():
            # Retrieve current histogram levels
            # histogram = self.mainWidget.getHistogramWidget()
            histogram = self.imageViewer.getHistogramWidget()
            histMin, histMax = histogram.getLevels()

            # Set histogram range and levels for mainWidget
            # self.mainWidget.setHistogramRange(histMin, histMax)
            self.imageViewer.setHistogramRange(histMin, histMax)
            # self.mainWidget.setLevels(histMin, histMax)
            self.imageViewer.setLevels(histMin, histMax)

            # Update finalHistMin and finalHistMax variables
            self.finalHistMin = histMin
            self.finalHistMax = histMax
        else:
            # Checkbox is checked, adjust histogram range within finalHistMin and finalHistMax bounds

            # Retrieve current histogram levels
            # histMin, histMax = self.mainWidget.getHistogramWidget().getLevels()
            histMin, histMax = self.imageViewer.getHistogramWidget().getLevels()

            # Compare and update finalHistMin if necessary
            if histMin > self.finalHistMin:
                self.finalHistMin = histMin

            # Compare and update finalHistMax if necessary
            if histMax < self.finalHistMax:
                self.finalHistMax = histMax

            # Set histogram range and levels within finalHistMin and finalHistMax bounds
            # self.mainWidget.setHistogramRange(self.finalHistMin, self.finalHistMax)
            self.imageViewer.setHistogramRange(self.finalHistMin, self.finalHistMax)
            # self.mainWidget.setLevels(self.finalHistMin, self.finalHistMax)
            self.imageViewer.setLevels(self.finalHistMin, self.finalHistMax)

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
            # if self.mainWidget.getView().sceneBoundingRect().contains(pos):
            if self.imageViewer.getView().sceneBoundingRect().contains(pos):
                # mouse_point = self.mainWidget.getView().mapSceneToView(pos)
                mouse_point = self.imageViewer.getView().mapSceneToView(pos)
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

