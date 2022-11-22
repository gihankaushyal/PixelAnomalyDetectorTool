import h5py
import numpy as np

from PyQt5 import QtWidgets as qtw


class sortTools:

    def __init__(self):
        pass

    def writeToFile(self, eventsList, fileName):
        f = open(fileName, 'w')
        for key in eventsList:
            for eventNumber,x1,x2 in eventsList[key]:
                # f.write('%s //%i \n' % (key, i))
                f.write(key)
                f.write(' ')
                f.write('//')
                f.write(str(eventNumber))
                f.write(' ')
                f.write(str(x1))
                f.write(' ')
                f.write(str(x2))
                f.write('\n')

        f.close()

    def sortFrames(self, file_name):
        """
            this method only works on single cxi file. for multiple files code needs to be changed
            function for sorting good frames from the bad frames based on the average intensity ratio of the peak of the
            curve vs crest of the curve
        """

        try:
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

            self.writeToFile(goodEvents, 'goodEventsGUI.list')
            self.writeToFile(badEvents, 'badEventsGUI.list')

            qtw.QMessageBox.information(self, 'Success', "Done Sorting")

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

    def advanceSortFrames(self, file_name, orderOfFit):
        tag = str(file_name).split('/')[-1].split('.')[0].split('-')[-1]

        goodEvents = {}
        badEvents = {}

        # goodList to store all the events with expected pixel intensities for the file
        goodList = []
        # badList to store all the events with detector artifacts for the file
        badList = []
        try:
            with h5py.File(file_name, "r") as f:
                data = f['entry_1']['data_1']['data'][()]

            for i in range(len(data)):
                frame = data[i]

                avgIntensities = []
                for j in range(10, 186):
                    avgIntensities.append(np.average(frame[2112:2288, j]))

                fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=int(orderOfFit))
                # calculating the inflection points (second derivative of the forth order polynomial)
                # print(fit)
                try:
                    x1 = round((-6 * fit[1] + np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]))
                    x2 = round((-6 * fit[1] - np.sqrt(36 * fit[1] * fit[1] - 96 * fit[0] * fit[2])) / (24 * fit[0]))
                except IndexError:
                    print("Index Error")
                    # qtw.QMessageBox.information(self, 'Error', 'Please try a higher order polynomial')
                except ValueError:
                    print('Value Error')
                    # qtw.QMessageBox.information(self, 'Skip', 'Calculation Error! \n \n Skipping the frame')
                    continue

                if x1 in range(130, 140):
                    goodList.append((i,x1,x2))
                else:
                    badList.append((i,x1,x2))

            goodEvents[str(file_name)] = goodList
            badEvents[str(file_name)] = badList

            self.writeToFile(goodEvents, 'goodEvents-advanceSearch-%s.list' % tag)
            self.writeToFile(badEvents, 'badEvents-advanceSearch-%s.list' % tag)

            print('done sorting')
            # qtw.QMessageBox.information(self, 'Success', "Done Sorting")

        except FileNotFoundError:
            # qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)
            print("%s is not found" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")


class plotTools:

    def __init__(self):
        pass

    def returnMaxPixel(self, coeff, x_range):
        """
            :type coeff: output of a numpy curve fit.
            :type xRange : a tuple with (min, max)

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

    def plotCurve(self, file_name, plotWidget, event_number=0):
        """

        :type plotWidget: pyqtgraph plotWidget object
        """
        try:
            with h5py.File(file_name, "r") as f:
                data = f['entry_1']['data_1']['data'][()]

            frame = data[int(event_number)]

            avgIntensities = []

            for i in range(10, 185):
                avgIntensities.append(np.average(frame[2112:2288, i]))

            plotWidget.clear()
            plotWidget.plot(list(np.linspace(10, 185, 175)), avgIntensities)
            plotWidget.setTitle('average intensity over the selected panel', size='15pt')
            plotWidget.setLabel('left', "Avg. Pixel intensity")
            plotWidget.setLabel('bottom', "Pixel Number")
            # self.buttonClicked = 'plotCurve'
            # self.sortButton.setEnabled(True)
            # self.nextButton.setEnabled(True)
            # self.previousButton.setEnabled(True)

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

    def plotFit(self, file_name, plotWidget, eventNumber=0, deg=4):
        """ fileName(str) : name of the file to be open
                eventNumber(int) : event number for the file
                deg (int) : order of the fit ex: is the fit a straight line (1) or quadratic (2 or more)
                :type plotWidget: pyqtgraph plotWidget object
        """
        try:
            filename = file_name
            with h5py.File(filename, "r") as f:
                data = f['entry_1']['data_1']['data'][()]

            frame = data[int(eventNumber)]

            avgIntensities = []
            degry = int(deg)

            for i in range(10, 186):
                avgIntensities.append(np.average(frame[2112:2288, i]))

            fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=degry)

            plotWidget.clear()
            plotWidget.plot(range(10, 186), avgIntensities, name='data')
            plotWidget.plot(range(10, 186), np.polyval(fit, range(10, 186)), name='fit', pen=pg.mkPen(color='r',
                                                                                                      width=2))
            plotWidget.setTitle('fitting a polynomial to average intensity over the selected panel', size='15pt')
            plotWidget.setLabel('left', "Avg. Pixel intensity")
            plotWidget.setLabel('bottom', "Pixel Number")
            plotWidget.addLegend()
            # self.buttonClicked = 'plotFit'
            # self.advanceSortButton.setEnabled(True)
            # self.nextButton.setEnabled(True)
            # self.previousButton.setEnabled(True)

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")

        except IndexError:
            qtw.QMessageBox.critical(self, 'Fail', 'Value you entered is out of bound')

    def plotMaxPixels(self, file_name, plotWidget):
        try:
            y = self.returnMaxPixelsList(file_name, deg=6)
            x = range(len(y))

            plotWidget.clear()
            plotWidget.plot(x, y, pen=None, symbol='o')
            plotWidget.setTitle('change of the pixel with the highest average intensity', size='15pt')
            plotWidget.setLabel('left', "Pixel Number")
            plotWidget.setLabel('bottom', "Frame Number")

        except FileNotFoundError:
            qtw.QMessageBox.critical(self, 'Fail', "Couldn't find file %s" % file_name)

        except ValueError:
            qtw.QMessageBox.critical(self, 'Fail', "Please Enter a file path")
