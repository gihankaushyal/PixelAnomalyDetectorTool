import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# import pandas as pd
import h5py


def returnMaxPix(coeff, xRange):
    ''' coeff (object): out put of a numpy curve fit.
        xRange (tup) : a tuple with (min, max)

        returns the value where the coeff is maximized
    '''

    storingList = []
    for i in range(xRange[0], xRange[1]):
        storingList.append((np.polyval(coeff, i), i))
    storingList.sort()
    return storingList[-1][1]


def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['r_squared'] = 1 - (((1 - (ssreg / sstot)) * (len(y) - 1)) / (len(y) - degree - 1))

    return results


def plotfit(fileName, eventNumber=1, deg=1):
    ''' fileName(str) : name of the file to be open
        eventNumber(int) : event number for the file
        deg (int) : order of the fit ex: is the fit a straight line (1) or quadratic (2 or more)
    '''

    filename = fileName
    with h5py.File(filename, "r") as f:
        data = f['entry_1']['data_1']['data'][()]

    frame = data[eventNumber]

    avgIntensities = []

    for i in range(10, 186):
        avgIntensities.append(np.average(frame[2112:2288, i]))

    fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=deg)

    fig, ax = plt.subplots()
    ax.plot(range(10, 186), avgIntensities, label='data')
    ax.plot(range(10, 186), np.polyval(fit, range(10, 186)), label='fit')
    ax.set_yticks(np.linspace(800, 2000, 7))
    ax.set_xticks(np.linspace(0, 200, 21))
    ax.legend()
    plt.show()


def maxPixels(fileName, deg=1, *args):
    ''' fileName(str) : name of the file to be open
        deg (int) : order of the fit ex: is the fit a straight line (1) or quadratic (2 or more)
        *args(list) : expects a list of events to be considered. **TO BE IMPLEMENTED**
    '''
    filename = fileName
    maxPixel = []
    with h5py.File(filename, "r") as f:
        data = f['entry_1']['data_1']['data'][()]

    for i in range(len(data)):
        frame = data[i]

        avgIntensities = []
        for j in range(10, 186):
            avgIntensities.append(np.average(frame[2112:2288, j]))

        fit = np.polyfit(np.arange(10, 186), avgIntensities, deg=deg)

        maxPixel.append(returnMaxPix(fit, (10, 186)))
    return maxPixel
