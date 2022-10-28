import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# import pandas as pd
import h5py
from pathlib import Path

# following code will look at a user specified panel of the LCLS ePix detector and plot vertically average pixel
# intensity against the pixel position to aid with debugging detector calibration issues.


# user manually has to go through the *cxi file or the *stream file at the moment to identify the good and bad frames
# by eye.
# tuple has the format of (*cxi file name, event number)

# bad_list keeps track of the file number and frame number of the bad panel p6a0
bad_list = [(1, 20), (1, 6), (1, 41), (1, 57), (1, 33), (11, 20), (11, 13), (1, 24), (1, 58), (1, 51)]

# good_list keeps track of the file number and frame number of the bad panel p6a0
good_list = [(1, 55), (11, 8), (11, 17), (1, 25), (1, 37), (11, 10), (1, 12), (11, 48), (1, 3), (11, 51)]


def plotCurve(ax_name, input_list, fileName, plot_title):
    for tup in input_list:
        filename = fileName % tup[0]

        with h5py.File(filename, "r") as f:
            data = f['entry_1']['data_1']['data'][()]

        frame = data[tup[1]]

        avgIntensities = []

        for i in range(10, 185):
            avgIntensities.append(np.average(frame[2112:2288, i]))

        ax_name.plot(list(np.linspace(10, 185, 175)), avgIntensities)
        ax_name.set_xlabel('pixel number')
        ax_name.set_ylabel('average intensity of the PANEL')
        ax_name.title.set_text(plot_title)
        ax_name.set_yticks(np.linspace(800, 2000, 7))
        ax_name.set_xticks(np.linspace(0, 200, 21))


fileName = '/Users/gketawal/PycharmProjects/InternalTool/r0484-snr5pix2/mfxp17218-r0484_%s.cxi'
fig = plt.figure(figsize=(15, 18))
ax1 = fig.add_axes([0.1, 0.6, 0.9, 0.15])
ax2 = fig.add_axes([0.1, 0.4, 0.9, 0.15])

# ploting the curves
plotCurve(ax1, bad_list, fileName, 'Bad List')
plotCurve(ax2, good_list, fileName, 'Good List')

# following code does the sorting of the frames by looking at the peak of the curve (pixels 60-80) and valley of the
# curve (pixel 165-185)

# sorting events from the cxi file based on the panel p6a0 intensity signature


folder = '/Users/gketawal/PycharmProjects/InternalTool/r0484-snr5pix2/'
files = Path(folder).glob('*.cxi')

goodEvents = {}
badEvents = {}

for file in files:
    # goodList to store all the events with expected pixel intensities for the file
    goodList = []

    # badList to store all the events with detector artifacts for the file
    badList = []

    with h5py.File(file, "r") as f:
        data = f['entry_1']['data_1']['data'][()]

    #   print(len(data)) #this line was intended to see if the code is actually reading file and also at the same time
    #   check to the number of data blocks in each cxi file

    # print('Reading %s and it has %i events' % (file, len(data)))
    # break

    for i in range(0, len(data)):
        frame = data[i]

        peakMeanIntensities = []
        for j in range(60, 80):
            peakMeanIntensities.append(np.average(frame[2112:2288, j]))
        # print(peakMeanIntensities)

        crestMeanIntensities = []
        for k in range(165, 185):
            crestMeanIntensities.append(np.average(frame[2112:2288, k]))
        # print(crestMeanIntensities)

        peakMean = np.average(peakMeanIntensities)
        crestMean = np.average(crestMeanIntensities)

        if peakMean / crestMean > 1:
            goodList.append(i)
        else:
            badList.append(i)

    #       print(goodList)
    #       print(badList)

    goodEvents[str(file)] = goodList
    badEvents[str(file)] = badList


# following code will write the good events and bad events in to separate *cxi files
# running into possible memory issue when the data set is getting bigger

def writeCxi(dictionary, name):
    fileName = name
    outFile = h5py.File(fileName, "w")
    entry_1 = outFile.create_group('entry_1')
    data_1 = entry_1.create_group('data_1')

    r = 1
    for key in dictionary.keys():
        with h5py.File(key, "r") as f:
            for i in dictionary[key]:
                frame = f['entry_1']['data_1']['data'][()][i]
                if r == 1:
                    dataBlock = frame
                elif r == 2:
                    dataBlock = np.stack((dataBlock, frame))
                else:
                    dataBlock = np.vstack((dataBlock, frame.reshape(1, 5632, 384)))
                r += 1

    data = data_1.create_dataset("data", data=dataBlock)
    outFile.close()


writeCxi(goodEvents, "goodFrames.cxi")
writeCxi(badEvents, "badFrames.cxi")

#follwing code will write good and bad frames to separate file

def writetofile(eventsList,fileName):
    f = open(fileName, 'w')
    for key in eventsList:
        for i in eventsList[key]:
            f.write('%s //%i \n' % (key,i))
    f.close()


writetofile(goodEvents, 'googEvents.list')
writetofile(badEvents, 'badEvents.list')