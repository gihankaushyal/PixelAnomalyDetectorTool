import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# import pandas as pd
import h5py

# bad_list keeps track of the file number and frame number of the bad panel p6a0
bad_list = [(1, 20), (1, 6), (1, 41), (1, 57), (1, 33), (11, 20), (11, 13), (1, 24), (1, 58), (1, 51)]

# good_list keeps track of the file number and frame number of the bad panel p6a0
good_list = [(1, 55), (11, 8), (11, 17), (1, 25), (1, 37), (11, 10), (1, 12), (11, 48), (1, 3), (11, 51)]

fig = plt.figure(figsize=(15, 18))
ax1 = fig.add_axes([0.1, 0.6, 0.9, 0.15])
ax2 = fig.add_axes([0.1, 0.4, 0.9, 0.15])
for tup in bad_list:
    filename = '/bioxfel/data/2020/LCLS-2020-Aug-FrommeP172-P182/analysis/gihan/data-new-hitfindings/r0484-snr5pix2/mfxp17218-r0484_%s.cxi' % \
               tup[0]

    with h5py.File(filename, "r") as f:
        data = f['entry_1']['data_1']['data'][()]

    frame = data[tup[1]]

    avgIntensities = []

    for i in range(10, 185):
        avgIntensities.append(np.average(frame[2112:2288, i]))

    ax1.plot(list(np.linspace(10, 185, 175)), avgIntensities)
    ax1.set_xlabel('pixel number')
    ax1.set_ylabel('average intensity of the PANEL')
    ax1.title.set_text('Bad Frames')
    ax1.set_yticks(np.linspace(800, 2000, 7))
    ax1.set_xticks(np.linspace(0, 200, 21))

for tup in good_list:
    filename = '/bioxfel/data/2020/LCLS-2020-Aug-FrommeP172-P182/analysis/gihan/data-new-hitfindings/r0484-snr5pix2/mfxp17218-r0484_%s.cxi' % tup[0]

    with h5py.File(filename, "r") as f:
        data = f['entry_1']['data_1']['data'][()]

    frame = data[tup[1]]

    avgIntensities = []

    for i in range(10, 185):
        avgIntensities.append(np.average(frame[2112:2288, i]))

    ax2.plot(list(np.linspace(10, 185, 175)), avgIntensities)
    ax2.set_xlabel('pixel number')
    ax2.set_ylabel('average intensity of the PANEL')
    ax2.title.set_text('Good Frames')
    ax2.set_yticks(np.linspace(800, 2000, 7))
    ax2.set_xticks(np.linspace(0, 200, 21))
