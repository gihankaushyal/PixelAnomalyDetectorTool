import h5py
import numpy as np

def compare_and_remove_datasets(cxi_file1, cxi_file2):
    datasets = [
        '/LCLS/detector_1/EncoderValue',
        '/LCLS/fiducial',
        '/LCLS/photon_energy_eV',
        '/LCLS/timestamp',
        '/entry_1/data_1/data',
        '/entry_1/result_1/nPeaks',
        '/entry_1/result_1/peakMaximumValue',
        '/entry_1/result_1/peakNPixels',
        '/entry_1/result_1/peakSNR',
        '/entry_1/result_1/peakTotalIntensity',
        '/entry_1/result_1/peakXPosRaw',
        '/entry_1/result_1/peakYPosRaw'
    ]

    with h5py.File(cxi_file1, 'r') as f1, h5py.File(cxi_file2, 'r+') as f2:
        fiducial1 = f1['/LCLS/fiducial'][()]
        fiducial2 = f2['/LCLS/fiducial'][()]

        common_indices = np.in1d(fiducial2, fiducial1)

        for dataset in datasets:
            data = f2[dataset][()]
            reduced_data = data[~common_indices]

            # Resize the dataset in the second file
            f2[dataset].resize(reduced_data.shape)

            # Assign the reduced data to the resized dataset
            f2[dataset][...] = reduced_data

def main():
    cxi_file1 = '/Users/gketawal/PycharmProjects/InternalTool/r0484-snr5pix2/r484_10-cpy.cxi'
    cxi_file2 = '/Users/gketawal/PycharmProjects/InternalTool/r0484-snr5pix2/r484_10-cpy-2.cxi'

    compare_and_remove_datasets(cxi_file1, cxi_file2)

if __name__ == '__main__':
    main()