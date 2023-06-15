import h5py
import glob
import numpy as np
import numpy as np


# def create_output_structure(output, input_file):
#     with h5py.File(input_file, 'r') as f:
#         f.copy('/LCLS', output)
#         f.copy('/entry_1', output)
#
#         datasets = [
#             '/LCLS/detector_1/EncoderValue',
#             '/LCLS/fiducial',
#             '/LCLS/photon_energy_eV',
#             '/LCLS/timestamp',
#             '/entry_1/data_1/data',
#             '/entry_1/result_1/nPeaks',
#             '/entry_1/result_1/peakMaximumValue',
#             '/entry_1/result_1/peakNPixels',
#             '/entry_1/result_1/peakSNR',
#             '/entry_1/result_1/peakTotalIntensity',
#             '/entry_1/result_1/peakXPosRaw',
#             '/entry_1/result_1/peakYPosRaw'
#         ]
#
#         for dataset in datasets:
#             output_shape = list(output[dataset].shape)
#             output_shape[0] = 0
#             output[dataset].resize(output_shape)


# def append_data(output, input_file):
#     with h5py.File(input_file, 'r') as f:
#         datasets = [
#             '/LCLS/detector_1/EncoderValue',
#             '/LCLS/fiducial',
#             '/LCLS/photon_energy_eV',
#             '/LCLS/timestamp',
#             '/entry_1/data_1/data',
#             '/entry_1/result_1/nPeaks',
#             '/entry_1/result_1/peakMaximumValue',
#             '/entry_1/result_1/peakNPixels',
#             '/entry_1/result_1/peakSNR',
#             '/entry_1/result_1/peakTotalIntensity',
#             '/entry_1/result_1/peakXPosRaw',
#             '/entry_1/result_1/peakYPosRaw'
#         ]
#
#         for dataset in datasets:
#             data = f[dataset][()]
#             output_data = output[dataset]
#             output_shape = list(output_data.shape)
#             output_shape[0] += data.shape[0]
#             output_data.resize(output_shape)
#             output_data[-data.shape[0]:] = data


# def combine_cxi_files(input_files, output_file):
#     with h5py.File(output_file, 'w') as output:
#         create_output_structure(output, input_files[0])
#
#         for file in input_files:
#             append_data(output, file)


def fiducialsFromCXI(cxiFile):
    """
    :param cxiFile: in put cxi file
    :return: an array with fiducils
    """
    with h5py.file(cxiFile, 'r') as f:
        fiducials = np.array(f['LCLS']['fiducial'])[()]
        return fiducials


def compareFiducials(arr1, arr2):
    """
    comparing the two arrays with fiducials
    :param arr1: arrys of hits
    :param arr2: array of hits + non hits
    :return: an arrya of arr2 - arr1
    """
    return list(set(arr1) & set(arr2))


def deleteDataFromFiducials(list, cxiFile, ):
    """
    delete the images/slices from the np arrays matches to the location of the fiducials
    :param list: fiducials to be deleted
    :param cxiFile: the source cxi file
    :return: a new cxi file with the given fiducials deleted
    """
    fiducials = fiducialsFromCXI(cxiFile)

    with h5py.File(cxiFile, 'r') as f:
        data = f['entry_1']['data_1']['data'][()]

    for element in list:
        locationToDelete = np.where(fiducials == element)
        fiducials = np.delete(fiducials, locationToDelete)
        data = np.delete(data, locationToDelete, axis=0)

    return fiducials
    return data


def extractDataFromFiducials(list, cxiFile, ):
    """
    extract the images/slices from the np arrays matches to the location of the fiducials
    :param list: fiducials to be deleted
    :param cxiFile: the source cxi file
    :return: a new cxi file with the given fiducials deleted
    """
    fiducials = fiducialsFromCXI(cxiFile)

    with h5py.File(cxiFile, 'r') as f:
        data = f['entry_1']['data_1']['data'][()]

    locationToExtract = []
    for element in list:
        locationToExtract.append(np.where(fiducials == element))

    dataOut = np.array([])
    for location in locationToExtract:
        dataOut.append(data[location])

    return list
    return dataOut


def createCXIFile(fileName, fiducials, data):
    """
    Creating a CXI file using given parameters
    :param fileName: name of the file to be saved
    :param fiducials: fiducials
    :param data: data
    :return: cxi file
    """

    with h5py.File(fileName, 'w') as f:
        group1 = f.create_group('LCLS')
        group2 = f.create_group('entry_1')
        group3 = group2.create_group('data_1')

        group1.create_dataset('fiducial', data=fiducials)
        group3.create_dataset('data', data=data)


def main():
    # input_files = glob.glob('/Users/gketawal/PycharmProjects/InternalTool/r0484-snr5pix2/*.cxi')
    # output_file = '../../combined_data.cxi'
    # combine_cxi_files(input_files, output_file)
    pass

if __name__ == '__main__':
    main()
