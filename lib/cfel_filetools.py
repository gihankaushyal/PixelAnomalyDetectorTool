# -*- coding: utf-8 -*-
#
#	CFEL file handling tools
#	Anton Barty
#

import os
import sys
import csv
import h5py
import glob
import numpy as np
import subprocess



# Needed for dialog_pickfile()
import PyQt5
import PyQt5.QtWidgets

""" Comment added by Dominik. With that line the GUI isn't rendered. The function of
    the line is not clear to me.
qtApp = PyQt5.QtWidgets.QApplication(sys.argv)
"""


#
# Launch a subprocess (eg: viewer or analysis script) without blocking the GUI
# Separate routine makes it easy change this globally if needed
#
def spawn_subprocess(cmdarr, wait=False, test=False, shell=False):
    command = str.join(' ', cmdarr)
    print(os.getcwd())
    print(command)

    if test:
        return

    # Shell requires command to be a string and not an array
    if shell:
        cmdarr = command

    if wait:
        subprocess.run(cmdarr)
    else:
        subprocess.Popen(cmdarr, shell=shell)



def dialog_pickfile(write=False, directory=False, multiple=False, path=False, filter='*.*', qtmainwin=None):
    """
    :param write:
    :param directory:
    :param multiple:
    :param path:
    :param filter:
    :return:
    See: http://doc.qt.io/qt-4.8/qfiledialog.html
    """
    if qtmainwin == None:
        qtApp = PyQt5.QtWidgets.QApplication(sys.argv)

    if path==False:
        path= ''

    if write==True:
        caption = 'Select destination file'
        file, _ = PyQt5.QtWidgets.QFileDialog.getSaveFileName(qtmainwin, caption, path, filter)
        return file

    elif directory==True:
        caption = 'Select directory'
        dirname = PyQt5.QtWidgets.QFileDialog.getExistingDirectory(qtmainwin, caption, path)
        return dirname

    elif multiple==True:
        caption = 'Select Files'
        files, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileNames(qtmainwin, caption, path, filter)
        return files

    else:
        caption = 'Select File'
        file, _ = PyQt5.QtWidgets.QFileDialog.getOpenFileName(qtmainwin, caption, path, filter)
        return file

#end dialog_pickfile()



def file_search(pattern, recursive=True, iterator=False):
    """
    :param pattern:
    :param recursive: True/False (default=True, '**' matches directories)
    :param iterator:
    :return:
    """
    if iterator:
        files = glob.iglob(pattern, recursive=recursive)
    else:
        files = glob.glob(pattern, recursive=recursive)
    return files
#end file_search()


def read_h5(filename='', field="/data/data"):
    """
    Read a simple HDF5 file

    if n_elements(filename) eq 0 then $
        filename = dialog_pickfile()
    """

    # Open a file selection dialog if no filename is provided
    if filename == '':
        filename = dialog_pickfile()
        if filename == '':
            return


    # Open HDF5 file, read specified field, clean up and exit
    try:
        with h5py.File(filename, 'r') as fp:
            data = fp[field][:]
            fp.close()
    except:
        print('Error reading field ', field, ' from ', filename)
        data = np.nan


    # return
    return data
# end read_h5




def write_h5(data_in, filename='', field="data/data"):
    """
    Write a simple HDF5 file
    """

    # Open a file selection dialog if no filename is provided
    if filename == '':
        filename = dialog_pickfile(write=True)
        if filename == '':
            return


    # Simple HDF5 writing with no data conversion
    try:
        with h5py.File(filename, 'w') as fp:
            dset = fp.create_dataset(field, data=data_in)
            fp.close()
    except:
        print('Error writing field ', field, ' to ', filename)

# end write_h5


#
#   Write selected dict keys to csv file
#
#   Provide a list of keys to force consistent ordering
#   Order of keys in the dict can be different each time
#   Seriously - try it and see with print(dict.keys())
#
def dict_to_csv(filename, dict, keys):

    ncol = len(keys)
    nrows = len(dict[keys[0]])

    # Check all keys are in the dict
    for k in keys:
        if not k in dict.keys():
            print("Error in dict_to_csv")
            print("Requested key is not in dict")
            print("Requested key: ", k)
            print("Available keys: ", dict.keys())
            return


    # Check all lines are the same length
    for k in keys:
        if len(dict[k]) != nrows:
            print("Error in dict_to_csv")
            print("Dict element ", k, "does not have the same dimensions")
            print("nlines: ", k, ' = ', len(dict[k]))
            print("nlines: ", keys[0], ' = ', nrows)
            return
        #endif
    #endfor

    # Write columns with header row
    with open(filename, 'w') as f:
        #w = csv.writer(sys.stderr)
        w = csv.writer(f)
        w.writerow(keys)

        for row in range(0, nrows):
            str_out = []
            for k in keys:
                str_out.append(dict[k][row])

            w.writerow(str_out)
        #endfor

        f.close()
    #endwith
#end dict_to_csv



#
#   Read CSV file into a dictionary using the header row as dict entry names
#   Result will be a blank dict {} if file does not exist
#
#   >>> import pandas as pd
#   >>> csv = pd.read_csv('example.csv')
#   >>> csv
#
def csv_to_dict(filename):

    # Open CSV file
    result = {}

    with open(filename, 'r', newline='') as f:

        reader = csv.DictReader(f)

        for row in reader:
           for column, value in row.items():
                result.setdefault(column.strip(), []).append(value.strip())

        f.close()

    # Strip blanks from field names
    fieldnames = list(reader.fieldnames)
    for item, field in enumerate(fieldnames):
        fieldnames[item] = field.strip()

    # Add field names to dict
    result.update({'fieldnames' : fieldnames})

    return result
#end csv_to_dict


#
#   Read event data using the appropriate reader for the format
#
def read_event(event_list, eventID, data=False, mask=False, peaks=False, photon_energy=False, camera_length=False, num_frames=False, slab_size=False):
    """
    Read an event from file
    Calls file-reading function for different file formats as specified in the event list 'format' field
    :param event_list:
    :param eventID:
    :param data:
    :param mask:
    :param peaks:
    :param photon_energy:
    :param camera_length:
    :param num_frames:
    :param slab_size:
    :return:
    """

    # Assume reading has not worked unless we get what we expect
    fail = True

    if event_list['format'][eventID] == 'cxi':
        event_data = read_cxi(event_list['filename'][eventID], event_list['event'][eventID], data=data, peaks=peaks, mask=mask, photon_energy=photon_energy, camera_length=camera_length, num_frames=num_frames, slab_size=slab_size)
        fail = True
    #end cxi

    elif event_list['format'][eventID] == 'cheetah_h5':
        data_array = read_h5(event_list['filename'][eventID], field=event_list['h5field'][eventID])
        # Need to add reading of peaks, photon energy, detector distance
        # Lower priority since this is an old file format
        if data is np.nan:
            fail = True
        # Need to add: Fail if data is not 2D array

        event_data = {
            'fail' : fail,
            'data': data_array,
            'data_shape': data_array.shape,
            'mask': [0],
            'nframes': 1,
            'EncoderValue': np.nan,
            'photon_energy_eV': np.nan,
            'n_peaks': 0,
            'peakXPosRaw': [0],
            'peakYPosRaw': [0]
        }

    elif event_list['format'][eventID] == 'generic_h5':
        data_array = read_h5(event_list['filename'][eventID], field=event_list['h5field'][eventID])
        fail = False

        event_data = {
            'fail' : fail,
            'data': data_array,
            'data_shape': data_array.shape,
            'mask': [0],
            'nframes': 1,
            'EncoderValue': np.nan,
            'photon_energy_eV': np.nan
        }
    #end generic_h5

    else:
        print("Unsupported file format: ", event_list['format'][eventID])
        fail = True
        exit(1)
    #end error


    return event_data



def read_cxi(filename, frameID=0, data=False, mask=False, peaks=False, photon_energy=False, camera_length=False, num_frames=False, slab_size=False):
    """ 
    Read a frame from multi-event CXI file
    Also read mask and peak lists if requested
    Would be smarter to read the requested stuff at once and return it all at the same time from the same file handle
    :param filename:
    :param frameID:
    :param mask:
    :param peaks:
    :param photon_energy:
    :param camera_length:
    :param slab_size:
    :return:
    """

    # Open CXI file
    hdf5_fh = h5py.File(filename, 'r')


    # Peak list
    if peaks == True:
        try:
            n_peaks = hdf5_fh['/entry_1/result_1/nPeaks'][frameID]
            peakXPosRaw = hdf5_fh['/entry_1/result_1/peakXPosRaw'][frameID]
            peakYPosRaw = hdf5_fh['/entry_1/result_1/peakYPosRaw'][frameID]
            peak_xy = (peakXPosRaw.flatten(), peakYPosRaw.flatten())
        except:
            n_peaks = 0
            peakXPosRaw = np.nan
            peakYPosRaw = np.nan
    else:
        n_peaks = 0
        peakXPosRaw = np.nan
        peakYPosRaw = np.nan


    # Masks
    if mask == True:
        try:
            mask_array = hdf5_fh['/entry_1/data_1/mask'][frameID, :, :]
        except:
            mask_array = np.nan
    else:
        mask_array = np.nan


    # Photon energy
    if photon_energy == True:
        try:
            photon_energy_eV = hdf5_fh['/instrument/photon_energy_eV'][frameID]
        except:
            photon_energy_eV = np.nan
    else:
        photon_energy_eV = np.nan


    # Camera length
    if camera_length == True:
        try:
            EncoderValue = hdf5_fh['/instrument/detector_1/EncoderValue'][frameID]
        except:
            EncoderValue = np.nan
    else:
        EncoderValue = np.nan

    # Array dimensions
    if slab_size == True:
        size = hdf5_fh['/entry_1/data_1/data'].shape
        data_shape = size[1:]
    else:
        size = [0,0,0]
        data_shape = [0,0]

    # Number of frames
    if num_frames == True:
        # For files which have finished being written this can be inferred from the data array shape
        # For files still being written there are blank frames at the end, so look for non-zero entries in x_pixel_size
        # The minimum of these two values is the number of events actually written so far
        #pix_size = hdf5_fh['entry_1/instrument_1/detector_1/data'][:]
        size = hdf5_fh['/entry_1/data_1/data'].shape
        nframes_1 = size[0]

        pix_size = hdf5_fh['entry_1/instrument_1/detector_1/x_pixel_size'][:]
        nframes_2 = len(np.where(pix_size != 0 )[0])

        nframes = np.min([nframes_1, nframes_2])
    else:
        nframes = -1

    # Image data
    if data == True:
        data_array = hdf5_fh['/entry_1/data_1/data'][frameID, :, :]
        data_shape = data_array.size
    else:
        data_array = np.nan


    # Close file
    hdf5_fh.close()


    # Build return structure
    result = {
        'data' : data_array,
        'data_shape' : data_shape,
        'stack_shape' : size,
        'mask' : mask_array,
        'nframes' : nframes,
        'EncoderValue' : EncoderValue,
        'photon_energy_eV' : photon_energy_eV,
        'n_peaks' : n_peaks,
        'peakXPosRaw' : peakXPosRaw,
        'peakYPosRaw' : peakYPosRaw
    }
    return result
# end read_cxi



def list_events(pattern='./*.cxi', field='data/data'):
    """
    :param file_pattern: Single filename, or search string
    :param field: HDF5 field from which to draw data, can be different for each file, default='data/data'
    :return: List of filenames, eventID and HDF5 field

    reload:
    import importlib
    importlib.reload(lib.cfel_filetools)
    from lib.cfel_filetools import *
    """

    # "field==none" means "use default value"
    if field=='none':
        field = 'data/data'

    # Find all files matching pattern
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        print('No files found matching pattern: ', pattern)
    list_of_files = glob.glob(pattern, recursive=True)
    list_of_files.sort()
        

    # List the found files (sanity check)
    #print('Found files:')
    #for filename in glob.iglob(pattern, recursive=True):
    #    print(filename)

    # Create empty event list
    filename_out = []
    eventid_out = []
    fieldname_out = []
    format_out = []


    #print('Found files:')
    #for filename in glob.iglob(pattern, recursive=True):
    for filename in list_of_files:

        basename = os.path.basename(filename)
        dirname = os.path.dirname(filename)
        #print(dirname, basename)

        # CXI file
        if filename.endswith(".cxi"):
            # Number of events in file
            nframes = read_cxi(filename, num_frames=True)['nframes']
            if nframes == 0:
                filename_short = basename
                # The following line has been commented by Dominik to avoid 
                # verbose console output
                #print(filename_short, '    ', nframes)
                continue

            # Default location for data in .cxi files is not data/data

            # But leave option for passing a different hdf5 data path on the command line
            cxi_field = field
            if cxi_field == 'data/data':
                cxi_field = '/entry_1/data_1/data'

            # Generate lists for this file
            cxi_filename = [filename] * nframes
            cxi_eventid = list(range(nframes))
            cxi_fieldname = [cxi_field] * nframes
            cxi_format = ['cxi'] * nframes

            # Append to main list
            filename_out.extend(cxi_filename)
            eventid_out.extend(cxi_eventid)
            fieldname_out.extend(cxi_fieldname)
            format_out.extend(cxi_format)
            #endif

        # Assume .h5 file is a single frame data file (for now)
        # Be more clever about generalising this later on
        # (eg: if number of dimensions of field = 2, it's an image; if number of dimensions = 3 it's a slab)
        if basename.endswith(".h5") and basename.startswith("LCLS"):
            nframes = 1
            filename_out.extend([filename])
            eventid_out.extend([0])
            fieldname_out.extend(['data/data'])
            format_out.extend(['cheetah_h5'])
            #endif

        elif basename.endswith(".h5"):
            nframes = 1
            filename_out.extend([filename])
            eventid_out.extend([0])
            fieldname_out.extend([field])
            format_out.extend(['generic_h5'])
            #endif


        #filename_short = filename.split('/')[-1]
        filename_short = basename
        # The following line has been commented by Dominik to avoid 
        # verbose console output
        #print(filename_short, '    ', nframes)


    #endfor

    nevents = len(filename_out)
    #print('Events found: ', nevents)


    # Build return structure
    result = {
        'nevents' : nevents,
        'filename': filename_out,
        'event': eventid_out,
        'h5field': fieldname_out,
        'format': format_out
    }
    return result

# end find_cheetah_images


