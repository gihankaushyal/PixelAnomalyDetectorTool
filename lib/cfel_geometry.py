#
#   CrystFEL Geometry parser
#   Mostly due to Valerio Mariani
#
#   Tested using Anaconda / Python 3.4
#

import h5py
import numpy
import os.path

from .geometry_parser.GeometryFileParser import *

def read_pixelmap(filename):
    """
    This function reads a Cheetah style pixelmap from an HDF5 file. The pixel
    coordinates are given in meters in the fields "x", "y" and "z"

    Args:
        filename (string): The path to the geometry file on the harddrive

    Returns:
        list: List containg the pixel map
    """

    # Open HDF5 pixelmap file
    try:
        with h5py.File(filename, 'r') as fp:
            x = fp['x'][:]
            y = fp['y'][:]
    except:
        print("Error reading the pixelmap file: ", filename)
        exit()

    # Correct for pixel size (meters --> pixels)
    # Currently hard coded for CSPAD
    # We can figure this out from the pixel map
    dx = 110e-6
    x /= dx
    y /= dx
    

    # Calculate radius
    r = numpy.sqrt(numpy.square(x) + numpy.square(y))

    return x, y, r, dx


def read_geometry(geometry_filename, quiet=True):
    """
    This functions reads the geometry file and returns pixel map needed for the
    cxiview.py program. Depending on the file format the function decides how
    to read the geometry.

    Args:
        geometry_filename (string): Path to the geometry file on the harddrive
        quiet (bool): True if detailed information of the geometry shall be
            printed, false otherwise
    
    Returns:
        dict: Geometry dictionary cxiview.py needs to display the diffraction 
            image

    Note:
        Axes are transposed such that images appear in the same orientation as
        in hdfsee, cheetah/IDL and pyQtGraph

        Return unit for geometry is pixels (CrystFEL unit 'res = 1/pix_size' is 
        depreciated). The structure of the returned dictionary is as follows:

        result_dict = {
            'x' : x.flatten(),      # In pixels
            'y' : y.flatten(),      # In pixels
            'r' : r.flatten(),      # In pixels
            'dx' : dx_m,
            'coffset' : coffset,
            'shape' : img_shape,
            'clen' : img_shape
        }
    """

    result_dict = {}

    extension = os.path.splitext(geometry_filename)[1]
    # read the geometry differently depending on the file format
    if extension == ".geom":
        parser = GeometryFileParser(geometry_filename)
        result_dict = parser.pixel_map_for_cxiview()

    elif extension == ".h5":
        x, y, r, dx_m = read_pixelmap(geometry_filename)
        coffset = float('nan')
        # clen is not neccessarily an integer so we choose as default the None 
        # type
        clen = None

        # find the smallest size of cspad_geom that contains all
        # xy values but is symmetric about the origin
        M = 2 * int(max(abs(x.max()), abs(x.min()))) + 2
        N = 2 * int(max(abs(y.max()), abs(y.min()))) + 2


        # convert x y values to i j values Minus sign for y-axis because Python
        # takes (0,0) in top left corner instead of bottom left corner

        # Note to Valerio: Do not add the offset to convert to image
        # coordinates staring at (0,0) as we may want the actual pixel
        # coordinates This means do not center array here --> it is done in
        # pixel_remap instead Returning actual coordinates (x,y) is better for
        # other operations such as radial averages

        x = x
        y = -y
        img_shape = (M, N)

        result_dict = {
            'x' : x.flatten(),
            'y' : y.flatten(),
            'r' : r.flatten(),
            'dx' : dx_m,
            'coffset' : coffset,
            'shape' : img_shape,
            'clen' : clen
        }
    else:
        print("Error reading geometry file: " + geometry_filename) 
        print("Unknown geometry file format: " + extension)
        exit()

    # Print a sanity check unless suppressed
    if not quiet:
        print('----------')
        print('Geometry info:')
        print('X range (pix): ', x.min(), x.max())
        print('Y range (pix): ', y.min(), y.max())
        print('R range (pix): ', r.min(), r.max())
        print('Pixel size (m): %4.6f' % (dx_m))
        print("Geometry shape: ", x.shape)
        print("Geometry elements: ", x.flatten().shape)
        print("Assembled image size: ", img_shape)

    return result_dict
