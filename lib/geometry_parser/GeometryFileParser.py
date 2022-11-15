#
# Author: Dominik Michels 
# Date: August 2016
#

import sys
import re
import pprint
import textwrap
import numpy

from .BeamCharacteristicFlags import *
from .PanelFlags import *
from .ParserError import *

class GeometryFileParser:
    """
    This class provides the functionality to parse the CrystFEL geometry file.
    """

    def __init__(self, filename = ""):
        """
        Constructor of the class.

        Args:
            filename (string): Filename of the geometry file.

        Note:
            The filename is optional.
        """

        self._lines = []
        self.filename = filename
        self.dictionary = {}
        self.dictionary['panels'] = {}
        self.dictionary['beam_characteristics'] = {}
        self.dictionary['bad_regions'] = {}
        self.dictionary['rigid_groups'] = {}
        self.dictionary['rigid_group_collections'] = {}

        self.error_list = []


    def dump(self):
        """
        This methods dumps the contents of the geometry dictionary.
        """

        #pprint.pprint(self.dictionary['panels'])
        #pprint.pprint(self.dictionary['beam_characteristics'])
        #pprint.pprint(self.dictionary['bad_regions'])
        #pprint.pprint(self.dictionary['rigid_groups'])
        #pprint.pprint(self.dictionary['rigid_group_collections'])
        pprint.pprint(self.dictionary)


    def _read_geometry_file(self, filename):
        """
        This methods reads the geometry file line by line into the self._lines
        list. Comments and empty lines are removed.

        Args:
            filename (string): Filename of the geometry file.
        """

        try:
            with open(filename, "r") as f:
                for line in f:
                    line_parsed_comments = line.split(";", 1)[0].strip()
                    if line_parsed_comments:
                        self._lines.append(line_parsed_comments)
        except IOError:
            print("Error reading the geometry file: ", self.filename)
            exit()


    def _match_pattern(self, line, pattern):
        regex = re.compile(pattern, re.VERBOSE)

        return regex.match(line)


    def _get_property_from_match(self, match, index):
        try:
            return match.group(index).strip()
        except IndexError:
            return ""
        

    def _match_rigid_group_information(self, line):
        """
        This methods checks whether the current line has information about rigid
        groups in the detector

        Args:
            line (string): Line containing the rigid group information

        Returns:
            re.MatchObject: The match object of the line

        """

        rigid_group_information_pattern = """
            ^[\s]*          
            ((rigid_group)
            (?!_collection)
            [A-Za-z0-9_]+)     
            [\s]*          
            =
            [\s]*             
            (([A-Za-z0-9_]+)[\s]*  (,[\s]*[A-Za-z0-9_]+[\s]*)*)$
        """

        return self._match_pattern(line, rigid_group_information_pattern)


    def _match_rigid_group_collection_information(self, line):
        """
        This methods checks whether the current line has information about rigid
        group collections in the detector

        Args:
            line (string): Line containing the rigid group information

        Returns:
            re.MatchObject: The match object of the line

        """

        rigid_group_information_pattern = """
            ^[\s]*           
            (rigid_group_collection
            [A-Za-z0-9_]+)    
            [\s]*             
            =
            [\s]*             
            (([A-Za-z0-9_]+)[\s]*  (,[\s]*[A-Za-z0-9_]+[\s]*)*)$
        """

        return self._match_pattern(line, rigid_group_information_pattern)


    def _get_rigid_group_information(self, match):
        """
        This method returns the rigid group information from the given match.

        Args:
            match (re.MatchObject): Match object containing the rigid group 
                information

        Returns:
            list: list of all the rigid groups

        """

        try:
            information = match.group(3)
            stripped_information = "".join(information.split())
            return stripped_information.split(",")
        except IndexError:
            return []


    def _get_rigid_group_collection_information(self, match):
        """
        This method returns the rigid group collection information from the 
        given line.

        Args:
            match (re.MatchObject): MatchObject containing the rigid group 
            collection information

        Returns:
            list: list of all the rigid groups collections

        """

        try:
            information = match.group(2)
            stripped_information = "".join(information.split())
            return stripped_information.split(",")
        except IndexError:
            return []


    def _match_bad_region_information(self, line):
        """
        This methods checks whether the current line has information about bad
        regions in the detector.

        Args:
            line (string): Line containing the local panel information

        Returns:
            re.MatchObject: The match object of the line

        """

        bad_region_pattern = """
            ^[\s]*              # whitespaces at the beginning
            (?=bad)             
            ([A-Za-z0-9_]+)     # bad region name
            \/
            ([A-Za-z0-9_]+)     # property name 
            [\s]*          
            =
            (.*$)               # property value
        """

        return self._match_pattern(line, bad_region_pattern)
    

    def _get_bad_region_information(self, match):
        """
        This method returns the bad region information from the given match.

        Args:
            match (string): MatchObject containing the bad region information

        Returns:
            string: Bad region information

        """

        try:
            return match.group(3).strip()
        except IndexError:
            return "" 


    def _match_local_panel_information(self, line):
        """
        This methods checks if the current line contains local panel 
        information.

        Args:
            line (string): Line containing the local panel information

        Returns:
            re.MatchObject: The match object of the line

        """

        local_panel_information_pattern = """
            ^[\s]*            # whitespaces at the beginning
            (?!bad)             # the panel information must not begin with bad
            ([A-Za-z0-9_]+)     # panel name
            \/
            ([A-Za-z0-9_]+)     # property name 
            [\s]*          
            =
            (.*$)               # property value
        """

        return self._match_pattern(line, local_panel_information_pattern) 


    def _get_panel_name(self, match):
        """
        This method returns the panel name from a given match.

        Args:
            match (re.MatchObject): Match object containing the panel name

        Returns:
            string: Panel name
        """

        try:
            return match.group(1).strip()
        except IndexError:
            return ""


    def _get_bad_region_name(self, match):
        """
        This method returns the name of the bad region from a given match.

        Args:
            match (re.MatchObject): Match object containing the bad region name

        Returns:
            string: Bad region name

        """

        try:
            return match.group(1).strip()
        except IndexError:
            return ""


    def _get_local_panel_information(self, match):
        """
        This method returns the local panel information from the given match.

        Args:
            match (re.MatchObject): Match obejct containing the local panel 
            information

        Returns:
            string: Local panel information

        """

        try:
            return match.group(3).strip()
        except IndexError:
            return ""


    def _match_global_panel_information(self, line):
        """
        This method checks whether the current line has global panel
        information.

        Returns:
            match (re.MatchObject): Match object containing the global panel
                information
        """

        global_panel_information_pattern = """
            ^[\s]*            
            (?!rigid_group)             # may not be a rigid group
            (?!rigid_group_collection)  # may not be a rigid group collection
            ([A-Za-z0-9_]+)             # property name 
            [\s]*          
            =
            (.*$)                       # property value
        """
        
        return self._match_pattern(line, global_panel_information_pattern)


    def _get_global_panel_information(self, match):
        """
        This method returns the global panel information from the given match.

        Args:
            match (re.MatchObject): Match object containing the global panel 
            information

        Returns:
            string: Global panel information

        """

        try:
            return match.group(2).strip()
        except IndexError:
            return ""

        
    def _match_beam_characteristics_information(self, line):
        """
        This method checks whether the current line has beam characteristics
        information.

        Returns:
            match (re.MatchObject): Match object containing the beam
                characteristics information

        Note:
            The layout of the beam characteristic information is exactly the
            same as the layout of the global panel information. Therefore this
            method just wraps the match_global_panel_information method.
        """

        return self._match_global_panel_information(line)


    def _get_beam_characteristics_information(self, match):
        """
        This method returns the beam characteristics information from the given 
        match.

        Args:
            match (re.MatchObject): Match object containing the beam
            characteristics information

        Returns:
            string: Global panel information

        Note:
            The layout of the beam characteristic information is exactly the
            same as the layout of the global panel information. Therefore this
            method just wraps the _get_global_panel_information method.

        """
        
        return self._get_global_panel_information(match)


    def _get_flag(self, line, line_matches):
        """
        This method returns the CrystFEL geometry flag (eg. ss, fs,
        photon_energy_eV, ...) from a given line. If no flag is found an empty
        string is returned.

        Args:
            line (string): Line containing the beam characteristics information

        Returns:
            string: Flag

        """

        flag = ""

        if (line_matches['beam_characteristics_information']
            or line_matches['global_panel_information']
            or line_matches['rigid_group_information']
            or line_matches['rigid_group_collection_information']):

            flag_pattern = """
                ([\s]*)
                =
                (.*)
            """

            regex = re.compile(flag_pattern, re.VERBOSE)
            flag = re.sub(regex, "", line)
        elif (line_matches['local_panel_information']
            or line_matches['bad_region_information']):
            flag_pattern1 = """
                (^[\s]*)
                [A-Za-z0-9_]+
                \/
            """

            flag_pattern2 = """
                ([\s]*)
                =
                (.*)
            """

            regex1 = re.compile(flag_pattern1, re.VERBOSE)
            regex2 = re.compile(flag_pattern2, re.VERBOSE)
            sub_line = re.sub(regex1, "", line)
            flag = re.sub(regex2, "", sub_line)
        if (line_matches['local_panel_information'] 
            or line_matches['beam_characteristics_information']):
            # check if found flags are valid
            for key in BeamCharacteristicFlags.list + PanelFlags.list:
                if key == flag:
                    return flag
            return ""
        else:
            return flag


    def _convert_type(self, key, value):
        if (key == "ss" or key == "fs"):
            stripped = "".join(value.split())
            check_pattern = """
            (^[-+]? 
            \s*
            (?: (?: \d* \. \d+) | (?: \d+ \.? ))?
            \s*
            x
            [-+]? 
            \s*
            (?: (?: \d* \. \d+) | (?: \d+ \.? ))?
            \s*
            y$)
            |
            (^[-+]?
            \s*
            (?: (?: \d* \. \d+) | (?: \d+ \.? ))?
            \s*
            [xy]{1})$
            """
            
            regex = re.compile(check_pattern, re.VERBOSE)
            if not regex.match(stripped):
                raise ParserError("Unable to parse the key " + key  + 
                    " with value: " + value)

            group_pattern = """
            ^([-+] (?= .*x))?   # Positive lookaheads in this and the next line
                                # are needed to check if [-+] is really 
                                # belonging to the x and not to the y variable. 
                                # This is relevant for the case when no x but a
                                # y variabe is present.
            ((?: (?: \d* \. \d+) | (?: \d+ \.? )) (?=\s*x))?
            \s*
            (x)?
            \s*
            ([-+])?  
            ((?: (?: \d* \. \d+) | (?: \d+ \.? )))?
            \s*
            (y)?$
            """

            regex = re.compile(group_pattern, re.VERBOSE)
            match = regex.match(stripped)
            if match:
                x_sign = match.group(1)
                x_value = match.group(2)
                x_key = match.group(3)
                y_sign = match.group(4)
                y_value = match.group(5)
                y_key = match.group(6)
                # debug prints may also be handy later
                # print("x_sign: ", x_sign)
                # print("x_key: ", x_key)
                # print("x_value: ", x_value)
                # print("y_sign: ", y_sign)
                # print("y_key: ", y_key)
                # print("y_value: ", y_value)
                if x_sign is None:
                    x_sign = "+"
                if y_sign is None:
                    y_sign = "+"
                if (x_key is not None and y_key is not None):
                    if x_value is None:
                        x_value = "1.0"
                    if y_value is None:
                        y_value = "1.0"
                    return {'x' : float(x_sign + x_value), 
                        'y' : float(y_sign + y_value)}
                elif (x_key is None and y_key is not None):
                    if y_value is None:
                        y_value = "1.0"
                    return {'x' : 0.0, 'y' : float(y_sign + y_value)}
                elif (x_key is not None and y_key is None):
                    if x_value is None:
                        x_value = "1.0"
                    return {'x' : float(x_sign + x_value), 'y' : 0.0}
                else:
                    raise ParserError("Unable to parse the key " + key \
                        + " with value: " + value)
        else:
            try:
                return int(value)
            except (ValueError, TypeError):
                try:
                    return  float(value)
                except (ValueError, TypeError):
                    return value


    def check_geometry(self, filename = ""):
        """
        This methods checks if the given geometry file or the geometry file
        stored in the object fulfills the definition standards of a valid
        CrystFEL geometry file.

        Args:
            filename (string): Path to the geometry file

        Returns:
            bool: True if the geometry file fulfills the standards, False
                otherwise

        Note:
            The filename is optional. If no filename is given the filename
            stored in self.filename is used.  
        """

        self._parse(filename, exit_on_error = False)

        num_errors = len(self.error_list)
        if num_errors > 0:
            if num_errors == 1:
                num_errors_string = "error"
            else:
                num_errors_string = "errors"
            print(textwrap.fill("The geometry file does not fulfill the " +
                "CrystFEL geometry standards. " + str(num_errors) + " " + 
                num_errors_string + " occured.", 80))
            for error in self.error_list:
                print("")
                print(textwrap.fill("Error: " + error[1], 80))
                print(textwrap.fill("Line: " + error[0], 80))
            self.error_list = []
            self.dictionary = {}
            return False
        else:
            print("The geometry file fulfills the CrystFEL geometry standards.")
            return True

    
    def parse(self, filename = ""):
        """
        This methods parses the geometry file. 

        Args:
            filename (string): Path to the geometry file

        Note:
            The filename is optional. If no filename is given the filename
            stored in self.filename is used.  
        """

        self._parse(filename, True)
            

    def _pixel_map(self):
        """
        This method returns the pixelmap needed for the cxiviewer.py

        Returns:
            x: slab-like pixel map with x coordinate of each slab pixel in 
                the reference system of the detector
            y: slab-like pixel map with y coordinate of each slab pixel in 
                the reference system of the detector
            z: slab-like pixel map with distance of each pixel from the center
                of the reference system.  

        """

        if not self.dictionary['panels']:
            self.parse()

        try:
            max_slab_fs = numpy.array([self.dictionary['panels'][k]['max_fs']  
                for k in self.dictionary['panels'].keys()]).max()
            max_slab_ss = numpy.array([self.dictionary['panels'][k]['max_ss'] 
                for k in self.dictionary['panels'].keys()]).max()


            x = numpy.zeros((max_slab_ss+1, max_slab_fs+1), dtype=numpy.float32)
            y = numpy.zeros((max_slab_ss+1, max_slab_fs+1), dtype=numpy.float32)


            for p in self.dictionary['panels'].keys():
                # get the pixel coords for this asic
                i, j = numpy.meshgrid(numpy.arange(
                    self.dictionary['panels'][p]['max_ss']
                    - self.dictionary['panels'][p]['min_ss'] + 1),
                    numpy.arange(self.dictionary['panels'][p]['max_fs'] 
                    - self.dictionary['panels'][p]['min_fs'] + 1), indexing='ij')

                # make the y-x ( ss, fs ) vectors, using complex notation
                dx  = self.dictionary['panels'][p]['fs']['y'] + 1J * \
                    self.dictionary['panels'][p]['fs']['x']
                dy  = self.dictionary['panels'][p]['ss']['y'] + 1J * \
                    self.dictionary['panels'][p]['ss']['x']
                #print("here")
                r_0 = self.dictionary['panels'][p]['corner_y'] + 1J * \
                    self.dictionary['panels'][p]['corner_x']

                r   = i * dy + j * dx + r_0

                y[self.dictionary['panels'][p]['min_ss']: \
                    self.dictionary['panels'][p]['max_ss'] + 1, \
                    self.dictionary['panels'][p]['min_fs']: \
                    self.dictionary['panels'][p]['max_fs'] + 1] = r.real
                x[self.dictionary['panels'][p]['min_ss']: \
                    self.dictionary['panels'][p]['max_ss'] + 1, \
                    self.dictionary['panels'][p]['min_fs']: \
                    self.dictionary['panels'][p]['max_fs'] + 1] = r.imag
        except KeyError:
            print(textwrap.fill("Error: The geometry file does not provide " +
                "sufficient information to construct the pixelmap.", 80))
            exit()

        r = numpy.sqrt(numpy.square(x) + numpy.square(y))
        return x,y,r


    def pixel_map_for_cxiview(self):
        """
        This method returns the information needed for the program cxiview.py

        Returns:
            dict: Dictionary the cxiviewer.py needs to display the information
                correctly. The keys are: x, y, r, dx, coffset, shape, clen
        """

        x, y, r = self._pixel_map()

        M = 2 * int(max(abs(x.max()), abs(x.min()))) + 2       
        N = 2 * int(max(abs(y.max()), abs(y.min()))) + 2

        y = -y
        img_shape = (M, N)

        coffset = float('nan')
        clen = float('nan')
        res = float('nan')
        dx_m = float('nan')

        try:
            panel_dict = next(iter(self.dictionary['panels'].values()))
        except KeyError:
            print("Error: The geometry file does not contain panel information.")
            exit()
        try:
            coffset = panel_dict['coffset']
        except KeyError:
            pass
        try:
            res = panel_dict['res']
            dx_m = 1.0/res
        except KeyError:
            pass
        try:
            clen = panel_dict['clen']
        except KeyError:
            pass

        result_dict = {
            'x' : x.flatten(),
            'y' : y.flatten(),
            'r' : r.flatten(),
            'dx' : dx_m,
            'coffset' : coffset,
            'shape' : img_shape,
            'clen' : clen
        }

        return result_dict


    def _parse(self, filename = "", exit_on_error = True):
        """
        This methods parses the geometry file.

        Args:
            filename (string): Path to the geometry file
            exit_on_error (bool): If True the program exits when an error in
                the parsing process occurs. If False the program stores the
                errors and line content and can give a complete error report.

        """

        if exit_on_error:
            # reset the error list
            self.error_list = []

        if filename != "":
            self.filename = filename

        self._read_geometry_file(self.filename)

        beam_flags = BeamCharacteristicFlags.list
        panel_flags = PanelFlags.list

        # we setup a dictionary storing current global panel information
        global_panel_properties = {}
        panels = set()
        bad_regions = set()

        for line in self._lines:
            # match all the known patterns in the line
            line_matches = {}
            line_matches['beam_characteristics_information'] = \
                self._match_beam_characteristics_information(line)
            line_matches['local_panel_information'] = \
                self._match_local_panel_information(line)
            line_matches['global_panel_information'] = \
                self._match_global_panel_information(line)
            line_matches['rigid_group_information'] = \
                self._match_rigid_group_information(line)
            line_matches['rigid_group_collection_information'] = \
                self._match_rigid_group_collection_information( line)
            line_matches['bad_region_information'] = \
                self._match_bad_region_information(line)

            try:
                # every line has to contain a flag, get the flag first
                flag = self._get_flag(line, line_matches)
                if flag == "":
                    raise ParserError(
                        "A valid CrystFEL geometry flag could not be found. " +
                        "Either the line has an invalid form or no CrystFEL " +
                        "geometry flag is present. ")

                # rigid groups
                if(line_matches['rigid_group_information']):
                    information = self._get_rigid_group_information(
                        line_matches['rigid_group_information'])
                    self.dictionary['rigid_groups'][flag] = \
                        self._convert_type(flag, information)
                # rigid group collections
                elif(line_matches['rigid_group_collection_information']):
                    information = self._get_rigid_group_collection_information(
                        line_matches['rigid_group_collection_information'])
                    self.dictionary['rigid_group_collections'][flag] = \
                        self._convert_type(flag, information)
                # search for panel characteristics
                # check first if the flag is for one panel only or global
                # information valid for many panels
                elif line_matches['local_panel_information']:
                    panel_name = self._get_panel_name(
                        line_matches['local_panel_information'])
                    # check whether the panel has already appeared in the 
                    # geometry
                    if panel_name not in panels:
                        panels.add(panel_name)
                        self.dictionary['panels'][panel_name] = {}
                        # set the current global information about panels in the
                        # dictionary
                        for key in global_panel_properties:
                           self.dictionary['panels'][panel_name][key] = \
                            global_panel_properties[key] 
                    # set the local line information in the dictionary
                    information = self._get_local_panel_information(
                        line_matches['local_panel_information'])
                    self.dictionary['panels'][panel_name][flag] = \
                        self._convert_type(flag, information)
                # global panel information has the same shape as beam 
                # characteristic information, we can just separate them by the
                # flag
                elif line_matches['global_panel_information']:
                    if flag in BeamCharacteristicFlags.list:
                        information = \
                            self._get_beam_characteristics_information(
                            line_matches['global_panel_information'])
                        self.dictionary['beam_characteristics'][flag] = \
                            self._convert_type(flag, information)
                    else: 
                        information = self._get_global_panel_information(
                            line_matches['global_panel_information'])
                        global_panel_properties[flag] = \
                            self._convert_type(flag, information)
                # bad regions
                elif line_matches['bad_region_information']:
                    bad_region_name = self._get_bad_region_name(
                        line_matches['bad_region_information'])
                    if bad_region_name not in bad_regions:
                        bad_regions.add(bad_region_name)    
                        self.dictionary['bad_regions'][bad_region_name] = {}
                    information = self._get_bad_region_information(
                        line_matches['bad_region_information'])
                    self.dictionary['bad_regions'][bad_region_name][flag] = \
                        self._convert_type(flag, information)
                else:
                    raise ParserError("The given line has an invalid form.")
                   
            except ParserError as e:
                if exit_on_error:
                    print(textwrap.fill("Error: " + e.args[0], 80))
                    print(textwrap.fill("Line: " + line, 80))
                    print("")
                    exit()
                else:
                    self.error_list.append((line, e.args[0]))
