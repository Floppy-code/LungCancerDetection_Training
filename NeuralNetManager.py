#3rd party imports
import medpy
import medpy.io
import matplotlib.pyplot as plt
import numpy as np
import os

#My imports
from CTScanModule import CTScanModule

SOURCE_PATH = 'D:/LungCancerCTScans/SPIE-AAPM Lung CT Challenge' #TODO

class NeuralNetManager:
    """Manager used to create and train new neural networks using CTScanModules"""

    def __init__(self):
        self._ct_scan_list = []


    #Iterates over CT scans and returns matching one or None
    def search_scan_by_id(self, id):
        for scan in self._ct_scan_list:
            if scan.id == id:
                return scan
        return None


    #Iterates over CT scans and returns matching one or None
    def search_scan_by_name(self, name):
        for scan in self._ct_scan_list:
            if scan.name == name:
                return scan
        return None


    #Loads all CT scans from .csv file, in a case that a patient has more than one nodule, it tries to
    #find an existing patient first
    def load_ct_scans(self, path_file = None):
        #FORMAT
        #Path, name, x, y, z
        if path_file == None:
            print("[!] Path file not specified!")
            return

        file = open(path_file, 'r') #Reading the .csv file
        for line in file.readlines():
            line = line.split(';')
            
            #Splitting the data into relevant groups
            name = line[1]
            data = None #Optimization so CT scan is not loaded if the patient already exists
            nodule_position = (int(line[2]), int(line[3]), int(line[4]))

            #Checking if the patient already exists
            existing_scan = self.search_scan_by_name(name)
            if (existing_scan is not None):
                existing_scan.add_nodule_location(nodule_position)
            else:
                data = self.get_dicom_data(line[0]) #Late data loading
                CT_scan_object = CTScanModule(name, len(self._ct_scan_list), data, nodule_position)
                self._ct_scan_list.append(CT_scan_object)


    #Loads DICOM data using MedPy
    def get_dicom_data(self, path_to_data):
        try:
            print("[*] Loading {}".format(path_to_data))
            dicom_data, dicom_header = medpy.io.load(path_to_data)
        except:
            print("[!] Loading failed, file: {}".format(path_to_data))
        return dicom_data