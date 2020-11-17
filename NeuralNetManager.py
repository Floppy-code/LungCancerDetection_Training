#3rd party imports
import medpy
import medpy.io
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping

#My imports
from CTScanModule import CTScanModule
import NeuralNet

class NeuralNetManager:
    """Manager used to create and train new neural networks using CTScanModules"""

    def __init__(self):
        self._ct_scan_list = []
        self._last_used_scan = -1 #K-Fold validation
        self._number_of_folds = 10


    def generate_fset(self, fold_to_use):
        """Returns a touple with ((training_fset, training_lset), (validation_fset, validation_lset))"""
        #Feature set shape - (len(fset), 512, 512, 1)
        #STEPS
        #1. Generate all 10 fsets
        #2. Keep 9 together and use 1 as validation
        #3. Test on the rest...
        training_fset = None
        training_lset = None

        validation_fset = None
        validation_lset = None

        print("**Generating validation Feature and Label set")
        validation_fset = self.get_fset_single(fold_to_use)
        validation_lset = self.get_lset_single(fold_to_use)

        for i in range(0, len(self._ct_scan_list)):
            if (i != fold_to_use):
                print("**Generating training Feature and Label set for fold: {}/{}".format(i + 1, self._number_of_folds))
                if (training_fset is None):
                    training_fset = self.get_fset_single(i)
                else:
                    training_fset = np.concatenate((training_fset, self.get_fset_single(i)), axis = 0)
                
                if (training_lset is None):
                    training_lset = self.get_lset_single(i)
                else:
                    training_lset = np.append(training_lset, self.get_lset_single(i))

                print("**Shapes: {} | {}".format(training_fset.shape, training_lset.shape))

        print("**Reshaping validation sets")
        validation_fset = validation_fset.reshape(len(validation_fset), 512, 512, 1)
        print('**Reshaping training sets')
        training_fset = training_fset.reshape(len(training_fset), 512, 512, 1)

        return ((training_fset, training_lset), (validation_fset, validation_lset))


    def train_model(self):
        for current_fold in range(0, self._number_of_folds):
            #Data
            data = self.generate_fset(current_fold)
            
            feature_set_training = data[0][0]
            label_set_training = data[0][1]

            feature_set_validation = data[1][0]
            label_set_validation = data[1][1]

            ##DEBUG
            ##Checking if the dataset is alright
            #for i in range(0, len(label_set_training)):
            #    if label_set_training[i] == True:
            #        plt.imshow(feature_set_training[i], cmap = 'gray')
            #        plt.show()

            #Training
            model = NeuralNet.get_neural_net_WH()
            model.fit(feature_set_training, label_set_training, batch_size = 16, epochs = 10, verbose = 1, validation_data = (feature_set_validation, label_set_validation))


    def get_fset_single(self, ct_scan_index):
        if (ct_scan_index >= len(self._ct_scan_list)):
            raise Exception("Scan index out of bounds!")
        
        return self._ct_scan_list[ct_scan_index].ct_scan_array


    def get_lset_single(self, ct_scan_index):
        #Label set is a Numpy array with index corresponding to true or false value of slice
        if (ct_scan_index >= len(self._ct_scan_list)):
            raise Exception("Scan index out of bounds!")
        
        ct_scan_module = self.search_scan_by_id(ct_scan_index)
        s_count = ct_scan_module.slice_count
        label_set = np.full((s_count), False)
        
        for loc in ct_scan_module.nodule_location:
            label_set[s_count - loc[2]] = True

        return label_set

        
    def print_CT_scan_modules(self):
        for scan in self._ct_scan_list:
            print("{}. {}".format(scan.id, scan.name))


    def show_CT_scan(self, ct_scan_id, mode, slice_no):
        scan = self.search_scan_by_id(ct_scan_id)
        if (scan == None):
            raise Exception("Scan not found!")
        scan.show_slice(mode, int(slice_no))

        
    #Iterates over CT scans and returns matching one or None
    def search_scan_by_id(self, id):
        for scan in self._ct_scan_list:
            if scan.id == int(id):
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
            raise Exception("Path file not specified!")

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
                CT_scan_object.normalize_data()
                CT_scan_object.transpose_ct_scan()
                self._ct_scan_list.append(CT_scan_object)


    #Loads DICOM data using MedPy
    def get_dicom_data(self, path_to_data):
        try:
            print("**Loading {}".format(path_to_data))
            dicom_data, dicom_header = medpy.io.load(path_to_data)
        except:
            print("[!] Loading failed, file: {}".format(path_to_data))
        return dicom_data