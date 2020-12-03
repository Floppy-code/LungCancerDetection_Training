#3rd party imports
import medpy
import medpy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random

from keras.callbacks import EarlyStopping

#My imports
from CTScanModule import CTScanModule
import NeuralNet

#=============== GLOBAL VARIABLES ===============
#Saves CTScanModule objects to disk to improve load times
CT_SCAN_CACHING = True  
CT_SCAN_CACHING_PATH = '.scan_cache'

#Saves training stats after every epoch into file
TRAINING_HISTORY = True  
TRAINING_HISTORY_VAL_ACC = 'val_acc_training.txt'
TRAINING_HISTORY_ACC = 'acc_training.txt'

#Label set implementation
ONE_HOT = 0
HEATMAP = 1

#Data augumentation
UNDERSAMPLING = 1
OVERSAMPLING = 2

#CNN Model
MODEL_SAVE_PATH = 'last_model.h5'


class NeuralNetManager:
    """Manager used to create and train new neural networks using CTScanModules"""

    def __init__(self):
        self._ct_scan_list = []
        self._last_used_scan = -1 #K-Fold validation
        self._number_of_folds = 10


    def train_model(self):
        model_name = 'WH_2x2_Heatmap+Undersampling' #Used for statistics
        for current_fold in range(0, self._number_of_folds):
            #Data
            data = self.generate_fset(current_fold, UNDERSAMPLING, HEATMAP)
            
            feature_set_training = data[0][0]
            label_set_training = data[0][1]

            feature_set_validation = data[1][0]
            label_set_validation = data[1][1]

            #DEBUG
            #Checking if the dataset is alright
            #for i in range(0, len(label_set_training)):
            #    if label_set_training[i] == True:
            #        plt.imshow(feature_set_training[i], cmap = 'gray')
            #        plt.show()
            #for i in range(0, len(label_set_training)):
            #    if label_set_training[i] == True:
            #        print(label_set_training[i])
            #        print(label_set_training[i - 1])

            #Training
            #model = NeuralNet.get_neural_net_WH()
            model = NeuralNet.get_neural_net_WH()
            history = model.fit(feature_set_training, label_set_training,
                                batch_size = 16,
                                epochs = 128,
                                verbose = 1,
                                validation_data = (feature_set_validation, label_set_validation))
            
            #Saving TODO
            model.save(MODEL_SAVE_PATH)
            break

            #Saving
            if (TRAINING_HISTORY):
                self.save_training_statistics(TRAINING_HISTORY_VAL_ACC, history.history['val_acc'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_ACC, history.history['acc'], model_name, current_fold)


    def random_sampler(self, feature_set, label_set, mode = 1):
        """Implementation of random over and undersampler"""
        #Modes: 1 - undersampling
        #       2 - oversampling
        #Return (feature_set, label_set)
        positive_indexes = [] #For oversampler
        negative_indexes = [] #For undersampler

        positive = 0
        for i in range(0, len(label_set)):
            if(len(label_set[i]) > 1):
                if (np.argmax(label_set[i]) != 64): #TODO Rework
                    positive += 1
                    positive_indexes.append(i)
            else:
                if (np.max(label_set[i]) == 1.0):
                    positive += 1
                    positive_indexes.append(i)

        if (mode == UNDERSAMPLING): #Random undersmapler implementation
            copies_fset = []
            copies_lset = []
            negative = 0
            while(negative != positive):
                rand_index = random.randint(0, label_set.shape[0])
                if (rand_index not in positive_indexes and rand_index not in negative_indexes):
                    negative_indexes.append(rand_index)
                negative += 1
                print('\r**Undersampling {}/{}'.format(negative, positive), end = '')
                if (negative == positive):
                    print('...DONE')

            for index in positive_indexes:
                copies_fset.append(np.copy(feature_set[index]))
                copies_lset.append(np.copy(label_set[index]))
                   
            for index in negative_indexes:
                copies_fset.append(np.copy(feature_set[index]))
                copies_lset.append(np.copy(label_set[index]))

            copies_fset = np.array(copies_fset)
            copies_fset = np.reshape(copies_fset, (len(copies_fset), feature_set.shape[1], feature_set.shape[2]))

            copies_lset = np.array(copies_lset)

            return ((copies_fset, copies_lset))

        elif (mode == OVERSAMPLING): #Random oversampler implementation
            negative = label_set.shape[0] - positive
            copies_fset = []
            copies_lset = []
            while (positive != negative):
                rand_index = random.randint(0, len(positive_indexes) - 1)
                copies_fset.append(np.copy(feature_set[positive_indexes[rand_index]]))
                copies_lset.append(np.copy(label_set[positive_indexes[rand_index]]))

                positive_indexes.append(len(label_set) - 1)
                positive += 1

                print('\r**Oversampling {}/{}'.format(positive, negative), end = '')
                if (positive == negative): 
                    print('...DONE')


            copies_fset = np.array(copies_fset)
            copies_fset = np.reshape(copies_fset, (len(copies_fset), feature_set.shape[1], feature_set.shape[2]))

            copies_lset = np.array(copies_lset)

            feature_set = np.concatenate((feature_set, copies_fset), axis = 0)
            label_set = np.concatenate((label_set, copies_lset), axis = 0)

            return (feature_set, label_set)

        
    def generate_fset(self, fold_to_use, data_aug_options = 0, label_set_mode = ONE_HOT):
        """Returns a touple with ((training_fset, training_lset), (validation_fset, validation_lset))"""
        #Feature set shape - (len(fset), 512, 512, 1)
        #STEPS
        #1. Generate all 10 fsets
        #2. Keep 9 together and use 1 as validation
        #3. Test on the rest...

        #DATA AUGUMENTATION
        #0 - do nothing
        #1 - undersampling
        #2 - oversampling

        if (data_aug_options < 0 and data_aug_options > 2):
            raise Exception("Invalid data augumentation option!")
        if (label_set_mode < 0 and label_set_mode > 1):
            raise Exception("Invalid label set mode!")

        training_fset = None
        training_lset = None

        validation_fset = None
        validation_lset = None

        print("**Generating validation Feature and Label set")
        validation_fset = self.get_fset_single(fold_to_use)
        validation_lset = self.get_lset_single(fold_to_use, label_set_mode)

        #Generating training and validation sets, splitting folds...
        for i in range(0, len(self._ct_scan_list)):
            if (i != fold_to_use):
                print("**Generating training Feature and Label set for fold: {}/{}".format(i + 1, self._number_of_folds))
                if (training_fset is None):
                    training_fset = self.get_fset_single(i)
                else:
                    training_fset = np.concatenate((training_fset, self.get_fset_single(i)), axis = 0)
                
                if (training_lset is None):
                    if (label_set_mode == ONE_HOT):
                        training_lset = self.get_lset_single(i, label_set_mode)
                    else:
                        training_lset = np.array(self.get_lset_single(i, label_set_mode))
                else:
                    if (label_set_mode == ONE_HOT):
                        training_lset = np.append(training_lset, self.get_lset_single(i, label_set_mode))
                    else:
                        training_lset = np.concatenate((training_lset, np.array(self.get_lset_single(i, label_set_mode))), axis = 0)

        print("**Shapes: {} | {}".format(training_fset.shape, training_lset.shape))
        validation_lset = np.array(validation_lset)

        #Data augumentation
        if (data_aug_options != 0):
            temp_training = None
            temp_validation = None

            if (data_aug_options == UNDERSAMPLING):
                temp_training = self.random_sampler(training_fset, training_lset, UNDERSAMPLING)
                temp_validation = self.random_sampler(validation_fset, validation_lset, UNDERSAMPLING)
            elif (data_aug_options == OVERSAMPLING):
                temp_training = self.random_sampler(training_fset, training_lset, OVERSAMPLING)
                temp_validation = self.random_sampler(validation_fset, validation_lset, OVERSAMPLING)
            
            training_fset = temp_training[0]
            training_lset = temp_training[1]
            
            validation_fset = temp_validation[0]
            validation_lset = temp_validation[1]

        #Reshaping for CNN
        print("**Reshaping validation sets")
        validation_fset = validation_fset.reshape(len(validation_fset), validation_fset.shape[1], validation_fset.shape[2], 1)
        if (label_set_mode == HEATMAP):
            validation_lset = np.reshape(validation_lset, (len(validation_lset), validation_lset.shape[1]))
        print('**Reshaping training sets')
        training_fset = training_fset.reshape(len(training_fset), training_fset.shape[1], training_fset.shape[2], 1)
        if (label_set_mode == HEATMAP):
            training_lset = np.reshape(training_lset, (len(training_lset), training_lset.shape[1]))

        return ((training_fset, training_lset), (validation_fset, validation_lset))


    def save_training_statistics(self, file_path, data_to_save, model_name, model_fold):
        save_file = open(file_path, 'a+')
        save_file.write(model_name + ' fold: {}'.format(model_fold))
        for value in data_to_save:
            save_file.write(';{}'.format(value))
        save_file.write('\n')
        save_file.close()
    
            
    def get_fset_single(self, ct_scan_index):
        if (ct_scan_index >= len(self._ct_scan_list)):
            raise Exception("Scan index out of bounds!")
        
        return self._ct_scan_list[ct_scan_index].ct_scan_array


    def get_lset_single(self, ct_scan_index, lset_mode = 0):
        #Label set is a Numpy array
        #Modes - 0. One hot encoding
        #        1. Heatmaps (1D numpy arrays)
        if (ct_scan_index >= len(self._ct_scan_list)):
            raise Exception("Scan index out of bounds!")
        
        if (lset_mode == ONE_HOT):
            ct_scan_module = self.search_scan_by_id(ct_scan_index)
            slice_count = ct_scan_module.slice_count
            label_set = np.full((slice_count), 0.0) #Fils label set with 0.0 values
        
            #Iterates over all nodule locations and adds 1.0 when z axis is positive
            for loc in ct_scan_module.nodule_location:
                label_set[slice_count - loc[2]] = 1.0

            return label_set
        
        elif (lset_mode == HEATMAP):
            #Heatmaps have an area of 16x16 pixels and each slice has 16x16 heatmaps for a total of 256 h. maps.
            #[!] WARNING: This implementation assumes that the input image is 256x256 px. [!]
            heatmap_size = (8 * 8) + 1 #row * col
            heatmap_width = 8 #values per row
            heatmap_value_size = 32 #16px per one value
            label_set = []
            ct_scan_module = self.search_scan_by_id(ct_scan_index)
            slice_count = ct_scan_module.slice_count
            
            #Generates and fills all heatmaps with 0.0 values
            for i in range(0, slice_count):
                label_set.append(np.full(heatmap_size, 0.0))

            for location in ct_scan_module.nodule_location:
                #Finds the correct heat map on z axis
                heatmap = label_set[slice_count - location[2]]
                #[!] WARNING: nodule_location is in format (y, x, z). [!]
                x = int(location[1] / heatmap_value_size)
                y = int(location[0] / heatmap_value_size)
                self.set_element(heatmap, heatmap_width, x, y, 1.0)

            ##TESTING!!! TODO
            for heatmap in label_set:
                if (np.max(heatmap) == 0.0):
                    heatmap[len(heatmap) - 1] = 1.0

            #DEBUG
            #reshaped_heatmap = np.reshape(heatmap, (16, 16))
            #print(reshaped_heatmap.shape)
            #print(reshaped_heatmap)

            return label_set

    def set_element(self, array, width, row, col, value):
        """Utility function, sets value in 1D using 2D location"""
        array[width * row + col] = value
            
        
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

        #Load scan modules from cache
        if (CT_SCAN_CACHING):
                    #Check if cache folder exists
            if (CT_SCAN_CACHING_PATH not in os.listdir('.') and CT_SCAN_CACHING):
                os.mkdir(CT_SCAN_CACHING_PATH)

            #Loads all scan modules which were previously cached
            cached_scan_files = os.listdir(CT_SCAN_CACHING_PATH)
            cached_scan_names = []
            for filename in cached_scan_files:
                cached_scan_names = filename.split('.')[0]

            #Load from file
            for file in cached_scan_files:
                file = open(os.path.join(CT_SCAN_CACHING_PATH, file), 'rb')
                loaded_scan = pickle.load(file)
                isLoaded = False
                for scan in self._ct_scan_list:
                    if scan.name == loaded_scan.name:
                        isLoaded = True
                if (not isLoaded):
                    self._ct_scan_list.append(loaded_scan)
                    print('Scan "{}" loaded from cache.'.format(loaded_scan.name))
                else:
                    print('Scan "{}" already loaded!'.format(loaded_scan.name))

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
                if (existing_scan.name not in cached_scan_names): #Check if the nodule is not being added for the secound time
                    existing_scan.add_nodule_location(nodule_position)
            else:
                data = self.get_dicom_data(line[0]) #Late data loading
                CT_scan_object = CTScanModule(name, len(self._ct_scan_list), data, nodule_position)             
                CT_scan_object.normalize_data()
                CT_scan_object.transpose_ct_scan()
                self._ct_scan_list.append(CT_scan_object)

                #Save scan modules to cache
                if (CT_SCAN_CACHING):
                    #Save the object ONLY if its not already cached
                    if (CT_scan_object.name not in cached_scan_names):
                        file = open(os.path.join(CT_SCAN_CACHING_PATH,CT_scan_object.name) + '.csm', 'wb')
                        pickle.dump(CT_scan_object, file)
                        print('**Scan "{}" saved to cache.'.format(CT_scan_object.name))


    #Loads DICOM data using MedPy
    def get_dicom_data(self, path_to_data):
        try:
            print("**Loading {}".format(path_to_data))
            dicom_data, dicom_header = medpy.io.load(path_to_data)
        except:
            print("[!] Loading failed, file: {}".format(path_to_data))
        return dicom_data


    def resize_ct_scans(self, new_resolution):
        """Resizes all available CT scan datasets to resolution x resolution px"""
        counter = 1
        for ct_scan in self._ct_scan_list:
            print('**Resizing dataset {}, {}/{}'.format(ct_scan.name, counter, len(self._ct_scan_list)))
            ct_scan.resize_image(new_resolution)
            counter += 1