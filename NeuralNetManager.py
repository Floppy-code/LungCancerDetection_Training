#3rd party imports
import medpy
import medpy.io
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import math
import gc
from scipy import ndimage

from keras.callbacks import EarlyStopping

#My imports
from CTScanModule import CTScanModule
import NeuralNet

#=============== GLOBAL VARIABLES ===============
#Saves CTScanModule objects to disk to improve load times / data loading
CT_SCAN_CACHING = True  
CT_SCAN_CACHING_PATH = '.scan_cache'
RESIZE_ON_LOAD = True
RESIZE_DEFAULT_VAL = 224

#Saves training stats after every epoch into file
TRAINING_HISTORY = True  
TRAINING_HISTORY_VAL_ACC = 'val_acc_training.txt'
TRAINING_HISTORY_ACC = 'acc_training.txt'
TRAINING_HISTORY_VAL_LOSS = 'val_loss_training.txt'
TRAINING_HISTORY_LOSS = 'loss_training.txt'

#Label set implementation
ONE_HOT = 0
HEATMAP = 1

#Data augumentation
NONE = 0
UNDERSAMPLING = 1
OVERSAMPLING = 2

#CNN Model
SAVE_MODELS = True
MODEL_SAVE_PATH = '.old_models'

#Annotations file path
ANOTATIONS_PATH = 'D:/LungCancerCTScans/LUNA16/annotations.csv'


class NeuralNetManager:
    """Manager used to create and train new neural networks using CTScanModules"""

    def __init__(self):
        self._ct_scan_list = []
        self._last_used_scan = -1 #K-Fold validation
        self._number_of_folds = 5


    def train_model(self):
        model_name = 'LUNA16_NonAugumentedDataset_VGG16' #Used for statistics
        for current_fold in range(0, self._number_of_folds):
            #Data
            data = self.generate_fset(current_fold, OVERSAMPLING, ONE_HOT)
            
            feature_set_training = data[0][0]
            label_set_training = data[0][1]

            feature_set_validation = data[1][0]
            label_set_validation = data[1][1]

            unique, counts = np.unique(label_set_training, return_counts=True)
            print(dict(zip(unique, counts)))

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
            model = NeuralNet.get_neural_net_VGG16()
            #model = NeuralNet.get_neural_net_VGG19()
            history = model.fit(feature_set_training, label_set_training,
                                batch_size = 32,
                                epochs = 15,
                                verbose = 1,
                                validation_data = (feature_set_validation, label_set_validation),
                                shuffle = True)
            
            if (SAVE_MODELS and MODEL_SAVE_PATH not in os.listdir('.')):
                os.mkdir(MODEL_SAVE_PATH)
                model.save(os.path.join(MODEL_SAVE_PATH, "{}_fold{}.h5".format(model_name, current_fold)))
            elif (SAVE_MODELS):
                model.save(os.path.join(MODEL_SAVE_PATH, "{}_fold{}.h5".format(model_name, current_fold)))

            #Saving
            if (TRAINING_HISTORY):
                self.save_training_statistics(TRAINING_HISTORY_VAL_ACC, history.history['val_acc'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_ACC, history.history['acc'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_VAL_LOSS, history.history['val_loss'], model_name, current_fold)
                self.save_training_statistics(TRAINING_HISTORY_LOSS, history.history['loss'], model_name, current_fold)


    def random_sampler(self, feature_set, label_set, mode = 1):
        """Implementation of random over and undersampler"""
        #Modes: 1 - undersampling
        #       2 - oversampling
        #Return (feature_set, label_set)
        positive_indexes = [] #For oversampler
        negative_indexes = [] #For undersampler

        positive = 0
        #[!] WARNING: Random sampler for heatmaps not implemented yet!
        for i in range(0, len(label_set)):
            if (label_set[i] == 1.0):
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

            return (copies_fset, copies_lset)

        elif (mode == OVERSAMPLING): #Random oversampler implementation
            negative = label_set.shape[0] - positive
            copies_fset = []
            copies_lset = []
            while (positive != negative):
                rand_index = random.randint(0, len(positive_indexes) - 1)

                copies_fset.append(np.copy(feature_set[positive_indexes[rand_index]]))
                copies_lset.append(np.copy(label_set[positive_indexes[rand_index]]))

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


    def generate_fset(self, fold_to_use, data_aug_options = NONE, label_set_mode = ONE_HOT):
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
        training_lset = np.array([])

        validation_fset = None
        validation_lset = np.array([])

        print("**Generating validation Feature and Label set")
        #How many sets will be in a fold
        dataset_count_to_use = math.ceil(len(self._ct_scan_list) / self._number_of_folds) 
        
        #Select indexes supposed to be in training and validation folds
        validation_scans = [x for x in range(fold_to_use * dataset_count_to_use, fold_to_use * dataset_count_to_use + dataset_count_to_use)]
        print(validation_scans)

        #Iterate over all sets and add them to corresponding sets
        for i in range(0, len(self._ct_scan_list)):
            print('\r**Processing dataset: {}/{}'.format(i + 1, len(self._ct_scan_list)), end = '')
            if (i not in validation_scans):
                if (training_fset is None): 
                    training_fset = self.get_fset_single(i)
                else:
                    training_fset = np.concatenate((training_fset, self.get_fset_single(i)), 0)
                training_lset = np.concatenate((training_lset, self.get_lset_single(i)))
            else:
                if (validation_fset is None):
                    validation_fset = self.get_fset_single(i)
                else:
                    validation_fset = np.concatenate((validation_fset, self.get_fset_single(i)), 0)
                validation_lset = np.concatenate((validation_lset, self.get_lset_single(i)))
        print('')


        #Data augumentation - Random sampling
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

            #[!] MANUAL Garbage Collection!
            gc.collect()

        #Data augumentation - Random image rotation
        random_rotation = True
        if (random_rotation):
            #Iterates over all images in feature sets and rotates them by random degree
            print_counter = 0
            print("**Rotating training feature set.")
            for train_img in training_fset:
                print("\r**Rotating image: {}/{}".format(print_counter + 1, training_fset.shape[0]), end = '')
                if (print_counter + 1 == training_fset.shape[0]):
                    print("...DONE")
                random_angle = random.randint(0, 360)
                train_img = ndimage.rotate(train_img, random_angle, order = 0)
                print_counter += 1

            print_counter = 0
            print("**Rotating validation feature set.")
            for validate_img in validation_fset:
                print("\r**Rotating image: {}/{}".format(print_counter + 1, validation_fset.shape[0]), end = '')
                if (print_counter + 1 == validation_fset.shape[0]):
                    print("...DONE")
                random_angle = random.randint(0, 360)
                validate_img = ndimage.rotate(validate_img, random_angle, order = 0)
                print_counter += 1

        #Reshaping for CNN
        print("**Reshaping validation sets")
        validation_fset = validation_fset.reshape(len(validation_fset), validation_fset.shape[1], validation_fset.shape[2], 1)
        print('**Reshaping training sets')
        training_fset = training_fset.reshape(len(training_fset), training_fset.shape[1], training_fset.shape[2], 1)
        print("**Shapes: Training {} {} \n          Validation: {} {}".format(training_fset.shape, training_lset.shape, validation_fset.shape, validation_lset.shape))

        return ((training_fset, training_lset), (validation_fset, validation_lset))


    def save_training_statistics(self, file_path, data_to_save, model_name, model_fold):
        save_file = open(file_path, 'a+')
        save_file.write(model_name + ' fold_{}'.format(model_fold))
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
            label_set = np.full(slice_count, 0.0) #Fils label set with 0.0 values
        
            #Iterates over all nodule locations and adds 1.0 when z axis is positive
            for loc in ct_scan_module.nodule_location:
                label_set[loc[2] - 1] = 1.0
                label_set[loc[2]] = 1.0
                label_set[loc[2] + 1] = 1.0

            return label_set
        
        
        elif (lset_mode == HEATMAP):
            raise Exception("Heatmaps not implemented!")
            
        
    def print_CT_scan_modules(self):
        for scan in self._ct_scan_list:
            print("{}. {} Nodes: {}".format(scan.id, scan.name, scan.nodule_count))


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


    def load_nodes_single(self, path_file, scan_instance):
        '''Iterates over list of anotations and appends them to the corresponding CTScanModule instance'''
        #Filter out datasets that are not loaded
        if path_file == None:
            raise Exception("Path file not specified!")
        if scan_instance == None:
            raise Exception("Nullpointer to scan instance!")
        
        annotations_file = open(path_file, 'r')
        for line in annotations_file.readlines():
            line = line.split(',')
            if (line[0] == scan_instance.name):
                nodule_location = (float(line[1]), float(line[2]), float(line[3]))
                scan_instance.add_nodule_location(nodule_location, 1)
                

    def load_ct_scans(self, path_file):
        '''Loads ct scans as 3D numpy array and saves them as an CTScanModule instance'''
        #FORMAT
        #Path, name, x, y, z
        if path_file == None:
            raise Exception("Path file not specified!")

        #Load scan modules from cache
        cached_scan_names = []
        if (CT_SCAN_CACHING):
                    #Check if cache folder exists
            if (CT_SCAN_CACHING_PATH not in os.listdir('.') and CT_SCAN_CACHING):
                os.mkdir(CT_SCAN_CACHING_PATH)

            #Loads all scan modules which were previously cached
            cached_scan_files = os.listdir(CT_SCAN_CACHING_PATH)
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
                    print('**Scan "{}" loaded from cache.'.format(loaded_scan.name))
                else:
                    print('Scan "{}" already loaded!'.format(loaded_scan.name))
                file.close()

        file = open(path_file, 'r') #Reading the .csv file
        for line in file.readlines():
            #Find all .mhd paths from the subfolder
            line = line.replace('\n', '')
            files_in_folder = os.listdir(line)
            files_in_folder = [filename for filename in files_in_folder if '.mhd' in filename]

            #Load .mhd files from folder
            for file in files_in_folder:
                name = file.split(".mhd")[0]
                #Checking if the scan module isnt already loaded
                if (name not in [scan.name for scan in self._ct_scan_list]):
                    dicom_data, dicom_header = self.get_dicom_data(os.path.join(line, file))
                
                    #Create new CTScanModule instance for data
                    scan_instance = CTScanModule(name, len(self._ct_scan_list), dicom_data, dicom_header)
                    scan_instance.normalize_data()
                    scan_instance.transpose_ct_scan()
                    self.load_nodes_single(ANOTATIONS_PATH, scan_instance)
                    if (RESIZE_ON_LOAD):
                        scan_instance.resize_images(RESIZE_DEFAULT_VAL)
                    self._ct_scan_list.append(scan_instance)

                    #Save scan modules to cache
                    if (CT_SCAN_CACHING):
                        #Save the object ONLY if its not already cached
                        if (scan_instance.name not in cached_scan_names):
                            file = open(os.path.join(CT_SCAN_CACHING_PATH,scan_instance.name) + '.csm', 'wb')
                            pickle.dump(scan_instance, file)
                            print('**Scan "{}" saved to cache.'.format(scan_instance.name))
                            file.close()


    #Loads DICOM data using MedPy
    def get_dicom_data(self, path_to_data):
        try:
            print("**Loading {}".format(path_to_data))
            dicom_data, dicom_header = medpy.io.load(path_to_data)
        except:
            print("[!] Loading failed, file: {}".format(path_to_data))
        return (dicom_data, dicom_header)


    def resize_ct_scans(self, new_resolution):
        """Resizes all available CT scan datasets to resolution x resolution px"""
        counter = 1
        for ct_scan in self._ct_scan_list:
            print('**Resizing dataset {}, {}/{}'.format(ct_scan.name, counter, len(self._ct_scan_list)))
            ct_scan.resize_images(new_resolution)
            counter += 1