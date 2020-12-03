import numpy as np
import matplotlib.pyplot as plt
import cv2 

class CTScanModule:
    """CTScanModule holds all information related to a CT scan of one patient."""

    def __init__(self, name, id, scan_array, nodule_location):
        self._name = name
        self._id = id
        self._ct_scan_array = scan_array
        self._slice_count = self._ct_scan_array.shape[2]
        self._nodule_location = [nodule_location] #List with touples of locations (x, y ,z)
        self._nodule_count = len(self._nodule_location)
        self._is_normalized = False
        self._is_transposed = False
        self._resolution = (self._ct_scan_array.shape[0], self._ct_scan_array.shape[1]) #Image resolution e.g. (512,512)

    @property
    def name(self):
        return self._name

    @property
    def ct_scan_array(self):
        return self._ct_scan_array

    @property
    def id(self):
        return self._id

    @property
    def nodule_count(self):
        return self._nodule_count

    @property
    def is_normalized(self):
        return self._is_normalized

    @property
    def slice_count(self):
        return self._slice_count

    @property
    def nodule_location(self):
        return self._nodule_location

    @property
    def resolution(self):
        return self._resolution

    @property
    def is_transposed(self):
        return self._is_transposed


    def add_nodule_location(self, location):
        self._nodule_count += 1
        self._nodule_location.append(location)


    #Modes:
    #s - show slice using slice number
    #n - show nodule using id
    def show_slice(self, mode = 's', slice_no = 0):
        size = 7 #Used for figure size
        
        if (mode == 's'):
            z_location = self._slice_count - slice_no
            slice = np.copy(self._ct_scan_array[z_location])

            plt.figure(figsize = (size, size))
            plt.imshow(slice, cmap = 'grey')
            plt.show()
        
        elif (mode == 'n'):
            z_location = self._slice_count - self._nodule_location[slice_no][2] #Slice with nodule is SliceCount - NoduleLoc
            x_location = self._nodule_location[slice_no][1]
            y_location = self._nodule_location[slice_no][0]
        
            slice = np.copy(self._ct_scan_array[z_location])

            #Draw square around the nodule
            offset = 8 #px
            for x in range(x_location - offset, x_location + offset):
                for y in range(y_location - offset, y_location + offset):
                    if (x == x_location - offset or x == x_location + offset - 1):
                        slice[x][y] = 1.0
                    elif (y == y_location - offset or y == y_location + offset - 1):
                        slice[x][y] = 1.0

            #Show the edited slice
            plt.figure(figsize = (size, size))
            plt.imshow(slice, cmap = 'gray')
            plt.show()
        else:
            raise Exception("Invalid mode!")


    #Transposes the np array to (z, x, y):
    def transpose_ct_scan(self):
        print("**Transposing {}".format(self._name))
        
        self._ct_scan_array = np.transpose(self._ct_scan_array, (2, 1, 0))
        self._is_transposed = True


    #By default DICOM images have values ranging from -2000 to 3000
    #Normalization for NN is needed
    def normalize_data(self):
        print("**Normalizing {}".format(self._name))

        min_val = np.min(self._ct_scan_array)
        self._ct_scan_array = self._ct_scan_array + (min_val * (-1))
        max_val = np.max(self._ct_scan_array)
        self._ct_scan_array = self._ct_scan_array / max_val

        self._ct_scan_array = self._ct_scan_array.astype('float32', copy = False)
        self._is_normalized = True


    def resize_image(self, resolution):
        #Steps:
        #1. Calculate the scale for resizing
        #2. Resize the image
        #3. Multiply node locations by scale to make them accurate again
        #4. Update image resolution variable
        if self._resolution[0] != self._resolution[1]:
            raise Exception("Variable axis resolution not implemented!")
        if not self._is_transposed:
            raise Exception("CT scans are not transposed yet!")
        
        scale = resolution / self._resolution[0]
        self._resolution = (resolution, resolution)
        resized_array = []

        for i in range(0, self._ct_scan_array.shape[0]):
            try:
                resized_array.append(cv2.resize(self._ct_scan_array[i], self._resolution))
            except:
                print("[!]There was an error trying to resize image data!")

        self._ct_scan_array = np.array(resized_array)
        for i in range(0, self._nodule_count):
            x = self._nodule_location[i][0] * scale
            y = self._nodule_location[i][1] * scale
            self.nodule_location[i] = (int(x), int(y), self._nodule_location[i][2])
        
        print('**Images in dataset "{}" resized!'.format(self._name))