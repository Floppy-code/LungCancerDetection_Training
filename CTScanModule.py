import numpy as np
import matplotlib.pyplot as plt

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


    def add_nodule_location(self, location):
        self._nodule_count += 1
        self._nodule_location.append(location)


    def show_nodule(self, nodule_id = 0):
        z_location = self._slice_count - self._nodule_location[nodule_id]
        x_location = self._nodule_location[nodule_id][1]
        y_location = self._nodule_location[nodule_id][2]
        
        slice = np.copy(self._ct_scan_array[z_location])

        #Draw square around the nodule
        offset = 15 #px
        for x in range(x_location - offset, x_location + offset):
            for y in range(y_location - offset, y_location + offset):
                if (x == x_location - offset or x == x_location + offset - 1):
                    slice[x][y] = 1.0
                elif (y == y_location - offset or y == y_location + offset - 1):
                    slice[x][y] = 1.0

        #Show the edited slice
        size = 10
        plt.figure(figsize = size)
        plt.imshow(slice, cmap = 'grey')


    #Transposes the np array to (z, x, y):
    def transpose_ct_scan(self):
        self._ct_scan_array = np.transpose(self._ct_scan_array, (2, 1, 0))


    #By default DICOM images have values ranging from -2000 to 3000
    #Normalization for NN is needed
    def normalize_data(self):
        min_val = np.min(self._ct_scan_array)
        self._ct_scan_array = self._ct_scan_array + (min_val * (-1))
        max_val = np.max(self._ct_scan_array)
        self._ct_scan_array = self._ct_scan_array / max_val