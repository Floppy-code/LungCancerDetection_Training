import numpy as np

class CTScanModule:
    """CTScanModule holds all information related to a CT scan of one patient."""

    def __init__(self, name, id, scan_array, nodule_location):
        self._name = name
        self._id = id
        self._ct_scan_array = scan_array
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


    def add_nodule_location(self, location):
        self._nodule_count += 1
        self._nodule_location.append(location)


    #By default DICOM images have values ranging from -2000 to 3000
    #Normalization for NN is needed
    def normalize_data(self):
        min_val = np.min(self._ct_scan_array)
        self._ct_scan_array = self._ct_scan_array + (min_val * (-1))
        max_val = np.max(self._ct_scan_array)
        self._ct_scan_array = self._ct_scan_array / max_val