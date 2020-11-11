class CTScanModule:
    """CTScanModule holds all information related to a CT scan of one patient."""

    def __init__(self, name, id, scan_array, nodule_location):
        self._name = name
        self._id = id
        self._ct_scan_array = scan_array
        self._nodule_location = [nodule_location] #List with touples of locations (x, y ,z)
        self._nodule_count = len(self._nodule_location)


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


    def add_nodule_location(self, location):
        self._nodule_count += 1
        self._nodule_location.append(location)