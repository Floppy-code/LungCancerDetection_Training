class CTScanModule:
    """CTScanModule holds all information related to a CT scan of one patient."""

    def __init__(self, name, id, scan_array, nodule_location):
        self._name = name
        self._id = id
        self._ct_scan_array = scan_array
        self._nodule_location = nodule_location #(x, y ,z)


    @property
    def name(self):
        return self._name


    @property
    def ct_scan_array(self):
        return self._ct_scan_array


    @property
    def id(self):
        return self._id