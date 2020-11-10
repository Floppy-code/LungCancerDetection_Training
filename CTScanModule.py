class CTScanModule:
    """CTScanModule holds all information related to a CT scan of one patient."""

    def __init__(self, name, scan_array, nodule_location):
        self._name = name
        self._ct_scan_array = scan_array
        self._nodule_location = nodule_location

    @property
    def name(self):
        return self._name

    @property
    def ct_scan_array(self):
        return self._ct_scan_array