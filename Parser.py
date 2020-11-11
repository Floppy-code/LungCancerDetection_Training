SOURCE_PATH = 'D:/LungCancerCTScans/SPIE-AAPM Lung CT Challenge'

class Parser:
    """Parser which controls the NNManager"""

    def __init__(self, manager):
        self._manager = manager


    def parse(self):
        while (True):
            print("> ", end='')
            keyboard_input = input()

            if (keyboard_input == 'l'):
                self._manager.load_ct_scans(SOURCE_PATH) #TODO
            elif (keyboard_input == 'p'):
                self._manager.print_CT_scan_modules()

            elif (keyboard_input == 'e'):
                return