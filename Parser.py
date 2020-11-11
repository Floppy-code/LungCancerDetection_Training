SOURCE_PATH = 'D:/LungCancerCTScans/SPIE-AAPM Lung CT Challenge/paths.csv'

class Parser:
    """Parser which controls the NNManager, quite rought around the edges for now..."""

    def __init__(self, manager):
        self._manager = manager


    def parse(self):
        while (True):
            self.print_request("")
            keyboard_input = input()

            if (keyboard_input == 'l'):
                self._manager.load_ct_scans(SOURCE_PATH) #TODO
            elif (keyboard_input == 'p'):
                self._manager.print_CT_scan_modules()
            elif (keyboard_input == 's'):
                self.print_request("ID")
                id = input()
                self.print_request('Mode')
                mode = input()
                self.print_request('Slice/Node number')
                slice = input()
                self._manager.show_CT_scan(id, mode, slice)
            elif (keyboard_input == 'e'):
                return

    def print_request(self, request):
        print(request + ">", end = '')