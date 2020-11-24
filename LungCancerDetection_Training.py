import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from NeuralNetManager import NeuralNetManager
from Parser import Parser

def main():
    manager = NeuralNetManager()
    #Skip parser for now
    #parser = Parser(manager)
    #parser.parse()

    manager.load_ct_scans('D:/LungCancerCTScans/SPIE-AAPM Lung CT Challenge/paths.csv')
    manager.resize_ct_scans(256)
    manager.train_model()

main()