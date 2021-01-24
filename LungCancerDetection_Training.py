import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from NeuralNetManager import NeuralNetManager
from Parser import Parser

def main():
    manager = NeuralNetManager()

    #parser = Parser(manager)
    #parser.parse()

    manager.load_ct_scans('D:/LungCancerCTScans/LUNA16/foldersToLoad.csv')
    manager.train_model()

main()