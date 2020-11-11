import os

from NeuralNetManager import NeuralNetManager

DATASET_FILE = 'D:\LungCancerCTScans\SPIE-AAPM Lung CT Challenge\paths.csv'

def main():
    manager = NeuralNetManager()
    manager.load_ct_scans(DATASET_FILE)


main()