import os

from NeuralNetManager import NeuralNetManager
from Parser import Parser

DATASET_FILE = 'D:\LungCancerCTScans\SPIE-AAPM Lung CT Challenge\paths.csv'

def main():
    manager = NeuralNetManager()
    parser = Parser(manager)
    parser.parse()
    

main()