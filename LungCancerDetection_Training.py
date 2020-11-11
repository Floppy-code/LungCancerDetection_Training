import os

from NeuralNetManager import NeuralNetManager
from Parser import Parser

def main():
    manager = NeuralNetManager()
    parser = Parser(manager)
    parser.parse()
    

main()