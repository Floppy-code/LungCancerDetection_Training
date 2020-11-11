import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from NeuralNetManager import NeuralNetManager
from Parser import Parser
import NeuralNet

def main():
    manager = NeuralNetManager()
    parser = Parser(manager)
    parser.parse()
    NeuralNet.get_neural_net_WH()

main()