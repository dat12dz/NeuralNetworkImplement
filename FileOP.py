import pickle

import NeuralNetworkBase
#i  mport NeuralNetwork as nn
def SaveModel(Nn):
    modelFile = open('Model.vuaIciTea','wb')
    pickle.dump(Nn,modelFile)
    modelFile.close()
def LoadModel() -> NeuralNetworkBase.NeuralNetworkBase:
    modelFile = open('Model.vuaIciTea','rb')
    nn =  pickle.load(modelFile)
    modelFile.close()
    return nn