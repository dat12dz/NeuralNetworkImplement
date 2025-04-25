from abc import ABC,abstractmethod

class NeuralNetworkBase(ABC):

    
    #init class với mảng độ dài cuiar layer
    @abstractmethod
    def __init__(self,LayerLengthList : list[float],activation):
        pass
    @abstractmethod
    def LinkingLayer(self):
        pass
    @abstractmethod
    def Predict(self,inputInt : list[float]):
        pass
    @abstractmethod
    def BackPropNeuron(self,PredictDigit):
        pass
    @abstractmethod
    def PrintValue(self):
        pass
    @abstractmethod
    def PrintWeight(self):
        pass
    @abstractmethod
    def PrintWeightChange(self):
        pass
    @abstractmethod
    def PrintActivationChange(self):
        pass
    @abstractmethod
    def train(self,dataset):
        pass
    @abstractmethod
    def GetPredictResult(self):
        pass
    @abstractmethod
    def CalcFinalBiasAndApply():
        pass