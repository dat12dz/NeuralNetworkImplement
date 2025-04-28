import numpy as np
import mathFunction 
import random
import math
import NeuralNetwork
import NeuralConst
import WeightPool
class Neural:
    
    def __init__(self,val = 0.0,b = 0.0,layer = 0,id = 0,activation = 0):
        self.val = val 
        self.prevNeural : list[Neural] = []
        self.wList : list[float] = []
        self.wListChange : list[float]  = []
        self.wListChangeCount : list[int] = []
        self.bias = b
        self.biasChange = 0.0
        self.biasChangeCount = 0.0
        self.z = 0.0
        self.DeltaValPerDeltaCost = 0.0
        self.DeltavalChangeCount = 0.0
        self.layer = layer
        self.id = id
        self.ActivationFunction = activation
        self.ExpectedVal = 0
    def getPrevActivation(self):
        res = []
        for i in  self.prevNeural:
            res.append(i.val)
        return np.array(res)
    
    def setNeuralWeight(self):
        res = []
        for i in  self.prevNeural:
            res.append(i.val)
        return np.array(res)
    def SetprevNeural(self,arr):
        self.prevNeural = arr
        if (self.layer != 0):
            self.wList = [np.random.normal(0, np.sqrt(2 / len(arr)))  for i in arr]
        arr0 = [0.0 for i in arr]
        self.wListChange = np.array(arr0)
        self.wListChangeCount =  arr0
        WeightPool.finalWeightChange[self.layer][self.id] = arr0
       
    def CalcThisNeural(self):
        self.z = np.dot(self.getPrevActivation(),self.wList) + self.bias
        self.val = mathFunction.TranslateActivationFunction(self.z,self.ActivationFunction)
        #print(f"layer {self.layer} id {self.id}  : {self.val}")

    def DeltaActivationPerDeltaZ(self):
        if (self.DeltaValPerDeltaCost  == 0):
            print("DeltaValPerDeltaCost havn't init yet Set it = 0.001")
            self.DeltaValPerDeltaCost = 0.01
        deltaSigmoidPerDeltaW = mathFunction.TranslateDeltaActivationFunction(self.z,self.ActivationFunction,self.ExpectedVal,self.val)
        return self.DeltaValPerDeltaCost * deltaSigmoidPerDeltaW

    def DeltaCostPerDeltaZ(self,ExpectedVal):          
        deltaActivationPerDeltaW = mathFunction.TranslateDeltaActivationFunction(self.z,self.ActivationFunction,self.ExpectedVal,self.val)
        DeltaCostPerDeltaSigmod = 1
        if (self.ActivationFunction != NeuralConst.ACTIVATION_SOFTMAX):
            DeltaCostPerDeltaSigmod = 2 * (self.val - ExpectedVal)
        return deltaActivationPerDeltaW * DeltaCostPerDeltaSigmod
    
    def AddDetltavalPerDeltaCost(self,x):
        self.DeltaValPerDeltaCost += x

    def CalcDeltaValPerDeltaCostAvg(self):
        #self.DeltaValPerDeltaCost /= self.DeltavalChangeCount
        self.DeltavalChangeCount = 0
        
    
    #Add bias change
    def AddBiasChange(self,x):
            self.biasChange += x
            self.biasChangeCount+= 1
    def AddWeightChange(self,index,x):
            self.wListChange[index] += x
            self.wListChangeCount[index] += 1
    
    # Average
    def WeightAvg(self):


        if (WeightPool.UsingMutilProcess == True):
            for weightChangeiD in range(len(self.wListChange)):
                WeightPool.finalWeightChange[self.layer][self.id][weightChangeiD] = NeuralNetwork.learningRate  * self.wListChange[weightChangeiD] / self.wListChangeCount[weightChangeiD]
            WeightPool.finalBiasChange[self.layer][self.id] = NeuralNetwork.learningRate * self.biasChange / self.biasChangeCount 
            return
        #weight avg
        for weightChangeiD in range(len(self.wListChange)):
           self.wList[weightChangeiD] -= NeuralNetwork.learningRate  * self.wListChange[weightChangeiD] / self.wListChangeCount[weightChangeiD]
           self.wListChangeCount[weightChangeiD] = 0
           self.wListChange[weightChangeiD] = 0
        #bias change
        if (self.biasChangeCount == 0):
            raise Exception(f"Layer {self.layer} id {self.id} biasChangeCount == 0 cannot divide")
        self.bias -= NeuralNetwork.learningRate * self.biasChange / self.biasChangeCount 
        self.biasChangeCount = 0
        self.biasChange = 0

    def TraceBackDeltaActivaition(self,ExpectedVal = -1):
        self.ExpectedVal = ExpectedVal
        biasChange = 0.0
        if (ExpectedVal != -1):
            biasChange = self.DeltaCostPerDeltaZ(ExpectedVal)
        else:
            biasChange = self.DeltaActivationPerDeltaZ()
        for i in range(len(self.wList)): 
            #if the last layer calc cost from delta c /  delta z
            self.prevNeural[i].AddDetltavalPerDeltaCost(self.wList[i] *  biasChange) 
            self.AddWeightChange(i,biasChange * self.prevNeural[i].val)
        self.AddBiasChange(biasChange)
        self.DeltaValPerDeltaCost = 0
