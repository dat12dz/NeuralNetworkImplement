import NeuralClassT as n
import mathFunction
from NeuralNetworkBase import NeuralNetworkBase
import WeightPool
import NeuralConst
import numpy as np

learningRate = 0.06
class NeuralNetwork(NeuralNetworkBase):
    #init class với mảng độ dài cuiar layer
    def __init__(self,LayerLengthList : list[float],activation):
        self.NeuralNetworkList : list[list[n.Neural]] = []
        # Lặp qua từng mảng layer
        arr0 = [[np.float32(0) for i in range(layerLen)] for layerLen in LayerLengthList]
        WeightPool.finalWeightChange = [[[] for i in range(layerLen)] for layerLen in LayerLengthList]
        WeightPool.finalBiasChange = arr0

        for i in range(0,len(LayerLengthList)):
            # thêm vào mảng neron N phần tử trống
            self.NeuralNetworkList.append([n.Neural(0,0.01,i,j,activation[i]) for j in range(LayerLengthList[i])])
        self.LinkingLayer()
        self.LayerLen = len(self.NeuralNetworkList)
        self.activationArr = activation
 

    def LinkingLayer(self):
    
        # Nếu Không phải layer 0
        for thisLayerID in range(1,len( self.NeuralNetworkList)): 
            prevLayerID  = thisLayerID - 1
            for thisNeural  in self.NeuralNetworkList[thisLayerID]:
                    #Linking Prev Layer với Neuron ở layer hiện tại
                    thisNeural.SetprevNeural([PrevNeuron for PrevNeuron in self.NeuralNetworkList[prevLayerID]])

#                    thisNeural.RandomWeight(4)    
        
    # Init class với mảng bias
    # def __init__(self,NeuralNetworkBiasArrr : list[list[float]]):
    #     self.NeuralNetworkList : list[list[n.Neural]] = [[n.Neural(0,NeuralNetworkBiasArrr[NeuralLayerBiasID][biasID]) 
    #                                                       for biasID in range(len(NeuralNetworkBiasArrr[NeuralLayerBiasID]))] 
    #                                                       for NeuralLayerBiasID in  range(len(NeuralNetworkBiasArrr))]

    #     self.LinkingLayer()
    # Dự đoán : Forwarding
    def Predict(self,inputInt : list[float]):
        # check độ dài của input
        if (len(inputInt) != len(self.NeuralNetworkList[0])):
            raise Exception("méo thể dự đoán vì input layer không đúng kích cỡ input")
        # Gán input
        for NeuralID in range(0,len(inputInt)):
            Neural = self.NeuralNetworkList[0][NeuralID]
            Neural.val = inputInt[NeuralID]

        # Lặp Layer từ 1
        for layerId in range(1,len(self.NeuralNetworkList)):
            #Lặp từng neron
            for Neural in self.NeuralNetworkList[layerId]:
                Neural.CalcThisNeural()
            if (self.activationArr[layerId] == NeuralConst.ACTIVATION_SOFTMAX):
                mathFunction.softMax(self.NeuralNetworkList[layerId])
    # Backpropagation
    def BackPropNeuron(self,PredictDigit):
        n = len(self.NeuralNetworkList)
        m = len(self.NeuralNetworkList[n - 1])
        for lastLayerNeuID in range(0,m):
            neu = self.NeuralNetworkList[n - 1][lastLayerNeuID]
            if (PredictDigit == lastLayerNeuID):
                neu.TraceBackDeltaActivaition(1)
            else:
                neu.TraceBackDeltaActivaition(0)

        for layerID in range(n - 2,0,-1):
            for Neural in self.NeuralNetworkList[layerID]:
                Neural.CalcDeltaValPerDeltaCostAvg()
                Neural.TraceBackDeltaActivaition(-1)
    # Print debug
    def PrintValue(self):
        for LayerID in range(0,len(self.NeuralNetworkList)):
            for NeuralID in range(0,len(self.NeuralNetworkList[LayerID])):
                print(f"Value of Layer {LayerID} and Neural {NeuralID} : {self.NeuralNetworkList[LayerID][NeuralID].val}")
    def PrintWeight(self):
        for LayerID in range(len(self.NeuralNetworkList)):
            for NeuralID in range(len(self.NeuralNetworkList[LayerID])):
                print(f"Weight of Layer {LayerID} and Neural {NeuralID} : {self.NeuralNetworkList[LayerID][NeuralID].wList}")
    def PrintWeightChange(self):
        for LayerID in range(len(self.NeuralNetworkList)):
            for NeuralID in range(len(self.NeuralNetworkList[LayerID])):
                print(f"Weight of Layer {LayerID} and Neural {NeuralID} : {self.NeuralNetworkList[LayerID][NeuralID].wListChange}")
    def PrintActivationChange(self):
        for LayerID in range(len(self.NeuralNetworkList)):
            for NeuralID in range(len(self.NeuralNetworkList[LayerID])):
                print(f"Weight of Layer {LayerID} and Neural {NeuralID} : {self.NeuralNetworkList[LayerID][NeuralID].DeltaValPerDeltaCost}")
    # Train data nhờ việc predict xong Backprop sử dụng kết quả của predict
    def train(self,dataset):
        n = len(dataset)
        if (n == 0):
           return {'b' : [] , 'w' : []}
        totalError = 0.0

        for dataSample in dataset:
            self.Predict(dataSample.img)
            error =  mathFunction.Error(dataSample.lable,self.NeuralNetworkList[self.LayerLen - 1])
            totalError += error
            self.BackPropNeuron(dataSample.lable)
        # in ra lỗi sau khi Backprop xong
        print(f"Total error: {totalError / n}")
        # tính tổng 
        for layerID in range(1,len(self.NeuralNetworkList)):
            for Neuron in self.NeuralNetworkList[layerID]:
                Neuron.WeightAvg()
        if (WeightPool.UsingMutilProcess):
            return { "b" :  WeightPool.finalBiasChange , "w" : WeightPool.finalWeightChange }
        else:
            return {}

    def CalcFinalBiasAndApply(self,allBiasChange,allWeightChange):
        for k in range(len(allBiasChange)):
            if (len(allBiasChange[k]) != 0):
                for i in range(0,self.LayerLen):
                    for j in range(0,len(self.NeuralNetworkList[i])):
                        thisNeural = self.NeuralNetworkList[i][j]
                        thisNeural.bias -= allBiasChange[k][i][j]
                        for LayerWeight in range(len(thisNeural.wList)):
                            thisNeural.wList[LayerWeight] -= allWeightChange[k][i][j][LayerWeight]

    
    def GetPredictResult(self):
        max = 0.0
        maxi = 0
        for i in range(0,len(self.NeuralNetworkList[self.LayerLen - 1])):
            neural = self.NeuralNetworkList[self.LayerLen - 1][i]
            if (neural.val > max):
                max = neural.val
                maxi = i
        return maxi      
    #backProp thay đổi tham số sau khi backprop đến đâu
    