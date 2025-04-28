import NeuralNetwork as nn
import ImageUtils
import NeuralConst
import FileOP
from multiprocessing import Pool
import numpy as np

core = 12
Nn = nn.NeuralNetwork([32*32,16,10],
                      [NeuralConst.ACTIVATION_RELU,
                       NeuralConst.ACTIVATION_RELU,
                       NeuralConst.ACTIVATION_SOFTMAX]) 
datasetAll = ImageUtils.load_handwritten_digits_dataset("./datasetNew")
dataset = datasetAll[0] # get train dataset
datasetTest = datasetAll[1]
#flatten and suffel datasetTrain
# dataset = ImageUtils.flatten_dataset(dataset)
# dataset = ImageUtils.DatasetToSuffledArray(dataset)
# dataset = ImageUtils.divBatch(core,dataset)

# #Nn = FileOP.LoadModel()
# print("Start traning")
# for i in range(0,1000):
#     try:
#         results = Pool(processes=core).map(Nn.train, dataset) 
#         biasset = [i['b'] for i in results]
#         weightSet = [i['w'] for i in results]
#         Nn.CalcFinalBiasAndApply(biasset,weightSet)
        
#     except KeyboardInterrupt:
#         print("\nsaving model")
#         FileOP.SaveModel(Nn)
#         print("exit")
#         exit()
# FileOP.SaveModel(Nn)

# print()
confusionMatrix =  np.array([[0 for i in range(0,10)] for i in range(0,10)])
# Predict
# Ealuate
datasetTest = ImageUtils.flatten_dataset(datasetTest)
Nn  = FileOP.LoadModel()
allcorrect = 0
all = 0
for testSo in range(0,10):
    correct = 0
    for arr in datasetTest[testSo]:
        Nn.Predict(arr)
        ans =  Nn.GetPredictResult( )
        confusionMatrix[ans][testSo] += 1
        #print(f"Hallo {ans}")
        if (ans == testSo):
            correct+= 1
            allcorrect += 1
        all+=1
    print(f"Tỉ lệ chính xác của số {testSo} là : {correct / len(datasetTest[testSo]) * 100}%" )
print(f"Tong ti le {allcorrect / all * 100} % ")
print("Confusion matrix:")
print(confusionMatrix)
print(all)

#Predict an img

Nn = FileOP.LoadModel()
img = ImageUtils.OpenIamgeAsFlatten('testcase/DOAN.jpg')
Nn.Predict(img)
print(f'ket qua : {Nn.GetPredictResult()}')