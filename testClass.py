import NeuralNetwork as nn
import ImageUtils
import NeuralConst
import FileOP
from multiprocessing import Pool
core = 12
Nn = nn.NeuralNetwork([32*32,16,8,10],
                      [NeuralConst.ACTIVATION_RELU,
                       NeuralConst.ACTIVATION_RELU,
                       NeuralConst.ACTIVATION_RELU,
                       NeuralConst.ACTIVATION_SOFTMAX]) 
dataset = ImageUtils.load_handwritten_digits_dataset("./dataset")
dataset = ImageUtils.flatten_dataset(dataset)
dataset = ImageUtils.DatasetToSuffledArray(dataset)
#dataset = ImageUtils.divBatch(core,dataset)

# je ai
# tu as
# il a
# nous avons
# vous avez
# vous sont
# Nn = FileOP.LoadModel()
# print("Start traning")
# for i in range(0,1000):
#     try:
#         Nn.train(dataset)
#     except KeyboardInterrupt:
#         print("\nsaving model")
#         FileOP.SaveModel(Nn)
#         print("exit")
#         exit()
# FileOP.SaveModel(Nn)
    


#Predict

Nn  = FileOP.LoadModel()
arr = ImageUtils.OpenIamgeAsFlatten("testcase/DOAN.jpg")
print(len(arr))
Nn.Predict(arr)
Nn.PrintValue()

print(f"Hallo {Nn.GetPredictResult( )}")