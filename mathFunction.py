import math
from typing import TYPE_CHECKING
import NeuralConst
import numpy
# tính hàm sigmoid
def sigmoid(x):
    return 1 /(1 + math.exp(-x))
# tính hàm relu
def ReLU(x):
    return max(0,x)
# tính hàm softmax
def softMax(LayerNeural):
    max_val = max(neural.val for neural in LayerNeural)
    exps = [numpy.exp(neural.val - max_val) for neural in LayerNeural]
    sum_exps = sum(exps)
    for i, exp_val in enumerate(exps):
        LayerNeural[i].val = exp_val / sum_exps
# tính đạo hàm sigmoid
def DetlaSigmod(x):
    expPowerMinusX = math.exp(-x)
    return expPowerMinusX / math.pow(1 + expPowerMinusX,2)
#tính đạo hàm relu
def deltaRelu(x):
    if (x < 0):
        return 0
    if (x > 0):
        return 1
    else:
        raise Exception("Không tìm thấy đạo hàm relu của x = 0")
# tính đạo hàm cho cross entropy - trường hợp đặc biệt của softmax
def deltaSoftMax(expectedVal,val):
    return val - expectedVal
# Dịch hàm activation tương ứng với mỗi neuron
def TranslateActivationFunction(x,act_const):
    if (act_const == NeuralConst. ACTIVATION_RELU):
        return ReLU(x)
    if (act_const == NeuralConst. ACTIVATION_SIGMOID):
        return sigmoid(x)
    if (act_const == NeuralConst. ACTIVATION_SOFTMAX):
        return x
#Dịch đạo hàm của activation tương ứng với mỗi neuron
def TranslateDeltaActivationFunction(x,act_const,expectedVal,val):
    if (act_const == NeuralConst. ACTIVATION_RELU):
        return deltaRelu(x)
    if (act_const == NeuralConst.ACTIVATION_SIGMOID):
        return DetlaSigmod(x)
    if (act_const == NeuralConst.ACTIVATION_SOFTMAX):
        return deltaSoftMax(expectedVal,val)
# Tính lỗi theo Cross-Entropy loss
def Error(ExpectedLayer, Lastlayer):
    a = numpy.zeros(len(Lastlayer))
    a[ExpectedLayer] = 1.0
    b = numpy.array([i.val for i in Lastlayer])
    return -numpy.sum(a * numpy.log(b + 1e-9))  #