# 模拟单个神经元
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 神经元都用类来做，因为有一些内置属性和内置方法
class Neuron:
    """
    权值和偏置是神经元的属性
    """
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self,inputs):
        total = np.dot(self.weights,inputs)+self.bias
        return sigmoid(total)



# 一个逻辑与的神经元
def AND(a,b):
    weights = np.array([20,20])
    bias = -30
    n = Neuron(weights,bias)
    x = np.array([a,b]) 
    ans = n.feedforward(x)
    #print(ans)
    return ans

# 一个逻辑或的神经元
def OR(a,b):
    weights = np.array([30,30])
    bias = -20
    n = Neuron(weights,bias)
    x = np.array([a,b]) 
    ans = n.feedforward(x)
    #print(ans)
    return ans

# 一个逻辑非的神经元
def NOT(a):
    weights = np.array([-30])
    bias = 20
    n = Neuron(weights,bias)
    x = np.array([a]) 
    ans = n.feedforward(x)
    #print(ans)
    return ans

# 与非门

def NAND(a,b):
    '''
    真值表
    A   B   output
    --------------
    0   0   1
    0   1   1
    1   0   1
    1   1   0
    '''
    ans =  NOT(AND(a,b)) # 其实这样只是逻辑相连，看看如何通过神经元相连
    #print(ans)
    return ans

# 异或门
def XOR(a,b):
    '''
    真值表
    A   B   output
    --------------
    0   0   1
    0   1   1
    1   0   1
    1   1   0

    用与门，或门，与非门构成异或门
    话说，有科学家发现了人脑可以用单神经元实现异或运算
    '''
    ans = AND(round(OR(a,b)),round(NAND(a,b)))
    print(ans)
    return ans

# 看看这些逻辑单元的拼凑能不能实现数字电路的功能
if __name__ == "__main__":
    '''
    weights = np.array([0,1,-2])
    bias = 4
    n = Neuron(weights,bias)
    x = np.array([2,3,2])
    print(n.feedforward(x))
    '''
    XOR(0,0)
