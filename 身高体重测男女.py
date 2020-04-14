# -*- coding: utf-8 -*

'''
Data
Name 	Weight (lb) 	Height (in) 	Gender
------------------------------------------------
Alice 	    133 	        65 	        F
Bob 	    160 	        72 	        M
Charlie 	152 	        70 	        M
Diana 	    120 	        60 	        F




Shift data by some amount to look better (usually by the mean):

Name 	Weight (minus 135) 	Height (minus 66) 	Gender
Alice 	    -2 	                -1 	                1
Bob 	     25 	             6 	                0
Charlie 	 17 	             4 	                0
Diana 	    -15 	            -6 	                1



Input Layer         Hidden Layer            Output Layer

    x1                  h1
                                                 o1
    x2                  h2

'''


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

# 神经元都用类来做，因为有一些内置属性和内置方法
class Neuron:
    """
    权值和偏置是神经元的属性，前馈是神经元的一个方法（sigmoid是前馈时选择的激活函数）
    """
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self,inputs):
        total = np.dot(self.weights,inputs)+self.bias
        return sigmoid(total)


# 神经网络没有再使用Neuron class，而是直接用公式算
class OurNeuralNetwork:
    '''
    A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
    '''
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self,x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 5000 # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                #print(y_preds)
                #print("Epoch %d loss: %.3f" % (epoch, loss))
                print("Epoch %d: weights = {w1 = %f, w2 = %f, w3 = %f, w4 = %f, w5 = %f, w6 = %f}, bias = {b1 = %f, b2 = %f, b3 = %f},  loss = %f"  %  (epoch, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.b1, self.b2, self.b3, loss))

if __name__ == "__main__":
    # Define dataset
    
    data = np.array([
    [-2, -1],  # Alice
    [25, 6],   # Bob
    [17, 4],   # Charlie
    [-15, -6], # Diana
    ])


    all_y_trues = np.array([
    1, # Alice
    0, # Bob
    0, # Charlie
    1, # Diana
    ])


    a = float(input("体重（kg）:"))
    b = float(input("身高（cm）:"))

    # 源程序用磅和英寸训练，把输入的公制转换成英制
    a = 2.2 * a - 135
    b = 0.39 * b - 66
        
    # Train our neural network!
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)

    david = np.array([a,b])
    n = network.feedforward(david)
    # print(n)
    if n > 0.5:
        print("有 %f %% 的可能性是女生" % (n*100))
    else:
        print("有 %f %% 的可能性是男生" % ((1-n)*100))
    
    while True:
        print("--------")
        a = float(input("体重（kg）:"))
        b = float(input("身高（cm）:"))

        # 源程序用磅和英寸训练，把输入的公制转换成英制
        a = 2.2 * a - 135
        b = 0.39 * b - 66
        david = np.array([a,b])
        n = network.feedforward(david)
        # print(n)
        if n > 0.5:
            print("有 %f %% 的可能性是女生" % (n*100))
        else:
            print("有 %f %% 的可能性是男生" % ((1-n)*100))
















    










