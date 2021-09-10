"""
Multi-class Classification Neural Network Using Sigmoid
Question 5 part 1
Followed tutorial from this website for understanding and creating this neural network:
  https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/
I pledge my honor that I have abided by the Stevens Honor System
-Noah Suttora
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def plotData(iris):
  x_index = 0                                                               # index of sepal length target
  y_index = 1                                                               # index of sepal width target
  formatter = plt.FuncFormatter(lambda i,
                                *args: iris.target_names[int(i)])           # format colorbar with correct target names
  plt.figure(figsize=(5, 4))                                                # size
  plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)  # plot sepal length and width data points
  plt.colorbar(ticks=[0, 1, 2], format=formatter)                           # format colorbar
  plt.xlabel(iris.feature_names[x_index])                                   # sepal length x-axis
  plt.ylabel(iris.feature_names[y_index])                                   # sepal width y-axis
  plt.show()                                                                # show plot

def splitData(iris):
  X = iris.data                                                       # get data from iris
  Y = iris.target                                                     # get label from iris
  x_train, x_test, y_train, y_test = train_test_split(X, Y,           # split data into train/test sets
                                                      test_size=0.2,  # 120 to train, 30 to test
                                                      shuffle=True)   # shuffle the data
  # print("train data shape: ", x_train.shape)                        # (120, 4)
  # print("train label shape: ", y_train.shape)                       # (120, )
  # print("test data shape: ", x_test.shape)                          # (30, 4)
  # print("test label shape: ", y_test.shape)                         # (30, )
  return x_train, x_test, y_train, y_test

def oneHotEncoding(y_train, y_test):
  onehot_encoder=OneHotEncoder(sparse=False)                    # one-hot encoder
  train_reshape = y_train.reshape(len(y_train), 1)              # reshape y training from (120,) to (120,1)
  test_reshape = y_test.reshape(len(y_test), 1)                 # reshape y testing from (30,) to (30,1)
  y_onehot_train = onehot_encoder.fit_transform(train_reshape)  # transform y training to one-hot encoder
  y_onehot_test = onehot_encoder.fit_transform(test_reshape)    # transform y testing to one-hot encoder
  return y_onehot_train, y_onehot_test

def normalize(x_train, x_test):
  sc = StandardScaler()                                       # normalizer
  sc.fit(x_train)                                             # mean & std of training data
  x_train_norm = sc.transform(x_train)                        # normalization of training data                            
  sc.fit(x_test)                                              # mean & std of training data
  x_test_norm = sc.transform(x_test)                          # normalization of testing data    
  # print("mean of train data: ", x_train_norm.mean(axis=0))  # normalized means
  # print("std of train data: ", x_train_norm.std(axis=0))    # 1's
  # print("mean of test data: ", x_test_norm.mean(axis=0))    # normalized means
  # print("std of test data: ", x_test_norm.std(axis=0))      # 1's
  return x_train_norm, x_test_norm

class Neural_Network(object):
  def __init__(self, x, y_onehot_train, y_onehot_test):
    self.X = x                                                # set data to training set
    self.onehot_train = y_onehot_train                        # training one-hot encoding
    self.onehot_test = y_onehot_test                          # testing one-hot encoding

    self.numInput = 4                                         # input layer
    self.numHidden = 10                                       # hidden layer
    self.numOutput = 3                                        # output layer
    
    self.wh = np.random.randn(self.numInput, self.numHidden)  # hidden weight (4,10)
    self.wo = np.random.randn(self.numHidden, self.numOutput) # output weight (10,3)
    self.bh = np.random.randn(self.numHidden)                 # hidden bias (10,)
    self.bo = np.random.randn(self.numOutput)                 # output bias (3,)
    
    self.lr = 0.01                                            # learning rate
    self.epoch = 1000                                         # iterations
    self.losses = []                                          # list to hold training losses
    self.accuracies = []                                      # list to hold training accuracies

  def sigmoid(self, z):
    return 1/(1+np.exp(-z)) # sigmoid activation function

  def sigmoid_prime(self, z):
    return self.sigmoid(z) * (1-self.sigmoid(z))  # sigmoid derivative activation function

  def softmax(self, z):
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims = True) # softmax activation function

  def cost(self, onehot, ao):
    return np.mean(np.square(np.subtract(onehot, ao)))  # mean squared error (MSE)

  def forward(self):
    self.zh = np.dot(self.X, self.wh) + self.bh   # calculate output values for each node in hidden layer
    self.ah = self.sigmoid(self.zh)               # activation function for zh
    self.zo = np.dot(self.ah, self.wo) + self.bo  # calculate output values for each node in output layer
    self.ao = self.softmax(self.zo)               # activation function for zo

  def backward(self):
    self.dcost_dzo = self.ao - self.onehot_train                          # dcost/dzo = ao - y
    self.dzo_dwo = self.ah                                                # dzo/dwo = ah
    self.dcost_wo = np.dot(self.dzo_dwo.T, self.dcost_dzo)                # dcost/dwo = dcost/dzo * dzo/dwo
    self.dcost_bo = self.dcost_dzo                                        # dcost/dbo = ao - y

    self.dzo_dah = self.wo                                                # dzo/dah = wo
    self.dcost_dah = np.dot(self.dcost_dzo, self.dzo_dah.T)               # dcost/dah = dcost/dzo * dzo/dah
    self.dah_dzh = self.sigmoid_prime(self.zh)                            # derivative activation function for zh
    self.dzh_dwh = self.X                                                 # dzh/dwh = X
    self.dcost_wh = np.dot(self.dzh_dwh.T, self.dah_dzh * self.dcost_dah) # dcost/wh = dzh/dwh * (dah/dzh * dcost/dah)
    self.dcost_bh = self.dcost_dah * self.dah_dzh                         # dcost/bh = dcost/dah * dah/dzh

    self.wh -= self.dcost_wh * self.lr                                    # update hidden weight
    self.wo -= self.dcost_wo * self.lr                                    # update output weight
    self.bh -= self.dcost_bh.sum(axis=0) * self.lr                        # update hidden bias
    self.bo -= self.dcost_bo.sum(axis=0) * self.lr                        # update output bias

  def plot_loss(self):
    plt.plot(self.losses)   # plot loss values
    plt.xlabel("Epoch")     # x-axis label
    plt.ylabel("Loss")      # y-axis label
    plt.title("Loss Plot")  # plot title
    plt.show()              # show plot

  def accuracy(self, onehot, ao):
    correct = 0                           # variable to accumulate correct predictions
    idx = 0                               # index for one-hot following indexing in ao
    for i in ao:                          # iterate through the output from forward
      ao_max = np.argmax(i)               # get the index of the max value of the output from forward
      onehot_max = np.argmax(onehot[idx]) # get the index of the max value of the one-hot encoding
      if ao_max == onehot_max:            # check if the indices are the same
        correct += 1                      # if same, correct guess
      idx += 1                            # increment index
    accuracy = (correct/len(ao)) * 100    # calculate accuracy of ao compared to one-hot for current epoch
    return accuracy

  def plot_accuracy(self):
    plt.plot(self.accuracies)   # plot accuracy values
    plt.xlabel("Epoch")         # x-axis label
    plt.ylabel("Accuracy")      # y-axis label
    plt.title("Accuracy Plot")  # plot title
    plt.show()                  # show plot

  def train(self):
    for i in range(self.epoch):                             # iterate through epochs       
      self.forward()                                        # forward propagation first
      self.backward()                                       # backward propagation with output obtained from forward
      loss = self.cost(self.onehot_train, self.ao)          # compute loss of expected versus output
      self.losses.append(loss)                              # append loss to losses list
      accuracy = self.accuracy(self.onehot_train, self.ao)  # compute accuracy of training data
      self.accuracies.append(accuracy)                      # append accuracy to accuracies list
      print('Training loss: {}, Trainning Accuracy: {}'
            .format(loss, accuracy))                        # print loss and accuracy for each epoch

  def predict(self, x_test):
    self.X = x_test                                     # set data to testing set
    self.forward()                                      # forward propagation for output
    test_acc = self.accuracy(self.onehot_test, self.ao) # calculate accuracy of testing output
    print("----------------------------------")         # separate training loss/accuracy and testing accuracy in terminal
    print('Test Accuracy: {}'.format(test_acc))         # print testing accuracy

def main():
  iris = load_iris()                                              # load iris dataset
  plotData(iris)                                                  # plot iris data
  x_train, x_test, y_train, y_test = splitData(iris)              # split data into train/test sets
  y_onehot_train, y_onehot_test = oneHotEncoding(y_train, y_test) # one-hot encode labels
  x_train_norm, x_test_norm = normalize(x_train, x_test)          # normalize x train and test

  NN = Neural_Network(x=x_train_norm,
                      y_onehot_train=y_onehot_train,
                      y_onehot_test=y_onehot_test)                # create nueral network class object
  NN.train()                                                      # train nueral network for specified epochs
  NN.plot_loss()                                                  # plot losses
  NN.plot_accuracy()                                              # plot accuracies
  NN.predict(x_test_norm)                                         # predict labels for testing data 

if __name__ == '__main__':
  main()