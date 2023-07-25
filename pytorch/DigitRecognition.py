import torch as torch
import csv
import numpy as np
import matplotlib.pyplot as plt

lr = 0.3
mini_batch_size = 500
iterations = 2000

def read_csv_into_2d_list(file_name):
    data_2d_list = []
    with open(file_name, newline='') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            data_2d_list.append(row)
    return data_2d_list

mnist_data = read_csv_into_2d_list("mnist_train.csv")
print("Read data, Number of images in the dataset: ", len(mnist_data))

trainAmount = min(len(mnist_data), 10000)
X = np.zeros((trainAmount, 784), dtype=np.float32)
Y = np.zeros((trainAmount, 10), dtype=np.float32)
X_test = np.zeros((1000, 784), dtype=np.float32)
Y_test = np.zeros((1000, 10), dtype=np.float32)

for i in range(0, trainAmount + 1000):
    if i < trainAmount:
        for j in range(0, 784):
            X[i][j] = float(mnist_data[i][j+1]) / 255.0
        Y[i][int(float(mnist_data[i][0]))] = 1.0
    else:
        for j in range(0, 784):
            X_test[i - trainAmount][j] = float(mnist_data[i][j+1]) / 255.0
        Y_test[i - trainAmount][int(float(mnist_data[i][0]))] = 1.0

#initialize
Weights1 = torch.randn(784, 10) * 0.01
Bias1 = torch.randn(1, 10)
Weights2 = torch.randn(10, 10) * 0.01
Bias2 = torch.randn(1, 10)
X_train = torch.from_numpy(X)
Y_train = torch.from_numpy(Y)

def relu(x): return torch.max(torch.zeros(x.size()), x)
def softmax(x): return torch.exp(x) / torch.sum(torch.exp(x))
def relu_deriv(x): return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
def softmax_deriv(x):
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True) * (1 - exp_x / torch.sum(exp_x, dim=1, keepdim=True))

listLoss = []
for i in range(iterations):
    #get a random mini-batch
    random_indices = torch.randperm(X_train.shape[0])
    random_indices = random_indices[:mini_batch_size]
    X_train_mini_batch = X_train[random_indices]
    Y_train_mini_batch = Y_train[random_indices]

    #forward
    forwardFirstLayer = torch.matmul(X_train_mini_batch, Weights1) + Bias1
    forwardFirstLayer = relu(forwardFirstLayer)
    forwardSecondLayer = torch.matmul(forwardFirstLayer, Weights2) + Bias2
    forwardSecondLayer = relu(forwardSecondLayer)

    #calculate the -logloss
    log_probs = torch.log_softmax(forwardSecondLayer, dim=1)
    logloss = -torch.sum(log_probs * Y_train_mini_batch) / X_train_mini_batch.shape[0]

    #backward
    #calculate the gradients
    dlogloss = (forwardSecondLayer - Y_train_mini_batch) * softmax_deriv(forwardSecondLayer)
    dWeights2 = torch.matmul(forwardFirstLayer.T, dlogloss) / X_train_mini_batch.shape[0]
    dBias2 = torch.sum(dlogloss, 0) / X_train_mini_batch.shape[0]
    dforwardFirstLayer = torch.matmul(dlogloss, Weights2.T) * relu_deriv(forwardFirstLayer)
    dWeights1 = torch.matmul(X_train_mini_batch.T, dforwardFirstLayer) / X_train_mini_batch.shape[0]
    dBias1 = torch.sum(dforwardFirstLayer, 0) / X_train_mini_batch.shape[0]

    #update the weights
    Weights1 = Weights1 - lr * dWeights1
    Bias1 = Bias1 - lr * dBias1
    Weights2 = Weights2 - lr * dWeights2
    Bias2 = Bias2 - lr * dBias2

    listLoss.append(logloss)
    if i % 100 == 0:
        print("Iteration: ", i)
        print("Loss: ", logloss)

#test on 1000 test images
correct = 0
testImages = torch.from_numpy(X_test)
anserImages = torch.from_numpy(Y_test)
for n in range(0, 1000):
    forwardFirstLayer = torch.matmul(testImages[n], Weights1) + Bias1
    forwardFirstLayer = relu(forwardFirstLayer)
    forwardSecondLayer = torch.matmul(forwardFirstLayer, Weights2) + Bias2
    if torch.argmax(forwardSecondLayer) == torch.argmax(anserImages[n]):
        correct += 1

print("Test Accuracy:", (correct / 10))

#plot the loss
plt.plot(listLoss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()