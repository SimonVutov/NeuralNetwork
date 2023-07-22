import csv
import numpy as np

def read_csv_into_2d_list(file_name):
    data_2d_list = []
    with open(file_name, newline='') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            data_2d_list.append(row)
    return data_2d_list

mnist_data = read_csv_into_2d_list("mnist_train.csv")

input = np.zeros((len(mnist_data), 784), dtype=float)
output = np.zeros((len(mnist_data), 10), dtype=float)

for i in range(0, len(mnist_data)):
    for j in range(0, 784):
        input[i][j] = float(mnist_data[i][j+1]) / 255.0
    output[i][int(float(mnist_data[i][0]))] = 1.0

for i in range(0, 27):
    print(input[1][i*28:(i+1)*28])
print(output[1])