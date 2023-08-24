import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, dataframe, augmentation=True):
        self.data = dataframe.values
        self.augmentation = augmentation
        
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), # rotate up to 15 degrees and translate up to 10% in x and y direction
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx, 1:].reshape(28, 28).astype('uint8')
        label = int(self.data[idx, 0])
        
        if self.augmentation:
            img = self.apply_augmentation(img)
            
        img = self.transforms(img)  # Use the data augmentation transforms
        return img, label
    
    def apply_augmentation(self, img):
        img = self.transforms(img)
        return img

# Load data from CSV
data = pd.read_csv('mnist_train.csv')

# Split into train and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Create custom datasets and data loaders
train_dataset = CustomDataset(train_data, augmentation=True) # Enable augmentation for training
val_dataset = CustomDataset(val_data, augmentation=False)    # Disable augmentation for validation

# Define the batch size you want
batch_size = 64  # Change this to your desired batch size

# Create data loaders with the specified batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define a simple neural network with dropout
class SimpleNN(nn.Module): # 784, 128, 64, 32, 10
    def __init__(self, dropout_prob=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Initialize the network, loss function, and optimizer
dropout_prob = 0.0
net = SimpleNN(dropout_prob)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

# Training loop
num_epochs = 330
for epoch in range(num_epochs):
    net.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Validation Loss: {val_loss/len(val_loader):.4f} | "
          f"Validation Acc: {(correct/total)*100:.2f}%")
