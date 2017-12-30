# This is just cheating lol

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable

import glob # Why even bother doing shit yourself

class TiDataSet(Dataset):
    DATA_DIR = '../images/4/' # here we assume that in that folder there are downsampled pngs of size (28x28) of the ti dataset

    def __init__(self, transform=None, data_dir=None):
        data_dir = self.DATA_DIR if data_dir is None else data_dir

        names = glob.glob(data_dir + "*.png") # name expansion holy shit
        
                  

        #dummy image loader
        N = len(names)
        images = np.zeros((N,28,28,1),   dtype=np.uint8) # how do you python?
        labels = np.zeros((N),           dtype=np.int64) # somewhere if make this 32 bit shit goes bananas

        for n in range(N):
            images[n] = cv2.imread(names[n], 0).reshape((28,28,1)) #Yes we have 1 channel, better make it explicit or .ToTensor will freak the heck out
            # images[n] = Image.open(names[n]).convert('L')
            # images[n].reshape((28,28,1))
            labels[n] = int(names[n].split('/')[-1][0:2])-1 # poof
            #print(labels[n])
        
        self.transform = transform
        self.images    = images  #torch.from_numpy(images)
        self.labels    = labels  #torch.from_numpy(labels)
        
        print(len(self.images), len(self.labels))


    def __getitem__(self, index):
        # print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        img   = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        # print ('\tcalling Dataset:__len__')
        return len(self.images)

dataset = TiDataSet()
print("Hi!")

#######################################

# Hyper Parameters
num_epochs = 10
batch_size = 1
learning_rate = 0.100

train_dataset = TiDataSet(transform=transforms.ToTensor()) # .ToTensor lol this is just to ez
test_dataset  = TiDataSet(transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset    = train_dataset,
                                           batch_size = batch_size, 
                                           shuffle    = False)

test_loader  = torch.utils.data.DataLoader(dataset    = test_dataset,
                                           batch_size = batch_size, 
                                           shuffle    = False)

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
cnn = CNN()

print("Network Structure:")
print(cnn)
print(train_loader)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')
