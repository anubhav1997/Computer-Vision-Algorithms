import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.nn.functional as F

from torch import nn 
import matplotlib.pyplot as plt
import seaborn as sn 
import pandas as pd 


# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)

# #The compose function allows for multiple transforms
# #transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
# #transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)

# test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)






# Hyperparameters
num_epochs = 5
num_classes = 4
batch_size = 100
learning_rate = 0.001

# DATA_PATH = 'C:\\Users\Andy\PycharmProjects\MNISTData'
# MODEL_STORE_PATH = 'C:\\Users\Andy\PycharmProjects\pytorch_models\\'


# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
# test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

train_dataset = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# train_dataset = train_dataset[np.where(train_dataset==2 | train_dataset==1 | train_dataset==3 | train_dataset==4)]
# test_dataset = 

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        
        # self.drop_out = nn.Dropout()
        # self.fc1 = nn.Linear(32 * 32 * 64, 1000)
        self.fc1 = nn.Linear(16384, 1000)
        
        self.fc2 = nn.Linear(1000, 4)




    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out



model = ConvNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass

        # labels = labels.data
        # print labels, type(labels) 
        # exit(0)

        images = images.numpy()
        labels = labels.numpy()
        l = []
        I = []
        for i in range(len(labels)):
            if(labels[i]==1 or labels[i]==2 or labels[i]==3 or labels[i]==0):
                l.append(labels[i])
                I.append(images[i])
        # images = images[np.argwhere(labels==1 or labels==2 or labels ==3 or labels==4)]
        # labels = labels[np.argwhere(labels==1 | labels==2 | labels ==3 | labels==4)]

        labels = torch.from_numpy(np.array(l))
        images = torch.from_numpy(np.array(I))


        # print labels
        images = Variable(images)
        labels = Variable(labels)
        
        #Clear the gradients
        optimizer.zero_grad()



        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (float(correct) / total) * 100))


# Test the model

predictions = []
label = []

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.numpy()
        labels = labels.numpy()
        l = []
        I = []
        for i in range(len(labels)):
            if(labels[i]==1 or labels[i]==2 or labels[i]==3 or labels[i]==0):
                l.append(labels[i])
                I.append(images[i])
                # label.append(labels[i])

        
        labels = torch.from_numpy(np.array(l))
        images = torch.from_numpy(np.array(I))

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        if(len(predictions)==0):
            predictions = predicted.numpy()
            label = labels.numpy()
            # print label.shape
            # print predictions.shape
        else: 
            predictions = np.append(predictions, predicted.numpy())
            label = np.append(label, labels.numpy())
            # print labels.shape 
            # print predictions.shape 

        
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((float(correct)/ total) * 100))

# Save the model and plot
torch.save(model.state_dict(),'conv_net_model.ckpt')


# Reference: https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(label, predictions)

# fig = 

fig = plt.figure(figsize=(7, 7))
df_cm = pd.DataFrame(conf_matrix, range(4),range(4))
# plt.figure(figsize = (10,10))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) #font size
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
fig.savefig('abc.png')










