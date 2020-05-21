import numpy as np
import cv2
import os
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
#    transforms.Normalize(mean = (0.896, 0.644, 0.854), std = (0.096, 0.204, 0.135))
    ])

images = torch.zeros(9,3,224,224)



for i in range(9):
#    file_name_curr = 'crop00' + str(i+1) +'.jpg'
    file_name_curr = 'crop00' + str(i+1) +'.jpg'
    img1 = cv2.imread(file_name_curr)
    print 'img',i,img1.shape
    img2 = transform1(img1)
    print img2
    images[i] = img2

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(1),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(1),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
    return model

model = alexnet(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
print model 

inputs = Variable(images)
outputs = model(inputs)
print outputs.data
print outputs
# np.savetxt('page1.txt',outputs.data.numpy())

