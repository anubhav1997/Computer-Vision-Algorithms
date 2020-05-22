import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import cv2
from PIL import Image

new_model = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(new_model.classifier.children())[:-1])
new_model.classifier = new_classifier

scaler = transforms.Scale((224, 224))
to_tensor = transforms.ToTensor()

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict



for i in range(5):
	filename = 'cifar-10-batches-py/data_batch_' + str(i+1)
	dict = unpickle(filename)
	if(i==0):
		trainX = dict['data']
		trainY = dict['labels']
	else:

		trainX = np.append(trainX, dict['data'], axis=0)
		trainY = np.append(trainY, dict['labels'], axis=0)


trainX = np.array(trainX)
trainY = np.array(trainY)
print trainX.shape

train = []

for i in range(len(trainX)):
	x = np.transpose(np.reshape(trainX[i],(3, 32,32)), (1,2,0))
	x = cv2.resize(x, (224, 224))
	x = np.reshape(x,(672,224))
	x = Image.fromarray(np.uint8(x))
	# x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	
	# print x.shape 
	# x = cv2.resize(x, (x.shape[0]//4, x.shape[1]//4))
	x = Variable(to_tensor(scaler(x))).unsqueeze(0)
	output = new_classifier(x)
	train.append(output)


trainX = np.array(train) 
print trainX.shape 

del train 

dict = unpickle('cifar-10-batches-py/test_batch')
testX = dict['data']
testY = dict['labels']

train = []

for i in range(len(testX)):
	x = np.transpose(np.reshape(testX[i],(3, 32,32)), (1,2,0))
	
	x = Variable(to_tensor(scaler(x))).unsqueeze(0)
	output = new_classifier(x)
	train.append(output)


testX = np.array(train) 
del train 


n_classes = np.max(trainY)+1


# print output
print testX
