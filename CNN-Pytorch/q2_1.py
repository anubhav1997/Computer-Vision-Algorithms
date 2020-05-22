import matplotlib
matplotlib.use('Agg')

import torchvision.models as models
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.nn.functional as F

from torch import nn 
import cv2 
from sklearn.svm import SVC
import seaborn as sn 
import matplotlib.pyplot as plt

# import matplotlib
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
num_classes = 10
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




alexnet = models.alexnet(pretrained=True)


# Test the model
alexnet.eval()

features = []
Y = []

with torch.no_grad():
	correct = 0
	total = 0
	feat = []
	lab = []
	for images, labels in train_loader:
		# print 'heyyyy'
		# print type(images)
		images = images.numpy()
		# print type(images)
		images = images.reshape(len(images), 32, 32, 3)

		# print images.shape 

		imgx = []
		# print 'hiii', type(imgx)

		for i in range(len(images)):
			# print type(imgx)
			x = cv2.resize(images[i], (244, 244))
			# print type(x)
			x = x.reshape(3, 244, 244)
			imgx.append(x)
			# print type(imgx)



		# print 'cdsc'
		# print 'Nooooooo'
			



		imgx = np.array(imgx)
		imgx = torch.from_numpy(imgx)

		labels = labels.numpy()
		# lab.append(labels)
		outputs = alexnet(imgx)
		outputs = outputs.numpy()
		# print outputs.shape

		if(len(lab)==0):
			lab = labels
		else:

			lab = np.append(lab, labels)

		if(len(feat)==0):
			feat = outputs
		else:

			feat = np.append(feat, outputs, axis=0)

		# feat.append(outputs)
		# print np.array(feat).shape 

		# _, predicted = torch.max(outputs.data, 1)
		# total += labels.size(0)
		# correct += (predicted == labels).sum().item()


	# print 'heyyy'
	# print np.array(feat).shape 

	if(len(features)==0):
		features = feat
	else: 

		features = np.append(features, feat, axis=0)
	
	print features.shape 
	
	if(len(Y)==0):
		Y = lab
	else: 

		Y = np.append(Y, lab)
	
	# print features.shape 
	
	# Y.append(lab)
	print np.array(Y).shape 
    # print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
# torch.save(model.state_dict(),'conv_net_model.ckpt')

features = np.array(features)
print features.shape 



features_test = []
Y_test = []


with torch.no_grad():
	correct = 0
	total = 0
	feat = []
	lab = []
	for images, labels in test_loader:
		# print 'heyyyy'
		# print type(images)
		images = images.numpy()
		# print type(images)
		images = images.reshape(len(images), 32, 32, 3)

		# print images.shape 

		imgx = []
		# print 'hiii', type(imgx)

		for i in range(len(images)):
			# print type(imgx)
			x = cv2.resize(images[i], (244, 244))
			# print type(x)
			x = x.reshape(3, 244, 244)
			imgx.append(x)
			# print type(imgx)



		# print 'cdsc'
		# print 'Nooooooo'
			



		imgx = np.array(imgx)
		imgx = torch.from_numpy(imgx)

		labels = labels.numpy()
		
		# lab = append(labels)
		outputs = alexnet(imgx)
		outputs = outputs.numpy()
		# print outputs.shape

		if(len(feat)==0):
			feat = outputs
		else:

			feat = np.append(feat, outputs, axis=0)

		if(len(lab)==0):
			lab = labels
		else:

			lab = np.append(lab, labels)


		# feat.append(outputs)
		print np.array(feat).shape 

		# _, predicted = torch.max(outputs.data, 1)
		# total += labels.size(0)
		# correct += (predicted == labels).sum().item()


	# print 'heyyy'
	if(len(features_test)==0):
		features_test = feat
	else: 

		features_test = np.append(features_test, feat, axis=0)
	
	print features_test.shape

	# Y_test.append(lab)
	if(len(Y_test)==0):

		Y_test = lab
	else: 

		Y_test = np.append(Y_test, lab)

    # print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))



print features.shape 


print 'heyyy'
clf = SVC(gamma='auto', max_iter=10000)
print 'hiiii'
clf.fit(features, Y) 

print 'xxxxx'

preds = clf.predict(features_test)
count = 0

print 'blahhhh'

for i in range(len(preds)):
	if(preds[i]==Y_test[i]):
		count+=1

print float(count)/len(preds)


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(Y_test, preds)

# fig = 

fig = plt.figure(figsize=(7, 7))
df_cm = pd.DataFrame(conf_matrix, range(10),range(10))
# plt.figure(figsize = (10,10))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) #font size
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()
fig.savefig('abc.png')





# print preds