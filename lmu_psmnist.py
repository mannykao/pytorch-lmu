#!/usr/bin/env python
# coding: utf-8

# # Legendre Memory Units

# ## Start
# To reset the notebook, run from this point

# In[26]:


#get_ipython().run_line_magic('reset', '-f')


# ## Imports

# In[27]:

import argparse
import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from scipy.signal import cont2discrete

#
# mck:
from mkpyutils import fileutils

from src.lmu2 import *


# ### Imports and Constants
# In[31]:
import random
#from tqdm.notebook import tqdm
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms



# In[34]:
N_x = 1 # dimension of the input, a single pixel
N_t = 784
N_h = 212 # dimension of the hidden state
N_m = 256 # dimension of the memory
N_c = 10 # number of classes 
THETA = N_t
N_b = 100 # batch size
N_epochs = 10 	#10
LEARN_A = False
LEARN_B = False



# ### Model
# In[36]:
class Model(nn.Module):
	""" A simple model for the psMNIST dataset consisting of a single LMU layer and a single dense classifier """

	def __init__(self, input_size, output_size, hidden_size, memory_size, theta, learn_a = False, learn_b = False):
		super(Model, self).__init__()
		self.lmu = LMU(input_size, hidden_size, memory_size, theta, learn_a, learn_b, psmnist = True)
		self.classifier = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		_, (h_n, _) = self.lmu(x) # [batch_size, hidden_size]
		output = self.classifier(h_n)
		return output # [batch_size, output_size]


# ### Functions

# #### Utils

# In[37]:
def disp(img):
	""" Displays an image """
	if len(img.shape) == 3:
		img = img.squeeze(0)
	plt.imshow(img, cmap = "gray")
	plt.axis("off")
	plt.tight_layout()
	plt.show()


# In[38]:

def dispSeq(seq, rows = 8):
	""" Displays a sequence of pixels """
	seq = seq.reshape(rows, -1) # divide the 1D sequence into `rows` rows for easy visualization
	disp(seq)


# #### Model
# In[39]:

def countParameters(model):
	""" Counts and prints the number of trainable and non-trainable parameters of a model """
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")


# In[40]:
def train(model, loader, optimizer, criterion):
	""" A single training epoch on the psMNIST data """

	epoch_loss = 0
	y_pred = []
	y_true = []
	
	model.train()
	for batch, labels in tqdm(loader):

		torch.cuda.empty_cache()

		batch = batch.to(DEVICE)
		labels = labels.long().to(DEVICE)

		optimizer.zero_grad()

		output = model(batch)
		loss = criterion(output, labels)
		
		loss.backward()
		optimizer.step()

		preds  = output.argmax(dim = 1)
		y_pred += preds.tolist()
		y_true += labels.tolist()
		epoch_loss += loss.item()

	# Loss
	avg_epoch_loss = epoch_loss / len(loader)

	# Accuracy
	epoch_acc = accuracy_score(y_true, y_pred)

	return avg_epoch_loss, epoch_acc


# In[41]:
def validate(model, loader, criterion):
	""" A single validation epoch on the psMNIST data """

	epoch_loss = 0
	y_pred = []
	y_true = []
	
	model.eval()
	with torch.no_grad():
		for batch, labels in tqdm(loader):

			torch.cuda.empty_cache()

			batch = batch.to(DEVICE)
			labels = labels.long().to(DEVICE)

			output = model(batch)
			loss = criterion(output, labels)
			
			preds  = output.argmax(dim = 1)
			y_pred += preds.tolist()
			y_true += labels.tolist()
			epoch_loss += loss.item()
			
	# Loss
	avg_epoch_loss = epoch_loss / len(loader)

	# Accuracy
	epoch_acc = accuracy_score(y_true, y_pred)

	return avg_epoch_loss, epoch_acc


# ### Main
# #### Data
# In[42]:

if __name__ == "__main__":
	args = ourargs(title="Sequential LMU (Voelker)")

	THETA = args.theta 		#784
	N_b = args.batchsize 	#100 # batch size
	N_epochs = args.epochs 	#1

	# Connect to GPU
	DEVICE = initCuda()

	SEED = 0
	setSeed(SEED)
	
	transform = transforms.ToTensor()
	mnist_train = datasets.MNIST("/content/", train = True, download = True, transform = transform)
	mnist_val   = datasets.MNIST("/content/", train = False, download = True, transform = transform)

	permute_file = fileutils.my_path(__file__)/"examples/permutation.pt"	# created using torch.randperm(784)
	perm = torch.load(permute_file).long() 
	ds_train = psMNIST(mnist_train, perm)
	ds_val   = psMNIST(mnist_val, perm) 

	dl_train = DataLoader(ds_train, batch_size = N_b, shuffle = True, num_workers = 2)
	dl_val   = DataLoader(ds_val, batch_size = N_b, shuffle = True, num_workers = 2)

	# In[43]:
	# Example of the data
	eg_img, eg_label = ds_train[0]
	print("Label:", eg_label)
	dispSeq(eg_img)

	# #### Model
	# In[44]:
	model = Model(
		input_size  = N_x,
		output_size = N_c,
		hidden_size = N_h, 
		memory_size = N_m, 
		theta = THETA, 
		learn_a = LEARN_A,
		learn_b = LEARN_B
	)
	model = model.to(DEVICE)

	# In[45]:
	countParameters(model) # as stated in the paper, the model has ≈102K parameters

	# #### Optimization
	# In[46]:
	optimizer = optim.Adam(params = model.parameters())

	# In[47]:
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(DEVICE)

	# In[48]:
	train_losses = []
	train_accs = []
	val_losses = []
	val_accs = []

	for epoch in range(N_epochs):
		print(f"Epoch: {epoch+1:02}/{N_epochs:02}")

		train_loss, train_acc = train(model, dl_train, optimizer, criterion)
		val_loss, val_acc = validate(model, dl_val, criterion)

		train_losses.append(train_loss)
		train_accs.append(train_acc)
		val_losses.append(val_loss)
		val_accs.append(val_acc)

		print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
		print(f"Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%")
		print()

	# In[49]:
	# Learning curves

	plt.plot(range(N_epochs), train_losses)
	plt.plot(range(N_epochs), val_losses)
	plt.ylabel("Loss")
	plt.xlabel("Epochs")
	plt.legend(["Train", "Val."])
	plt.show()

	plt.plot(range(N_epochs), train_accs)
	plt.plot(range(N_epochs), val_accs)
	plt.ylabel("Accuracy")
	plt.xlabel("Epochs")
	plt.legend(["Train", "Val."])
	plt.show()

