#!/usr/bin/env python
# coding: utf-8

# # Legendre Memory Units

# ## Start
# To reset the notebook, run from this point

# In[26]:


#get_ipython().run_line_magic('reset', '-f')


# ## Imports

# In[27]:

import numpy as np
import random

from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from scipy.signal import cont2discrete

#
# mck:
import datasets.utils.projconfig as projconfig
from datasets.mnist import mnist
from mk_mlutils.dataset import datasetutils
from datasets.utils.xforms import GreyToFloat

from src.lmu2 import *
from src.lmuapp import *


# ### Imports and Constants
# In[31]:
#from tqdm.notebook import tqdm
from tqdm import tqdm
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
class LMUModel(nn.Module):
	""" A simple model for the psMNIST dataset consisting of a single LMU layer and a single dense classifier """

	def __init__(self, input_size, output_size, hidden_size, memory_size, theta, learn_a = False, learn_b = False):
		super(LMUModel, self).__init__()
		self.lmu = LMU(input_size, hidden_size, memory_size, theta, learn_a, learn_b, psmnist = True)
		self.dropout = nn.Dropout(p = 0.5)
		self.classifier = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		_, (h_n, _) = self.lmu(x) # [batch_size, hidden_size]
		#x = self.dropout(h_n)
		output = self.classifier(h_n)
		return output # [batch_size, output_size]


# ### Main
if __name__ == "__main__":
	title = "Sequential LMU (Voelker)"
	mnist_dir = projconfig.getMNISTFolder()
	print(f"{title} is superced by 'lmu_fft_psmnist --model lmu")
#	getch()

	args = ourargs(title=title)
	THETA = args.theta 		#784
	N_b = args.batchsize 	#100 # batch size
	N_epochs = args.epochs 	#10 should be enough, defaults to 1
	N_validate = args.validate #validate interval, defaults to 5

	# Connect to GPU
	SEED = 0
	DEVICE = torchutils.onceInit(kCUDA=True, seed=SEED)
	print(f"{DEVICE}")

	#1: use SeqMNIST or psMNIST
	permute = getSeqMNISTtype(args.d)
	print(f"SeqMNIST({mnist_dir}, permute={permute})")

	seqmnist_train = mnist.SeqMNIST(split="train", permute=permute, imagepipeline=GreyToFloat())
	seqmnist_test  = mnist.SeqMNIST(split="test", permute=permute, imagepipeline=GreyToFloat())
	ds_train, ds_test = seqmnist_train, seqmnist_test

	if args.trset == 'test':
		ds_train, ds_test = ds_test, ds_train

	ds_val = datasetutils.getBalancedSubset(ds_test, fraction=.1, useCDF=True)

	dl_train = DataLoader(ds_train, batch_size = N_b, shuffle = True, num_workers = 1)
	dl_test  = DataLoader(ds_test, batch_size = N_b, shuffle = False, num_workers = 1, pin_memory = False)
	dl_val   = DataLoader(ds_val, batch_size = N_b, shuffle = False, num_workers = 1)

	# In[43]:
	# Example of the data
	eg_img, eg_label = ds_train[0]
	print("Label:", eg_label)
	#dispSeq(eg_img)

	# #### Model
	# In[44]:
	model = LMUModel(
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
	val_loss, val_acc = 0, 0
	last_validate = -1

	for epoch in range(N_epochs):
		print(f"Epoch: {epoch+1:02}/{N_epochs:02}")

		train_loss, train_acc = train(DEVICE, model, dl_train, optimizer, criterion)

		#validate interval?
		if ((epoch+1) % N_validate) == 0: 	#validate using dl_val set
			val_loss, val_acc = validate(DEVICE, model, dl_val, criterion)
			last_validate = epoch

		losses(train_loss, train_acc, val_loss, val_acc)

		train_losses.append(train_loss)
		train_accs.append(train_acc)
		val_losses.append(val_loss)
		val_accs.append(val_acc)
	#end of epochs	

	#use full test-set	
	tst_loss, tst_acc = validate(DEVICE, model, dl_test, criterion)
	losses(train_loss, train_acc, tst_loss, tst_acc)
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

