#!/usr/bin/env python
# coding: utf-8

# # Legendre Memory Units

#
# Exported from lmu_fft_psmnist.ipynb. Rewritten for more reuse and enhancements.
#
from collections import namedtuple
import numpy as np
import random

from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import fft
from torch.nn import init
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from scipy.signal import cont2discrete
#
# mck:
import datasets.utils.projconfig as projconfig
from datasets.mnist import mnist
from datasets.mnist.getdb import getMNIST
from datasets.utils.xforms import GreyToFloat
from mk_mlutils.dataset import datasetutils
from mk_mlutils.pipeline import torchbatch
from mk_mlutils.utils import torchutils

from src.lmufft import *
from src.lmuapp import *


kPlot=False

# In[35]:
#LMU_config = namedtuple("LMU_config", [N_x, N_t, N_h, N_m, N_c, THETA, LEARN_A, LEARN_B])
"""
kLMU_config = LMU_config(
	N_x,	# = 1 # dimension of the input, a single pixel
	N_t,	# = 784
	N_h,	# = 346 # dimension of the hidden state
	N_m,	# = 468 # dimension of the memory
	N_c,	# = 10 # number of classes 
	THETA,	# = 784
	LEARN_A,# = False
	LEARN_B #= False
)
"""

N_x = 1 	# dimension of the input, a single pixel
N_t = 784
N_h = 212 # dimension of the hidden state
N_m = 256 # dimension of the memory
#N_h = 346	# dimension of the hidden state
#N_m = 468	# dimension of the memory
N_c = 10	# number of classes 
THETA = 784
N_b = 100	# batch size
N_epochs = 1 	#15
LEARN_A = False
LEARN_B = False


# ### Model
class LMUModel(nn.Module):
	""" A simple model for the psMNIST dataset consisting of a single LMU layer and a single dense classifier """
	def __init__(self, 
		input_size, 
		output_size, 
		hidden_size, 
		memory_size, 
		seq_len,		#not used (yet) by LMU() 
		theta, 
		learn_a = False, 
		learn_b = False
	):
		super().__init__()
		self.lmu = LMU(input_size, hidden_size, memory_size, theta, learn_a, learn_b, psmnist = True)
		self.dropout = nn.Dropout(p = 0.5)
		self.classifier = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		_, (h_n, _) = self.lmu(x) # [batch_size, hidden_size]
		#x = self.dropout(h_n)
		output = self.classifier(h_n)
		return output # [batch_size, output_size]

	def __str__(self):
		return f"LMUModel(hidden_size={self.lmu.hidden_size}, memory_size={self.lmu.memory_size})"
#end of LMUModel

class LMUFFTModel(nn.Module):
	""" A simple model for the psMNIST dataset consisting of a single LMUFFT layer and a single dense classifier """
	def __init__(self, 
		input_size, 
		output_size, 
		hidden_size, 
		memory_size, 
		seq_len, 
		theta,
		learn_a = False,	#not used by LMUFFT (yet) 
		learn_b = False		#not used by LMUFFT (yet)
	):
		super().__init__()
		self.lmu_fft = LMUFFT(input_size, hidden_size, memory_size, seq_len, theta)
		self.dropout = nn.Dropout(p = 0.5)
		self.classifier = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		_, h_n = self.lmu_fft(x) # [batch_size, hidden_size]
		x = self.dropout(h_n)
		output = self.classifier(x)
		return output # [batch_size, output_size]

	def __str__(self):
		return f"LMUModelFFT(hidden_size={self.lmu_fft.hidden_size}, memory_size={self.lmu_fft.memory_size}, {self.lmu_fft})"
#end of LMUFFTModel

def getSeqMNISTtype(kind:str) -> str:
	""" args.d """
	valid = {'row', 'psMNIST', 'psLMU'}
	return kind if kind in valid else None 		#could use getattr()


if __name__ == "__main__":
	title = "Parallel LMU with fft"
	mnist_dir = projconfig.getMNISTFolder()	#"/content/"
	print(f"{title}")

	kFashion = True		#fashionmist or mnist

	args = ourargs(title=title)
	THETA = args.theta 		#784
	N_b = args.batchsize 	#100 # batch size
	N_epochs = args.epochs 	#15
	N_t = args.t 	 	 	#784
	N_h = args.h 			#346 # dimension of the hidden state
	N_m = args.m 			#468 # dimension of the memory
	N_validate = args.validate #validate interval, defaults to 5

	SEED = 0
	# Connect to GPU
	DEVICE = torchutils.onceInit(kCUDA=True, seed=SEED)

	#1: use SeqMNIST or psMNIST
	permute = getSeqMNISTtype(args.p)
	print(f"{permute=}")

	seqmnist_train = getMNIST(kind='seqmnist', split='train', imagepipeline=GreyToFloat(), kFashion=kFashion, permute=permute)
	seqmnist_test  = getMNIST(kind='seqmnist', split='test', imagepipeline=GreyToFloat(), kFashion=kFashion, permute=permute)
	#seqmnist_train = mnist.SeqMNIST(split="train", permute=permute, imagepipeline=GreyToFloat(), kFashion=kFashion)
	#seqmnist_test  = mnist.SeqMNIST(split="test", permute=permute, imagepipeline=GreyToFloat(), kFashion=kFashion)
	ds_train, ds_test = seqmnist_train, seqmnist_test
	print(f"{seqmnist_train}")

	if args.trset == 'test':
		ds_train, ds_test = ds_test, ds_train

	ds_val = datasetutils.getBalancedSubset(ds_test, fraction=.1, useCDF=True)

	dl_train = DataLoader(ds_train, batch_size = N_b, shuffle = True, num_workers = 1, pin_memory = False)
	dl_test  = DataLoader(ds_test, batch_size = N_b, shuffle = False, num_workers = 1, pin_memory = False)
	dl_val   = DataLoader(ds_val, batch_size = N_b, shuffle = False, num_workers = 1, pin_memory = False)

	#create out batch builder
	#dl_train = batch.Bagging(ds_train, batchsize=N_b, shuffle=False)

	# Example of the data
	eg_img, eg_label = ds_train[0]
	print("Label:", eg_label)
	#dispSeq(eg_img)

 	# #### Model
	if args.model == "lmu":
		#  def __init__(self, units=212, order=256, theta=28**2):
		#N_h = 212 # dimension of the hidden state
		#N_m = 256 # dimension of the memory

		model = LMUModel(
			input_size = N_x, 
			output_size = N_c, 
			hidden_size = N_h, 
			memory_size = N_m, 
			seq_len = N_t, 
			theta = THETA,
			learn_a = LEARN_A,
			learn_b = LEARN_B
		)
	else:	
#		N_h = 346 # dimension of the hidden state
#		N_m = 468 # dimension of the memory
#		N_h = 212 # dimension of the hidden state
#		N_m = 256 # dimension of the memory

		model = LMUFFTModel(
			input_size = N_x, 
			output_size = N_c, 
			hidden_size = N_h, 
			memory_size = N_m, 
			seq_len = N_t, 
			theta = THETA,
			learn_a = LEARN_A,
			learn_b = LEARN_B
		)

	print(model)
	torchutils.dumpModelSize(model)
	model = model.to(DEVICE)
	#countParameters(model)

	# #### Optimization
	optimizer = optim.Adam(params = model.parameters())

	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(DEVICE)

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
		if ((epoch+1) % N_validate) == 0:
			val_loss, val_acc = validate(DEVICE, model, dl_val, criterion)
			last_validate = epoch

		train_losses.append(train_loss)
		train_accs.append(train_acc)
		val_losses.append(val_loss)
		val_accs.append(val_acc)

		losses(train_loss, train_acc, val_loss, val_acc)
	#end of epochs
	accuracy_scores(train_accs, val_accs)
	print()

	if args.testmode == 1:
		print("Final test:")
		#use full test-set	
		tst_loss, tst_acc = validate(DEVICE, model, dl_test, criterion)
		losses(train_loss, train_acc, tst_loss, tst_acc)
		print()

	# Learning curves
	if kPlot:
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

