#!/usr/bin/env python
# coding: utf-8

# # Legendre Memory Units

#
# Exported from lmu_fft_psmnist.ipynb. Rewritten for more reuse and enhancements.
#
import argparse
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
from datasets.mnist import mnist
from datasets.utils.xforms import GreyToFloat
from mk_mlutils.pipeline import torchbatch

from src.lmu2 import *
from src.lmuapp import *


# In[35]:
N_x = 1 # dimension of the input, a single pixel
N_t = 784
N_h = 346 # dimension of the hidden state
N_m = 468 # dimension of the memory
N_c = 10 # number of classes 
THETA = 784
N_b = 100 # batch size
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
#end of LMUFFTModel


if __name__ == "__main__":
	args = ourargs(title="Parallel LMU with fft")

	THETA = args.theta 		#784
	N_b = args.batchsize 	#100 # batch size
	N_epochs = args.epochs 	#15
	N_t = args.t 	 	 	#784
	N_h = args.h 			#346 # dimension of the hidden state
	N_m = args.m 			#468 # dimension of the memory

	# Connect to GPU
	DEVICE = initCuda()

	SEED = 0
	setSeed(SEED)

	transform = transforms.ToTensor()
	mnist_train = datasets.MNIST("/content/", train = True, download = True, transform = transform)
	mnist_val   = datasets.MNIST("/content/", train = False, download = True, transform = transform)

	perm = load_permutation(__file__)
	ds_train = psMNIST(mnist_train, perm)
	ds_val   = psMNIST(mnist_val, perm)

	seqmnist_train = mnist.SeqMNIST(split="train", permute='psLMU', imagepipeline=GreyToFloat())
	seqmnist_test  = mnist.SeqMNIST(split="test", permute='psLMU', imagepipeline=GreyToFloat())

	ds_train, ds_test = seqmnist_train, seqmnist_test

	if args.trset == 'test':
		ds_train, ds_val = ds_val, ds_train

	dl_train = DataLoader(ds_train, batch_size = N_b, shuffle = True, num_workers = 1, pin_memory = True)
	dl_val   = DataLoader(ds_val, batch_size = N_b, shuffle = False, num_workers = 1, pin_memory = True)

	#dl_train = torchbatch.getBatchAsync(DEVICE, )


	# Example of the data
	eg_img, eg_label = ds_train[0]
	print("Label:", eg_label)
	#dispSeq(eg_img)

	# #### Model
	if args.model == "lmu":
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
	model = model.to(DEVICE)
	countParameters(model)

	# #### Optimization
	optimizer = optim.Adam(params = model.parameters())

	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(DEVICE)

	train_losses = []
	train_accs = []
	val_losses = []
	val_accs = []

	for epoch in range(N_epochs):
		print(f"Epoch: {epoch+1:02}/{N_epochs:02}")

		train_loss, train_acc = train(DEVICE, model, dl_train, optimizer, criterion)
		val_loss, val_acc = validate(DEVICE, model, dl_val, criterion)

		train_losses.append(train_loss)
		train_accs.append(train_acc)
		val_losses.append(val_loss)
		val_accs.append(val_acc)

		print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
		print(f"Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%")
		print()

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

