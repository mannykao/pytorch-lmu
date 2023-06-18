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

from lmufft import *
from lmuapp import *


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


def getSeqMNISTtype(kind:str) -> str:
	""" args.d """
	valid = {'row', 'psMNIST', 'psLMU'}
	return kind if kind in valid else None 		#could use getattr()

def main(title:str):
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
	device = torchutils.onceInit(kCUDA=True, seed=SEED)

	#1: use SeqMNIST or psMNIST
	dataset = args.dataset 		#fashion|mnist
	kFashion = (dataset == 'fashion')
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

 	# #### Model
	if args.model == "lmu":
		#  def __init__(self, units=212, order=256, theta=28**2):

		model = LMUModel(
			input_size = N_x, 
			output_size = N_c, 
			hidden_size = N_h,	#212: dimension of the hidden state 
			memory_size = N_m,	#256: dimension of the memory 
			seq_len = N_t, 
			theta = THETA,
			learn_a = LEARN_A,
			learn_b = LEARN_B
		)
	else:	
		model = LMUFFTModel(
			input_size = N_x, 
			output_size = N_c, 
			hidden_size = N_h,	#212: dimension of the hidden state 
			memory_size = N_m,	#256: dimension of the memory 
			seq_len = N_t, 
			theta = THETA,
			learn_a = LEARN_A,
			learn_b = LEARN_B
		)

	print(model)
	torchutils.dumpModelSize(model)
	model = model.to(device)

	# #### Optimization
	optimizer = optim.Adam(params = model.parameters())
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(device)

	train_losses, train_accs, val_losses, val_accs = training(device, model, 
		dl_train, dl_val, dl_test,
		optimizer, criterion, 
		N_epochs, 
		N_validate=args.validate,
		testmode=args.testmode,
	)

	# Learning curves
	if args.plot:
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


if __name__ == "__main__":
	title = "Parallel LMU with fft"
	main(title)

