"""
Title: lmuapp: extracted from the beginning of lmu_fft_psmnist.py. Same as src/lmu.py with 'psmnist' extensions.
	
Created on Sun Apr 23 17:44:29 2023

@author: Manny Ko
"""
import argparse
import random
import numpy as np
from matplotlib import pyplot as plt

#from tqdm.notebook import tqdm
from tqdm import tqdm

import torch
from torch import nn
#from torch import fft
#from torch.nn import init
#from torch.nn import functional as F

from sklearn.metrics import accuracy_score

from mkpyutils.fileutils import my_path


def setSeed(seed):
	""" Set all seeds to ensure reproducibility """
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def initCuda():
	# Connect to GPU
	if torch.cuda.is_available():
		DEVICE = "cuda"
		# Clear cache if non-empty
		torch.cuda.empty_cache()
		# See which GPU has been allotted 
		print(torch.cuda.get_device_name(torch.cuda.current_device()))
	else:
		DEVICE = "cpu"
	return DEVICE	

def countParameters(model):
	""" Counts and prints the number of trainable and non-trainable parameters of a model """
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")

# #### Utils
def disp(img):
	""" Displays an image """
	if len(img.shape) == 3:
		img = img.squeeze(0)
	plt.imshow(img, cmap = "gray")
	plt.axis("off")
	plt.tight_layout()
	plt.show()

def dispSeq(seq, rows = 8):
	""" Displays a sequence of pixels """
	seq = seq.reshape(rows, -1) # divide the 1D sequence into `rows` rows for easy visualization
	disp(seq)

def load_permutation(myfilepath:__file__) -> torch.Tensor:
	permute_file = my_path(myfilepath)/"examples/permutation.pt"	# created using torch.randperm(784)
	perm = torch.load(permute_file).long()
	return perm

def train(DEVICE, model, loader, optimizer, criterion):
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

def validate(DEVICE, model, loader, criterion):
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
	

def ourargs(title:str):
	parser = argparse.ArgumentParser(description=title,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', type = str, metavar="lmu|fft",
						default = 'fft', help = 'Sequential or Parallel LMU')
	parser.add_argument('--batchsize', type=int, default=100, metavar='N',
						help='input batch size for training (default: 100)')
	parser.add_argument('--epochs', type=int, default=1, metavar='N',
						help='number of epochs to train (default: 1)')
	parser.add_argument('--theta', type=int, default=784, metavar='delay/window size',
						help='delay theta (default: 784)')
	parser.add_argument('--trset', type = str, metavar="test<n>|train<n>",
						default = 'test', help = 'dataset used for training and testing')
	parser.add_argument('--t', type=int, default=784, metavar='N',
						help='number of time steps (default: 784)')
	parser.add_argument('--h', type=int, default=346, metavar='N',
						help='number of hidden states (default: 346)')
	parser.add_argument('--m', type=int, default=468, metavar='N',
						help='number of memory states (default: 468)')
	#N_t = 784
	#N_h = 346 # dimension of the hidden state
	#N_m = 468 # dimension of the memory

	args = parser.parse_args()
	return args