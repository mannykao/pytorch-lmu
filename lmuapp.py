"""
Title: lmuapp: extracted from the beginning of lmu_fft_psmnist.py. Same as src/lmu.py with 'psmnist' extensions.
	
Created on Sun Apr 23 17:44:29 2023

@author: Manny Ko
"""
import argparse
import random, time
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
from mk_mlutils.pipeline import batch, torchbatch
from mk_mlutils.utils import torchutils


	
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
	parser.add_argument('--dataset', type=str, default='fashion', choices=('fashion','mnist'), help='dataset'),
	parser.add_argument('--trset', type = str, metavar="test<n>|train<n>",
						default = 'test', help = 'dataset used for training and testing')
	parser.add_argument('--t', type=int, default=784, metavar='N',
						help='number of time steps (default: 784)')
	parser.add_argument('--h', type=int, default=212, metavar='N',
						help='number of hidden states (default: 212)')
	parser.add_argument('--m', type=int, default=256, metavar='N',
						help='number of memory states (default: 256)')
	parser.add_argument('--p', type = str, metavar="None|row|psMNIST|psLMU",
						default = 'row', help = 'Permutation for SeqMNIST')
	parser.add_argument('--testmode', type=int, default=1, metavar='N', help='final test control')
	parser.add_argument('--validate', type=int, default=1, help='validate interval')
	parser.add_argument('--plot', action='store_true', default=False, help='whether to plot the training and validate results.')

	#N_t = 784
	args = parser.parse_args()
	return args

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
#	print(f"{permute_file=}")
	perm = torch.load(permute_file).long()
	return perm

def train(DEVICE, model, loader, optimizer, criterion):
	""" A single training epoch on the psMNIST data """
	epoch_loss = 0
	y_pred = []
	y_true = []
	
	torch.cuda.empty_cache()	#mck: just once/epoch
	start = time.time()

	model.train()
	for batch, labels in tqdm(loader):

#		torch.cuda.empty_cache()		#mck: don't do this in inner loop - saved 4.7s/iter (10k)

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

	end = time.time()
	print(f"time:{(end - start):.2f} seconds")

	return avg_epoch_loss, epoch_acc	

def validate(DEVICE, model, loader, criterion):
	""" A single validation epoch on the psMNIST data """
	epoch_loss = 0
	y_pred = []
	y_true = []

	torch.cuda.empty_cache()	#mck: just once/epoch
	start = time.time()
	
	model.eval()
	with torch.no_grad():
		for batch, labels in tqdm(loader):

#			torch.cuda.empty_cache()		#mck: don't do this in inner loop - saved 26.8s/iter(60k)

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
	end = time.time()

	# Accuracy
	epoch_acc = accuracy_score(y_true, y_pred)
	print(f"time:{(end - start):.2f} seconds")

	return avg_epoch_loss, epoch_acc
	
def losses(train_loss:float, train_acc:float, val_loss:float, val_acc:float):
	print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
	print(f"Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%")

def accuracy_scores(train_acc:list, val_acc:list):
	print("Train Losses: ", end='')
	for t_acc in train_acc:
		print(f"{t_acc*100:.2f}%, ", end='')
	print("")	
	print("Val Losses: ", end='')
	for v_acc in val_acc:
		print(f"{v_acc*100:.2f}%, ", end='')
	print("")	

def getSeqMNISTtype(kind:str) -> str:
	valid = {'row', 'psMNIST', 'psLMU'}
	return kind if kind in valid else 'psLMU'

def training(
	device, 
	model, 
	dl_train, dl_val, dl_test,
	optimizer, criterion, 
	N_epochs:int, 
	N_validate:int=1,
	testmode:int=1,
) -> tuple:
	train_losses = []
	train_accs = []
	val_losses = []
	val_accs = []
	val_loss, val_acc = 0, 0
	last_validate = -1

	for epoch in range(N_epochs):
		print(f"Epoch: {epoch+1:02}/{N_epochs:02}")

		train_loss, train_acc = train(device, model, dl_train, optimizer, criterion)

		#validate interval?
		if ((epoch+1) % N_validate) == 0:
			val_loss, val_acc = validate(device, model, dl_val, criterion)
			last_validate = epoch

		train_losses.append(train_loss)
		train_accs.append(train_acc)
		val_losses.append(val_loss)
		val_accs.append(val_acc)

		losses(train_loss, train_acc, val_loss, val_acc)
	#end of epochs
	accuracy_scores(train_accs, val_accs)
	print()

	if testmode == 1:
		print(f"Final test: ({len(dl_test)*dl_test.batch_size})")
		#use full test-set	
		tst_loss, tst_acc = validate(device, model, dl_test, criterion)
		losses(train_loss, train_acc, tst_loss, tst_acc)
		print()

	return train_losses, train_accs, val_losses, val_accs
			