"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Fri Mar 3 17:44:29 2023

@author: Manny Ko.
"""
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

#from datasets.flowers import flowers
import datasets.utils.projconfig as projconfig
from datasets.mnist import mnist
from datasets.utils.xforms import GreyToFloat

from mk_mlutils.dataset import datasetutils

from src.lmu2 import *
from src.lmuapp import *

kPlot=False


def compare2datasets(dataset1, dataset2, kLabelsOnly=False) -> bool:
	print(f"compare2datasets({getattr(dataset1, 'name', 'unknown')}, {getattr(dataset2, 'name', 'unknown')})")
	result = True
	for idx, entry1 in enumerate(dataset1):
		entry2 = dataset2[idx]
		result &= (entry1[1] == entry2[1]) 	#comparse label

		if not kLabelsOnly:
			pix1 = entry1[0].numpy()
			pix2 = entry2[0] #/255.
			if not (np.isclose(pix1, pix2, atol=1e-5).all()):
				print(f"[{idx}]: failed")
				return False
	return True


if __name__ == "__main__":
	mnist_dir = projconfig.getMNISTFolder()	#"/content/"
	transform = transforms.ToTensor()
	mnist_train = datasets.MNIST(mnist_dir, train = True, download = True, transform = transform)
	mnist_val   = datasets.MNIST(mnist_dir, train = False, download = True, transform = transform)

	#1: use lmu's psMNIST
	perm = load_permutation(__file__)
	ds_train = psMNIST(mnist_train, perm)
	ds_val   = psMNIST(mnist_val, perm) 
	psmnist = ds_val

	#2: use our SeqMNIST(permute='psLMU') which loads the same 'permutation.pt'
	seqmnist = mnist.SeqMNIST(split="test", permute='psLMU', imagepipeline=GreyToFloat())

	img0, lab0 = seqmnist[0]
	print("Label:", lab0, "dtype", img0.dtype)
	if kPlot: dispSeq(img0)

	eg_img, eg_label = psmnist[0]
	print("Label:", eg_label, "dtype", eg_img.numpy().dtype)
	if kPlot: dispSeq(eg_img.numpy())

	print(img0.shape, eg_img.shape)

	pix1 = eg_img.numpy()
	pix2 = img0 #/255.
	assert(np.isclose(pix1, pix2).all())

	#3: compare the 2 datasets entry by entry
	compare2datasets(psmnist, seqmnist, kLabelsOnly=False)

	ds_val = datasetutils.getBalancedSubset(mnist_val, fraction=.2, useCDF=True)

