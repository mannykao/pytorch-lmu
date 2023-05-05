

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from datasets.flowers import flowers
from datasets.mnist import mnist

from src.lmu2 import *
from src.lmuapp import *

kPlot=False


if __name__ == "__main__":
	transform = transforms.ToTensor()
	mnist_train = datasets.MNIST("/content/", train = True, download = True, transform = transform)
	mnist_val   = datasets.MNIST("/content/", train = False, download = True, transform = transform)

	#1: use lmu's psMNIST
	perm = load_permutation(__file__)
	ds_train = psMNIST(mnist_train, perm)
	ds_val   = psMNIST(mnist_val, perm) 

	#2: use our SeqMNIST(permute='psLMU') which loads the same 'permutation.pt'
	seqmnist = mnist.SeqMNIST(split="train", permute='psLMU')
	img0, lab0 = seqmnist[0]
	print("Label:", lab0)
	if kPlot: dispSeq(img0)

	eg_img, eg_label = ds_train[0]
	print("Label:", eg_label)
	if kPlot: dispSeq(eg_img)

	print(img0.shape, eg_img.shape)

	pix1 = eg_img.numpy()
	pix2 = img0/255.
	assert(np.isclose(pix1, pix2).all())

	for idx, entry1 in enumerate(ds_train):
		entry2 = seqmnist[idx]
		assert(entry1[1] == entry2[1])

		pix1 = entry1[0].numpy()
		pix2 = entry2[0]/255.
		if not (np.isclose(pix1, pix2, atol=1e-5).all()):
			print(f"[{idx}]: failed")

