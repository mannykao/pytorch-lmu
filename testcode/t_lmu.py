"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Feb 1 17:44:29 2023

@author: Manny Ko.
"""
import torch
from ssmutils.lmu.fftlmu import lmu, lmufft


if __name__ == "__main__":
	#test sequence copied from README.md

	model = lmu.LMU(
		input_size = 1,
		hidden_size = 212,
		memory_size = 256,
		theta = 784
	)
	print(model)

	x = torch.rand(100, 784, 1) # [batch_size, seq_len, input_size]
	output, (h_n, m_n) = model(x)
	#print(h_n, m_n)

	lmufft = lmufft.LMUFFT(
		input_size = 1,
		hidden_size = 346,
		memory_size = 468, 
		seq_len = 784, 
		theta = 784
	)
	print(lmufft)

	x = torch.rand(100, 784, 1) # [batch_size, seq_len, input_size]
	output, h_n = model(x)
	
	#
	# now you are ready to try to run examples/lmu_psmnist and lmu_fft_psmnist
	#
	