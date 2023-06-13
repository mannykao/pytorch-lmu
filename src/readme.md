"""
Title: PyTorch implementation of

	"Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks" [Voelker 2019]
&	""	
	
Created on Fri Mar 3 17:44:29 2023

@author: Manny Ko.
"""

lmu:

lmu2: extracted from the beginning of lmu_fft_psmnist.py. 
	  Same as src/lmu.py with 'psmnist' extensions.

lmuapp: shared application level code for both the Sequential-LMU (lmu_psmnist) and fft-LMU (lmu_fft_psmnist).


lmu_psmnist.py - Sequential-LMU, torch port of [Voelker 2019]:
--------------------------------------------------------------
KerasLMU: https://github.com/abr/neurips2019
log: output/torch-lmu-train<n>.png
	 output/output.log

lmu_fft_psmnist - Parallel LMU torch port of [Chilkuri]
-------------------------------------------------------
[Parallelizing Legendre Memory Unit Training](https://arxiv.org/abs/2102.11417), by Chilkuri N and Eliasmith C

log: output/lmu-fft-train<n>.png
	 output/output.log

