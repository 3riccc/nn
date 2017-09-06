import numpy as np
import matplotlib as plt

import pickle

with open('res/x_origin.pickle','rb') as f1:
	X = pickle.load(f1)
with open('res/xt_origin.pickle','rb') as f2:
	XT = pickle.load(f2)
with open('res/Y.pickle','rb') as f3:
	Y = pickle.load(f3)
with open('res/YT.pickle','rb') as f4:
	YT = pickle.load(f4)

X = np.array(X)
XT = np.array(XT)
Y = np.array(Y)
YT = np.array(YT)


# sigmoid
def sigmoid(z):
	s = 1./(1. + np.exp(-z))
	return s

def ReLU(z):
	if z > 0:
		s = z
	else:
		s = 0
	return s

def initialize_with_zeros(dim):
	w = np.zeros((dim,1))
	b = 0
	return w,b

dim = 2
w,b = initialize_with_zeros(dim)

print(X.shape[1])
