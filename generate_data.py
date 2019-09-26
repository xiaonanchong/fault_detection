import numpy as np
import math
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("x_path", help="path to store data x", type=str)
parser.add_argument("y_path", help="path to store data y", type=str)
parser.add_argument('sample_num', help='number of samples', type=int)
args = parser.parse_args()

# generate 2*N training data
N = args['sample_num']/2
L = 2000
l = 200

x = np.empty((N*2, L))
y = np.empty((N*2, 1))
for i in range(N):
	s1 = np.random.normal(10, 20, L) 
	s2 = np.random.normal(5, 20, L)
	s3 = np.random.normal(20, 2, L)
	f1 = np.random.normal(4, 20, l)

	s = s1 #+ s2 + s3
	
	f = s
	index = np.random.randint(0, L-l)
	for j in range(l):
		f[j+index] += f1[j]
	
	x[2*i] = s
	x[2*i+1] = f
	y[2*i] = 1
	y[2*i+1] = 0

# shuffle data
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

np.save(args['x_path']+'.npy', x)
np.save(args['y_path']+'.npy', y)
