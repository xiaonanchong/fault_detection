import numpy as np
import math
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("x_path", help="path to store data x", type=str)
parser.add_argument("y_path", help="path to store data y", type=str)
parser.add_argument('sample_num', help='number of samples', type=int)
args = parser.parse_args()

# generate 2*N training data
N = args.sample_num
L = 2000
l = 200

x = np.empty((N*2, L))
y = np.empty((N*2, 2))
for i in range(N):
	s1 = np.random.normal(1.0, 2.0, L) 
	s2 = np.random.normal(0.5, 2.0, L)
	s3 = np.random.normal(2.0, 2.0, L)
	f0 = np.random.normal(1.4, 3.0, l)

	s = (s1+s2+s3)
	f1 = np.zeros(L)
	index = np.random.randint(0, L-l)
	for j in range(l):
		f1[j+index] += f0[j]
	f = (s1+s2+s3)+f1

	print('check:',np.array_equal(f,s))
	
	x[2*i] = s
	x[2*i+1] = f
	y[2*i] = [1,0]
	y[2*i+1] = [0,1]

# shuffle data
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

np.save(args.x_path+'.npy', x)
np.save(args.y_path+'.npy', y)
