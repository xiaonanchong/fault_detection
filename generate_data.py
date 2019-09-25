import numpy as np
import math

# generate 2*N training data
N = 10000
L = 2000
l = 200

x = np.empty((N*2, L))
y = np.empty((N*2, 1))
for i in range(N):
	s1 = np.random.normal(10, 20, L) 
	s2 = np.random.normal(5, 20, L)
	s3 = np.random.normal(20, 2, L)
	f1 = np.random.normal(4, 20, l)

	s = s1 + s2 + s3
	
	f = s
	index = random.randint(0, L-l)
	for j in range(l):
		f[j+index] += f1[index]
	
	x[2*i] = s
	x[2*i+1] = f
	y[2*i] = 1
	y[2*i+1] = 0

np.save('x.npy', x)
np.save('y.npy', y)
