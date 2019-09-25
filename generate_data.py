import numpy as np
import math

# generate 2*N training data
N = 2
L = 2000


s1 = np.random.normal(10, 20, N) 
s2 = np.random.normal(5, 20, N)
s3 = np.random.normal(20, 2, N)

s = s1 + s2 + s3

'''
x_train = np.empty((N*2, L))
y_train = np.empty((N*2, 1))
for i in range(N):

	s1 = math.sqrt(20) * np.matlib.randn(L) + 10
	s2 = math.sqrt(20) * np.matlib.randn(L) + 5
	s3 = math.sqrt(2) * np.matlib.randn(L) + 20
	print(x1)
	
	sigma = 2
	mu = 0
	x2 = sigma * np.matlib.randn(L) + mu
	print(x2)
	
	x_train[2*i] = x1
	x_train[2*i+1] = x2
	y_train[2*i] = 1
	y_train[2*i+1] = 0
'''