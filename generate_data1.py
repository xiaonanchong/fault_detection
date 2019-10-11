import numpy as np
import math
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('x_path', help='path to store data x', type=str)
parser.add_argument('y_path', help='path to store data y', type=str)
parser.add_argument('sample_num', help='number of samples to generate', type=int)
args=parser.parse_args()

N = args.sample_num
L = 2000
l = 200

x = np.empty((N*2, L))
y = np.empty((N*2, 2))

for i in range(N):
	noise = np.random.normal(0.0, 0.2, L)
	signal1 = np.array([math.sin(k-0.3) for k in range(L)])
	signal2 = np.array([math.sin(2*k)*1.2 for k in range(L)])
	fault_singal = [math.cos(10*k)*0.23 for k in range(l)]
	
	s = (signal1 + signal2 + noise)
	
	f1 = np.zeros(L)
	index = np.random.randint(0,L-l)
	for j in range(l):
		f1[j+index] += fault_singal[j]
	f = (signal1 + signal2 + noise)+ f1
	
	x[2*i] =s
	x[2*i+1] =f
	y[2*i] =[1,0]
	y[2*i+1] =[0,1]
	
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

np.save(args.x_path+'.npy', x)
np.save(args.y_path+'.npy', y)
