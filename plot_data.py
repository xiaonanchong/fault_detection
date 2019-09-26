import numpy as np
from matplotlib import pyplot as plt
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("x_path", help="path to store data x", type=str)
parser.add_argument("y_path", help="path to store data y", type=str)
args = parser.parse_args()

x = np.load(args.x_path+'.npy')
y = np.load(args.y_path+'.npy')
'''
L = 2000
l = 200
s1 = np.random.normal(1.0, 2.0, L) 
f1 = np.random.normal(1.0, 2.0, l)
s = s1
f = s
index = np.random.randint(0, L-l)
for j in range(l):
	f[j+index] += f1[j]

plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(s)
plt.plot(f)
plt.legend(['normal signal', 'fault signal'], loc='upper left')
plt.show()
plt.savefig('data_sample.png')