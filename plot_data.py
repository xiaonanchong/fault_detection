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
f1 = np.random.normal(0.4, 2.0, l)
s = s1
f = s
index = np.random.randint(0, L-l)
for j in range(l):
	f[j+index] += f1[j]

print('check:',f==s)
	
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(s, 'b.')
plt.subplot(2,2,2)
plt.plot(f, 'r.')
plt.subplot(2,2,3)
plt.plot(f, 'r.', s, 'b.', )
plt.savefig('data.png')

'''
plt.title("nomral signal") 
plt.plot(s)
plt.savefig('normal_data_sample.png')

plt.title("fault signal") 
plt.plot(f)
plt.savefig('fault_data_sample.png')

plt.plot(f)
plt.plot(s)
plt.legend(['fault signal', 'normal signal'])
plt.savefig('data_sample.png')
'''
