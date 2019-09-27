import numpy as np
from matplotlib import pyplot as plt

L = 2000
l = 200

s1 = np.random.normal(1.0, 2.0, L)
s2 = np.random.normal(0.5, 2.0, L)
s3 = np.random.normal(2.0, 2.0, L) 
f0 = np.random.normal(0.4, 2.0, l)

s = (s1+s2+s3)
f1 = np.zeros(L)
index = np.random.randint(0, L-l)
for j in range(l):
	f1[j+index] += f0[j]
f = (s1+s2+s3)+f1

print('check:',np.array_equal(f,s))
	
k1=-15
k2=20

plt.figure(1)
plt.subplot(2,2,1)
plt.ylim((k1,k2))
plt.plot(s, 'b.')

plt.subplot(2,2,2)
plt.ylim((k1,k2))
plt.plot(f, 'r.')

plt.subplot(2,2,3)
plt.ylim((k1,k2))
plt.plot(f, 'r.', s, 'b.', )

plt.subplot(2,2,4)
plt.ylim((k1,k2))
plt.plot(f1, 'g.')

plt.savefig('data.png')
