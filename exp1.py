import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import numpy.matlib

# generate 2*N training data
N = 1000
L = 2000

x_train = np.empty((N*2, L))
y_train = np.empty((N*2, 1))
for i in range(N):
	sigma = 1
	mu = 0
	x1 = sigma * np.matlib.randn(L) + mu
	print(x1)
	
	sigma = 2
	mu = 0
	x2 = sigma * np.matlib.randn(L) + mu
	print(x2)
	
	x_train[2*i] = x1
	x_train[2*i+1] = x2
	y_train[2*i] = 1
	y_train[2*i+1] = 0

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=L))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32)
