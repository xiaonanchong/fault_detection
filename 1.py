import tensorflow as tf
import keras
import numpy as np
import math
import random
from keras.models import Sequential
from keras.layers import Dense, Activation

# generate training data
N = 3 #1000
L = 20 #2000

x_train = []
y_train = []
for i in range(N):
	r1 = random.gauss(0,0.5)
	r2 = random.gauss(0,0.5)
	x1 = [math.sin(i*r1 + r2) for i in range(L)]
	print(x1)
	
	noise = [random.uniform(0,1) for i in range(L)]
	x2 = x1 + noise
	
	x_train.append(x1)
	x_train.append(x2)
	y_train.append([1])
	y_train.append([0])
	
print(x_train)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=L))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32)