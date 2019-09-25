import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import numpy.matlib
import random
import matplotlib.pyplot as plt

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
	
	sigma = 1.1
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

x_trian = random.shuffle(x_train)
history = model.fit(x_train, y_train, validation_split=0.33, epochs=1, batch_size=32)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("train.png")
