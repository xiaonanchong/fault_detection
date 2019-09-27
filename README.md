# fault_detection  

python generate_data.py x y 1000

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

python generate_data.py x2 y2 500  

for i in range(N):

	s1 = np.random.normal(1.0, 2.0, L) 
	s2 = np.random.normal(0.5, 2.0, L)
	s3 = np.random.normal(2.0, 2.0, L)
	f1 = np.random.normal(1.0, 2.0, l)

	s = s1 #+ s2 + s3
	
	f = s
	index = np.random.randint(0, L-l)
	for j in range(l):
		f[j+index] += f1[j]

![accuracy](https://github.com/xiaonanchong/fault_detection/blob/master/train_acc.png)
![loss](https://github.com/xiaonanchong/fault_detection/blob/master/train_loss.png)


	model = Sequential()
	model.add(Conv1D(filters=1024, kernel_size=4, strides=1, padding='valid', activation='relu')) 
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) 
	model.add(Conv1D(filters=512, kernel_size=8, strides=2, padding='valid', activation='relu')) 
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) 
	model.add(Conv1D(filters=64, kernel_size=16, strides=4, padding='valid', activation='relu')) 
	model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid')) 
	model.add(Flatten())
	model.add(Dense(1), activation='sigmoid')
669/669 [==============================] - 336s 503ms/step - loss: 7.0664 - accuracy: 0.5007 - val_loss: 8.1808 - val_accuracy: 0.4924
Epoch 2/100  
669/669 [==============================] - 343s 513ms/step - loss: 7.9988 - accuracy: 0.5037 - val_loss: 8.1808 - val_accuracy: 0.4924
Epoch 3/100  
