# fault_detection  
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	conv1d_1 (Conv1D)            (None, 497, 512)          8704
	_________________________________________________________________
	max_pooling1d_1 (MaxPooling1 (None, 248, 512)          0
	_________________________________________________________________
	conv1d_2 (Conv1D)            (None, 121, 256)          1048832
	_________________________________________________________________
	max_pooling1d_2 (MaxPooling1 (None, 60, 256)           0
	_________________________________________________________________
	conv1d_3 (Conv1D)            (None, 57, 128)           131200
	_________________________________________________________________
	max_pooling1d_3 (MaxPooling1 (None, 28, 128)           0
	_________________________________________________________________
	conv1d_4 (Conv1D)            (None, 27, 64)            16448
	_________________________________________________________________
	max_pooling1d_4 (MaxPooling1 (None, 13, 64)            0
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 832)               0
	_________________________________________________________________
	dense_1 (Dense)              (None, 32)                26656
	_________________________________________________________________
	dense_2 (Dense)              (None, 2)                 66
	=================================================================
![accuracy](https://github.com/xiaonanchong/fault_detection/blob/master/train_acc.png)
![loss](https://github.com/xiaonanchong/fault_detection/blob/master/train_loss.png)

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
