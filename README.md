# fault_detection  
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
