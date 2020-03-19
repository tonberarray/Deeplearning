
import numpy as np
from scipy import special
# import matplotlib.pyplot as plt

# 生成(3,3)形状的数值在[-0.5,0.5]之间的矩阵
array = np.random.rand(3,3) - 0.5
print(array)

l = [[1,2],[3,4]]
print("origin l is {0}".format(l))
ll = np.array(l, ndmin=2)
print(ll) #[[1 2]
#			[3 4]]
print(ll.T)

class Network(object):
	def __init__(self,inputnodes, hiddennodes,outputnodes, learnrate):
		# 初始化神经网络，设置输入层，中间层，输出层的节点数,
		# 学习率,激活函数
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes		
		#设置神经网络的学习率
		self.learnrate = learnrate
		"""初始化神经网络各节点的权重值，网络有三层，
		需要设置两个权值矩阵，wih是输入层到中间层的权值矩阵
		who是中间层到输出层的权值矩阵"""
		self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
		self.who = np.random.rand(self.onodes, self.hnodes) - 0.5	
		# 设置激活函数sigmoid
		self.activation = lambda x: special.expit(x)

	def	train(self,inputs_list,targets_list):
		# 根据输入的训练数据调整神经网络各个连接点的之间的权值
		# 先将传入的数据转换成numpy能处理的数组
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T
		
		# 模型训练
		# 计算中间层从输入层接收到的信号量	
		hiddeninputs = np.dot(self.wih,inputs)
		# 计算中间层经过激活函数后输出的信号量
		hiddenoutputs = self.activation(hiddeninputs)
		# 计算输出层中间层接收到的信号量	
		final_inputs = np.dot(self.who,hiddenoutputs)
		# 计算输出层经过激活函数后输出的信号量
		final_outputs = self.activation(final_inputs)
		# 计算误差
		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)
		# 根据误差量调整各层网络节点上的权值
		self.who += self.learnrate * np.dot((output_errors*final_outputs*(1-final_outputs)),
											np.transpose(hiddenoutputs))
		self.wih += self.learnrate * np.dot((hidden_errors*hiddenoutputs*(1-hiddenoutputs)),
											np.transpose(inputs))
		pass

	def predict(self,inputs):
		# 将训练的模型用于预测
		# 计算中间层从输入层接收到的信号量	
		hiddeninputs = np.dot(self.wih, inputs)
		# 计算中间层经过激活函数后输出的信号量
		hiddenoutputs = self.activation(hiddeninputs)
		# 计算输出层中间层接收到的信号量	
		final_inputs = np.dot(self.who, hiddenoutputs)
		# 计算输出层经过激活函数后输出的信号量
		final_outputs = self.activation(final_inputs)

		return final_outputs
		pass

	def fit(self,train_data,epochs,validation_data):
		# 神经网络训练
		for e in range(epochs):
			print("epoch:  {0}/{1}".format(e+1,epochs))
			score = []	
			for image in train_data:
				values = image.split(',')
				inputs = (np.asfarray(values[1:]))/255.0
				targets = np.zeros(self.onodes)
				targets[int(values[0])] = 1
				self.train(inputs,targets)
	
			# 验证训练的神经网络
			for image in validation_data:
				values = image.split(',')
				correct_number = int(values[0])
				# print('图片对应的数字为:', correct_number)
				inputs = (np.asfarray(values[1:]))/255.0
				outputs = self.predict(inputs)
				label = np.argmax(outputs)
				# print('output result is:', label)
				if label == correct_number:
					score.append(1)
				else:
					score.append(0)	
			test_size = len(score)
			correct_sum = score.count(1)
			print("test data size is ",test_size)
			print("the correct sum is ",correct_sum)		
			accuracy = correct_sum / test_size
			print("accuracy: ",accuracy)
		pass

def main():
	# 神经网络初始参数设置
	inputnodes = 784
	hiddennodes = 100
	outputnodes = 10
	learnrate = 0.3
	network = Network(inputnodes,hiddennodes,outputnodes,learnrate)

	# 设置网络训练的循环次数
	epochs = 7 
	# 加载数据
	file_name = 'mnist_train.csv'	
	with open(file_name, 'r',encoding='utf-8') as f1:
		train_data = f1.readlines()
	file = 'mnist_test.csv'
	with open(file, 'r',encoding='utf-8') as f2:
		validation_data = f2.readlines()	
	# 神经网络训练
	network.fit(train_data, epochs, validation_data)
	
if __name__ == '__main__':
	main()

"""
[[ 0.0883626   0.01804596 -0.28455273]
 [-0.47562868 -0.36569163  0.01197926]
 [ 0.17577157  0.10377199  0.42572511]]
origin l is [[1, 2], [3, 4]]
[[1 2]
 [3 4]]
[[1 3]
 [2 4]]
epoch:  1/7
test data size is  10000
the correct sum is  9384
accuracy:  0.9384
epoch:  2/7
test data size is  10000
the correct sum is  9527
accuracy:  0.9527
epoch:  3/7
test data size is  10000
the correct sum is  9592
accuracy:  0.9592
epoch:  4/7
test data size is  10000
the correct sum is  9576
accuracy:  0.9576
epoch:  5/7
test data size is  10000
the correct sum is  9647
accuracy:  0.9647
epoch:  6/7
test data size is  10000
the correct sum is  9620
accuracy:  0.962
epoch:  7/7
test data size is  10000
the correct sum is  9622
accuracy:  0.9622
[Finished in 618.6s]
"""
