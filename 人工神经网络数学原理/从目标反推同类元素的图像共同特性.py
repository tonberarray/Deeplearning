
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

""" sigmoid 函数的反函数y=ln[x/(1-x)],
在scipy函数库中对应为special.logit(x)"""

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
		# 激活函数sigmoid的反函数
		self.inverse_activation = lambda x: special.logit(x)

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

	def backpredict(self,targets_list):
		# 将结果向量装置以便反推输入信号,
		# 整个过程为predict()方法的反向运算
		final_outputs  = np.array(targets_list,ndmin=2).T
		final_inputs = self.inverse_activation(final_outputs)
		hiddenoutputs = np.dot(self.who.T,final_inputs)

		# 将信号量规整到（0,1）之间
		hiddenoutputs -= np.min(hiddenoutputs)
		hiddenoutputs /= np.max(hiddenoutputs)
		hiddenoutputs *= 0.98
		hiddenoutputs += 0.01

		hiddeninputs = self.inverse_activation(hiddenoutputs)
		inputs = np.dot(self.wih.T, hiddeninputs)
		# 将信号量规整到（0,1）之间
		inputs -= np.min(inputs)
		inputs /= np.max(inputs)
		inputs *= 0.98
		inputs += 0.01

		return inputs
		pass


# 神经网络实例化
inputnodes = 784
hiddennodes = 100
outputnodes = 10
learnrate = 0.3
network = Network(inputnodes,hiddennodes,outputnodes,learnrate)

# 神经网络训练
# 加入epochs，设置网络训练的循环次数
epochs = 10 
file_name = 'mnist_test.csv'	
with open(file_name, 'r',encoding='utf-8') as f:
	train_data = f.readlines()
for e in range(epochs):
	for image in train_data:
		values = image.split(',')
		inputs = (np.asfarray(values[1:]))/255.0*0.99 + 0.01
		targets = np.zeros(outputnodes) + 0.01
		targets[int(values[0])] = 0.99
		network.train(inputs,targets)

# 反向观测相同元素的图像共性
label = 6
targets =np.zeros(outputnodes)+0.01
targets[label] = 0.99
print(targets)
image_data = network.backpredict(targets)
image_array = image_data.reshape(28,28)

plt.figure()
# plt.imshow(image_array,cmap='Greys', interpolation="None")
plt.imshow(image_array,cmap='gray')
plt.show()