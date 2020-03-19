import os 
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop


# # include_top=False 只加载卷积层参数,分类层参数未加载
conv_base = VGG16(weights='imagenet',include_top=False,
	input_shape=(150,150,3))
conv_base.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 150, 150, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
"""

# 图片数据的路径
base_dir = 'cats_and_dogs_small'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

# 对导入的数据数据进行处理
datagen = ImageDataGenerator(rescale=1/255.0)
batch_size = 20

def  extract_features(directory,sample_count):
	features = np.zeros(shape=(sample_count,4,4,512))
	labels = np.zeros(shape=(sample_count,))
	generator = datagen.flow_from_directory(directory,
		target_size=(150,150),batch_size=batch_size,
		class_mode='binary')
	i = 0
	for inputs_batch, labels_batch in generator:
		# 把图片放入VGG16卷积层，对图片信息进行抽取
		features_batch = conv_base.predict(inputs_batch)
		# features_batch的结构为4*4*512
		features[i*batch_size:(i+1)*batch_size] = features_batch
		labels[i*batch_size:(i+1)*batch_size] = labels_batch
		i += 1
		if i *batch_size>=sample_count:
			break
		return features,labels	
	pass

# extract_features返回数据的格式为(sample_count,4,4,512)
train_features,train_labels = extract_features(train_dir,2000)
validation_features,validation_labels = extract_features(validation_dir,1000)
test_features,test_labels = extract_features(test_dir,1000)

# 对每条数据进行扁平化处理
train_features = train_features.reshape(2000,-1)
validation_features = validation_features.reshape(1000,-1)
test_features = test_features.reshape(1000,-1)

# 构建自己的网络层对输出的数据分类
model = Sequential()
model.add(Dense(256,activation='relu',input_dim=4*4*512))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=2e-5),
	loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_features,train_labels,epochs=30,
	batch_size=20,validation_data=(validation_features,validation_labels))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='train acc')
plt.plot(epochs,val_acc,'r',label='validation acc')
plt.title('Train and Validation accuracy')
plt.show()

plt.plot(epochs,loss,'bo',label='train loss')
plt.plot(epochs,val_loss,'r',label='validation loss')
plt.title('Train and Validation loss')
plt.show()		

# 对已经训练好的模型参数输入数据训练再调整，使其更符合实际项目需求
model = Sequential()
model.add(conv_base)
model.add(Dense(256,activation='relu',input_dim=4*4*512))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

conv_base.trainable = True
trainable_flag = False
# 一旦读取到'block5_conv1'时，意味着来到卷积网络的最高三层
# 可以使用conv_base.summary()来查看卷积层的信息
for layer in conv_base.layers:
	if layer.name == 'block5_conv1':
		trainable_flag = True
	if trainable_flag:
		# 当trainable == True 意味着该网络层可以更改，
		# 要不然该网络层会被冻结，不能修改
		layer.trainable = True
	else:
		layer.trainable = False

model.summary()
		
#把图片数据读取进来
datagen = ImageDataGenerator(rescale=1/255.0)
train_generator = datagen.flow_from_directory(train_dir,
		target_size=(150,150),batch_size=20,
		class_mode='binary')
validation_generator = datagen.flow_from_directory(validation_dir,
		target_size=(150,150),batch_size=20,
		class_mode='binary')

model.compile(optimizer=RMSprop(lr=2e-5),
	loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_generator,epochs=30,
	batch_size=20,validation_data=validation_generator,
	validation_steps=50)