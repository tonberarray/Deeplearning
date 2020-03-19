import os, shutil
# shutil python一个用于对文件，压缩包等进行操作的模块 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


# 数据包被解压的路径
original_dir = 'train'
# 构建一个专门的数据存储路径
base_dir = 'cats_and_dogs_small'
os.makedirs(base_dir,exist_ok=True)
# 构造路径存储训练数据，校验数据以及测试数据
train_dir = os.path.join(base_dir,'train')
os.makedirs(train_dir,exist_ok=True)
test_dir = os.path.join(base_dir,'test')
os.makedirs(test_dir,exist_ok=True)
validation_dir = os.path.join(base_dir,'validation')
os.makedirs(validation_dir,exist_ok=True)

# 构造专门存储猫图片的路径，用于训练网络
train_cats_dir = os.path.join(train_dir,'cats')
os.makedirs(train_cats_dir,exist_ok=True)
# 构造专门存储狗图片的路径，用于训练网络
train_dogs_dir = os.path.join(train_dir,'dogs')
os.makedirs(train_dogs_dir,exist_ok=True)

# 构造专门存储猫图片的路径，用于校验
valid_cats_dir = os.path.join(validation_dir,'cats')
os.makedirs(valid_cats_dir,exist_ok=True)
# 构造专门存储狗图片的路径，用于校验
valid_dogs_dir = os.path.join(validation_dir,'dogs')
os.makedirs(valid_dogs_dir,exist_ok=True)

# 构造专门存储猫图片的路径，用于测试
test_cats_dir = os.path.join(test_dir,'cats')
os.makedirs(test_cats_dir,exist_ok=True)
# 构造专门存储狗图片的路径，用于测试
test_dogs_dir = os.path.join(test_dir,'dogs')
os.makedirs(test_dogs_dir,exist_ok=True)

# 把前1000张猫图片复制到训练路径
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src= os.path.join(original_dir,fname)
	dst= os.path.join(train_cats_dir,fname)
	shutil.copyfile(src, dst)
# 把接着的500张猫图片复制到校验路径
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
	src= os.path.join(original_dir,fname)
	dst= os.path.join(valid_cats_dir,fname)
	shutil.copyfile(src, dst)
# 把接着的500张猫图片复制到测试路径
fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
	src= os.path.join(original_dir,fname)
	dst= os.path.join(test_cats_dir,fname)
	shutil.copyfile(src, dst)

# 把前1000张狗图片复制到训练路径
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src= os.path.join(original_dir,fname)
	dst= os.path.join(train_dogs_dir,fname)
	shutil.copyfile(src, dst)
# 把接着的500张狗图片复制到校验路径
fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
	src= os.path.join(original_dir,fname)
	dst= os.path.join(valid_dogs_dir,fname)
	shutil.copyfile(src, dst)
# 把接着的500张狗图片复制到测试路径
fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
	src= os.path.join(original_dir,fname)
	dst= os.path.join(test_dogs_dir,fname)
	shutil.copyfile(src, dst)
	
print('total train cat images:',len(os.listdir(train_cats_dir)))  
print('total validation cat images:',len(os.listdir(valid_cats_dir))) 
print('total test cat images:',len(os.listdir(test_cats_dir)))
print('total train dog images:',len(os.listdir(train_dogs_dir)))
print('total validation dog images:',len(os.listdir(valid_dogs_dir)))
print('total test dog images:',len(os.listdir(test_dogs_dir)))


model = Sequential()
model.add(Conv2D(32,(3,3),activation ='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation ='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation ='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation ='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
# model.add(Dropout(0.6))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=1e-4),
	loss='binary_crossentropy',
	metrics=['accuracy'])
model.summary()


# 把像素点的值除以255，使之在0到1之间
train_datagen = ImageDataGenerator(rescale= 1./255)
valid_datagen = ImageDataGenerator(rescale= 1./255)

# generator 实际上是将数据批量读入内存，使得代码能以for in 的方式去方便的访问
train_generator = train_datagen.flow_from_directory(train_dir,
	target_size=(150,150),batch_size=20,class_mode='binary')
# class_mode 让每张读入的图片对应一个标签值，
# 我们上面一下子读入20张图片，因此还附带着一个数组(20, ),
#标签数组的具体值没有设定，由我们后面去使用
valid_generator = valid_datagen.flow_from_directory(validation_dir,
	target_size=(150,150),batch_size=20,class_mode='binary')

for data_batch, labels_batch in train_generator:
	print(data_batch.shape)
	print(labels_batch.shape)
	print(labels_batch)
	break

history = model.fit_generator(train_generator,steps_per_epoch=100,
	epochs=13,validation_data=valid_generator,validation_steps=50)
print(history.history.keys())
model.save('cats_and_dogs_small.h5')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'b',label='train acc')
plt.plot(epochs,val_acc,'r',label='validation acc')
plt.title('Train and Validation accuracy')
plt.show()

plt.plot(epochs,loss,'b',label='train loss')
plt.plot(epochs,val_loss,'r',label='validation loss')
plt.title('Train and Validation loss')
plt.show()