from keras.models import load_model
from keras import models
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# 把训练的网络加载进来
model = load_model('cats_and_dogs_small.h5')
# model.summary()

"""
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
"""

img_path = 'cat.739.jpg'
# 加载图片设置大小为150*150
img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
# 像素点取值变换到[0,1]之间
img_tensor/=255.
print(img_tensor.shape)
plt.figure()
plt.imshow(img_tensor[0])
plt.show()


'''
我们把网络的前8层，也就是含有卷积和max pooling的网络层抽取出来，
下面代码会把前八层网络的输出结果放置到数组layers_outputs中
'''
layer_outputs =[layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input,outputs=layer_outputs)
#执行下面代码后，我们能获得卷积层和max pooling层对图片的计算结果
activations = activation_model.predict(img_tensor)
#我们把第一层卷积网络对图片信息的识别结果绘制出来
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.figure()
plt.matshow(first_layer_activation[0,:,:,7],cmap='viridis')
plt.show()

layer_names = []
for layer in model.layers[:8]:
	layer_names.append(layer.name)

images_per_row = 16	
for layer_name,layer_activation in zip(layer_names,activations):
	# layer_activation的结构为(1, width, height, array_len)
    # 向量中的元素个数
    n_features = layer_activation.shape[-1]
    # 获得切片的宽和高
    size = layer_activation.shape[1]
    # 在做卷积运算时，我们把图片进行3*3切片，
    # 然后计算出一个含有32个元素的向量，
    # 这32个元素代表着网络从3*3切片中抽取的信息
    # 我们把这32个元素分成6列，绘制在一行里
    n_cols = n_features // images_per_row
    display_grid =np.zeros((size*n_cols,images_per_row*size))
    for col in range(n_cols):
    	for row in range(images_per_row):
    		channel_image = layer_activation[0,:,:,col*images_per_row + row]
    		# 这32个元素中，不一定每个元素对应的值都能绘制到界面上，
    		# 所以我们对它做一些处理，使得它能画出来
    		channel_image -= channel_image.mean()
    		channel_image /= channel_image.std()
    		channel_image *= 64
    		channel_image += 128
    		channel_image = np.clip(channel_image,0,255).astype('uint8')
    		display_grid[col*size:(col+1)*size,row*size:(row+1)*size] = channel_image

    scale =1./size
    plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)	
    plt.imshow(display_grid,aspect='auto',cmap='viridis')
    plt.savefig('{}.png'.format(layer_name),dpi=515)
    # plt.show()	
    