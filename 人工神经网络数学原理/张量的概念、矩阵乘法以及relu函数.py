
# coding: utf-8

# In[24]:

# 张量的概念
import numpy as np
x = np.array(12)
print(x)
print(x.ndim)


# In[6]:


x = np.array([12, 3, 6, 14])
print(x)
print(x.ndim)


# In[7]:


x = np.array([
    [1,2,3,4,5],
    [6,1,8,9,10],
    [11,12,13,14,15]
])

print(x.ndim)


# In[9]:


x = np.array([
    [
        [1,2],
        [3,4]
    ],
    [
        [5,6],
        [7,8]
    ],
    [
        [9,10],
        [11,12]
    ]
])

print(x.ndim)
print(x.shape)


# In[12]:


from keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
my_slice = train_images[10:100]
print(my_slice.shape)


# In[15]:

# relu激活函数 
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy() #确保操作不对x产生影响
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = max (x[i][j], 0)
    return x

x = np.array([
    [1, -1],
    [-2, 1]
])

print(naive_relu(x)) # [[1,0],[0,1]]

    


# In[20]:

# 矩阵的点乘算法
def naive_vector_dot(x,y):
    """assert断言函数，只有命题为真才能执行下一步，如果命题为假则抛出异常"""
    assert len(x.shape) == 1 
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
        
    return z

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(naive_vector_dot(x,y)) # 32.0
    


# In[23]:


def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z

x = np.array([
    [1,2],
    [3,4]
])

y = np.array(
    [5, 6]
)

print(naive_matrix_vector_dot(x,y)) # [17.,39.]

