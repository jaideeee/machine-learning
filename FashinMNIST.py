import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

"""
downloaded from : https://www.kaggle.com/zalando-research/fashionmnist 
classify images into following category.

0. t-shirt
1. trouser
2. pullover
3. dress
4. coat
5. sandal
6. shirt
7. sneaker
8. Bag
9. Ankle Boot

cnn : 
1) conv2d (3,3) feature map of 32 layers 
2) max pooling of (2,2)
3) Flatten  
4) Dense
5) Dropout
6) softmax output

result train set => 93%
test set 91%
"""
def read_file():
    """
    in windows sometime direct paths dont work , then use os. path .join
    """
    fashion_mnist_dir = r'C:\Users\lenovo\Downloads\fashionmnist'
    test_file = 'fashion-mnist_test.csv'
    train_file = 'fashion-mnist_train.csv'
    train_path = os.path.join(fashion_mnist_dir, train_file)
    test_path = os.path.join(fashion_mnist_dir,test_file)
    return train_path, test_path

#1.read file
df_train = pd.read_csv(r'C:\Users\lenovo\Downloads\fashionmnist\fashion-mnist_train.csv')
df_test = pd.read_csv(r'C:\Users\lenovo\Downloads\fashionmnist\fashion-mnist_test.csv')
#2. convert to numpy array of float32 type.
train_data = np.array(df_train,dtype='float32')
test_data = np.array(df_test,dtype='float32')
#3. seperate data and labels.
X_train = train_data[:,1:]/255
y_train = train_data[:,0]
X_test =test_data[:,1:]/255
y_test = test_data[:,0]
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
# import random
# print(y_train[2])
# plt.imshow(X_train[2].reshape(28,28))
# plt.show()


from sklearn.model_selection import train_test_split
X_train,X_validate, y_train , y_validate = train_test_split(X_train,y_train,test_size=.2,random_state=0)
print('train',X_train.shape)
print('test',X_test.shape)
print('validate',X_validate.shape)


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20, batch_size=512,verbose=1, validation_data=(X_validate,y_validate))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('loss', test_loss ,' test_acc' , test_acc)