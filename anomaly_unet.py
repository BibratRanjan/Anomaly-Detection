import numpy as np
import os
import math
import cv2
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, merge, UpSampling2D 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from time import time

batch_size = 1
height = 128
width = 128

channels = 3
lr = 1e-7


def get_input_data():
    #     print 'Sending'
    imgs = './UMN_neuro/Train/'
    folders = [v for v in os.listdir(imgs)]
    for f in folders:
         flist = [v for v in sorted(os.listdir(os.path.join(imgs, f)))]
         batches = math.floor((len(flist) - channels) / batch_size)
         for b in range(batches):
             rv = np.zeros((batch_size, height, width, channels), np.float32)
             rl = np.zeros((batch_size, height, width, 1))
             path = os.path.join(imgs, f)
             for i in range(batch_size):
                rv[i,..., 0] = cv2.cvtColor(cv2.imread(path +'/'+flist[b * batch_size + i]),cv2.COLOR_BGR2GRAY)
                rv[i, ..., 1] = cv2.cvtColor(cv2.imread(path +'/'+flist[b * batch_size + i + 1]),cv2.COLOR_BGR2GRAY)
                rv[i, ..., 2] = cv2.cvtColor(cv2.imread(path +'/'+flist[b * batch_size + i + 2]),cv2.COLOR_BGR2GRAY)
                rl[i, ..., 0] = cv2.cvtColor(cv2.imread(path +'/'+flist[b * batch_size + i + 3]),cv2.COLOR_BGR2GRAY)
             yield rv, rl



#input = np.asarray([rv for rv, rl in get_input_data()]).reshape((2826,128,128,3))
#output = np.asarray([rl for rv, rl in get_input_data()]).reshape((2826,128,128,1))
#
#np.save('unm_train_input',input)
#np.save('unm_train_output',output)
             
input = np.load('unm_train_input.npy')
output = np.load('unm_train_output.npy')             

adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)


def unet(pretrained_weights = None,input_size = (128,128,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    model = Model(input = inputs, output = conv10)

    return model

pretrained_model = load_model('./weights/unet-unm-4.hdf5')

#model = unet()
#model.compile(optimizer = adam, loss = 'mse', metrics = ['mae'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model_checkpoint = ModelCheckpoint('./weights/unet-unm-5.hdf5', monitor='loss',verbose=1, save_best_only=True)
pretrained_model.fit(input,output,batch_size=40, epochs=100,validation_split=0.1,callbacks=[model_checkpoint, tensorboard])




