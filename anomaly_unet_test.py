import numpy as np
import os
import math
import cv2

from keras.models import load_model

batch_size = 1
height = 128
width = 128

channels = 3
lr = 1e-7

#
def get_input_data():
    #     print 'Sending'
    imgs = './UMN_neuro/Test/'
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
#


input = np.asarray([rv for rv, rl in get_input_data()]).reshape((3866,128,128,3))
output = np.asarray([rl for rv, rl in get_input_data()]).reshape((3866,128,128,1))

np.save('total_input_unm.npy',input)
np.save('total_output_unm.npy',output)

pretrained_model = load_model('./weights/unet-unm-5.hdf5')

print(pretrained_model.summary())

# model = unet(pretrained_weights = pretrained_weights)
pred = pretrained_model.predict(input)
prediction = pred.reshape((3866,128,128))
output = output.reshape((3866,128,128))

prediction = prediction.clip(0,255)
prediction[(prediction>110)&(prediction<=140)] = 125
prediction[(prediction<=110)&(prediction>=0)] = 0
prediction[(prediction>140)&(prediction<=255)] = 255
prediction = prediction.astype('uint8')

error = np.zeros((3866,1))
for i in range(3866):
    dif = np.linalg.norm(np.subtract(prediction[i,:,:],output[i,:,:]),'fro')
    error[i,0] = dif
    # dif = np.floor(dif).astype('uint8')
    # cv2.imwrite('./unm_images/train_dif_images/'+str(i).zfill(3)+'.jpg',dif.reshape((128,128)))

np.save('unm_total_error.npy',error)
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit(input,output,batch_size=40, epochs=10,validation_split=0.1,callbacks=[model_checkpoint])

# model.save('unet-model-10.h5')
