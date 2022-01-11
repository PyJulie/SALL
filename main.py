from keras.layers import Conv2D
from keras.layers import BatchNormalization
from spp.SpatialPyramidPooling import SpatialPyramidPooling
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Input, Dense, Dropout, Activation, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
import keras.backend as K
from loss.loss import *
from loss.metirc import *
import os
import glob
import random
import numpy as np
from scipy import misc
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.generator import generator

dropout_rate = 0.5
img_size = (299,299)
task_categories = [5, 4, 2]
T = 3
batch_size = 32

input_image = Input(shape=(img_size[0],img_size[1],3))
base_model = InceptionV3(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')
base_model.summary()

input_image = Input(shape=(img_size[0],img_size[1],3))
base_model = InceptionV3(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')
output = base_model.output
output1 = Dense(task_categories[0],name='dense_1')(Dropout(dropout_rate)(output))
output2 = Dense(task_categories[1],name='dense_2')(Dropout(dropout_rate)(output))
output3 = Dense(task_categories[2],name='dense_3')(Dropout(dropout_rate)(output))
output1 = Lambda(lambda x: x/T)(output1)
output2 = Lambda(lambda x: x/T)(output2)
output3 = Lambda(lambda x: x/T)(output3)
output1 = Activation('softmax')(output1)
output2 = Activation('softmax')(output2)
output3 = Activation('softmax')(output3)
output = [output1,output2,output3]
model = Model(inputs=input_image, outputs=output)
optimizer = Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss=[knowledge_distillation_loss_1,knowledge_distillation_loss_2], metrics=[knowledge_distillation_acc_1,knowledge_distillation_acc_2])
model.summary()


base_image_dir = './data'
files = ['normal','NPDRI','NPDRII','NPDRIII','PDR','small_drusen','big_drusen','CNV','arter']
dic = {
# task 1
 '0':'normal',
 '1': 'NPDRI',
 '2':'NPDRII',
 '3':'NPDRIII',
 '4':'PDR',
 # task 2
 '5':'small_drusen',
 '6':'big_drusen',
 '7':'CNV',
 # task 3
 '8':'arter'
}


train_samples = []
test_samples = []
for file in files:
    sub_path = os.path.join(base_image_dir,file)
    for _ in os.listdir(sub_path)[:int(0.75*len(os.listdir(sub_path)))]:
        train_samples.append(_)
        
for file in files:
    sub_path = os.path.join(base_image_dir,file)
    for _ in os.listdir(sub_path)[int(0.75*len(os.listdir(sub_path))):]:
        test_samples.append(_)

np.random.shuffle(train_samples)
np.random.shuffle(test_samples)

dic_image = {}
with open('pretrain/299_softmax_ground_truth.txt','r') as f:
    while True:
        image_name = f.readline()
        if not image_name:
            break
        label = f.readline()
        dic_image[image_name[:-1]] = label[:-1]


train_generator = data_generator(train_samples,batch_size,base_image_dir,dic,dic_image)
test_generator = data_generator(test_samples,batch_size,base_image_dir,dic,dic_image)


best_weights_filepath = 'checkpoints/299_yes_multi_task_t_100.hdf5'
earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1, mode = 'auto')
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = False)

history = model.fit_generator(test_generator,steps_per_epoch=len(train_samples)/batch_size,epochs=batch_size,
	validation_data=test_generator, validation_steps=len(test_samples)/batch_size, callbacks=[earlyStopping, saveBestModel])