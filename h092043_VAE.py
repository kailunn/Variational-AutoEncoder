import numpy as np
import pandas as pd
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns 
import cv2
import tensorflow
from keras.layers.normalization import BatchNormalization

#############################################################
#data pre-process
# binary

data =np.load(r"C:\Users\test\Desktop\DL hw2\VAE\VAE_dataset\TibetanMNIST.npz")  #image/label
img = data['image']  # shape = 12000,28,28
label = data['label'] # 12000
ret, img_b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # img_b : binary processed
img_b =np.reshape(img_b, (12000, 28, 28, 1))


##############################################################
#Encoder
from keras.models import Model
##############################################################
img_input = layers.Input(shape = (28, 28, 1))
encoder = layers.Conv2D(64, (5,5), activation='relu')(img_input)
encoder = layers.MaxPooling2D((2,2))(encoder)
encoder = layers.Conv2D(64, (3,3), activation='relu')(encoder)
encoder = layers.MaxPooling2D((2,2))(encoder)
encoder = layers.Conv2D(32, (3,3), activation='relu')(encoder)
encoder = layers.MaxPooling2D((2,2))(encoder)
encoder = layers.Flatten()(encoder)
encoder = layers.Dense(16)(encoder)


##############################################################
#find mean and variace
##############################################################

def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tensorflow.shape(distribution_variance)[0]
    random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
    return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random

distribution_mean = layers.Dense(2, name='mean')(encoder)
distribution_variance = layers.Dense(2, name='log_variance')(encoder)
latent_encoding = layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])

encoder_model = tensorflow.keras.Model(img_input, latent_encoding)
#encoder_model.summary()



##################################################################
#Decoder
##################################################################
#inputs of decoder is the outputs fron encoder => checkout the shape of the last layer(laten space)? 
'''

decoder_input = tensorflow.keras.layers.Input(shape=(2))
decoder = tensorflow.keras.layers.Dense(64)(decoder_input)
decoder = tensorflow.keras.layers.Reshape((1, 1, 64))(decoder)
decoder = tensorflow.keras.layers.Conv2DTranspose(64, (3,3), activation='relu')(decoder)

decoder = tensorflow.keras.layers.Conv2DTranspose(64, (3,3), activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling2D((2,2))(decoder)

decoder = tensorflow.keras.layers.Conv2DTranspose(64, (3,3), activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling2D((2,2))(decoder)

decoder_output = tensorflow.keras.layers.Conv2DTranspose(1, (5,5), activation='relu')(decoder)

decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
decoder_model.summary()
'''
# =================
# Decoder
# =================

# Definition
latent_dim = 2
num_channels = 1

d_i   = layers.Input(shape=(latent_dim, ), name='decoder_input')
x     = layers.Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
x     = BatchNormalization()(x)
x     = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
cx    = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
cx    = BatchNormalization()(cx)
cx    = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
cx    = BatchNormalization()(cx)
o     = layers.Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
# Instantiate decoder
decoder = Model(d_i, o, name='decoder')
decoder.summary()




'''
input_de = layers.Input(shape=(2))

decoder_dense_layer1 = layers.Dense(units=numpy.prod(shape_before_flatten), name="decoder_dense_1")(input_de)
decoder_reshape = layers.Reshape(target_shape=shape_before_flatten)(decoder_dense_layer1)
decoder_conv_tran_layer1 = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=1, name="decoder_conv_tran_1")(decoder_reshape)
decoder_norm_layer1 = layers.BatchNormalization(name="decoder_norm_1")(decoder_conv_tran_layer1)
decoder_activ_layer1 = layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)

decoder_conv_tran_layer2 = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2, name="decoder_conv_tran_2")(decoder_activ_layer1)
decoder_norm_layer2 = layers.BatchNormalization(name="decoder_norm_2")(decoder_conv_tran_layer2)
decoder_activ_layer2 = layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)

decoder_conv_tran_layer3 = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2, name="decoder_conv_tran_3")(decoder_activ_layer2)
decoder_norm_layer3 = layers.BatchNormalization(name="decoder_norm_3")(decoder_conv_tran_layer3)
decoder_activ_layer3 = layers.LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)

decoder_conv_tran_layer4 = layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), padding="same", strides=1, name="decoder_conv_tran_4")(decoder_activ_layer3)
decoder_output = layers.LeakyReLU(name="decoder_output")(decoder_conv_tran_layer4 )
decoder = Model(input_de, decoder_output, name="decoder_model")

##################################################################
#encoded = encoder_model(img_input)
#decoded = decoder_model(encoded)

'''
