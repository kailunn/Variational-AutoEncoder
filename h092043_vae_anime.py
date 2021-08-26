

import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image


   
    
#img = cv2.imread(r'C:\Users\test\Desktop\DL hw2\VAE\VAE_dataset\data\3.png')
#print(img.shape)

#######################################################################
# load data
#######################################################################
# C:\Users\test\Desktop\DL hw2\VAE\VAE_dataset


############################################################################################
# Data & model configuration
#img_width, img_height = input_train.shape[1], input_train.shape[2]
batch_size = 128
no_epochs = 1
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 3

data =np.load(r"C:\Users\test\Desktop\DL hw2\VAE\VAE_dataset\animation.npz")  #image/label
img = np.array(data['arr_0']) #['image']  # shape = 22551,64,64,3
#label = data['label'] # 12000
#ret, img_b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # img_b : binary processed
#print(img.shape)
img_b =np.reshape(img, (21551, 64, 64, 3))
input_train = img_b

img_width, img_height = input_train.shape[1], input_train.shape[2]

'''
# Reshape data
input_train = input_train.reshape(input_train.shape[0], img_height, img_width, num_channels)
#input_test = input_test.reshape(input_test.shape[0], img_height, img_width, num_channels)
input_shape = (img_height, img_width, num_channels)
'''
# Parse numbers as floats
input_train = input_train.astype('float32')
#input_test = input_test.astype('float32')

# Normalize data
#input_train = input_train / 255
#input_test = input_test / 255



###################################################################
#Encoder
###################################################################

# Definition
i = Input(shape=(64,64,3), name='encoder_input')
cx = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(i)
cx = BatchNormalization()(cx)
cx = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
cx = BatchNormalization()(cx)
x = Flatten()(cx)
x = Dense(20, activation='relu')(x)
x = BatchNormalization()(x)
mu = Dense(latent_dim, name='latent_mu')(x)
sigma = Dense(latent_dim, name='latent_sigma')(x)

# Get Conv2D shape for Conv2DTranspose operation in decoder
conv_shape = K.int_shape(cx)

# Define sampling with reparameterization trick
def sample_z(args):
  mu, sigma = args
  batch = K.shape(mu)[0]
  dim = K.int_shape(mu)[1]
  eps = K.random_normal(shape=(batch, dim))
  return mu + K.exp(sigma / 2) * eps

# Use reparameterization trick to ....??
z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])

# Instantiate encoder
encoder = Model(i, [mu, sigma, z], name='encoder')
#encoder.summary()

####################################################################
# Decoder
####################################################################

# Definition
d_i = Input(shape=(latent_dim, ), name='decoder_input')
x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
x = BatchNormalization()(x)
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
cx = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
cx = BatchNormalization()(cx)
cx = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
cx = BatchNormalization()(cx)
o = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)

# Instantiate decoder
decoder = Model(d_i, o, name='decoder')
#decoder.summary()

##############################################################################
# Variational AutoEncoder
##############################################################################

# Instantiate VAE
vae_outputs = decoder(encoder(i)[2])
vae = Model(i, vae_outputs, name='vae')
#vae.summary()


##############################################################################
# Define loss ###ELOBO
##############################################################################

def kl_reconstruction_loss(true, pred):
  # Reconstruction loss
  reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
  
  # KL divergence loss
  kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
  kl_loss = K.sum(kl_loss, axis=-1) ##### KL divergence #分布相似low
  kl_loss *= -0.5
  # Total loss = 50% rec + 50% KL divergence loss
  
  return K.mean(reconstruction_loss + kl_loss) #kl_loss


# Compile VAE
vae.compile(optimizer='adam', loss=kl_reconstruction_loss)


###########################################################
# Train variational AutoEncoder
#vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_split = validation_split)
###########################################################
history = vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_split = validation_split)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

###########################################################






pred = vae.predict(input_train)

print(pred.shape)





########################################################################################
# plot results
########################################################################################

#============================
#real
#============================
for i in range(1000,1064):
      img = input_train[i+1, ]
      plt.subplot(8,8,i-1000+1)
      plt.imshow(img, cmap= 'gray')

plt.show()


#============================
#reconstructed
#============================

for i in range(1000,1064):
      img = pred[i+1, ]
      plt.subplot(8,8,i-1000+1)
      plt.imshow(img, cmap= 'gray')

plt.show()





