import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Cropping2D, concatenate, BatchNormalization, Input
from __future__ import absolute_import

def unet(input_shape, n_classes):
  inputs = Input(shape = input_shape)
  x = inputs

  conv_blocks2 =    [[Conv2D(64, 3, padding='same', activation='relu'), 
                      Conv2D(64, 3, padding='same',activation='relu')],  
                   
                    [Conv2D(128, 3,  padding='same', activation='relu'),
                      Conv2D(128, 3,  padding='same', activation='relu')],
                   
                    [Conv2D(256, 3,  padding='same', activation='relu'),
                      Conv2D(256, 3,  padding='same', activation='relu')],
                     
                    [Conv2D(512, 3,  padding='same', activation='relu'),
                      Conv2D(512, 3,  padding='same', activation='relu')],
                     
                    [Conv2D(1024, 3, padding='same', activation='relu'),
                      Conv2D(1024, 3, padding='same', activation='relu')]]


  " downsampling: "
  conv_blocks =    [[Conv2D(64, 3,  padding='same', activation='relu'), 
                      Conv2D(64, 3,  padding='same', activation='relu')],  
                   
                    [Conv2D(128, 3,  padding='same', activation='relu'),
                      Conv2D(128, 3,  padding='same', activation='relu')],
                   
                    [Conv2D(256, 3,  padding='same', activation='relu'),
                      Conv2D(256, 3,  padding='same', activation='relu')],
                     
                    [Conv2D(512, 3,  padding='same', activation='relu'),
                      Conv2D(512, 3,  padding='same', activation='relu')],
                     
                    [Conv2D(1024, 3, padding='same', activation='relu'),
                      Conv2D(1024, 3, padding='same', activation='relu')]]

  " UpSampling: "
  upconvs =       [Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu'),
                    Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu'),
                    Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
                    Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')]


  copy = []
  for block in conv_blocks:
    for conv in block:
      x = conv(x)
    if block == conv_blocks[-1]:
      break
    copy.append(x)
    x = MaxPooling2D()(x)

  conv_blocks2.reverse()
  copy.reverse()
  for i, block in enumerate(conv_blocks2[0:-1]):
    x = upconvs[i](x)
    x = concatenate([x, copy[i]])
    for conv in block:
      x = conv(x)

  x = Conv2D(n_classes, 1, activation='sigmoid')(x)
  return tf.keras.Model(inputs=inputs, outputs=x)







