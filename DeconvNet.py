from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Input, ZeroPadding2D

from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt

ds, ds_info = tfds.load('oxford_iiit_pet', with_info=True)
print(ds_info)
#   add a channel dimension cause mnist have a (width, height) shape
#   x_train = x_train[..., tf.newaxis]
#   x_test = x_test[..., tf.newaxis]

#   normalize input
#   x_train, x_test = x_train/255.0, x_test/255.0

#   function to apply on all data using map function
@tf.function
def parse_image(data):
  image = tf.image.convert_image_dtype(data['image'], tf.float32)
  image = tf.image.resize(image, [224, 224])
  mask = tf.image.resize(data['segmentation_mask'], [224, 224])

  #   onde no tensor 'mask' tiver valor maior que 1, a função where decide se vai botar 1(valor desejado) ou continua com o msm valor
  mask -= 1
  mask = tf.where(tf.greater(mask, 1), tf.ones_like(mask), mask)

  return image, mask

#   batch and shuffle the data
train_data = ds['train'].map(parse_image).shuffle(3000).batch(32)

test_data = ds['test'].map(parse_image).batch(32)

for image, mask in train_data.take(1):
  plt.imshow(np.squeeze(image.numpy()[0]))
  plt.title('image')
  plt.show()

  plt.imshow(np.squeeze(mask.numpy()[0]))
  plt.title('mask')
  plt.show()
  break

def conv_block(depth, filters, kernel_size=3):
  convs = []

  for i in range(depth):
    convs += [tf.keras.layers.Conv2D(filters, kernel_size, activation=tf.nn.relu)]
  return convs
  
def _init_classificator():
  layers = [tf.keras.layers.Dense(4096, activation=tf.nn.relu),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(4096, activation=tf.nn.relu),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(1000, activation=tf.nn.softmax)]
  return layers

def _init_feature_extractor():
  layers =  [*conv_block(2, 64), 
              tf.keras.layers.MaxPooling2D(3, 2),
              *conv_block(2, 128),
              tf.keras.layers.MaxPooling2D(),
              *conv_block(3, 256),
              tf.keras.layers.MaxPooling2D(),
              *conv_block(3, 512),
              tf.keras.layers.MaxPooling2D(),
              *conv_block(3, 512),
              tf.keras.layers.MaxPooling2D()]
  return layers

#   create Deconv Net model
class Deconv_Net(tf.keras.Model):
  def __init__(self):
    super(Deconv_Net, self).__init__()

    '''
      DOWNSAMPLING
    '''
    self.conv1_1 = Conv2D(64, 3, padding='same', activation='relu')
    self.conv1_2 = Conv2D(64, 3, padding='same', activation='relu')
    self.max_pool1 = MaxPooling2D(strides=2)

    self.conv2_1 = Conv2D(128, 3, padding='same', activation='relu')
    self.conv2_2 = Conv2D(128, 3, padding='same', activation='relu')
    self.max_pool2 = MaxPooling2D(strides=2)

    self.conv3_1 = Conv2D(256, 3, padding='same', activation='relu')
    self.conv3_2 = Conv2D(256, 3, padding='same', activation='relu')
    self.conv3_3 = Conv2D(256, 3, padding='same', activation='relu')
    self.max_pool3 = MaxPooling2D(strides=2)

    self.conv4_1 = Conv2D(512, 3, padding='same', activation='relu')
    self.conv4_2 = Conv2D(512, 3, padding='same', activation='relu')
    self.conv4_3 = Conv2D(512, 3, padding='same', activation='relu')
    self.max_pool4 = MaxPooling2D(strides=2)
  
    self.conv5_1 = Conv2D(512, 3, padding='same', activation='relu')
    self.conv5_2 = Conv2D(512, 3, padding='same', activation='relu')
    self.conv5_3 = Conv2D(512, 3, padding='same', activation='relu')
    self.max_pool5 = MaxPooling2D(strides=2)

    '''
      'FULLY CONNECTED' LAYERS
    '''
    self.fc_1 = Conv2D(4096, 7, activation='relu')
    self.fc_2 = Conv2D(4096, 1, activation='relu')

    '''
      UPSAMPLING
    '''
    self.upconv1_1 = Conv2DTranspose(512, 7, activation='relu')
    self.upsamp1 = UpSampling2D()

    self.upconv2_1 = Conv2DTranspose(512, 3, padding='same', activation='relu')
    self.upconv2_2 = Conv2DTranspose(512, 3, padding='same', activation='relu')
    self.upconv2_3 = Conv2DTranspose(512, 3, padding='same', activation='relu')
    self.upsamp2 = UpSampling2D()

    self.upconv3_1 = Conv2DTranspose(512, 3, padding='same', activation='relu')
    self.upconv3_2 = Conv2DTranspose(512, 3, padding='same', activation='relu')
    self.upconv3_3 = Conv2DTranspose(256, 3, padding='same', activation='relu')
    self.upsamp3 = UpSampling2D()

    self.upconv4_1 = Conv2DTranspose(256, 3, padding='same', activation='relu')
    self.upconv4_2 = Conv2DTranspose(256, 3, padding='same', activation='relu')
    self.upconv4_3 = Conv2DTranspose(128, 3, padding='same', activation='relu')
    self.upsamp4 = UpSampling2D()

    self.upconv5_1 = Conv2DTranspose(128, 3, padding='same', activation='relu')
    self.upconv5_2 = Conv2DTranspose(64, 3, padding='same', activation='relu')
    self.upsamp5 = UpSampling2D()

    self.upconv6_1 = Conv2DTranspose(64, 3, padding='same', activation='relu')
    self.upconv6_2 = Conv2DTranspose(64, 3, padding='same', activation='relu')

    self.output1 = Conv2DTranspose(21, 1, padding='same', activation='softmax')


  def call(self, inputs):

    x = self.conv1_1(inputs)
    x = self.conv1_2(x)
    x = self.max_pool1(x)
    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.max_pool2(x)
    x = self.conv3_1(x) 
    x = self.conv3_2(x) 
    x = self.conv3_3(x) 
    x = self.max_pool3(x) 
    x = self.conv4_1(x) 
    x = self.conv4_2(x) 
    x = self.conv4_3(x) 
    x = self.max_pool4(x)
    x = self.conv5_1(x) 
    x = self.conv5_2(x) 
    x = self.conv5_3(x) 
    x = self.max_pool5(x)
    x = self.fc_1(x) 
    x = self.fc_2(x) 
    x = self.upconv1_1(x) 
    x = self.upsamp1(x) 
    x = self.upconv2_1(x) 
    x = self.upconv2_2(x) 
    x = self.upconv2_3(x) 
    x = self.upsamp2(x)
    x = self.upconv3_1(x) 
    x = self.upconv3_2(x) 
    x = self.upconv3_3(x) 
    x = self.upsamp3(x) 
    x = self.upconv4_1(x) 
    x = self.upconv4_2(x) 
    x = self.upconv4_3(x) 
    x = self.upsamp4(x) 
    x = self.upconv5_1(x) 
    x = self.upconv5_2(x) 
    x = self.upsamp5(x)
    x = self.upconv6_1(x) 
    x = self.upconv6_2(x) 
    return self.output1(x)
  def model(self):
    x = Input(shape=(224, 224, 3))
    return tf.keras.Model(inputs=[x], outputs=self.call(x))

deconvnet = Deconv_Net()
model = deconvnet.model()
model.summary()

#   define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

#   define avg loss and accuracy for visualize during train
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
  #   training=True is needed if there are layers with different
  #   behavior during training versus test (like dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(image, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 10

for epoch in range(EPOCHS):
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_data.take(1):
    train_step(images, labels)
  
  for images, labels in test_data.take(1):
    test_step(images, labels)
  
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

