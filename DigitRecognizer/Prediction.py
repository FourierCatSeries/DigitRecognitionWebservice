# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import cv2
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
USER_IMAGE = 'user_image'
# load a singal image file from user input image file name. the input image has to be put in the same directory as this script
# return both the gray scale image and the original image
def load_input_image(image_name):
      original_image = cv2.imread(image_name)
      # convert the image into grayscale
      gray_scale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
      # format the image to mnist style
      resized_image = cv2.resize((gray_scale), (28,28), interpolation=cv2.INTER_AREA)
      ## testing code
      cv2.imwrite(os.path.join('test', image_name), resized_image)
      ##
      reshaped_image = np.reshape(resized_image, (1,784))
      return reshaped_image, original_image
# recieve the image file object passed from flask webservice, the image_file_str is the string data of the image file
def recieve_image(image_file_str, img_name):
      np_img = np.fromstring(image_file_str, np.uint8)
      original_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR) 
      gray_scale = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
      # format the image to mnist style
      resized_img = cv2.resize((gray_scale), (28,28), interpolation=cv2.INTER_AREA)
      ## reshape it for feeding into the mnist trained deepCNN model
      reshaped_img = np.reshape(resized_img, (1,784))
      return reshaped_img, original_img

def save_user_image(image, image_name, dir = USER_IMAGE):
      if not os.path.exists(dir):
         os.mkdir(dir)
      # get the full path of file to save
      image_path = os.path.join(dir, image_name)
      cv2.imwrite(image_path, image)
      return image_path

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def Predict(tf_sess, y_conv, x, x_img, keep_prob):
      output = tf_sess.run(y_conv, feed_dict = {x : x_img, keep_prob : 1})
      predicted_label = np.argmax(output)
      for value in output:
          print(str(value) , ' ')
      return predicted_label

class DigitRecognizer():
      
  def __init__(self):
      self.y_ = tf.placeholder(tf.float32, [None, 10])
      self.x = tf.placeholder(tf.float32, [None, 784])
      self.keep_prob = None
      self.y_conv = None
      self.y_conv, self.keep_prob = deepnn(self.x)
      self.model_dir = ""
      self.model_name = ""
      self.tf_sess = None
  def Load_Model(self, model_dir, model_name):
        self.model_dir = model_dir
        self.model_name = model_name
        parameter_saver = tf.train.Saver()
        self.tf_sess = tf.Session()
        self.tf_sess.run(tf.global_variables_initializer())
        model_path = os.path.join(self.model_dir, self.model_name)
        parameter_saver.restore(self.tf_sess, model_path)
        print("Model restored from %s." % model_path)
        return 
  def Predict_Label(self, x_img):
        output = self.tf_sess.run(self.y_conv, feed_dict = {self.x : x_img, self.keep_prob : 1})
        predicted_label = np.argmax(output)
        return predicted_label
              
  # not exactly know how tf.session works, can the loading procedure be seperated from prediction?(under different tf.session?)
  def Load_and_Predict(self, x_img, model_dir, model_name):
        self.model_dir = model_dir
        self.model_name = model_name
        parameter_saver = tf.train.Saver()
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          model_path = os.path.join(self.model_dir, self.model_name)
          parameter_saver.restore(sess, model_path)
          print("Model restored from %s." % model_path)
          output = sess.run(self.y_conv, feed_dict = {self.x : x_img, self.keep_prob : 1})
          predicted_label = np.argmax(output)
        for value in output:
            print(str(value) , ' ')
        return predicted_label

      
def main(input_image, model_dir, model_name):
  # get client input image of hand written digit.
  x_image, original_image = load_input_image(input_image)

  # saving the original image (will be replaced by cassandra module)
  saved_path = save_user_image(original_image, input_image)
  print('User input image has been saved as %s' % saved_path)
     

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  # create a saver to save and restore all the parameters 
  parameter_saver = tf.train.Saver()
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, model_dir, model_name)
    parameter_saver.restore(sess, model_path)
    print("Model restored from %s." % model_path)
    # make prediction
    predict_label = Predict(sess, y_conv, x, x_image, keep_prob)
    
    print('The user input hand written digit is recognized as: %i' % predict_label)
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--input_image', default = '8.png', type = str)
  parser.add_argument('--model_dir', default = 'trained_models', type = str)
  parser.add_argument('--model_name', default = 'deepCNN.ckpt', type = str)

  args = parser.parse_args()
  main(args.input_image, args.model_dir, args.model_name)
