# Define the CNN net using tensorflow
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mimg
#import Grandpa_utils as gu # self defined 

class SRnet(object):
    def __init__(self, phase='train'):
        self.lr = 1e-4
        self.input_size32 = 32
        self.batch_size = 32
        self.output_size = 20
        self.kernel_size1 = 9
        self.kernel_size2 = 1
        self.kernel_size3 = 5
        self.image_channel = 3
        self.feature_maps1 = 64
        self.feature_maps2 = 32
        self.scale_factor = 2
        self.phase = phase
        self.tfDataPath = r'g:\Jupyter\ImagePro\Code\Super_Resolution\tfData'
        self.files = list(map(lambda x: self.tfDataPath+'\\'+ x, os.listdir(self.tfDataPath)))
        print(self.files)
        
        self.filename_queue = tf.train.string_input_producer(self.files)
        
        self.reader = tf.TFRecordReader()
        
        _, self.serialized_example = self.reader.read(self.filename_queue)
        
        self.features = tf.parse_single_example(self.serialized_example, features={
        'label': tf.FixedLenFeature([], tf.string), 
        'img_raw': tf.FixedLenFeature([], tf.string)}
        )
        
        self.img = tf.decode_raw(self.features['img_raw'], tf.uint8)
        self.img = tf.reshape(self.img, [16, 16, 3])
        #self.img = tf.map_fn()
        self.label = tf.decode_raw(self.features['label'], tf.uint8)
        self.label = tf.reshape(self.label, [32, 32, 3])
        
        
        self.images, self.labels = tf.train.shuffle_batch([self.img, self.label], batch_size=32, 
                                                         capacity=200, min_after_dequeue=100)
        self.images = tf.image.resize_bicubic(self.images, (self.images.shape[1]*2, self.images.shape[2]*2))
        self.images = tf.cast(self.images, tf.float32) / 255.0 * 2 - 1
        
        self.labels = (tf.cast(self.labels, tf.float32) / 255.0) * 2 - 1
        #self.images = tf.placeholder(tf.float32, [None, None, None, 3])
        
        #self.label = tf.placeholder(tf.float32, [None, self.input_size32, self.input_size32, 3])
        
        self.logits = self.build_network(self.images, self.phase)
    
        self.logits_scale_back = tf.clip_by_value(tf.round((self.logits + 1) / 2 * 255.0), 0, 255) 

        # get the loss.
        self.loss = self.loss_func(self.logits, self.labels)
        
        
    def build_network(self, images, phase):
        # ground truth data placeholder.
        if phase == 'train':
            padding = 'VALID'
        else:
            padding = 'SAME'
        #x_input = tf.placeholder(tf.float32, (None, self.input_size32, self.input_size32, self.image_channel))
        #y = tf.placeholder(tf.float32, (None, self.output_size, self.output_size, self.image_channel))
        #images = tf.image.resize_bicubic(images, [images.shape[1]*self.scale_factor, images.shape[2]*self.scale_factor])
        # layer 1
        with tf.variable_scope('layer_1') as scope:
            W1 = tf.get_variable('layer1_W1', initializer=tf.truncated_normal((self.kernel_size1, self.kernel_size1, self.image_channel, self.feature_maps1), 
                                                                              stddev= 0.001))
            b1 = tf.get_variable('layer1_b1', initializer=tf.constant(0.001))
            #print(W1.shape, b1.shape)
            conv1 = tf.nn.conv2d(images, W1, strides=[1,1,1,1], padding=padding) + b1
            net = tf.nn.relu(conv1)# 24*24
            
        with tf.variable_scope('layer_2') as scope:
            W2 = tf.get_variable('layer2_W2', initializer=tf.truncated_normal((self.kernel_size2, self.kernel_size2, self.feature_maps1, self.feature_maps2), 
                                                                              stddev= 0.001))
            b2 = tf.get_variable('layer2_b2', initializer=tf.constant(0.001))
            conv2 = tf.nn.conv2d(net, W2, strides=[1,1,1,1], padding=padding) + b2
            #tf.nn.batch_normalization()
            net = tf.nn.relu(conv2) # 24*24            
        
        with tf.variable_scope('layer_3') as scope:
            W3 = tf.get_variable('layer3_W3', initializer=tf.truncated_normal((self.kernel_size3, self.kernel_size3, self.feature_maps2, self.image_channel), 
                                                                              stddev= 0.001))
            b3 = tf.get_variable('layer3_b3', initializer=tf.constant(0.001))
            conv3 = tf.nn.conv2d(net, W3, strides=[1,1,1,1], padding=padding) + b3
            net =  conv3 # 20*20
            
        return net
    
    def loss_func(self, logits, y_label):
        '''
        input:
            logits: output of network
            y_label: corresponding X images
            
        output: MSE of inner 20*20 area.
        '''
        
        y_label20 = y_label[:,6:26, 6:26,:]
        #assert logits.shape == [64, 20, 20, 3], 'logits shape wrong'
        #assert y_label20.shape == [64, 20, 20, 3], 'labels shape wrong'
        SE = tf.square(logits - y_label20)
        MSE = tf.reduce_mean(SE)
        
        return MSE