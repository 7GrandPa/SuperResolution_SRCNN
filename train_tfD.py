import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
#import matplotlib.image as mimg
from datetime import datetime
from SRNet_tfD import SRnet
from Image_Set_tfD import Image_set
# def train solver class.
class Solver(object):
    def __init__(self, data, net):
        self.batch_size = 128
        self.lr = 0.0002
        self.data = data
        self.net = net
        self.max_iter = 1000000
        self.model_path = r'g:/Jupyter/ImagePro/Code/Super_Resolution/Model/'
        self.log_dir = r'g:/Jupyter/ImagePro/Code/Super_Resolution/Log/'
        self.global_step = 0
        #self.saver = tf.train.Saver(self.global_step, max_to_keep=None)
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.net.loss)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        #self.sess.run(tf.local_variables_initializer())
        #self.sess.run(tf.initialize_all_variables())
        
        self.saver = tf.train.Saver()
        
        self.sum_op = self.summary_func()
        
    def summary_func(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.net.loss)
            #tf.summary.scalar('accuracy', self.accuracy)
            #tf.summary.histogram('hist_loss', self.loss)
        sum_op = tf.summary.merge_all()
        return sum_op
    
    def train(self):
        
        # 1. data load
        #print('get_data')
        #labels, images = self.data.get_sub_images(self.scale_factor)
        #print(labels.shape, images.shape)
        
        #sb_t = time.time()
        #images = self.sess.run(
            #tf.image.resize_bicubic(images, [images.shape[1]*self.scale_factor, images.shape[2]*self.scale_factor]))
        #print(labels.shape, images.shape)
        #eb_t = time.time()
        
        #print('Data obtained, after %.4f seconds of bicubic' % (eb_t - sb_t))
        
        time_stamp = '{0:%Y-%m-%dT%H-%M-%S/}'.format(datetime.now())
        graph_path = self.log_dir + time_stamp
        writer = tf.summary.FileWriter(graph_path, self.sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for iters in range(self.max_iter):
            #self.sess.run(self.net.iterator.initializer())
            # random choose index
            self.global_step = self.global_step + 1
            #idx = np.random.randint(0,images.shape[0], self.batch_size)
            #images_feed = self.sess.run(tf.image.resize_bicubic(images[idx,:,:,:], 
                                                                #[images.shape[1]*self.scale_factor, images.shape[2]*self.scale_factor]))
            #feed_dict = {self.net.images: images[idx,:,:,:] / 255.0 * 2 - 1,
                         #self.net.label : labels[idx,:,:,:] / 255.0 * 2 - 1} 
                         
            train_t = time.time()
            loss, _, summary = self.sess.run([self.net.loss, self.train_op, self.sum_op])
            train_et = time.time()
            #train_et = time.time()
            writer.add_summary(summary, global_step=self.global_step)
            if (iters) % 1000 == 0:
                
                print('After iter %d, with lr is %.7f, loss is: %.7f, time cost %.4f' % (iters, self.lr, loss, train_et-train_t))
                
            if iters % 3000 ==0:
                self.lr = self.lr*0.9
                self.saver.save(self.sess, self.model_path+'srnet.ckpt', global_step=self.global_step)
        coord.request_stop()
        coord.join(threads)
        
        
def main():
    data = Image_set()
    net = SRnet()    
    
    solver = Solver(data, net)
    print('Start training...')
    solver.train()
    print('Done training!')
    
if __name__ == '__main__':
    main()