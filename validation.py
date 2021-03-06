import tensorflow as tf
import numpy as np
import os
#import matplotlib.image as mimg
#import cv2
#import matplotlib.pyplot as plt
import time
import scipy.misc
from SRNet import SRnet
from Image_Set import Image_set
from PIL import Image
import argparse
import shutil
# def train solver class.
class Solver(object):
    def __init__(self, net):
        self.batch_size = 128
        #self.lr = 0.0001
        #self.data = data
        self.net = net
        #self.max_iter = 1000000
        self.model_path = r'g:/Jupyter/ImagePro/Code/Super_Resolution/Model/'
        
        #self.global_step = tf.train.create_global_step()
        
        #self.saver = tf.train.Saver(self.global_step, max_to_keep=None)
        #self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.net.loss)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
    # inference
    def inference(self, inf_img):
        #ckpt = tf.train.get_checkpoint_state(self.model_path)
        #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)         
        #output_img = self.net.build_network(inf_img)
        bstart_t = time.time()
        #inf_img_bicu = self.sess.run(tf.image.resize_bicubic(inf_img, [inf_img.shape[1]*2, inf_img.shape[2] * 2]))
        inf_img_bicu = inf_img
        bend_t = time.time()
        feed_dict = {self.net.images: inf_img_bicu/255.0*2 -1}
        output_img = self.sess.run(self.net.logits_scale_back, feed_dict=feed_dict)
        inf_t = time.time()
        
        #b_t = bend_t - bstart_t
        #inf_t = inf_t - bend_t
        print('with %d imgs, bucubic time is %.4f, and inference time is %.4f' %(inf_img.shape[0], bend_t-bstart_t, inf_t-bend_t))
        return output_img
    
    def PSNR(self, orig_img, new_image):
        mse = np.average(np.square(orig_img-new_image))
        #print(mse)
        
        max_value = 2**8-1
        #print(max_value / np.sqrt(mse))
        psnr = 20 * np.log10(max_value / np.sqrt(mse))
        
        return psnr

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=r'g:\Jupyter\ImagePro\Code\Super_Resolution\Test\Set5',type=str)
    parser.add_argument('--phase', default='test',type=str)
    args = parser.parse_args()
    data_path = args.data_path
    phase = args.phase
    #args.test_path
    
    data = Image_set(data_path, phase)
    net = SRnet(phase)
    solver = Solver(net)
    
    inf_img_orig, inf_img_blur = data.get_sub_images(2)
    print(len(inf_img_blur))
    #print(inf_img_blur.shape)
    print('Start evaluating...')
    if phase == 'test' and os.path.exists('g:/Jupyter/ImagePro/Code/Super_Resolution/outTest/'):
        shutil.rmtree('g:/Jupyter/ImagePro/Code/Super_Resolution/outTest/')
    
    for i in range(len(inf_img_blur)):
        
        with tf.Session() as sess:
            
            inf_img_bb = sess.run(tf.image.resize_bicubic(
                np.expand_dims(inf_img_blur[i], axis=0), [inf_img_blur[i].shape[0]*2, inf_img_blur[i].shape[1]*2]))
            print(inf_img_bb.shape)
            
            gen_img = np.array(solver.inference(inf_img_bb), np.uint8)
            
            if phase == 'validation' and data_path.endswith('5'):
                output_dir = 'outSet5'
            elif phase == 'validation' and data_path.endswith('14'):
                output_dir = 'outSet14'
            else:
                output_dir = 'outTest'
                if not os.path.exists('g:/Jupyter/ImagePro/Code/Super_Resolution/'+ output_dir+'/'):
                    os.mkdir('g:/Jupyter/ImagePro/Code/Super_Resolution/'+ output_dir+'/')

            scipy.misc.imsave('g:/Jupyter/ImagePro/Code/Super_Resolution/'+ output_dir+'/%d_orig.png' % i, inf_img_orig[i])
            scipy.misc.imsave('g:/Jupyter/ImagePro/Code/Super_Resolution/'+ output_dir+'/%d_post.png' % i, gen_img[0])
            scipy.misc.imsave('g:/Jupyter/ImagePro/Code/Super_Resolution/'+ output_dir+'/%d_before.png' % i, inf_img_bb[0])
            try:
                print(solver.PSNR(inf_img_orig[i], gen_img[0]))
            except:
                pass

    print('Done evaluating!')
    
    
    
    #print(gen_img)
    
    #plt.imshow(gen_img[0])
    #plt.show()
 
if __name__ == '__main__':
    main()