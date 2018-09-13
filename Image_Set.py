import numpy as np
#import matplotlib.image as mimg
#import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import psutil
#import scipy.

class Image_set(object):
    def __init__(self, phase='train', scale_factor=2):
        self.train_image_path = r'g:/Jupyter/ImagePro/Code/Super_Resolution/Train/'
        #self.test_image_path = r'g:/Jupyter/ImagePro/Code/Super_Resolution/Test/'
        self.test_set_5_path = r'g:\Jupyter\ImagePro\Code\Super_Resolution\Test\Set5'
        self.test_set_14_path = r'g:\Jupyter\ImagePro\Code\Super_Resolution\Test\Set14'
        self.cache_path = r'g:/Jupyter/ImagePro/Code/Super_Resolution/Cache'
        self.phase = phase
        self.strides = 14
        self.image_size = 32
        self.image_size2 = 33
        self.batch_size = 32
        self.alpha = 0.5
        self.gau_size = 3
        self.gau_sigma = 1.5
        self.scale_factor = 3 # 3,4
       
    # get sub-images of size 32*32 for upscaling factor [2, 4] and 33*33 for upscaling factor 3    
    def get_sub_images(self, factor=2):
        if self.phase == 'train':
        # 1st to find if there are cache files. if yes, then load it. else gen it.
            cache_file32_X = self.cache_path+r'\sub_images_'+str(self.image_size)+'_'+self.phase+'X'+'.npy'
            cache_file33_X = self.cache_path+r'\sub_images_'+str(self.image_size2)+'_'+self.phase+'X'+'.npy'
            cache_file32_Y_2 = self.cache_path+r'\sub_images_'+str(self.image_size)+'_'+self.phase+'Y2'+'.npy'
            cache_file32_Y_4 = self.cache_path+r'\sub_images_'+str(self.image_size)+'_'+self.phase+'Y4'+'.npy'
            cache_file33_Y_3 = self.cache_path+r'\sub_images_'+str(self.image_size2)+'_'+self.phase+'Y3'+'.npy'        
            if os.path.isfile(cache_file32_X) and os.path.isfile(cache_file33_X):
                if factor == 2:
                    print('data exist')
                    #32x
                    sub_images32_X = np.load(cache_file32_X)
    
                    sub_images32_Y_2 = np.load(cache_file32_Y_2)
    
                    return sub_images32_X, sub_images32_Y_2
                if factor == 4:
                    #32x
                    sub_images32_X = np.load(cache_file32_X)
    
                    #32y4
                    sub_images32_Y_4 = np.load(cache_file32_Y_4)
        
                    return sub_images32_X, sub_images32_Y_4
                if factor == 3:
                    # 33x
                    sub_images33_X = np.load(cache_file33_X)
    
                    #33y3
                    sub_images33_Y_3 = np.load(cache_file33_Y_3)
                    print('33y3 get')
                    return sub_images33_X, sub_images33_Y_3
                print('Loading data from cache file, done.')
                #return sub_images32_X, sub_images32_Y_2, sub_images32_Y_4, sub_images33_X, sub_images33_Y_3
            else:
                
                print('Start generate sub_images from train images.')
                sub_images32_X, sub_images32_Y_2, sub_images32_Y_4, sub_images33_X, sub_images33_Y_3 = self.gen_train_sub_images(self.train_image_path)
                #sub_images32_Y, sub_images33_Y = self.
                
                print('Gen sub_images successfully, and save it to cache...')
                if os.path.exists(self.cache_path):
                    np.save(cache_file32_X, sub_images32_X)
                    np.save(cache_file32_Y_2, sub_images32_Y_2)
                    np.save(cache_file32_Y_4, sub_images32_Y_4)
                    np.save(cache_file33_X, sub_images33_X)
                    np.save(cache_file33_Y_3, sub_images33_Y_3)

                    print('Cache saved.')
                else:
                    print('creating cache dirs, and save data to cache...')
                    os.mkdir(self.cache_path)                    
                    np.save(cache_file32_X, sub_images32_X)
                    np.save(cache_file32_Y_2, sub_images32_Y_2)
                    np.save(cache_file32_Y_4, sub_images32_Y_4)
                    np.save(cache_file33_X, sub_images33_X)
                    np.save(cache_file33_Y_3, sub_images33_Y_3)
                    print('Cache saved.')
                if factor == 2:
                    return sub_images32_X, sub_images32_Y_2
                if factor == 4:
                    return sub_images32_X, sub_images32_Y_4
                if factor == 3:
                    return sub_images33_X, sub_images33_Y_3
                #return sub_images32_X, sub_images32_Y_2, sub_images32_Y_4, sub_images33_X, sub_images33_Y_3
            
        else:# for self.phase == 'test'
            test_cache_file32_X = self.cache_path+r'\sub_images_'+str(self.image_size)+'_'+self.phase+'X'+'.npy'
            test_cache_file33_X = self.cache_path+r'\sub_images_'+str(self.image_size2)+'_'+self.phase+'X'+'.npy'
            test_cache_file32_Y_2 = self.cache_path+r'\sub_images_'+str(self.image_size)+'_'+self.phase+'Y2'+'.npy'
            test_cache_file32_Y_4 = self.cache_path+r'\sub_images_'+str(self.image_size)+'_'+self.phase+'Y4'+'.npy'
            test_cache_file33_Y_3 = self.cache_path+r'\sub_images_'+str(self.image_size2)+'_'+self.phase+'Y3'+'.npy'            
            if os.path.exists(test_cache_file32_X):
                print('gen image from test set')
                if factor == 2:
                    ret_orig_img = np.load(test_cache_file32_X)
                    ret_down_img = np.load(test_cache_file32_Y_2)
                    return ret_orig_img, ret_down_img
                elif factor == 3:
                    ret_orig_img = np.load(test_cache_file33_X)
                    ret_down_img = np.load(test_cache_file33_Y_3)
                    return ret_orig_img, ret_down_img
                elif factor == 4:
                    ret_orig_img = np.load(test_cache_file32_X)
                    ret_down_img = np.load(test_cache_file32_Y_4)
                    return ret_orig_img, ret_down_img
                else:
                    print('pls enter correct factor')
                print('gen done')
            else:
                print('Start generate sub_images from test images.')
                sub_images32_X, sub_images32_Y_2, sub_images32_Y_4, sub_images33_X, sub_images33_Y_3 = self.gen_train_sub_images(self.test_set_14_path)            
                
                print('Gen sub_images successfully, and save it to cache...')
                if os.path.exists(self.cache_path):
                    np.save(test_cache_file32_X, sub_images32_X)
                    np.save(test_cache_file32_Y_2, sub_images32_Y_2)
                    np.save(test_cache_file32_Y_4, sub_images32_Y_4)
                    np.save(test_cache_file33_X, sub_images33_X)
                    np.save(test_cache_file33_Y_3, sub_images33_Y_3)

                    print('Cache saved.')
                else:
                    print('creating cache dirs, and save data to cache...')
                    os.mkdir(self.cache_path)                    
                    np.save(test_cache_file32_X, sub_images32_X)
                    np.save(test_cache_file32_Y_2, sub_images32_Y_2)
                    np.save(test_cache_file32_Y_4, sub_images32_Y_4)
                    np.save(test_cache_file33_X, sub_images33_X)
                    np.save(test_cache_file33_Y_3, sub_images33_Y_3)
                    print('Cache saved.')
                if factor == 2:
                    return sub_images32_X, sub_images32_Y_2
                if factor == 4:
                    return sub_images32_X, sub_images32_Y_4
                if factor == 3:
                    return sub_images33_X, sub_images33_Y_3
                
    def gen_train_sub_images(self, gen_files_path):
        '''
        '''
        sub_images32_X = []
        sub_images33_X = []
        sub_images32_Y_2 = []
        sub_images32_Y_4 = []
        sub_images33_Y_3 = []        
        file_list = os.listdir(gen_files_path)
        for f in file_list:
            stime = time.time()
            print(f)
            img = Image.open(gen_files_path+'/'+f)
            temp_img = np.array(img)
            height, width = temp_img.shape[0], temp_img.shape[1]
            

            # gen orig_size images
            
            if self.phase == 'test':
                sub_images32_X.append(temp_img)
                sub_images33_X = sub_images32_X
                
                if len(temp_img.shape) == 2:
                    temp_img = temp_img.reshape((temp_img.shape[0], temp_img.shape[1], 1))                
                
                _, img_32Y2 = self.gau_and_downscaling(temp_img, self.gau_size, self.gau_sigma, 2)
                sub_images32_Y_2.append(img_32Y2)
                _, img_32Y4 = self.gau_and_downscaling(temp_img, self.gau_size, self.gau_sigma, 4)
                sub_images32_Y_4.append(img_32Y4)                
                _, img_33Y3 = self.gau_and_downscaling(temp_img, self.gau_size, self.gau_sigma, 3)
                sub_images33_Y_3.append(img_33Y3)
                #plt.imshow(img_32Y2)
                #plt.show()
                                
            # gen 32 size sub_images
            if self.phase =='train':
                 
                h_iter = 0
                w_iter = 0
                while(h_iter <= height-self.image_size):
                    while(w_iter <= width-self.image_size):
                        # YCrCb
                        #temp_sub_image = self.RGB2YCrCb(temp_img[h_iter: h_iter+self.image_size, w_iter: w_iter+self.image_size,:])
                        temp_sub_image = (temp_img[h_iter: h_iter+self.image_size, w_iter: w_iter+self.image_size,:])
                        sub_images32_X.append(temp_sub_image)
                        # factor == 2
                        _, temp_sub_image_Y_2 = self.gau_and_downscaling(temp_sub_image, self.gau_size, 
                                                                            self.gau_sigma, 2)
                        sub_images32_Y_2.append(temp_sub_image_Y_2)
                        # factor == 4
                        _, temp_sub_image_Y_4 = self.gau_and_downscaling(temp_sub_image, self.gau_size, 
                                                                            self.gau_sigma, 4)
                        sub_images32_Y_4.append(temp_sub_image_Y_4)
                        w_iter = w_iter + self.strides# height move 14 strides
                    w_iter = 0 # reset to left.
                    h_iter = h_iter + self.strides # height move 14 strides
                # gen 33 sub_images
                h_iter = 0
                w_iter = 0
                while(h_iter <= height-self.image_size2):
                    while(w_iter <= width-self.image_size2):
                        # YCrCb
                        #temp_sub_image = self.RGB2YCrCb(temp_img[h_iter: h_iter+self.image_size2, w_iter: w_iter+self.image_size2,:])
                        temp_sub_image = (temp_img[h_iter: h_iter+self.image_size2, w_iter: w_iter+self.image_size2,:])
                        
                        sub_images33_X.append(temp_sub_image)
                        # factor == 3 
                        _, temp_sub_image_Y_3 = self.gau_and_downscaling(temp_sub_image, self.gau_size, self.gau_sigma, 3)
                        sub_images33_Y_3.append(temp_sub_image_Y_3)                 
                        w_iter = w_iter + self.strides# width move 14 pixels
                    w_iter = 0 # reset to left.
                    h_iter = h_iter + self.strides # height move 14 pixels  
                etime = time.time()
                print(etime-stime)
            
                sub_images32_X = np.array(sub_images32_X, np.uint8)
                sub_images32_Y_2 = np.asarray(sub_images32_Y_2, np.uint8)
                sub_images32_Y_4 = np.asarray(sub_images32_Y_4, np.uint8)
                sub_images33_X = np.asarray(sub_images33_X, np.uint8)
                sub_images33_Y_3 = np.asarray(sub_images33_Y_3, np.uint8)
        return sub_images32_X, sub_images32_Y_2, sub_images32_Y_4, sub_images33_X, sub_images33_Y_3
    
    # RGB to YCrCb
    def RGB2YCrCb(self, image):
        R = image[:,:,0]
        G = image[:,:,1]
        B = image[:,:,2]
        
        image_YCrCb = np.round(np.stack([0.257*R+0.564*G+0.098*B+16,
                                             -0.148*R-0.291*G+0.438*B+128,
                                    0.439*R-0.368*G-0.071*B+128]
                                            ,axis=-1))   
        return image_YCrCb
    
    # gaussion blur and downscaling.
    def gau_and_downscaling(self, image, gau_size, gau_sigma, down_size, down_type='average'):
        '''
        input:
            image: to be gau and downscaling
            gau_size: gaussion kernel size
            gau_sigma: gaussion kernel standard deviation
            down_size: scale factor,
            down_type: default-'average', can be 'min' or 'max'
        output:
            blur_image, that after gaussion blur
            blur_downscaling_image: that after gaussion and downscaling.
        '''
        # gaussion filter size is 3
           
        gau_filter = self.gau_filter(gau_size, gau_sigma)
        if image.shape[2] == 1:# in case the image is one channel
            gau_filter = np.expand_dims(gau_filter[:,:,0], axis=2)
        #print(gau_filter)
        #print(image.shape)
        pad_image = np.pad(image, ((int(gau_size/2), int(gau_size/2)), (int(gau_size/2), int(gau_size/2)), (0,0)), 'symmetric')# pad symmetricly
        print(image.shape)
        print(pad_image.shape)
        #print(pad_image.shape)
        h,w,_ = pad_image.shape
        blur_images = np.zeros_like(image, dtype=np.uint8)
        w_iter = 0
        h_iter = 0
        while(h_iter <= h-gau_size):
            while(w_iter <= w-gau_size):
                conv = np.sum(gau_filter*pad_image[h_iter:h_iter+gau_size, w_iter:w_iter+gau_size, :], axis=(0,1))
                #print(conv.shape)
                blur_images[h_iter, w_iter] = np.clip(np.round(conv), 0, 255)
                w_iter = w_iter + 1
            h_iter = h_iter + 1
            w_iter = 0
        # to be continued================
        # down sampling using factor 2, 3 or 4 
        #print(blur_images.shape)
        down_blur_images = np.zeros((int(image.shape[0]/down_size), int(image.shape[1]/down_size), 3))
        h, w, _ = blur_images.shape
        w_iter = 0
        h_iter = 0
        while(h_iter <= h-down_size):
            while(w_iter <= w-down_size):
                if down_type == 'average':
                    a_conv = np.average(blur_images[h_iter:h_iter+down_size, w_iter:w_iter+down_size, :], axis=(0,1))
                elif down_type == 'max':
                    a_conv = np.max(blur_images[h_iter:h_iter+down_size, w_iter:w_iter+down_size, :], axis=(0,1))
                else:
                    a_conv = np.min(blur_images[h_iter:h_iter+down_size, w_iter:w_iter+down_size, :], axis=(0,1))
                    
                down_blur_images[int(h_iter/down_size), int(w_iter/down_size)] = np.clip(np.round(a_conv), 0, 255) 
                w_iter = w_iter + down_size
            h_iter = h_iter + down_size
            w_iter = 0
        #print(down_blur_images[15,:,:])
        # bb go back to orig_size image.
        #rescale_size = (down_size * down_blur_images.shape[0], down_size * down_blur_images.shape[1], 3)
        #down_blur_rescale_images = self.bicubic_inter(down_blur_images, rescale_size)
        return blur_images, np.array(down_blur_images, dtype=np.uint8)
    
        
    def gau_filter(self, gau_size, gau_sigma):
        # only op in Y channel, so it is 2 dims.
        g_filter = np.zeros((gau_size, gau_size, 3))
        if gau_size % 2 == 0:
            print('gau_size is even, zeros returned.')
            return g_filter
        else:
            row, col = np.indices((gau_size, gau_size))
            row = abs(row - int(gau_size/2))
            col = abs(col - int(gau_size/2))
            g_filter_layer = 1 / (2*3.1415926*np.square(gau_sigma)) * np.exp(-(np.square(row)+np.square(col))/(2*np.square(gau_sigma)))
            g_filter_layer = g_filter_layer/ np.sum(g_filter_layer)
            g_filter[:,:,0] = g_filter_layer
            g_filter[:,:,1] = g_filter_layer
            g_filter[:,:,2] = g_filter_layer
            
            return g_filter
    
    def bicubic_inter(self, orig_image, gen_image_shape):
        '''
        input:
            orig_image: origin image used to gen bb image.
            gen_image_shape: the shape of generated image,i.e. (32,32,3)
        output:
            gen_image: generated image.
        '''
        gen_image = np.zeros((gen_image_shape))
        for i in range(gen_image_shape[0]):
            for j in range(gen_image_shape[1]):
                gen_coord = np.array([i+1, j+1, 3])
                x_index, y_index, x_mu, x_v = self.locate_pixels(orig_image, gen_image_shape, gen_coord)
                # get coefficent.
                #image_patch = orig_image[x_index[0]-1:x_index[-1], y_index[0]-1: y_index[-1],:]
                image_patch = orig_image[[x_index[0]-1,x_index[1]-1, x_index[2]-1, x_index[3]-1]]
                image_patch = image_patch[:,[y_index[0]-1, y_index[1]-1, y_index[2]-1, y_index[3]-1]]
                #print('test'+str(j))
                #print(y_index)
                #print(image_patch.shape)
                #print(image_patch)
                #print(image_patch)
                W_x_mu = np.reshape(np.array([self.bicubic_func(k) for k in x_mu]),(4,1))
                W_x_v = np.reshape(np.array([self.bicubic_func(l) for l in x_v]),(4,1))
                #print(W_x_mu.shape)
                
                W_x_v = W_x_v.transpose()
                #print(W_x_v.shape)
                W_i_j = np.matmul(W_x_mu, W_x_v)
                W_i_j = np.reshape(W_i_j, (4,4,1))
                #print(W_i_j)
                #W_i_j = np.transpose(W_i_j, (1,0,2))
                #print(W_i_j)
                W_i_j = np.tile(W_i_j, (1,1,3))
                #print(W_i_j)
                post_image_patch = image_patch * W_i_j
                #print(post_image_patch.shape)
                point_value = np.sum(post_image_patch, axis=(0,1))
                #print(point_value.shape)
                gen_image[i,j,:] = 256 - np.round(point_value)
                #print(point_value)
                #break
            #break        
        return gen_image
        
    
    def bicubic_func(self, x):
        if abs(x) <= 1:
            out = (self.alpha + 2) * abs(x)**3 - (self.alpha + 3) * abs(x) ** 2 + 1
            
        elif abs(x) < 2 and abs(x) > 1:
            out = self.alpha * abs(x) ** 3 - 5 * self.alpha * abs(x) ** 2 + 8 * self.alpha * abs(x) - 4 * self.alpha
            
        else:
            out = 0
        return out

    # find the nearest 16 pixels.
    def locate_pixels(self, orig_img, gen_shape, gen_coord):
        '''
        output: 16 nearest pixels index & 16 x-related outputs
        orig_img: orig_image,-array
        gen_shape: (M, N, 3) array  
        coord: gen_image pixel coord
        '''
        # 1. get the coord from gen_image to orig_image
        orig_size_x = orig_img.shape[0]
        orig_size_y = orig_img.shape[1]
        gen_coord = np.array(gen_coord)
        #orig_coord = 1.0 * orig_size_x / gen_shape[0] * gen_coord
        orig_x = gen_coord[0] * 1.0 * orig_size_x / gen_shape[0]
        orig_y = gen_coord[1] * 1.0 * orig_size_y / gen_shape[1]
        
        # 2. find valid nearest index seperately
        x_index = []
        y_index = []
        x_mu = []
        x_v = []
        mu = orig_x - int(orig_x)
        v = orig_y - int(orig_y)

        #find x, x_index starts from 1.

        if orig_x < 1:
            x_mu = [1.1-mu, 2.1-mu, 3.1-mu, 4.1-mu]
            x_index = [1,1,2,3]
        elif orig_x < 2:
            x_mu = [mu, 1-mu, 2-mu, 3-mu]
            x_index = [1,2,3,4]
        #elif orig_size_x - orig_x == 0:
            #x_mu = [3,2,1,0]
            #x_index = [orig_size_x-3, orig_size_x -2, orig_size_x-1, orig_size_x]
        elif orig_size_x - orig_x <= 1:
            x_mu = [2+mu, 1+mu, mu, 1-mu]
            x_index = [orig_size_x-2, orig_size_x -1, orig_size_x, orig_size_x]
        else:
            x_mu = [1+mu, mu, 1-mu, 2-mu]
            int_x = int(orig_x)
            x_index = [int_x-1, int_x, int_x+1, int_x+2]
            
        # find y        
        if orig_y < 1:
            x_v = [1.1-v, 2.1-v, 3.1-v, 4.1-v]
            y_index = [1,1,2,3]
        if orig_y < 2:
            x_v = [v, 1-v, 2-v, 3-v]
            y_index = [1,2,3,4] 
        elif orig_size_y - orig_y <= 1:
            x_v = [2+v, 1+v, v, 1-v]
            y_index = [orig_size_y-2, orig_size_y -1, orig_size_y, orig_size_y]
        else:
            x_v = [1+v, v, 1-v, 2-v]
            int_y = int(orig_y)
            y_index = [int_y-1, int_y, int_y+1, int_y+2]      
            
        return np.array(x_index), np.array(y_index), np.array(x_mu), np.array(x_v)       
    # bicubic interpolation to get the Low resolution image.
    # +++++++++++++++++++++++
    
    