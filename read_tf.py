import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
#a = Image.open(r'g:\Jupyter\ImagePro\ImageDataSet\ImageNet2013\ILSVRC2013_train_00000001.jpeg')
#a_np = np.array(a)
#print(a_np.shape)
#a_raw = a_np.tobytes()
#a_raw_np = a.tobytes()
#print(a_raw == a_raw_np)
#print(len(a_raw))


filenames = []


filename_queue = tf.train.string_input_producer(filenames)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={
'label': tf.FixedLenFeature([], tf.string),
'img_raw': tf.FixedLenFeature([], tf.string)
})

img = tf.decode_raw(features['img_raw'], tf.uint8)
img = tf.reshape(img, [16, 16,3])
label = tf.decode_raw(features['label'], tf.uint8)
label = tf.reshape(label, [32,32,3])

img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=32, capacity=1000, min_after_dequeue=200)
init = tf.initialize_all_variables()

with tf.Session() as sess:
     sess.run(init)
     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(coord=coord)
     
     imgs, labels = sess.run([img_batch, label_batch])
     for i in range(32):
          
          plt.subplot(121)
          plt.imshow(imgs[i])
          plt.subplot(122)
          plt.imshow(labels[i])
          plt.show()
          #plt.waitforbuttonpress()
          #time.sleep(3)
     coord.request_stop()
     coord.join(threads)