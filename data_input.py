# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import better_exceptions
except ImportError:
    pass

import os
import sys
import tarfile
import subprocess
import h5py
import numpy as np
import struct
import scipy.misc
import tensorflow as tf
import cv2
from time import sleep

class ImageProvider(object):
	def __init__(self, config):
		self.download_dir                          = config.download_path
		self.image_type                            = config.image_type
		self.data_url                              = config.data_url       
		self.label_dim                             = config.label_dim
		self.config                                = config

		#default generator, discriminator architecture , see ops.py for the meaning of each variables
		gen_info ={}
		gen_info['deconv1'] =   1, 64,64
		gen_info['deconv2'] =  64, 5, 2
		gen_info['deconv3'] =  32, 5, 2
		gen_info['deconv4'] =  16, 5, 2
		gen_info['deconv5'] =  16, 5, 2
		gen_info['deconv6'] =  1, 64, 1
		gen_info['is_input_image'] = True
		dis_info = {}
		dis_info['conv1'] =  32, 5, 2
		dis_info['conv2'] =  64, 5, 2
		dis_info['conv3'] =  128, 4, 2
		dis_info['conv4'] =  128, 4, 2
		dis_info['fc4_prev'] =  200
		dis_info['fc4']      =  1
		
		self.config.image_x=64
		self.config.image_y=64
		self.config.image_c =1 
		self.config.gen_info= gen_info
		self.config.dis_info= dis_info

		
		if(self.image_type =='project'):
			self.preprocess_project (self.download_dir)
			self.config.image_c=1
		self.data = h5py.File( os.path.join(self.data_dir, 'data.hy') ,  "r")
		self.train_ids = self.all_ids()
		#set up train batch tenstor
		self.batch_train_op = self.create_input_ops(self.config.batch_size)
		self.batch_test_op  = self.create_input_ops(100)

	
	def get_batch_train_op(self):		
		return self.batch_train_op
	def get_batch_test_op(self):		
		return self.batch_test_op

	# return batch tensor dequeue 
	def create_input_ops(self,batch_size, num_threads=10, scope='inputs', isTrain=True):
		input_ops ={}
		with tf.device("/cpu:0"), tf.name_scope(scope):
			
			self.isTrain= isTrain			

			if(isTrain): 
				data_id=self.train_ids
			else: 
				data_id=self.train_ids
			input_ops['id'] = tf.train.string_input_producer(
                                          tf.convert_to_tensor(data_id),
                                          capacity=128
                                          ).dequeue(name='input_ids_dequeue')

			a, b = self.get_data(data_id[0])

			
			def load_fn(id):
					# image [n, n], label: [m]
					a, b = self.get_data(id)
					return (id,
						a.astype(np.float32),
						b.astype(np.float32))

			input_ops['id'], input_ops['ori_image'], input_ops['seg_image'] = tf.py_func(
					load_fn, inp=[input_ops['id']],
					Tout=[tf.string, tf.float32, tf.float32],
					name='func_hp'
				)
			input_ops['id'].set_shape([])
			input_ops['ori_image'].set_shape(list(a.shape))
			input_ops['seg_image'].set_shape(list(b.shape))
			capacity = 2 * batch_size * num_threads        
			batch_ops = tf.train.batch(
					    input_ops,
					    batch_size=batch_size,
					    num_threads=num_threads,
					    capacity=capacity,
					)
		return batch_ops
	

	def all_ids(self):
		self.rs =  np.random.RandomState(123)
		id_filename = 'id.txt'

		id_txt = os.path.join(self.data_dir, id_filename)
		try:
			with open(id_txt, 'r') as fp:
				_ids = [s.strip() for s in fp.readlines() if s]
		except:
			raise IOError('Dataset not found. Please make sure the dataset was downloaded.')
		int_id = [int(i) for i in _ids]
		self.train_limit= max(int_id)
		self.rs.shuffle(_ids)
		return _ids


	def get_data(self, id):
		# preprocessing and data augmentation
		a = self.data[id]['ori_image'].value.astype(np.float32)
		b = self.data[id]['seg_image'].value.astype(np.float32)
		return a, b	


	def check_file(self, data_dir):
		if os.path.exists(data_dir):
			if os.path.isfile(os.path.join(data_dir,'data.hy')) and \
            			os.path.isfile(os.path.join(data_dir,'id.txt')):
				return True
			else:
				return False
		else:
			os.makedirs(data_dir)
			return False
	


	def preprocess_project(self, download_path, data_url='img_data' ):
		self.data_dir = os.path.join(download_path, 'cs599project')			
		if self.check_file(self.data_dir):
			print('Project data was preprocessed.')
			return
		
		self.data_ori_path = os.path.join(data_url,  'ori')
		self.data_seg_path = os.path.join(data_url,  'seg')

		from os import listdir
		from os.path import isfile, join
		ori_images = [f for f in listdir(self.data_ori_path) if isfile(join(self.data_ori_path, f))]
		seg_images = [f for f in listdir(self.data_seg_path) if isfile(join(self.data_seg_path, f))]

		f = h5py.File(os.path.join(self.data_dir, 'data.hy'), 'w')
		data_id = open(os.path.join(self.data_dir,'id.txt'), 'w')
	    
		assert(len(ori_images)== len(seg_images))
		i=0
		for k in ori_images:
			one_ori=cv2.resize(  (cv2.imread( os.path.join(self.data_ori_path,k),0)/256.0)*2-1, (68,68) ,  interpolation = cv2.INTER_CUBIC)
					
			k_seg = 'Out'+k[2:-4]+'_filtered.tif'
			print (k_seg)
			one_seg=cv2.resize(  (cv2.imread( os.path.join(self.data_seg_path,k_seg),0)/256.0 )*2-1, (68,68), interpolation = cv2.INTER_CUBIC)
			
			for d in range(4):
				grp = f.create_group(str(i))
				data_id.write(str(i)+'\n')
				i+=1				
				grp['ori_image'] = self.preprocess_image(one_ori, d)
				grp['seg_image'] = self.preprocess_image(one_seg, d)
				print (grp['ori_image'].shape)
		self.train_limit=i
		f.close()
		data_id.close()


	def preprocess_image( self, ori, direction):
		rows, cols = ori.shape
		if direction == 0:
			ori = np.reshape(ori, (rows,cols,1) )
			return ori[2:66, 2:66,:]
		elif direction == 1:
			r_m = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
		elif direction == 2:
			r_m = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
		elif direction == 3: 
			r_m = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
		
		ori=np.reshape( cv2.warpAffine(ori,r_m,(cols,rows)), (68,68,1))
		
		return ori[2:66, 2:66,:]


	
	
	
