from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
	import better_exceptions
except ImportError:
	pass

#This is the basic model for all wgan, gp_wgan, dcgan
# we change the loss function later in wgan, gp_wgan, dcgan 
import os
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import random
#from dataset import MnistProvider, render_fonts_image
from ops import *


class BaseModel(object):
	def __init__(self, config):

		self.config = config # configuration info  including the batch_size, and the generator, discriminator architecture
		self.batch_size = self.config.batch_size
		self.epoch      = self.config.epoch
                #self.batch_num  = int(mnist_data.get_train_num() / batch_size)
		self.z_dim      = self.config.z_dim # dimension of prior
		self.label_dim  = self.config.label_dim
		self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
		self.LAMBDA     = 0.1
		
		self.learning_rate = 0.0001
		self._build_model()
		random.seed(a=123)
		

	def get_feed_dict(self, batch_chunk, step=None, is_training=True, keep_prob=0.7, X_noise=None, z_weight=1):

		train_alpha =  np.random.uniform(0.0, 1.0, size=[ list(batch_chunk.shape)[0], 1, 1, 1]).astype(np.float32)
		#get the prior distribution(z), if there are labels, add conditions, if there are images(X_noise not none), use images as preconditions
		if(self.config.gen_info['is_input_image']):
			train_noise =  X_noise # Use provided images as prior distribuiton for generator,
		"""
		if self.label_dim>0: 
			train_label = batch_chunk['label']
			#use half of the dim for precondition (classes)
			
			if( not self.config.gen_info['is_input_image']):
				train_noise = np.random.uniform(-2, 2, size=[self.batch_size, self.rest_dim]).astype(np.float32)
				for i in range(self.z_dim_per_class):
					if(random.randint(0,2)%2):
						lb= train_label* np.random.uniform(1,2,size=[self.batch_size, self.label_dim])
					else:
						lb= train_label* np.random.uniform(-1,-2,size=[self.batch_size, self.label_dim])
					train_noise=np.concatenate( (lb ,train_noise), axis=1)				
			train_label = train_label+ np.random.normal(-0.1, 0.1, size=[self.batch_size, self.label_dim]).astype(np.float32)
		else:
			if( not self.config.gen_info['is_input_image']):
				train_noise = np.random.normal(0, 1, size=[self.batch_size, self.z_dim]).astype(np.float32) 
		"""
		if not is_training:
			keep_prob = 1

		fd = {
				    self.alpha: train_alpha,
				    self.Lamda: 10,	
				    self.real_image: batch_chunk, # [B, h, w, c]
				    self.z:  train_noise,
				    self.is_train: True,
				    self.keep_prob:keep_prob
			}		
		
		if self.label_dim > 0:
			fd[self.y_label] = train_label
		return fd 

	def _input_ops(self):
		with tf.variable_scope('input'):
			self.alpha = tf.placeholder(tf.float32, [None, 1, 1, 1], name='alpha') # weight for interpolation of improved wgan , only used for improved wgan
			self.Lamda = tf.placeholder(tf.float32, name='Lamda')                  # weight for loss of improved wgan, only used for imporved wgan
			self.real_image = tf.placeholder(tf.float32, [None, self.config.image_x, self.config.image_y,  self.config.image_c], name='input_image') 
			if not self.config.gen_info['is_input_image']:
				self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z') # prior distribuiton of gan
			else:
				self.z = tf.placeholder(tf.float32, shape=[None, self.config.image_x, self.config.image_y ,1], name='z') # prior distribuiton of gan
			self.is_train = tf.placeholder (tf.bool, name='is_train')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # for drop out
			self.y_label = tf.placeholder(tf.float32, shape=[None, self.label_dim], name='y_label') # use labels to train gan if any	
			
			self.fake_replay        = tf.placeholder(tf.float32, [None, self.config.image_x, self.config.image_y,  self.config.image_c], name='fake_image')
				
	def _build_optimizer(self):
		
	
		
		self.g_m=  self.g_loss+ 100*self.g_diff
		self.d_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)
		self.gf_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)
		self.g_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)
		#self.d_fakereplay  = tf.train.AdamOptimizer(learning_rate=0.0005 , beta1=0.5, beta2=0.9).minimize(self.d_fakereplay_op, var_list=self.d_vars)		
		
		self.d_gs =  self.d_optimizer_b.compute_gradients(self.d_loss, var_list=self.d_vars)
		self.g_gs =  self.g_optimizer_b.compute_gradients(self.g_m   , var_list=self.g_vars)
		self.gf_gs = self.gf_optimizer_b.compute_gradients(self.g_m  , var_list=self.g_vars)
		
		self.capped_d_gs =  [((tf.clip_by_value(gv[0], -5. , 5.), gv[1])  if not gv[0]==None else gv) for gv in self.d_gs]
		self.capped_g_gs =  [((tf.clip_by_value(gv[0], -5. , 5.), gv[1])  if not gv[0]==None else gv) for gv in self.g_gs]
		self.capped_gf_gs = [((tf.clip_by_value(gv[0], -5. , 5.), gv[1])  if not gv[0]==None else gv) for gv in self.gf_gs]
		
		self.d_optimizer  = self.d_optimizer_b.apply_gradients(self.capped_d_gs,global_step= self.global_step)
		self.gf_optimizer = self.gf_optimizer_b.apply_gradients(self.capped_gf_gs)
		self.g_optimizer  = self.g_optimizer_b.apply_gradients(self.capped_g_gs)
		

	def _build_model(self):
		# Define input variables
		self._input_ops()
		# Build a model and get logits
		self._model()
		# Compute loss
		self._build_loss()
		# Build optimizer
		self._build_optimizer()
		
		# Compute classification accuracy for fake, real image
		if self.label_dim >0:
			predict  = tf.argmax(self.real_logits, 1)
			correctY = tf.argmax(self.y_label,1)
			correct = tf.equal(predict, correctY) 
			self.right_class_acc = tf.reduce_mean( tf.cast( correct, tf.float32))

			predict = tf.argmax(self.fake_logits, 1)
			correct = tf.equal(predict, correctY) 
			self.fake_class_acc   = tf.reduce_mean(tf.cast(correct,  tf.float32))
			
	#build generator and discriminator
	#The detail of generator and discriminator is in ops.py 	
	def _model(self):
		self.z_dim_per_class = 5
		self.rest_dim = self.z_dim- self.z_dim_per_class* self.label_dim
		if not self.config.gen_info['is_input_image']:
			self.z_rand, self.z_pre   = tf.split( self.z, [self.rest_dim, self.z_dim_per_class* self.label_dim ], 1 ,name='split')
				
		if(self.label_dim>0):
			
			self.fake_image = generator(self.z_rand, self.is_train, self.keep_prob, self.config.gen_info, pre_cond=self.z_pre)
			self.real_logits,  self.real_feature = discriminator(self.real_image,self.is_train, self.keep_prob, self.config.dis_info)
			self.fake_logits,  self.fake_feature = discriminator(self.fake_image,self.is_train, self.keep_prob, self.config.dis_info, reuse=True)
			self.d_fakereplay_result, _ = discriminator(self.fake_replay, self.is_train, self.keep_prob, self.config.dis_info, reuse=True) 
			
		else: 
			self.fake_image = generator(self.z, self.is_train, self.keep_prob, self.config.gen_info)
			self.real_logits , self.real_feature= discriminator(self.real_image, self.is_train, self.keep_prob, self.config.dis_info)
			self.fake_logits , self.fake_feature= discriminator(self.fake_image, self.is_train, self.keep_prob, self.config.dis_info, reuse=True)			
			#self.d_fakereplay_result ,_= discriminator(self.fake_replay, self.is_train, self.keep_prob, self.config.dis_info, reuse=True)			
			self.g_diff = tf.reduce_mean(tf.square(self.fake_image- self.real_image))
		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'dis' in var.name]
		slim.model_analyzer.analyze_vars(self.d_vars, print_info=True)
		self.g_vars = [var for var in t_vars if 'gen' in var.name]
		slim.model_analyzer.analyze_vars(self.g_vars, print_info=True)
		
	
    

