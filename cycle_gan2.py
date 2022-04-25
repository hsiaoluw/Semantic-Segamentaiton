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


class CycleGanModel2(object):
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
		

	def get_feed_dict(self, batch_chunkA, batch_chunkB, step=None, is_training=True, keep_prob=0.7,  z_weight=1):

		train_alpha =  np.random.uniform(0.0, 1.0, size=[ list(batch_chunkA.shape)[0], 1, 1, 1]).astype(np.float32)
		#get the prior distribution(z), if there are labels, add conditions, if there are images(X_noise not none), use images as preconditions
		
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
				    self.Lamda: 100,	
				    self.domainB_image: batch_chunkB, # [B, h, w, c]
				    self.domainA_image: batch_chunkA,
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
			self.domainB_image = tf.placeholder(tf.float32, [None, self.config.image_x, self.config.image_y,  self.config.image_c], name='input_image') 
			if not self.config.gen_info['is_input_image']:
				self.domainA_image = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z') # prior distribuiton of gan
			else:
				self.domainA_image = tf.placeholder(tf.float32, shape=[None, self.config.image_x, self.config.image_y ,1], name='z') # prior distribuiton of gan
			self.is_train = tf.placeholder (tf.bool, name='is_train')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # for drop out
			self.y_label = tf.placeholder(tf.float32, shape=[None, self.label_dim], name='y_label') # use labels to train gan if any	
			
			self.fake_replay        = tf.placeholder(tf.float32, [None, self.config.image_x, self.config.image_y,  self.config.image_c], name='fake_image')
	def _build_loss(self):
		

		#self.da_loss =  tf.reduce_mean( tf.nn.sigmoid(self.realA_logits)- tf.nn.sigmoid(self.fakeA_logits) )#+ self.z_diff*self.z_weight
		#self.ga_loss =  self.Lamda* ( self.Aimage_diff)+ tf.reduce_mean( tf.nn.sigmoid(self.fakeA_logits))
		"""
		differences =  tf.add( self.fake_imageA , -1*self.domainA_image)
		interpolates = tf.add( self.domainA_image, tf.multiply(self.alpha, differences) )
		int_result, _ =discriminator(interpolates, self.is_train, self.keep_prob, self.config.dis_info, name='disA',reuse=True)
		gradients = tf.gradients(int_result, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
			
		self.da_loss +=  self.gradient_penalty* 0.1
		"""
		self.db_loss =  tf.reduce_mean( tf.nn.sigmoid(self.realB_logits)- tf.nn.sigmoid(self.fakeB_logits) )#+ self.z_diff*self.z_weight
		self.gb_loss =  self.Lamda* (self.Bimage_diff) + tf.reduce_mean( tf.nn.sigmoid(self.fakeB_logits))
		
		self.da_loss =  tf.reduce_mean( tf.nn.sigmoid(self.realA_logits)- tf.nn.sigmoid(self.fakeA_logits) )#+ self.z_diff*self.z_weight
		self.ga_loss =  self.Lamda* (self.Aimage_diff) + tf.reduce_mean( tf.nn.sigmoid(self.fakeA_logits))
		
		differences =  tf.add( self.fake_imageB , -1*self.domainB_image)
		interpolates = tf.add( self.domainB_image, tf.multiply(self.alpha, differences) )
		int_result, _ =discriminator(interpolates, self.is_train, self.keep_prob, self.config.dis_info, name= 'disB',reuse=True)
		gradients = tf.gradients(int_result, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		self.db_loss +=  self.gradient_penalty* 0.1
		
		differences =  tf.add( self.fake_imageA , -1*self.domainA_image)
		interpolates = tf.add( self.domainA_image, tf.multiply(self.alpha, differences) )
		int_result, _ =discriminator(interpolates, self.is_train, self.keep_prob, self.config.dis_info, name= 'disA',reuse=True)
		gradients = tf.gradients(int_result, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		self.da_loss +=  self.gradient_penalty* 0.1
		
		self.d_loss =  self.db_loss +  self.da_loss
		self.g_loss =  self.gb_loss + self.Lamda* (self.Bimage_diff+ self.Aimage_diff)+   self.ga_loss
				
	def _build_optimizer(self):
		
	
		
		
		self.d_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)
		self.gf_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)
		self.g_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)
		#self.d_fakereplay  = tf.train.AdamOptimizer(learning_rate=0.0005 , beta1=0.5, beta2=0.9).minimize(self.d_fakereplay_op, var_list=self.d_vars)		
		
		self.d_gs =  self.d_optimizer_b.compute_gradients(self.d_loss, var_list=[self.da_vars, self.db_vars])
		self.g_gs =  self.g_optimizer_b.compute_gradients(self.g_loss, var_list=[self.ga_vars, self.gb_vars])
		self.gf_gs = self.gf_optimizer_b.compute_gradients(self.g_loss,var_list=[self.ga_vars, self.gb_vars])
		
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
			self.fake_imageB = generator(self.domainA_image, self.is_train, self.keep_prob, self.config.gen_info, name='genA2B')
			self.fake_imageA = generator(self.domainB_image, self.is_train, self.keep_prob, self.config.gen_info, name='genB2A')
			
			self.reverse_imageB = generator(self.fake_imageA, self.is_train, self.keep_prob, self.config.gen_info, name='genA2B', reuse=True)
			self.reverse_imageA = generator(self.fake_imageB, self.is_train, self.keep_prob, self.config.gen_info, name='genB2A', reuse=True)
			
			self.realB_logits , _= discriminator(self.domainB_image, self.is_train, self.keep_prob, self.config.dis_info, name='disB')
			self.fakeB_logits , _= discriminator(self.fake_imageB, self.is_train, self.keep_prob, self.config.dis_info, reuse=True, name='disB')			
			
			self.realA_logits , _= discriminator(self.domainA_image, self.is_train, self.keep_prob, self.config.dis_info, name='disA')
			self.fakeA_logits , _= discriminator(self.fake_imageA,   self.is_train, self.keep_prob, self.config.dis_info, name='disA' ,reuse=True)
			#self.d_fakereplay_result ,_= discriminator(self.fake_replay, self.is_train, self.keep_prob, self.config.dis_info, reuse=True)			
			self.Bimage_diff = tf.reduce_mean(tf.square(self.reverse_imageB- self.domainB_image))
			self.Aimage_diff = tf.reduce_mean(tf.square(self.reverse_imageA- self.domainA_image))
		t_vars = tf.trainable_variables()
		
		self.da_vars = [var for var in t_vars if 'disA' in var.name]
		slim.model_analyzer.analyze_vars(self.da_vars, print_info=True)
		
		self.ga_vars = [var for var in t_vars if 'genB2A' in var.name]
		slim.model_analyzer.analyze_vars(self.ga_vars, print_info=True)
		
		self.db_vars = [var for var in t_vars if 'disB' in var.name]
		slim.model_analyzer.analyze_vars(self.da_vars, print_info=True)
		
		self.gb_vars = [var for var in t_vars if 'genA2B' in var.name]
		slim.model_analyzer.analyze_vars(self.ga_vars, print_info=True)
	
    

