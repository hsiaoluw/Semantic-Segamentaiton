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

class CycleGanModel(object):
	def __init__(self, config):

		self.config = config # configuration info  including the batch_size, and the generator, discriminator architecture
		self.batch_size = self.config.batch_size
		self.epoch      = self.config.epoch
                #self.batch_num  = int(mnist_data.get_train_num() / batch_size)
		self.z_dim      = 0
		self.label_dim  = 0
		self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
		self.LAMBDA     = 100
		self.test_num   = 100  #number of sample for testing 
                
		self._build_model()
		random.seed(a=123)
		

	def get_feed_dict(self, batch_chunkA, batch_chunkB,step=None, is_training=True, keep_prob=0.7):

		train_alpha =  np.random.uniform(0.0, 1.0, size=[ list(batch_chunkA.shape)[0], 1, 1, 1]).astype(np.float32)		
		if not is_training:
			keep_prob = 1
		fd = {
				    self.alpha: train_alpha,
				    self.Lamda: 100,	
				    self.domainA_image:batch_chunkA ,
				    self.domainB_image:batch_chunkB ,
				    self.keep_prob:keep_prob,
				    self.is_train : is_training
			}	
		
		return fd 

	def _input_ops(self):
		with tf.variable_scope('input'):
			self.alpha = tf.placeholder(tf.float32, [None, 1, 1, 1], name='alpha') # weight for interpolation of improved wgan , only used for improved wgan
			self.Lamda = tf.placeholder(tf.float32, name='Lamda')                  # weight for loss of improved wgan, only used for imporved wgan			
			self.domainA_image = tf.placeholder(tf.float32, [None, self.config.image_x, self.config.image_y,  self.config.image_c], name='A_image') 
			self.domainB_image = tf.placeholder(tf.float32, [None, self.config.image_x, self.config.image_y,  self.config.image_c], name='B_image') 
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') # for drop out		
			self.is_train = tf.placeholder (tf.bool, name='is_train')
				
	def _build_optimizer(self):
		
		
		#self.da_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)		
		#self.ga_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)
		self.db_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)		
		self.gb_optimizer_b = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0.5, beta2=0.9)
		
		#self.da_gs =  self.da_optimizer_b.compute_gradients(self.da_loss, var_list=[self.da_vars ])
		#self.ga_gs =  self.ga_optimizer_b.compute_gradients(self.ga_loss, var_list=[self.gbtoa_vars])
		self.db_gs =  self.db_optimizer_b.compute_gradients(self.db_loss, var_list=[self.db_vars ])
		self.gb_gs =  self.gb_optimizer_b.compute_gradients(self.gb_loss, var_list=[self.gatob_vars])
		
		#self.capped_da_gs =  [((tf.clip_by_value(gv[0], -5. , 5.), gv[1])  if not gv[0]==None else gv) for gv in self.da_gs]
		#self.capped_ga_gs =  [((tf.clip_by_value(gv[0], -5. , 5.), gv[1])  if not gv[0]==None else gv) for gv in self.ga_gs]
		self.capped_db_gs =  [((tf.clip_by_value(gv[0], -5. , 5.), gv[1])  if not gv[0]==None else gv) for gv in self.db_gs]
		self.capped_gb_gs =  [((tf.clip_by_value(gv[0], -5. , 5.), gv[1])  if not gv[0]==None else gv) for gv in self.gb_gs]
		
		#self.da_optimizer  = self.da_optimizer_b.apply_gradients(self.capped_da_gs, global_step= self.global_step)
		#self.ga_optimizer  = self.ga_optimizer_b.apply_gradients(self.capped_ga_gs)
		self.db_optimizer  = self.db_optimizer_b.apply_gradients(self.capped_db_gs, global_step= self.global_step)
		self.gb_optimizer  = self.gb_optimizer_b.apply_gradients(self.capped_gb_gs)
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
		
		differences =  tf.add( self.fake_imageB , -1*self.domainB_image)
		interpolates = tf.add( self.domainB_image, tf.multiply(self.alpha, differences) )
		int_result, _ =discriminator(interpolates, self.is_train, self.keep_prob, self.config.dis_info, name= 'dis',reuse=True)
		gradients = tf.gradients(int_result, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		self.db_loss +=  self.gradient_penalty* 0.1
		
		self.d_loss =  self.db_loss#+  self.da_loss+
		self.g_loss =  self.gb_loss #+ self.Lamda* (self.Bimage_diff+ self.Aimage_diff)+   self.ga_loss+ 

	def _build_model(self):
		# Define input variables
		self._input_ops()
		# Build a model and get logits
		self._model()
		# Compute loss
		self._build_loss()
		# Build optimizer
		self._build_optimizer()

	#build generator and discriminator
	#The detail of generator and discriminator is in ops.py 	
	def _model(self):
		
		
		self.fake_imageB = generator(self.domainA_image, self.is_train, self.keep_prob, self.config.gen_info, name='gen')
		#self.fake_imageA = generator(self.domainB_image, self.is_train, self.keep_prob, self.config.gen_info,  name='genBtoA')
		
		#self.reverB      = generator(self.fake_imageA  , self.is_train, self.keep_prob, self.config.gen_info, name='genAtoB', reuse=True)
		#self.reverA      = generator(self.fake_imageB  , self.is_train, self.keep_prob, self.config.gen_info, name='genBtoA', reuse=True)

		#self.realA_logits , self.realA_feature= discriminator(self.domainA_image, self.is_train, self.keep_prob, self.config.dis_info, name='disA')
		self.realB_logits , self.realB_feature= discriminator(self.domainB_image, self.is_train, self.keep_prob, self.config.dis_info, name='dis')

		#self.fakeA_logits , self.fakeA_feature= discriminator(self.fake_imageA, self.is_train, self.keep_prob, self.config.dis_info, name='disA', reuse=True)
		self.fakeB_logits , self.fakeB_feature= discriminator(self.fake_imageB, self.is_train, self.keep_prob, self.config.dis_info, name='dis', reuse=True)
		self. Bimage_diff = tf.reduce_mean(tf.square(self.fake_imageB- self.domainB_image))
		#self. Aimage_diff = tf.reduce_mean(tf.square(self.fake_imageA- self.domainA_image))
		
		t_vars = tf.trainable_variables()
		#self.da_vars = [var for var in t_vars if 'disA' in var.name]
		#slim.model_analyzer.analyze_vars(self.da_vars, print_info=True)
		self.db_vars = [var for var in t_vars if 'dis' in var.name]
		slim.model_analyzer.analyze_vars(self.db_vars, print_info=True)

		self.gatob_vars = [var for var in t_vars if 'gen' in var.name]
		slim.model_analyzer.analyze_vars(self.gatob_vars, print_info=True)
		#self.gbtoa_vars = [var for var in t_vars if 'genBtoA' in var.name]
		#slim.model_analyzer.analyze_vars(self.gbtoa_vars, print_info=True)
