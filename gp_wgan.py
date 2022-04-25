import os
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
from model import BaseModel
from ops import conv2d, deconv2d, lrelu, generator, discriminator

class gp_wgan(BaseModel):
	def __init__(self, config):
		super(gp_wgan, self).__init__(config)
	
	def _build_loss(self):
		
			if self.label_dim>0:
				self.real_label = tf.concat([self.y_label, tf.zeros([self.batch_size, 1])], axis=1)
				self.fake_label = tf.concat([(0.01)*tf.ones([self.batch_size, self.label_dim])/self.label_dim, 0.99*tf.ones([self.batch_size, 1])], axis=1)
			
				#self.real_label = self.real_label*2-1.
				#self.fake_label = self.fake_label*2-0.9
				self.d_real=tf.reduce_mean( tf.reduce_sum( tf.multiply( tf.nn.softmax(self.real_logits), self.real_label), axis=1) )
				self.d_fake=tf.reduce_mean( tf.reduce_sum( tf.multiply( tf.nn.softmax(self.fake_logits), self.fake_label), axis=1) )
				 # wgan-gp loss is same as wgan loss
				self.g_fake = tf.reduce_mean( tf.reduce_sum( tf.multiply( tf.nn.softmax( self.fake_logits), self.real_label), axis=1) )
				self.d_loss =  -(self.d_real +0.05*self.d_fake)#+ self.z_diff*self.z_weight# This optimizes the discriminator.
				self.g_loss =  - self.g_fake# This optimizes the generator.
				
				self.d_fakereplay_op        = -tf.reduce_mean(tf.reduce_sum( tf.multiply( tf.nn.softmax(self.d_fakereplay_result) , self.fake_label), axis=1) )
			else:
				self.d_loss = tf.reduce_mean(tf.nn.sigmoid(self.real_logits) - 0.5*tf.nn.sigmoid(self.fake_logits) ) #+ self.z_diff*self.z_weight
				self.g_loss = tf.reduce_mean(tf.nn.sigmoid(self.fake_logits) )
				self.d_fakereplay_op        = -tf.reduce_mean( tf.nn.sigmoid(self.d_fakereplay_result) )
		    # wgan-gp loss is same as wgan loss
			
			

			# wgan-gp gradient panelty
			
			differences =  tf.add( self.fake_image , -1*self.real_image)
			interpolates = tf.add( self.real_image, tf.multiply(self.alpha, differences) )
			if self.label_dim > 0:
				int_logits, int_feature  = discriminator(interpolates, self.is_train, self.keep_prob, self.config.dis_info,reuse=True)
				_, int_result            = tf.split(tf.nn.softmax(int_logits), [self.label_dim, 1], 1 ,name='split_int')
			else:
				int_result, _            =discriminator(interpolates, self.is_train, self.keep_prob, self.config.dis_info,reuse=True)
			gradients = tf.gradients(int_result, [interpolates])[0]
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
			self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
			
			self.d_loss +=  self.gradient_penalty* self.LAMBDA
			
			
	
