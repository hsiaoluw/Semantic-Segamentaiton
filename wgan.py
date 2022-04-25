import os
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
from model import BaseModel
from ops import conv2d, deconv2d, lrelu, generator, discriminator

class wgan(BaseModel):
	def __init__(self, config):
		super(wgan, self).__init__(config)
	
	def _build_loss(self):
			

			if self.label_dim>0:
				self.real_label = tf.concat([self.y_label, tf.zeros([self.batch_size, 1])], axis=1)
				self.fake_label = tf.concat([(0.02)*tf.ones([self.batch_size, self.label_dim])/self.label_dim, 0.98*tf.ones([self.batch_size, 1])], axis=1)
			
				self.d_real=tf.reduce_mean( tf.reduce_sum( tf.multiply( tf.nn.softmax(self.real_logits), self.real_label), axis=1) )
				self.d_fake=tf.reduce_mean( tf.reduce_sum( tf.multiply( tf.nn.softmax(self.fake_logits), self.fake_label), axis=1) )
				 # wgan-gp loss is same as wgan loss
				self.g_fake = tf.reduce_mean( tf.reduce_sum( tf.multiply( tf.nn.softmax( self.fake_logits), self.real_label), axis=1) )
				self.d_loss =  -(self.d_real +0.1*self.d_fake)#+self.z_diff*self.z_weight# This optimizes the discriminator.
				self.g_loss =  - self.g_fake# This optimizes the generator.
				
				self.d_fakereplay_op        = -tf.reduce_mean(tf.reduce_sum( tf.multiply( tf.nn.softmax(self.d_fakereplay_result) , self.fake_label), axis=1) )
			else:
				self.d_loss = tf.reduce_mean(tf.nn.sigmoid(self.real_logits) - 0.5*tf.nn.sigmoid(self.fake_logits) ) #+ self.z_diff*self.z_weight
				self.g_loss = tf.reduce_mean(tf.nn.sigmoid(self.fake_logits) )
				self.d_fakereplay_op        = -tf.reduce_mean( tf.nn.sigmoid(self.d_fakereplay_result) )

		    #weight clipping
		    # clip discriminator weights
			self.d_clip = [v.assign(tf.clip_by_value(v, -10, 10)) for v in self.d_vars]
