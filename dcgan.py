from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np

from model import BaseModel
from ops import conv2d, deconv2d, lrelu, generator, discriminator


class dcgan (BaseModel):
	def __init__(self, config):
		super(dcgan, self).__init__(config)
	
	def _build_loss(self):
			if self.label_dim>0:
				self.real_label = tf.concat([self.y_label, tf.zeros([self.batch_size, 1])], axis=1)
				self.fake_label = tf.concat([(0.05)*tf.ones([self.batch_size, self.label_dim])/self.label_dim, 0.95*tf.ones([self.batch_size, 1])], axis=1)
			
				self.d_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.real_logits, labels=self.real_label))
				self.d_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_logits, labels=self.fake_label))
				self.g_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fake_logits, labels=self.real_label))

				self.d_loss =  self.d_real + 0.1*self.d_fake#+self.z_diff*self.z_weight# This optimizes the discriminator.
				self.g_loss =  self.g_fake # This optimizes the generator.
				self.d_fakereplay_op =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.d_fakereplay_result, labels=self.fake_label))
			#no label only dicrminatre fake or real
			else:
				self.d_loss = -tf.reduce_mean(tf.log( 1-tf.nn.sigmoid(self.real_logits)+ 1e-8 ) +tf.log( tf.nn.sigmoid(self.fake_logits) + 1e-8))#+ self.z_diff*self.z_weight
				self.g_loss =  tf.reduce_mean(tf.log( tf.nn.sigmoid(self.fake_logits)+1e-8) )
				





		
