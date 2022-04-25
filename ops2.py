from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
	import better_exceptions
except ImportError:
	pass
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np

def lrelu(x, name='lrelu', leak=0.2):
	return tf.maximum(x, leak * x, name=name)

def huber_loss(labels, predictions, delta=1.0):
	residual = tf.abs(predictions - labels)
	condition = tf.less(residual, delta)
	small_res = 0.5 * tf.square(residual)
	large_res = delta * residual - 0.5 * tf.square(delta)
	return tf.where(condition, small_res, large_res)

#combine conv2d, batch normalization , dropout, activation fcn together, return last layer
#input: the previous layer to be convolutioned next
#output: the number of output channels(kernels)
# k : size of kernel in square k*k
# s : size of stride in square s*s, 
# if the input layer is n*n*c , this funciton return floor(n/s)*floor(n/s)*output
def conv2d(input, output, k, s, is_train,  keep_prob,stddev=0.02, name="conv2d", norm=True, activation_fn='lrelu'):
	with tf.variable_scope(name):

		conv = tf.layers.conv2d(input, output, kernel_size=[k, k], strides=[s, s], padding="SAME", \
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name= name+"_conv2",  bias_initializer=tf.zeros_initializer())
		bn = tf.layers.batch_normalization(conv, training=is_train, name=name+'_bn')
        
	if(norm):
		out = bn#tf.nn.avg_pool(bn, ksize=[1, 3,3, 1], strides=[1, 2,2,1], padding='SAME', name=name+'_pool')
	else:
		out = conv#tf.nn.avg_pool(conv, ksize=[1, 3,3, 1], strides=[1, 2,2,1], padding='SAME', name=name+'_pool')
	#out = tf.nn.avg_pool(out, ksize=[1, s,s, 1], strides=[1, s,s,1], padding='SAME', name=name+'_pool')
	if activation_fn == 'tanh':
		lrelu_act = tf.nn.tanh(out, name= name+"_tanh")
	else:
		lrelu_act = lrelu(out, name= name+"_lrelu")
	final_out =  tf.nn.dropout(lrelu_act, keep_prob )

	return final_out

#combine conv2dt, batch normalization , dropout, activation fcn together, return last layer
#input: the previous layer to be convolutioned next
#output: the number of output channels(kernels)
# k : size of kernel in square k*k
# s : size of stride in square s*s, 
# if the input layer is n*n*c , this funciton return (n*s)*(n*s)*output
def deconv2d(input, output, k, s, is_train,  keep_prob, name="deconv2d", stddev=0.02,activation_fn='lrelu', norm=True):
	with tf.variable_scope(name):
		
		deconv = tf.layers.conv2d_transpose(input, output, kernel_size=[k, k], strides=[s, s], padding="SAME", \
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name= name+"_deconv2",  bias_initializer=tf.zeros_initializer())
			
		if norm:
			bn = tf.layers.batch_normalization(deconv, training=is_train, name=name+'_bn')
		else:
			bn = deconv

		if activation_fn == 'relu':
		    out = tf.nn.relu(bn, name= name+"_relu")
		
		elif activation_fn == 'lrelu':
		    out = lrelu(bn, name= name+"_lrelu")

		elif activation_fn == 'tanh':
		    out = tf.nn.tanh(deconv, name= name+"_tanh")
		elif activation_fn == 'none':
		    out= bn
		else:
		    raise ValueError('Invalid activation function.')
		final_out =  tf.nn.dropout(out, keep_prob )
 	return final_out

def generator(input, is_train, keep_prob, config,reuse=False, pre_cond=None, name='gen'):

		        
		is_input_image= config['is_input_image'] # whether the prior distribution is image or not, if not the prior distribution is random need to use fc(fully connected layer) to contruct first layer
	
		c1, wid1 , h1 = config['deconv1']  # c1 number of channels of first layer, w1 : width of first layer, h1: height of first layer
		c2, k2 , s2 = config['deconv2']  # c2 number of channels fo second layer, s2 : stride of second, k2: kernel size
		c3, k3 , s3 = config['deconv3']  # for layer3,4,5  ci,si,ki is the same meaning fo each layer
		c4, k4 , s4 = config['deconv4']
		c5, k5 , s5 = config['deconv5']
		c6, k6 , s6 = config['deconv6']

		print ( c1, wid1, h1)
	    #img_ch = 1  # gray image
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()
			one = tf.Variable(1.0, trainable=False)
	
			if  is_input_image == False:
				input =  tf.concat((input, pre_cond), axis=1 )
				input =  ly.fully_connected( input,   c1 * wid1 * h1 , activation_fn=lrelu) 
				#input = tf.layers.batch_normalization(input,  training=is_train, name='input_bn1')
				w1 = tf.get_variable('w1', shape=[input.get_shape().as_list()[1], c1 * wid1 * h1], dtype=tf.float32,
								 initializer=tf.truncated_normal_initializer(stddev=0.02))
				print ( c1, wid1, h1)
				b1 = tf.get_variable('b1', shape=[c1*wid1*h1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
				flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
				# w1*h1*c1
				conv1 = tf.reshape(flat_conv1, shape=[-1, wid1, h1, c1], name='conv1')
				act1 = lrelu(conv1, name='act1')
			else:
				act1 = tf.reshape(input, shape=[-1, wid1, h1, c1], name='conv1')

			print('gen/act1 layer: ' + str(act1.get_shape()))

			dconv2 = conv2d(act1, 16, 6, 2, is_train, name='dconv2',   keep_prob= keep_prob, norm=True)
			print('gen/conv2 layer: ' + str(dconv2.get_shape()))
			dconv3 = conv2d(dconv2, 32, 6, 2, is_train, name='dconv3', keep_prob= keep_prob,  norm=True)
			print('gen/conv3 layer: ' + str(dconv3.get_shape()))
			dconv4 = conv2d(dconv3, 64, 4, 2, is_train, name='dconv4', keep_prob= keep_prob,  norm=True)
			print('gen/conv3 layer: ' + str(dconv4.get_shape()))
			dconv5 = conv2d(dconv4, 128, 4, 2, is_train, name='dconv5', keep_prob= keep_prob,  norm=True)
			
			conv4_1= deconv2d(dconv5 , 64, k4, 4, is_train, name='conv4_1', keep_prob= keep_prob, norm=True)
			conv4_2= deconv2d(conv4_1, 32, k4, 2, is_train, name='conv4_2', keep_prob= keep_prob, norm=True)
			print('gen/dconv4 layer: ' + str(conv4_2.get_shape()))
			conv5 = deconv2d(conv4_2,  16, k5, 2, is_train, name='conv5', keep_prob= one, norm=True)
			print('gen/dconv6 layer: ' + str(conv5.get_shape()))
			conv6 =  conv2d(conv5,  1,  6, 1, is_train, name='conv6', activation_fn = 'tanh', keep_prob=one, norm=False)
			print('gen/dconv6 layer: ' + str(conv6.get_shape()))
		return conv6*1.1



# input       : the input image
# config      : the configuration of discriminator architecture     
def discriminator( input,  is_train, keep_prob, config,reuse=False, name='dis'):

		c1, k1 , s1 = config['conv1']  # c1 number of channels fo second layer, s1 : stride of second, k1: kernel size
		c2, k2 , s2 = config['conv2']  # for layer2,3  ci,si,ki is the same meaning fo each layer
		c3, k3 , s3 = config['conv3']  # 
		c4, k4 , s4 = config['conv4']  # 
		d4_prev     = config['fc4_prev'] # the hidden layer neurons for classification
		d4          = config['fc4']      # the number of output classification
		
	
		with tf.variable_scope(name) as scope:
			if reuse:
				scope.reuse_variables()
			one = tf.Variable(1.0, trainable=False)
			conv1_t   = conv2d(input, c1,   k1, s1, is_train, name='conv1d', norm=False, keep_prob=one)			
			conv1_t_flat = tf.reshape(conv1_t, shape=[-1, (conv1_t.get_shape().as_list()[1])* (conv1_t.get_shape().as_list()[2])*c1])
			print('dis/conv1_t: ' + str(conv1_t.get_shape()))
			conv2_t = conv2d(conv1_t, c2, k2, s2, is_train, name='conv2d', norm=False, keep_prob= keep_prob)			
			print('dis/conv2_t: ' + str(conv2_t.get_shape()))
			conv3_t    = conv2d(conv2_t, c3, k3, s3, is_train, name='conv3d',keep_prob= keep_prob)		
			print('dis/conv3_t: ' + str(conv3_t.get_shape()))
			conv4_t    = conv2d(conv3_t, c4, k4, s4 , is_train, name='conv4d', norm=False,keep_prob= keep_prob)
			
			conv4t_shape=conv4_t.get_shape().as_list()			
			conv4_flat = tf.reshape(conv4_t, shape=[-1, (conv4t_shape[1])* (conv4t_shape[2])*c3])			
			print('dis/conv4_flat: ' + str(conv4_flat.get_shape()))
			
			if(d4>0):				
				fc4_class_fake_true        = ly.fully_connected( conv4_flat, d4+1              , activation_fn=None)
				return fc4_class_fake_true, input
			else:
				fc4_fake_true        = ly.fully_connected( conv4_flat, 1              , activation_fn=None)
				return fc4_fake_true, input




def resnet_blocks(input_res, num_features, is_train, name='_resnet', norm=True, keep_prob=1):

    out_res_1 = conv2d(input_res, num_features, 3, 1, is_train, name='resnet1_'+name , keep_prob= keep_prob , norm=False)
    out_res_2 = conv2d(out_res_1, num_features, 3, 1, is_train, name='resnet2_'+name , keep_prob= keep_prob , norm=norm)
    return (out_res_2 + input_res)



