from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
	import better_exceptions
except ImportError:
	pass
from pprint import pprint
import os
import sys
import tarfile
import subprocess
import h5py
import numpy as np
import struct
import scipy.misc
import time
import tensorflow as tf
from data_input import ImageProvider
from model import BaseModel
from cycle_gan import CycleGanModel
from cycle_gan2 import CycleGanModel2
from gp_wgan import gp_wgan
from wgan  import wgan
from dcgan import dcgan
from time import sleep
from display_data import render_image



class Train_and_Eval(object):
	def __init__(self,config,img_provider):
		self.config = config
        
		if self.config.mode == 'train':
			self.train_dir = './train_dir/%s-%s-%s-%s' % (
				config.gan_type,
				config.image_type,
				config.gen_type,
				time.strftime("%Y%m%d-%H%M%S")
			)
			if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
			self.f = open(self.train_dir+'/train_info.txt','w')
			

		else:
			self.eval_dir = './eval_dir/%s-%s' % (
				config.gan_type,
				config.image_type,
				#time.strftime("%Y%m%d-%H%M%S")
			)		
			if not os.path.exists(self.eval_dir):  os.makedirs(self.eval_dir)
			self.f = open(self.eval_dir+'/eval_info.txt','w')
        # --- input ops ---
		self.batch_size = config.batch_size
		self.batch_train = img_provider.get_batch_train_op() 
		self.batch_test  = img_provider.get_batch_test_op()
		self.img_manager = img_provider                                    
		self.batch_num   = img_provider.train_limit // self.batch_size
		self.output_save_step = self.batch_num
		self.epoch_num        = config.epoch
		self.is_gray = config.is_gray
        # --- create model ---
		if config.gan_type == 'dcgan' :
			self.model = dcgan(config)
		elif config.gan_type == 'wgan':
			self.model = wgan(config)
		elif config.gan_type == 'wgangp':
			self.model = gp_wgan(config)
		elif config.gan_type == 'cyclegan':
			self.model = CycleGanModel2(config)
		else:
			raise ValueError(config.gan_type)

		self.check_op = tf.no_op()

		#self.model.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
		self.saver = tf.train.Saver(max_to_keep=100)
		self.summary_writer = tf.summary.FileWriter(self.train_dir)
		self.checkpoint_secs = 100 # 10 min
		#self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.supervisor =  tf.train.Supervisor(
				init_op=init_op,
				logdir=self.train_dir,
				is_chief=True,
				saver=None,
				summary_op=None,
				summary_writer=self.summary_writer,
				save_summaries_secs=300,
				save_model_secs=self.checkpoint_secs,
				global_step=self.model.global_step
		)
		if self.config.gpu:
			session_config = tf.ConfigProto(
					allow_soft_placement=True,
					gpu_options=tf.GPUOptions(allow_growth=True),
					device_count={'GPU': 1},
			)
		else:
			session_config = tf.ConfigProto(
					allow_soft_placement=True,					
					device_count={'CPU': 1},
			)
		self.sess = self.supervisor.prepare_or_wait_for_session(config=session_config)

		self.ckpt_path = config.checkpoint
		if self.ckpt_path is not None:            
			self.saver.restore(self.sess, self.ckpt_path)
			#self.saver.restore(sess, tf.train.latest_checkpoint(model_path))
   			print("Model restore finished, current globle step: %d" % self.sess.run(self.model.global_step))
	def _train_one_step(self, sess,batch):
		
		batch_chunk = self.sess.run(batch)
		
		
		self.d_iter=1
		for i in range(self.d_iter):
			
			if   self.config.gan_type == 'cyclegan':
				feed_dict   = self.model.get_feed_dict(batch_chunk['ori_image'], batch_chunk['seg_image'],step=None, is_training=True, keep_prob=0.5 )	     
			elif self.config.gen_type == 'seg':
				feed_dict   = self.model.get_feed_dict(batch_chunk['seg_image'], step=None, is_training=True, keep_prob=0.5, X_noise=batch_chunk['ori_image'] )	     
			else:
				feed_dict   = self.model.get_feed_dict(batch_chunk['ori_image'], step=None, is_training=True, keep_prob=0.5, X_noise=batch_chunk['seg_image'] )
			
			if self.config.gan_type == 'cyclegan':
				fetch = [self.model.d_optimizer, self.model.global_step, self.model.d_loss, self.model.db_loss, self.model.fake_imageB ] 
			elif (self.config.gan_type =='wgangp'):
				fetch = [ self.model.d_optimizer, self.model.global_step, self.model.d_loss, self.model.fake_image, self.model.gradient_penalty]   
			else:
				fetch = [self.model.d_optimizer, self.model.global_step, self.model.d_loss , self.model.fake_image]   
			if ( (not self.d_stop) or self.step<1000):
				
				if not self.config.gan_type == 'cyclegan':
				#sess.run([self.model.d_gs, self.model.capped_d_gs], feed_dict)
					if(self.config.gan_type =='wgangp'):
						_, self.step ,self.dLoss , self.fake_img , self.gradient_penalty= sess.run(fetch, feed_dict)
					else:
						_, self.step ,self.dLoss , self.fake_img = sess.run(fetch, feed_dict)

					#minimize fake-replay, train z_decode discriminator part
					"""
					if self.first :
						self.replay = self.fake_img
						self.first = False
					else:
						self.replay = np.concatenate( (self.replay[:self.oldfake_sz], self.fake_img[self.oldfake_sz:]), axis=0)
						feed_dict[self.model.fake_replay] = self.replay
						_ = sess.run([self.model.d_fakereplay], feed_dict)					
						np.random.shuffle(self.replay)
					"""
				else:
					_,self.step ,self.dLoss, self.db_loss, self.fake_imageB  = sess.run(fetch, feed_dict)
					"""
						if self.first :
							self.replay = self.fake_imageA
							self.first = False
						else:
							self.replay = np.concatenate( (self.replay[:self.oldfake_sz], self.fake_imageA[self.oldfake_sz:]), axis=0)
							feed_dict[self.model.fake_replay] = self.replay
							_ = sess.run([self.model.d_fakereplay], feed_dict)					
							np.random.shuffle(self.replay)
					"""
		feed_dict[self.model.keep_prob]  = 0.7
		if self.config.gan_type == 'cyclegan':
			fetch = [self.model.g_optimizer, self.model.g_loss , self.model.fake_imageB, self.model.Bimage_diff, self.model.Aimage_diff]
			_, self.gLoss, self.fake_seg,  self.seg_loss, self.ori_loss = sess.run(fetch, feed_dict)
		else:
			fetch = [self.model.g_optimizer, self.model.g_loss , self.model.fake_image]
			_, self.gLoss, self.fake_image = sess.run(fetch, feed_dict)
		self.prev_gLoss            =  0.999* self.prev_gLoss          + 0.001* (self.gLoss)
		
		for j in range(1):
			
				fetch = [self.model.g_optimizer]#, self.model.zg_opt]   
				_ = sess.run(fetch, feed_dict)
				

		if (self.prev_gLoss > self.prev_gLoss_short):
			self.d_stop = False
			if self.g_iter <4:
				self.g_iter*=2
		
		else:
			self.d_stop = False
			self.g_iter =  max(1,(self.g_iter//2) )
		
		self.prev_gLoss_short = self.prev_gLoss
		
		
		if self.config.gan_type =='wgan':	
			sess.run(self.model.d_clip)	


	def train(self):
		self. g_iter =3
		self. prev_gLoss =0
		self. prev_gLoss_short=0
		self. threshold =10
		self.acc_real =0
		self.acc_fake=0
		self.d_iter =1
		self.d_stop = True
		self.first = True
		self.oldfake_sz = int(float(self.batch_size)*0.99)
		print ('batch size: %d, batch num per epoch: %d, epoch num: %d' % (self.batch_size, self.batch_num, self.epoch_num) )
		self.f.write('batch size: %d, batch num per epoch: %d, epoch num: %d\n' % (self.batch_size, self.batch_num, self.epoch_num) )
		print ( 'start training...')
	
		self.step= 1
		total_step = self.epoch_num* self.batch_num
		while self.step < total_step+2:
				self._train_one_step(self.sess ,self.batch_train)		
				
				if(self.step%2==1):
					if self.config.gan_type =='wgangp':
						print ('train:[%d/%d],d_loss:%f, g_loss:%f, gradient_penalty:%f' % (self.step//self.batch_num, self.step%self.batch_num, self.dLoss,  self.gLoss,self.gradient_penalty))
					elif self.config.gan_type== 'cyclegan':
						print ('train:[%d/%d],d_loss:%f, g_loss:%f, A_diff:%f, B_diff:%f' % (self.step//self.batch_num, self.step%self.batch_num, self.dLoss,  self.gLoss, self.ori_loss, self.seg_loss))
					else: 
						print ('train:[%d/%d],d_loss:%f, g_loss:%f' % (self.step//self.batch_num, self.step%self.batch_num, self.dLoss,  self.gLoss))
					self.f.write ('train:[%d/%d],d_loss:%f, g_loss:%f\n' % (self.step//self.batch_num, self.step%self.batch_num, self.dLoss,  self.gLoss))
				#self.step+=1
				if self.step % self.output_save_step == 1:
					ith_epoch = int(self.step//self.batch_num)
					save_path = self.saver.save( self.sess, os.path.join(self.train_dir, 'model'+ str(ith_epoch) + '.cptk'), global_step= self.model.global_step)
					self. run_test(self.batch_test)
					if not self.config.gan_type == 'cyclegan':
						if(self.is_gray):
							gen_images = np.asarray(self.gen_images, dtype=np.float32).reshape(
							[self.gen_images.shape[0], self.gen_images.shape[1], self.gen_images.shape[2]])
						else:
							gen_images = self.gen_images
						if self.config.gen_type == 'seg':
							curr_path = os.path.join(self.train_dir, str(ith_epoch)+ 'epoch_fake_seg.jpg')
						else:
							curr_path = os.path.join(self.train_dir, str(ith_epoch)+ 'epoch_fake_ori.jpg')
						
						np.clip(gen_images, -0.999, 0.999)
						render_image(gen_images, curr_path, 10, gray=self.is_gray)
					else:
						if(self.is_gray):
							fake_seg = np.asarray(self.fake_seg, dtype=np.float32).reshape([self.fake_seg.shape[0], self.fake_seg.shape[1], self.fake_seg.shape[2]])
							fake_ori = np.asarray(self.fake_ori, dtype=np.float32).reshape([self.fake_seg.shape[0], self.fake_seg.shape[1], self.fake_seg.shape[2]])
						
						else:
							fake_seg = self.fake_seg
							fake_ori = self.fake_ori
						
						fake_seg = np.clip(fake_seg, -0.999, 0.999)
						fake_ori = np.clip(fake_ori, -0.999, 0.999)
						curr_path = os.path.join(self.train_dir, str(ith_epoch)+ 'epoch_fake_seg.jpg')
						render_image(fake_seg, curr_path, 10, gray=self.is_gray)
						curr_path = os.path.join(self.train_dir, str(ith_epoch)+ 'epoch_fake_ori.jpg')
						render_image(fake_ori, curr_path, 10, gray=self.is_gray)
					#print (curr_path)
					#print (render_image(gen_images, curr_path, 16, gray=self.is_gray))	 
				     
	
	def run_test(self, batch, is_train=False, repeat_times=8):

		batch_chunk = self.sess.run(batch)
		
		if self.config.gan_type == 'cyclegan':
			feed_dict=self.model.get_feed_dict(batch_chunk['ori_image'], batch_chunk['seg_image'], is_training=False)
			self.ori_image, self.seg_image ,self.step, dloss, gloss,  self.fake_seg, self.fake_ori = self.sess.run([self.model.domainA_image, self.model.domainB_image,self.model.global_step, self.model.d_loss, self.model.g_loss, self.model.fake_imageB, self.model.fake_imageA ],feed_dict)
			curr_path = os.path.join(self.train_dir, str(int(self.step//self.batch_num))+ 'real_seg.jpg')
			
		
		elif self.config.gen_type == 'seg_image':
			feed_dict=self.model.get_feed_dict(batch_chunk['seg_image'], is_training=False, X_noise=batch_chunk['ori_image'])
			self.step, dloss, gloss, self.gen_images = self.sess.run([self.model.global_step, self.model.d_loss, self.model.g_loss, self.model.fake_image ],feed_dict)
		else:
			feed_dict=self.model.get_feed_dict(batch_chunk['ori_image'], is_training=False, X_noise=batch_chunk['seg_image'])	
			self.step, dloss, gloss, self.gen_images = self.sess.run([self.model.global_step, self.model.d_loss, self.model.g_loss, self.model.fake_image ],feed_dict)
	 	self.f.write ('step %d test: d_loss:%f,g_loss:%f \n' % ( self.step, dloss, gloss))					
	
		curr_path = os.path.join(self.train_dir, str(int(self.step//self.batch_num))+ 'real_seg.jpg')
		if self.is_gray:
			render_image(self.reshape_gray(batch_chunk['seg_image']), curr_path, 10, gray=self.is_gray)
		else:
			render_image(batch_chunk['seg_image'], curr_path, 10, gray=self.is_gray)

		curr_path = os.path.join(self.train_dir, str(int(self.step//self.batch_num))+ 'real_ori.jpg')
		if self.is_gray:
			render_image(self.reshape_gray( batch_chunk['ori_image']), curr_path, 10, gray=self.is_gray)
		else:
			render_image(batch_chunk['ori_image'], curr_path, 10, gray=self.is_gray)
	def reshape_gray(self,img):
		return  np.asarray(img, dtype=np.float32).reshape([img.shape[0], img.shape[1], img.shape[2]])
	
	def evaluate(self):
		#this two paramters are loss weight for training not important for evaluation
		

		self. run_test(self.batch_test)
		if(self.is_gray):
			gen_images = np.asarray(self.gen_images, dtype=np.float32).reshape(
			[self.gen_images.shape[0], self.gen_images.shape[1], self.gen_images.shape[2]])

		curr_path = os.path.join(self.eval_dir,  'eval.jpg')
		render_image(gen_images, curr_path, 10, gray)


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--epoch'     , type=int, default=10)
	parser.add_argument('--label_dim' , type=int, default=0)
	parser.add_argument('--z_dim' ,     type=int, default=256)       
	parser.add_argument('--checkpoint', type=str, default=None)
	parser.add_argument('--mode',       type=str, default='train', choices=['train', 'eval' ])
	parser.add_argument('--image_type', type=str, default='project', choices=['mnist', 'svhn', 'cifar10', 'project'])
	parser.add_argument('--gen_type',   type=str, default='seg'    , choices=['seg', 'ori'])
	parser.add_argument('--gan_type',   type=str, default='wgangp' , choices=['dcgan', 'wgan', 'wgangp','cyclegan'])
	parser.add_argument('--gpu',        type=bool, default=True )
	parser.add_argument('--download_path', type=str, default='datasets')
	parser.add_argument('--data_url', type=str, default='http://yann.lecun.com/exdb/mnist/')

	config = parser.parse_args()

	if config.image_type == 'mnist':
		config.data_url = 'http://yann.lecun.com/exdb/mnist/'
		config.is_gray = True
	elif config.image_type == 'svhn':
		config.data_url = 'http://ufldl.stanford.edu/housenumbers/'
		config.is_gray = False
	elif config.image_type == 'cifar10':
		config.data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
		config.is_gray = False
	elif config.image_type == 'project':
		config.data_url = 'img_data'
		config.is_gray = True
	else:
		raise ValueError(config.image_type)
	img_manager = ImageProvider(config)
	train_test_mng = Train_and_Eval(config, img_manager)
	if config.mode == 'train':
		train_test_mng.train()
	else:
		if ( config.checkpoint==None):
			print ('--checkpoints can not be NONE to evaluate a model')
			return 
		train_test_mng.evaluate()
		
	train_test_mng.f.close()
    
if __name__ == '__main__':
		main()
