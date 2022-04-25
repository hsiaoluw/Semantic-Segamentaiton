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
import cv2
from time import sleep


def check_file(data_dir, file_name):
	if os.path.exists(data_dir):
		if os.path.isfile(os.path.join(data_dir,file_name)):
			return True
		else:
			return False
	else:
			os.makedirs(data_dir)
			return False

data_dir = 'images'		
data_ori_path = os.path.join(data_dir,  'ori')
data_seg_path = os.path.join(data_dir,  'seg')
images = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]


in_name=[]
out_name=[]
for name in images:
	if name[:2]=='In':
		if(not check_file(data_ori_path, name)):
			cmd =  ['cp', os.path.join(data_dir,name), data_ori_path]
			subprocess.call(cmd)
			cmd =  ['rm', os.path.join(data_dir,name)]
			subprocess.call(cmd)
	else:
		if(not check_file(data_seg_path, name)):
			cmd =  ['cp', os.path.join(data_dir,name), data_seg_path]
			subprocess.call(cmd)
			cmd =  ['rm', os.path.join(data_dir,name)]
			subprocess.call(cmd)


ori_set = set(os.listdir(data_ori_path))
seg_set = set(os.listdir(data_seg_path))

print   ( len(os.listdir(data_ori_path)) )
print   ( len(os.listdir(data_seg_path)) )


for name in os.listdir(data_ori_path):
	ori_name = name
	name = 'Out'+name[2:-4]+'_filtered.tif'
	print (name)
	if name not in seg_set:
		cmd =  ['rm', os.path.join(data_ori_path,ori_name)]
		subprocess.call(cmd)
for name in os.listdir(data_seg_path):
	seg_name = name
	name = 'In'+name[3:-13]+'.tif'
	print (name)
	if name not in ori_set:
		cmd =  ['rm', os.path.join(data_seg_path,seg_name)]
		subprocess.call(cmd)

print   ( len(os.listdir(data_ori_path)) )
print   ( len(os.listdir(data_seg_path)) )



"""
in_name=[]
out_name=[]
for name in images:
	if name[:2]=='In':
		if(not check_file(data_ori_path, name)):
			cmd =  ['cp', os.path.join(data_dir,name), data_ori_path]
			subprocess.call(cmd)
	else:
		if(not check_file(data_seg_path, name)):
			cmd =  ['cp', os.path.join(data_dir,name), data_seg_path]
			subprocess.call(cmd)
"""
		
