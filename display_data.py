import os
import sys
import numpy as np
import struct
import scipy.misc

# to visualize the evaluate results
def render_image(x, path, img_per_row, unit_scale=True, gray=True):
	#print 'enter render_image'
	#print np.amax(x)
	#print np.amin(x)
	if unit_scale:
        # scale 0-1 matrix back to gray scale bitmaps
		np.clip(x, -0.999, 0.999)
		bitmaps =  ( ((x*255.0+255.0)/2).astype(dtype=np.int16))
	else:
		bitmaps = x
	
	#print np.amax(bitmaps)
	#print np.amin(bitmaps)
	if gray:
		num_imgs, h, w = x.shape
		width = img_per_row * w
		height = int(np.ceil(float(num_imgs) / img_per_row)) * h
		canvas = np.zeros(shape=(height, width), dtype=np.int16)
		canvas.fill(0)
		for idx, bm in enumerate(bitmaps):
			x = h * int(idx / img_per_row)
			y = w * int(idx % img_per_row)
			canvas[x: x + h, y: y + w] = bm
		scipy.misc.toimage(canvas).save(path)
		#print path
		return path

	else: 
		num_imgs, h, w, c = x.shape
		width = img_per_row * w
		height = int(np.ceil(float(num_imgs) / img_per_row)) * h
		canvas = np.zeros(shape=(height, width, c), dtype=np.int16)
    # make the canvas all white
		canvas.fill(0)
		for idx, bm in enumerate(bitmaps):
			x = h * int(idx / img_per_row)
			y = w * int(idx % img_per_row)
			canvas[x: x + h, y: y + w, :c] = bm
		scipy.misc.toimage(canvas).save(path)
		return path

