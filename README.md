Please see Semantic Segmentation of Ultra-Resolution 3d microscopic neural Imagery.pdf for
technique details and comparison results.

Environments:
Please pip install tensorflow_gpu, h5py
If you don't have gpu, use command --gpu False

The original dataset is in img_data.tar.gz, decompress it. 
Use --dataurl to indicate the location of source data dir, the original images are in /ori folder, segmented images are in /seg folder.

Option --gen_type  choices=['seg', 'ori']                 to generate orignal images or segemented images
Option --gan_type  choices=['dcgan', 'wgan', 'wgangp']    gans with different loss functions
Option --epoch     number of epochs
--
train_eval.py: the main train procedures, and evaluation funcitons
data_input.py: preprocess the data, store the data in to datasets, provide batch operators to be called

model.py : the model of gan to be inherited
gp_wgan, wgan, dcgan.py: the loss functions for different gans
ops.py : the auxiliary funcitons for model, generators, and discriminators are also defined here.


Run the command like:
python2 train_eval.py --gan_type dcgan --gen_type seg --epoch 20,  the result is in train_dir/[gan type]-project-[gen type]-[time]


Now the program only supports 2D since some of the data Ting-Ru gave is missing.
split_data.py : this file is the script to split the data(images) in ori, seg folder, and check whether each seg images has a ori image corresponding to it