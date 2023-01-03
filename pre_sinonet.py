import random
import os
import shutil
import sys
import time
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import sys
import vgi
import vgi.ct as ct
from vgi.ssim import SSIM
from vgi.ct import FanRec, createCircleMask
from dataset import CtDataset
from runet import RUNet
from skimage.metrics import structural_similarity as ssim
from evaluate import postproc

n_args = len(sys.argv)

# The model of sinogram synthesis
sino_syn_mdl = sys.argv[1] 
input_dir = sys.argv[2] 
output_dir = sys.argv[3] 
arg_chk = True
print('sino_syn_mdl:', sino_syn_mdl)
if not os.path.exists(sino_syn_mdl):
    arg_chk = False
    print('sino_syn_mdl is not existed')

print('input_dir', input_dir)
if not os.path.exists(input_dir):
    arg_chk = False
    print('input_dir is not existed')

if arg_chk == False:
    sys.exit()

if os.path.exists(output_dir):
    print('All files in output_dir will be deleted!', output_dir)
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
print('output_dir', output_dir)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  
else:
    device = None

time_s = time.time()

# Initializing models
sino_syn = RUNet()
sino_syn.to(device)
sino_syn.load_state_dict(torch.load(sino_syn_mdl))
sino_syn.eval() 

# Loading the inputs
patch_layout = (1, 1)
sino_ds = CtDataset(input_dir, None,
                  patch_layout = patch_layout,
                  device = device,
                  mem = False)

sino_shape = sino_ds.data_shape
n_sino_per_file = sino_ds.n_images
sino_set_shape = [n_sino_per_file] + sino_ds.data_shape
n_files = len(sino_ds.filenames)
n_total_imgs = len(sino_ds)

# Reconstruction configuration.
resolution = 512 # the size of each reconstructed image
rec_shape = (resolution, resolution)
rec_imgs_shape = (n_sino_per_file, resolution, resolution)


print('sino_set_shape', sino_set_shape)
print('sino_patch_size', sino_ds.patch_size)
print('sino_n_patches', sino_ds.n_patches)
print('n_sino_sets:', n_files)
print('rec_imgs_shape', rec_imgs_shape)
print('n_total_imgs :', n_total_imgs, flush=True)

n_angles, n_detectors = sino_shape
det_width = 1.0
source_origin = 512.
origin_det = 512.
ang_range = np.pi * 2
rec_angles = np.linspace(0, ang_range, num = n_angles, endpoint = False)
fbp = FanRec(rec_shape = rec_shape, 
             sino_shape = sino_shape, 
             angles = rec_angles,
             det_width = det_width,
             source_origin = source_origin,
             origin_det = origin_det)

mask = createCircleMask(shape = rec_shape, r = resolution / 2 * 0.99).astype(np.float32)

print('Phase 0 reconstruction', flush=True)
org_vranges = []
for filename in sino_ds.filenames:
    org_sino_set = sino_ds.rawdata(filename)
    rec_imgs = np.zeros(rec_imgs_shape, dtype = np.float32)
    for i, sino in enumerate(org_sino_set):
        rec_imgs[i] = fbp.reconstruct(sino).astype(np.float32)
    vrange = [np.min(rec_imgs), np.max(rec_imgs), np.mean(rec_imgs)]
    org_vranges += [vrange]
    print('vrange of', filename, ':', vrange, flush=True)

print('Phase 0 reconstruction finished, time:', time.time() - time_s, flush=True)

batch_size = 1 
n_batches = n_sino_per_file // batch_size

print('Synthesizing sinograms', flush=True)
for file_id in range(n_files):
    filename = sino_ds.filenames[file_id]
    sino_set = np.zeros(sino_set_shape, dtype = np.float32)
    for i in range(n_batches):
        img_id = i * batch_size
        img_ide = img_id + batch_size
        sino_batch = sino_ds.feed(model = sino_syn, file_id = file_id, image_id = img_id, batch_size = batch_size)
        sino_set[img_id:img_ide] = sino_batch.astype(np.float32)
    
    print('Synthesized sino set:', vgi.metric(sino_set), flush=True)
    org_sino_set = sino_ds.input_file
    for i in range(n_sino_per_file):
        org_sino = org_sino_set[i]
        vmin = np.min(org_sino) 
        vmax = np.max(org_sino)            
        sino_set[i] = vgi.normalize(sino_set[i], vmin, vmax)        
    print('Normalized sino set:', vgi.metric(sino_set), flush=True) 

    print('Phase 1 reconstruction', flush=True)
    rec_imgs = np.zeros(rec_imgs_shape, dtype = np.float32)
    for i, sino in enumerate(sino_set):
        rec_imgs[i] = fbp.reconstruct(sino).astype(np.float32)
    print('vrange:', np.min(rec_imgs), np.max(rec_imgs))
    rec_imgs = np.clip(rec_imgs, 0.0, rec_imgs.max())
    rec_imgs = vgi.normalize(rec_imgs* mask) * mask
    rec_imgs = postproc(rec_imgs, mask = mask, clip_min = True, clip_max = False, mean_align = True, ref_mean = org_vranges[file_id][2]) 
        
    output_path = output_dir + filename + '.npy'
    np.save(output_path, rec_imgs)    
    print('Phase 1 images saved:', output_path, flush=True)
print('Sinogram synthesis finished, time:', time.time() - time_s, flush=True)
