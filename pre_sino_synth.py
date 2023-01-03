import random
import os
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
from residual_models import SinoNet, ReconNet
from aunet import AUNet, init_weights
from skimage.metrics import structural_similarity as ssim
from evaluate import evaluate

n_args = len(sys.argv)

# The model of sinogram synthesis
model_id = 'unet' if n_args < 2 else sys.argv[1] 
    # 'unet': the standard U-net
    # 'runet': the residual U-Net
    # 'aunet': the attenion U-Net
# The loss function, 'L1' or 'SSIML1'
loss_name = 'L1' if n_args < 3 else sys.argv[2] 
# The number of sampling angles
n_smp_angles = 60 if n_args < 4 else int(sys.argv[3])
# SSIML1 alpha 
ssiml1_alpha = 0.8 if n_args < 5 else float(sys.argv[4])

print('model_id', model_id)
print('loss_name', loss_name)
print('n_smp_angles', n_smp_angles, flush=True)

sn_smp_angles = 'ang' + str(n_smp_angles)

# The paths of datasets, please modify them for your environment
# Each data item must be stored as a .npy file with the shape of (#sinograms, #angles, #detectors)
main_dir = 'D:/Data/MIDRC/'
data_dir = main_dir + 'fanflat/'
work_dir = data_dir + sn_smp_angles + '/'
train_input_dir = work_dir + 'sino_cubic/'
train_target_dir = data_dir + 'sino/'
test_input_dir = work_dir + 'sino_cubic_test/'
test_target_dir = data_dir + 'sino_test/'
gt_dir = main_dir + 'ct_test/' # the path of ground truths

patch_layout = (1, 1)

# Customizing the random seed
seed=8051
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.enabled = False

if torch.cuda.is_available():
    device = torch.device("cuda:0")  
else:
    device = None
    
# Data loading    
time_s = time.time()
train_ds = CtDataset(train_input_dir, train_target_dir,
                     patch_layout = patch_layout,
                     device = device,
                     mem = True)
test_ds = CtDataset(test_input_dir, test_target_dir,
                     patch_layout = patch_layout,
                     device = device,
                     mem = False)

n_train = len(train_ds)
n_test = len(test_ds)

print('Time:', time.time() - time_s)
print('data_shape', train_ds.data_shape)
print('patch_size', train_ds.patch_size)
print('n_patches', train_ds.n_patches)
print('|training|:', n_train)
print('|test|:', n_test, flush=True)

# Constructing the model
# Model constructing

print('Model constructing', flush=True)
#device = torch.device("cuda:0") if torch.cuda.is_available() else None
if model_id == 'unet':
    model = SinoNet(residual = False)
elif model_id == 'runet':   
    model = SinoNet(residual = True)
elif model_id == 'aunet':
    model = AUNet(img_ch = 1, output_ch = 1, residual = True)
model.to(device)
init_weights(model)

# Loss
if loss_name == 'L1':
    criterion = torch.nn.L1Loss(reduction='mean')
    sa = ''
elif loss_name == 'SSIML1':
    criterion = SSIM(loss = True)
    criterion.L1 = True
    criterion.alpha = ssiml1_alpha
    print('ssiml1_alpha', ssiml1_alpha, flush=True)
    sa = '_sa' + str(int(ssiml1_alpha * 100))
# create your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#print(model)

# Training
print('Training', flush=True)
model_name = model_id + '_' + loss_name # + sa
model_dir = work_dir + 'mdl/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
#print('torch.seed', torch.seed())        
resolution = 512 # the size of each reconstructed image
epochs = 20 # the number of traing epochs
batch_size = 1 
n_log_batches = 100 # log the training state per n_log_batches

# test data item
test_file_id = 6
test_image_id = 128
test_filename = test_ds.filenames[test_file_id]
_test_input, _test_target = test_ds.volume(test_filename, test_image_id)
test_input = vgi.toNumpy(_test_input.squeeze())
test_target= vgi.toNumpy(_test_target.squeeze())
test_vmin = np.min(test_input)
test_vmax = np.max(test_input)
test_gt = np.load(gt_dir + test_filename + '.npy')[test_image_id]

# Reconstruction configuration.
rec_shape = (resolution, resolution)
sino_shape = test_ds.data_shape
print('rec_shape', rec_shape)
print('sino_shape', sino_shape)
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

mask = createCircleMask(shape = rec_shape, r = resolution / 2).astype(np.float32)

time_s = time.time()
for i_e in range(0, epochs):
    n_batches = 0
    loss_batch_sum = 0.0
    itr = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    for i_b, (_input_patches, _target_patches) in enumerate(itr):  
        optimizer.zero_grad()
        _predictions = model(_input_patches)
        loss = criterion.forward(_predictions, _target_patches)
        loss.backward()
        optimizer.step()        
        loss_batch_sum += loss
        n_batches += 1
        if (i_b + 1) % n_log_batches == 0:
            t = time.time() - time_s  
            print('epoch:', i_e, 'batch:', i_b + 1, 'loss:', loss, 'time:', t, flush=True)                
    
    t = time.time() - time_s    
    print('epoch', i_e)
    print('time:', t, flush=True)
    loss_avg = loss_batch_sum / n_batches
    
    # Model save 
    if (i_e + 1) == epochs:
        if os.path.isfile(model_path):
            os.remove(model_path)
        model_path = model_dir + model_name + '_e' + str(i_e + 1) + '.mdl'
    else:
        model_path = model_dir + model_name + '_temp.mdl'
    torch.save(model.state_dict(), model_path)    
    print("Epoch:", i_e, ", Loss: ", loss_avg, flush=True)  
    
    # Test
    test_output = test_ds.feed(model = model, file_id = test_file_id, image_id = test_image_id)[0]
    print('test_output', vgi.metric(test_output))
    print('target', vgi.metric(test_target))
    
    #vgi.showImage(vgi.normalize(test_output))       
    test_recon = fbp.reconstruct(test_output) * mask
    print('test_recon', vgi.metric(test_recon), test_recon.shape)
    test_recon = vgi.normalize(test_recon) * mask
    #vgi.showImage(test_recon)
    
    test_d = test_recon - test_gt
    test_l1 = np.mean(np.abs(test_d))
    test_rmse = np.sqrt(np.mean(test_d**2)) 
    test_ssim = ssim(test_recon, test_gt, data_range = test_recon.max() - test_recon.min())
    err_L1 = np.mean(np.abs(test_output - test_target))
    print('proj err_L1:', err_L1)    
    print('test_l1:', test_l1)
    print('test_rmse:', test_rmse)
    print('test_ssim:', test_ssim, flush=True)    

    print('----------------------------------------')            
    # batch loop 
# epoch loop


# Outputting sinograms
print('Outputting sinograms')
# Model loading
model_path = model_dir + model_name + '_e20' + '.mdl'
print('model_path', model_path, flush=True)
model.load_state_dict(torch.load(model_path))
model.eval() # for testing

batch_size = 1
out_params = [[ train_ds, 
                train_input_dir, 
                work_dir + '_' + model_name + '/',
                work_dir + '_rec_' + model_name + '/'],
              [ test_ds, 
                test_input_dir, 
                work_dir + '_' + model_name + '_test/',
                work_dir + '_rec_' + model_name + '_test/',
                ]]
for out_param in out_params:
    ds = out_param[0]
    input_dir = out_param[1]
    out_dir = out_param[2]
    print('input_dir', input_dir)
    print('out_dir', out_dir, flush=True)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)        

    n_batches = ds.n_images // batch_size
    i_slice = ds.n_images // 2
    sino_shape = ds.data_shape
    output_shape = [ds.n_images] + ds.data_shape
    n_angles, n_detectors = sino_shape
    n_files = len(ds.filenames)

    print('output_shape:', output_shape, flush=True)
    for file_id in range( n_files):
        filename = ds.filenames[file_id]
        output_file = np.zeros(output_shape, dtype = np.float32)
        for i in range(n_batches):
            image_id = i * batch_size
            image_ide = image_id + batch_size
            output_batch = ds.feed(model = model, file_id = file_id, image_id = image_id, batch_size = batch_size)
            output_file[image_id:image_ide] = output_batch.astype(np.float32)
        
        print('output_file:', filename, output_file.shape, vgi.metric(output_file))
        for i in range(ds.n_images):
            vmin = np.min(ds.input_file[i])
            vmax = np.max(ds.input_file[i])            
            output_file[i] = vgi.normalize(output_file[i], vmin, vmax)        
        print('normalized output_file:', vgi.metric(output_file))    
        #vgi.showImage(vgi.normalize(output_file[i_slice]))
        
        out_path = out_dir + filename + '.npy'
        np.save(out_path, output_file)    
        print('Saved:', out_path, flush=True)

# Reconstruction
print('Reconstruction', flush=True)
for out_param in out_params:
    sino_dir = out_param[2]
    recon_dir = out_param[3]
    print('sino_dir', sino_dir)
    print('recon_dir', recon_dir)

    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)
    paths = vgi.getFiles(sino_dir)
    n = len(paths)
    print('#data:', n)

    # Reconstruction configuration.
    resolution = 512
    rec_shape = (resolution, resolution)
    sino_ini = np.load(paths[0])
    n_images, n_angles, n_detectors = sino_ini.shape
    i_slice = n_images // 2
    sino_shape = (n_angles, n_detectors)
    print('rec_shape', rec_shape)
    print('sino_shape', sino_shape, flush=True)
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
    mask = createCircleMask(shape = rec_shape, r = resolution / 2).astype(np.float32)

    for path in paths:
        _, filename, extname = vgi.parsePath(path)
        if extname != '.npy':
            continue    
        print('[sino]', path)
        sino_set = np.load(path)
        rec_imgs = np.zeros([n_images, resolution, resolution], dtype = np.float32)
        i = 0 
        for sino in sino_set:
            rec_img = fbp.reconstruct(sino).astype(np.float32)             
            rec_imgs[i] = rec_img 
            i += 1
        print('vrange:', np.min(rec_imgs), np.max(rec_imgs))
        #rec_imgs = vgi.normalize(rec_imgs) * mask 
        
        rec_imgs = np.clip(rec_imgs, 0.0, rec_imgs.max())
        rec_imgs = vgi.normalize(rec_imgs* mask) * mask
        
        recon_path = recon_dir + filename + '.npy'   
        np.save(recon_path, rec_imgs)        
        #vgi.showImage( rec_imgs[i_slice] )
        print('Saved:', recon_path, flush=True)

# Evaluation (it only evaluates the reconstruction results of test dataset)
print('Evaluation')
recon_dir = out_params[1][3]
print('recon_dir', recon_dir)
mean_mae, mean_mse, mean_ssim, mean_psnr = evaluate(recon_dir, gt_dir)
print('mean_mae, mean_mse, mean_ssim, mean_psnr')
print(mean_mae, mean_mse, mean_ssim, mean_psnr, flush=True)        