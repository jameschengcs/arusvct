import numpy as np
import os
import sys
import vgi
from vgi.ct import createCircleMask
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

ref_means = [0.31614921663988865, 0.35865365232842134, 0.321694478688629, 0.32748706128057975,
             0.31301958852873246, 0.32856736419276633, 0.1885250101617685, 0.13090186791437852]

def meannml(data, ref_mean):
    data_n = vgi.normalize(data)  
    vmean = np.mean(data_n)    
    w = ref_mean/ vmean
    data_n = data_n * w
    #data_n = np.clip(data_n, 0.0, 1.0)
    return data_n

def postproc(data, mask, clip_min, clip_max, mean_align, ref_mean):
    if clip_min:
        data = np.clip(data, 0.0, data.max())
    if mean_align:
        data = meannml(data * mask, ref_mean = ref_mean) * mask      
    else:
        data = vgi.normalize(data * mask) * mask
    if clip_max:
        data = np.clip(data, 0.0, 1.0)  
    return data

def evaluate(target_dir, gt_dir, mask_ratio = 0.99, 
             clip_min = True, clip_max = False, mean_align = True, 
             return_all = False, ref_means = None):
    
    L_ssim = []
    L_mae = []
    L_mse = []
    L_psnr = []
    paths = vgi.getFiles(target_dir)
    rec_shape = None
    mask = None #createCircleMask(shape = rec_shape, r = 512 / 2)
    k = 128
    
    i = 0
    ref_mean = None
    for path in paths:        
        _, filename, extname = vgi.parsePath(path)
        if extname != '.npy':
            continue 
        data = np.load(path) # (images, width, height)
        if rec_shape is None:
            rec_shape = data.shape[1:] # (width, height))
            mask = createCircleMask(shape = rec_shape, r = rec_shape[0] / 2 * mask_ratio)
        
        gt_path = gt_dir + filename + '.npy'
        gt_data = np.load(gt_path) * mask

        if ref_means is None:
            ref_mean = np.mean(gt_data)
        else:
            ref_mean = ref_means[i]
        
        data = postproc(data, mask = mask, clip_min = clip_min, clip_max = clip_max, mean_align = mean_align, ref_mean = ref_mean)    

        diff = data - gt_data
        v_mae = np.mean(np.abs(diff))
        v_mse = mse(data, gt_data) 
        v_ssim = ssim(data, gt_data, data_range = gt_data.max() - gt_data.min())
        v_psnr = psnr(data, gt_data, data_range = gt_data.max() - gt_data.min())
        #print(path)
        print(v_mae, v_mse, v_ssim, v_psnr)
        L_mae += [v_mae]
        L_mse += [v_mse]
        L_ssim += [v_ssim]
        L_psnr += [v_psnr]
        i += 1
        #break
    L_mae = np.array(L_mae)
    L_mse = np.array(L_mse)
    L_ssim = np.array(L_ssim)
    L_psnr = np.array(L_psnr)

    if return_all:
        return L_mae, L_mse, L_ssim, L_psnr
    else:
        mean_mae = float(np.mean(L_mae))
        mean_mse = float(np.mean(L_mse))
        mean_ssim = float(np.mean(L_ssim))
        mean_psnr = float(np.mean(L_psnr))
        return mean_mae, mean_mse, mean_ssim, mean_psnr
        
def display(target_dir, gt_dir, k = 128, filenames = None, mask_ratio = 0.99,
             clip_min = True, clip_max = False, mean_align = True, 
             ref_means = None, figsize = None, gt = True):    
    paths = vgi.getFiles(target_dir)
    rec_shape = None
    mask = None #createCircleMask(shape = rec_shape, r = 512 / 2)
    out = []
    i = 0
    ref_mean = None
    for path in paths:        
        _, filename, extname = vgi.parsePath(path)
        if extname != '.npy':
            continue 
        if not(filenames is None) and not(filename in filenames):
            continue
        data = np.load(path) # (images, width, height)
        if rec_shape is None:
            rec_shape = data.shape[1:] # (width, height))
            mask = createCircleMask(shape = rec_shape, r = rec_shape[0] / 2 * mask_ratio)
        
        gt_path = gt_dir + filename + '.npy'
        gt_data = np.load(gt_path) * mask

        if not(ref_means is None):
            ref_mean = ref_means[i]
        data = postproc(data, mask = mask, clip_min = clip_min, clip_max = clip_max, mean_align = mean_align, ref_mean = ref_mean)    
        vgi.showImage(vgi.normalize(data[k]), figsize = figsize)
        out += [data[k]]
        if gt:
            vgi.showImage(vgi.normalize(gt_data[k]), figsize = figsize)
    return out