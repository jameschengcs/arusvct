# Parallel-beam tomography 
# (c) 2022, Chang-Chieh Cheng, jameschengcs@nycu.edu.tw

import numpy as np
import copy
import sys
sys.path.append('../')
import vgi
import astra

__all__ = ('ParallelRec', 'padSino', 'createCircleMask')

def padSino(sino, pad_ratio = 1.5):
    n_ang, n_col = sino.shape
    n_det = int(n_col * pad_ratio)
    n_det_d = n_det - n_col
    n_det_d += n_det_d & 1 # let n_det be even
    n_deth = n_det_d // 2
    left_pad = np.tile(np.expand_dims(sino[:, 0], -1), (1, n_deth))
    right_pad = np.tile(np.expand_dims(sino[:, -1], -1), (1, n_deth))
    #print('left_pad', left_pad.shap
    sino_pad = np.zeros([n_ang, n_det])
    sino_pad[:, 0:n_deth] = left_pad
    sino_pad[:, -n_deth:] = right_pad
    sino_pad[:, n_deth:n_deth + n_col] = sino   
    return sino_pad

def createCircleMask(shape, r, center = (0, 0), smooth = True):
    h, w = shape[0:2]
    rr = r * r
    h_h = h // 2
    h_min, h_max = -h_h, h_h
    w_h = w // 2
    w_min, w_max = -w_h, w_h
    mask_x = np.arange(w_min, w_max) - center[0]
    mask_y = np.arange(h_min, h_max) - center[1]
    x, y = np.meshgrid(mask_x, mask_y)
    #print(y[0])
    
    if smooth:
        z = x*x / r + y*y / r
        mask = np.where(z <= r, 0.0, z - r )        
        mask = np.where(mask < 1.0, np.abs(mask - 1), 0)        
    else:
        z = x*x / rr + y*y / rr
        mask = np.where(z <= 1, 1, 0)
    mask = mask.astype(np.float32)
    return mask    

# ----------------------------------------------------
class ParallelRec:
    def __init__(self, rec_shape, sino_shape, scan_range = (0, np.pi), angles = None, sino = None, algo = 'FBP', iterations = 10, gpu = True):
        self.gpu = gpu
        self.rec_shape = rec_shape
        self.sino_shape = sino_shape
        self.scan_range = scan_range
        self.vol_geom = astra.create_vol_geom(self.rec_shape[0], self.rec_shape[1])
        # create_proj_geom('parallel', detector_spacing, det_count, angles)
        if angles is None:
            self.angles = np.linspace(self.scan_range[0], self.scan_range[1], self.sino_shape[0], False)
        else:
            self.angles = angles
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, self.sino_shape[1], self.angles)

        # For CPU-based algorithms, a "projector" object specifies the projection
        # model used. In this case, we use the "strip" model.
        # Available algorithms:
        # ART, SART, SIRT, CGLS, FBP
        if self.gpu:
            self.algo = algo + '_CUDA'
            self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
            self.cfg = astra.astra_dict(self.algo)
        else:
            self.algo = algo
            self.proj_id = astra.create_projector('strip', self.proj_geom, self.vol_geom)
            self.cfg = astra.astra_dict(self.algo)
        self.rec_id = astra.data2d.create('-vol', self.vol_geom)    
        self.sino_id  = astra.data2d.create('-sino', self.proj_geom, data = sino)
    
        self.cfg['ProjectorId'] = self.proj_id
        self.cfg['ReconstructionDataId'] = self.rec_id    
        self.cfg['ProjectionDataId'] = self.sino_id
        self.alg_id = astra.algorithm.create(self.cfg)  
        self.iterations = iterations
    
    def sino(self, img, keep_id = False):
        sinogram_id, sinogram = astra.create_sino(img, self.proj_id)
        if keep_id:
            return sinogram_id, sinogram
        else:
            sinogram = np.array(sinogram)
            astra.data2d.delete(sinogram_id)
            return sinogram

    def reconstruct(self, sino = None):
        if not(sino is None):
            astra.data2d.store(self.sino_id, sino)
            astra.algorithm.run(self.alg_id, self.iterations)
            rec = astra.data2d.get(self.rec_id)   
            rec = rec.astype(np.float32)
            return rec
    # Calling destructor
    def release(self):
        astra.data2d.delete(self.sino_id)
        astra.algorithm.delete(self.alg_id)
        astra.data2d.delete(self.rec_id)
        astra.projector.delete(self.proj_id) 
 