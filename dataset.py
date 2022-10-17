import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sys
import vgi
from vgi.ct import FanRec

def fanFlatFBP(rec_shape, sino_shape):
    # Reconstruction configuration.
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
    return fbp  

# Each dataset should be an npy file with the shape (images, height, width)
# For sino(720, 768)
#       patch_layout = (9, 8)
#       patch_size = (80, 96) /16 => (5, 6)
#       over_lap = patch_size / 2 = (40, 48)
# For reconstruction(512, 512)
#       patch_layout = (8, 8)
#       patch_size = (64, 64) /16 => (4, 4)
#       over_lap = patch_size / 2 = (32, 32)
class CtDataset(Dataset):
        def __init__(self, input_dir, target_dir, 
                     patch_layout = (9, 8),
                     normalize = False,
                     device = None,
                     mem = True,
                     np_dtype = np.float32,
                     torch_dtype = torch.float32):
            if device is None:
                self.device = torch.device("cpu")
            else:
                self.device = device
            self.np_dtype  = np_dtype
            self.torch_dtype = torch_dtype

            self.input_dir = input_dir
            self.target_dir = target_dir
            self.filenames = []   

            self.data_shape = [0, 0]
            self.n_images = 0
            paths_init = vgi.getFiles(self.input_dir)
            for path in paths_init:
                _, filename, extname = vgi.parsePath(path)
                if extname == '.npy':
                    self.filenames += [filename]
                    if self.n_images == 0:
                        data = np.load(path)
                        self.n_images, self.data_shape[0], self.data_shape[1] = data.shape                       
 
            n_dims = len(self.data_shape)       
            self.patch_layout = patch_layout 
            self.patch_size = [self.data_shape[i] // self.patch_layout[i] for i in range(n_dims)]
            self.overlap_shape = [self.patch_size[i] // 2 for i in range(n_dims)]
            loc = [list(range(0, self.data_shape[i] - self.overlap_shape[i], self.overlap_shape[i])) for i in range(n_dims) ]
            self.patch_ranges = np.array([[r, r + self.patch_size[0], c, c + self.patch_size[1]]  for r in loc[0] for c in loc[1]])
            self.n_patches = self.patch_ranges.shape[0]
            self.overlap_count = np.zeros(self.data_shape)
            for row, row_e, col, col_e in self.patch_ranges:
                self.overlap_count[row: row_e, col: col_e] += 1.0

            L = [['', i, p] for i in range(self.n_images) for p in range(self.n_patches)]
            self.idx = []
            for filename in self.filenames:
                Li = copy.deepcopy(L)
                for t in Li:
                    t[0] = filename
                self.idx += Li
            self.n = len(self.idx)

            self.mem = mem
            self.input_file = None
            self.target_file = None
            self.filename = None
            self.normalize = normalize            
            self.input_dataset = {}
            self.target_dataset = {}
            self.input_vranges = {}
            self.target_vranges = {}
            if self.mem:
                self.loadMem()

        # CtDataset::__init__

        def loadMem(self):         
            for filename in self.filenames:
                npy_name = filename + '.npy'
                input_path = self.input_dir + npy_name
                target_path = self.target_dir + npy_name
                input_data= np.load(input_path).astype(dtype = self.np_dtype)
                target_data = np.load(target_path).astype(dtype = self.np_dtype)  
                if self.normalize:
                    input_vr = (np.min(input_data), np.max(input_data))
                    target_vr = (np.min(target_data), np.max(target_data))
                    self.input_vranges[filename] = (input_vr)
                    self.target_vranges[filename] = (target_vr)
                    input_data = vgi.normalize(input_data) 
                    target_data = vgi.normalize(target_data) 

                self.input_dataset[filename] = input_data
                self.target_dataset[filename] = target_data 
        # CtDataset::loadMem


        def __len__(self):
            return self.n

        def item(self, idx):
            filename, image_id, patch_id = self.idx[idx]
            if filename != self.filename:
                if self.mem:
                    self.input_file = self.input_dataset[filename]
                    self.target_file = self.target_dataset[filename]  
                else:    
                    npy_name = filename + '.npy'
                    input_path = self.input_dir + npy_name
                    target_path = self.target_dir + npy_name
                    self.input_file = np.load(input_path).astype(dtype = self.np_dtype)
                    self.target_file = np.load(target_path).astype(dtype = self.np_dtype)                 
                    if self.normalize:
                        self.input_file = vgi.normalize(self.input_file)
                        self.target_file = vgi.normalize(self.target_file)                    
                self.filename = filename         

            row, row_e, col, col_e = self.patch_ranges[patch_id]    
            input_data = self.input_file[image_id, row:row_e, col:col_e]
            target_data = self.target_file[image_id, row:row_e, col:col_e]
                
            input_data = torch.tensor(input_data, dtype = self.torch_dtype, device = self.device)
            target_data = torch.tensor(target_data, dtype = self.torch_dtype, device = self.device)
            input_data = input_data.unsqueeze(0)
            target_data = target_data.unsqueeze(0)

            return input_data, target_data, filename, image_id, patch_id   
        # CtDataset::item          

        def __getitem__(self, idx):
            input_data, target_data, filename, image_id, patch_id = self.item(idx)
            return input_data, target_data

        def volume(self, filename, image_id = None, patch_id = None, target = True):
            if self.mem:
                self.input_file = self.input_dataset[filename]
                self.target_file = self.target_dataset[filename]
            else:
                npy_name = filename + '.npy'
                input_path = self.input_dir + npy_name
                target_path = self.target_dir + npy_name  

                self.input_file = np.load(input_path).astype(dtype = self.np_dtype)
                self.target_file = np.load(target_path).astype(dtype = self.np_dtype) 
                if self.normalize:
                    self.input_file = vgi.normalize(self.input_file)
                    self.target_file = vgi.normalize(self.target_file)                 
            self.filename = filename  
                
            input_data = torch.tensor(self.input_file, dtype = self.torch_dtype, device = self.device)
            input_data = input_data.unsqueeze(1)
            if target:
                target_data = torch.tensor(self.target_file, dtype = self.torch_dtype, device = self.device)
                target_data = target_data.unsqueeze(1)   
            else:
                target_data = None
            
            
            if image_id is None:
                if target:
                    return input_data, target_data 
                else:    
                    return input_data                           
            elif patch_id is None:
                if target:
                    return input_data[image_id].unsqueeze(0), target_data[image_id].unsqueeze(0)
                else:   
                    return input_data[image_id].unsqueeze(0)  
            else:
                row, row_e, col, col_e = self.patch_ranges[patch_id] 
                if target:   
                    return input_data[image_id, :, row:row_e, col:col_e].unsqueeze(0), target_data[image_id, :, row:row_e, col:col_e].unsqueeze(0)  
                else:         
                    return input_data[image_id, :, row:row_e, col:col_e].unsqueeze(0)   
        # CtDataset::volume  

        # For model output
        def loadInputPatches(self, file_id, image_s, patch_id, image_e = None, target = False):
            filename = self.filenames[file_id]
            if self.filename != filename:
                if self.mem:
                    self.input_file = self.input_dataset[filename]
                    self.target_file = self.target_dataset[filename]
                else:
                    npy_name = filename + '.npy'
                    input_path = self.input_dir + npy_name
                    target_path = self.target_dir + npy_name  

                    self.input_file = np.load(input_path).astype(dtype = self.np_dtype)
                    self.target_file = np.load(target_path).astype(dtype = self.np_dtype) 
                    if self.normalize:
                        self.input_file = vgi.normalize(self.input_file)
                        self.target_file = vgi.normalize(self.target_file)
                self.filename = filename
            if image_e is None:
                image_e = image_s + 1
            row, row_e, col, col_e = self.patch_ranges[patch_id]
            
            input_images = self.input_file[image_s:image_e] #[n, h, w]
            input_patches = input_images[..., row:row_e, col:col_e] #[n, h, w]
            _input_patches = torch.tensor(input_patches, dtype = self.torch_dtype, device = self.device)
            _input_patches = _input_patches.unsqueeze(1) #[n, 1, h, w]
            if target == False:
                return _input_patches
            target_images = self.target_file[image_s:image_e] #[n, h, w]
            target_patches = target_images[..., row:row_e, col:col_e] #[n, h, w]
            _target_patches = torch.tensor(target_patches, dtype = self.torch_dtype, device = self.device)
            _target_patches = _target_patches.unsqueeze(1) #[n, 1, h, w]
            return _input_patches, _target_patches
        # CtDataset::loadInputPatches  


        def feed(self, model, file_id, image_id, batch_size = 1, target = False):
            out_shape = [batch_size] + self.data_shape
            outputs = np.zeros(out_shape, dtype = self.np_dtype)
            for patch_id in range(self.n_patches):
                _input_patches = self.loadInputPatches(file_id = file_id, patch_id = patch_id,
                                                       image_s = image_id, image_e = image_id + batch_size, 
                                                       target = target)
                if target:
                    _target_patches = _input_patches[1]
                    _input_patches = _input_patches[0]
                if model is None:
                    _output_patches = _input_patches
                else:
                    _output_patches = model(_input_patches)
                output_patches = vgi.toNumpy(_output_patches.squeeze()).astype(self.np_dtype)
                row, row_e, col, col_e = self.patch_ranges[patch_id]
                outputs[:, row: row_e, col: col_e] += output_patches                
            outputs /= self.overlap_count     
            if target:
                return outputs, vgi.toNumpy(_target_patches.squeeze()) 
            else:
                return outputs
        # CtDataset::feed

#@ CtDataSet    


class SinoDataset(Dataset):
        def __init__(self, input_dir,
                     intervals = [8, 4, 2], 
                     device = None,
                     mem = True,
                     np_dtype = np.float32,
                     torch_dtype = torch.float32):
            if device is None:
                self.device = torch.device("cpu")
            else:
                self.device = device
            self.np_dtype  = np_dtype
            self.torch_dtype = torch_dtype

            self.input_dir = input_dir
            self.filenames = []   

            self.sino_shape = [0, 0]
            self.n_images = 0
            self.n_angles = 0
            self.n_detectors = 0
            paths_init = vgi.getFiles(self.input_dir)
            for path in paths_init:
                _, filename, extname = vgi.parsePath(path)
                if extname == '.npy':
                    self.filenames += [filename]
                    if self.n_images == 0:
                        data = np.load(path)
                        self.n_images, self.n_angles, self.n_detectors = data.shape    
                        self.sino_shape = (self.n_angles, self.n_detectors)
 

            self.intervals = intervals
            L = []       
            for i in range(self.n_images):     
                for d in self.intervals:
                    offset = d // 2
                    for k in range(self.n_angles):  
                        L += [['', i, k, (k + d) % self.n_angles, (k + offset) % self.n_angles]]

            self.idx = []
            for filename in self.filenames:
                Li = copy.deepcopy(L)
                for t in Li:
                    t[0] = filename
                self.idx += Li
            self.n = len(self.idx)

            self.pred_idx = []
            for d in self.intervals:
                n_angles_d = self.n_angles // d
                offset = d // 2
                Ld = []
                for k in range(n_angles_d):
                    s = k * d
                    Ld += [[s, (s + d) % self.n_angles, (s + offset) % self.n_angles ]] # [[sino_i, sino_j, sino_k]]
                    #print(s, (s + d) % n_angles, s + offset)
                self.pred_idx += [Ld]            

            self.mem = mem
            self.input_file = None
            self.filename = None        
            self.input_dataset = {}
            if self.mem:
                self.loadMem()

        # SinoDataset::__init__

        def loadMem(self):         
            for filename in self.filenames:
                npy_name = filename + '.npy'
                input_path = self.input_dir + npy_name
                input_data= np.load(input_path).astype(dtype = self.np_dtype)
                self.input_dataset[filename] = input_data
        # SinoDataset::loadMem


        def __len__(self):
            return self.n

        def loadData(self, filename):
            if filename != self.filename:
                if self.mem:
                    self.input_file = self.input_dataset[filename]
                else:    
                    npy_name = filename + '.npy'
                    input_path = self.input_dir + npy_name
                    self.input_file = np.load(input_path).astype(dtype = self.np_dtype)                
                self.filename = filename   
            return self.input_file     
        # SinoDataset::loadData

        def item(self, idx):
            filename, image_id, sino_i, sino_j, sino_k = self.idx[idx]
            data = self.loadData(filename)
            input_data = np.concatenate( 
                            [np.expand_dims(data[image_id, sino_i, :], 0),
                             np.expand_dims(data[image_id, sino_j, :], 0)])
            target_data = np.expand_dims(data[image_id, sino_k, :], 0)

            input_data = torch.tensor(input_data, dtype = self.torch_dtype, device = self.device)
            target_data = torch.tensor(target_data, dtype = self.torch_dtype, device = self.device)
            input_data = input_data.unsqueeze(0)
            target_data = target_data.unsqueeze(0)

            return input_data, target_data, filename, image_id, sino_i, sino_j, sino_k   
        # SinoDataset::item          

        def __getitem__(self, idx):
            input_data, target_data, filename, image_id, sino_i, sino_j, sino_k = self.item(idx)
            return input_data, target_data

         

        # For model output
        def loadBatch(self, image_id, sino_i, sino_j, batch_size = 1, file_id = None, data = None, _tensor = True):
            if data is None:
                data = self.loadData(self.filenames[file_id])

            image_e = (image_id + batch_size)
            image_e = image_e if image_e < self.n_images else self.n_images
            
            Ai = data[image_id:image_e, sino_i, :]
            Aj = data[image_id:image_e, sino_j, :]
            Ai = np.expand_dims(Ai, 1)
            Aj = np.expand_dims(Aj, 1)
            B = np.concatenate([Ai, Aj], axis = 1)
            B = np.expand_dims(B, 1)       
            if _tensor:
                return torch.tensor(B, dtype = self.torch_dtype, device = self.device)
            else:  
                return B
        # SinoDataset::loadBatch  


        def feed(self, model, file_id, image_id, batch_size = 1):
            out_shape = [batch_size] + list(self.sino_shape) # (batch_size, n_angles, n_detectors)
            data = self.loadData(self.filenames[file_id])
            outputs = np.zeros(out_shape, dtype = self.np_dtype)

            #print('SinoDataset::feed::data', data.shape)
            #print('SinoDataset::feed::outputs', outputs.shape)

            # initializing
            for sino_i, sino_j, sino_k in self.pred_idx[0]:
                _input = self.loadBatch(data = data, image_id = image_id, batch_size = batch_size, 
                                        sino_i = sino_i, sino_j = sino_j, _tensor = True)                
                input_data = vgi.toNumpy(_input.squeeze(axis = 1))
                outputs[:, sino_i, :] = input_data[:, 0, :] 
                outputs[:, sino_j, :] = input_data[:, 1, :] 

            # Interpolating
            for lv_idx in self.pred_idx:
                for sino_i, sino_j, sino_k in lv_idx:
                    _input = self.loadBatch(data = outputs, image_id = 0, batch_size = batch_size, 
                                            sino_i = sino_i, sino_j = sino_j, _tensor = True)
                    # (batch_size, 1, 2, n_detectors)
                    #print('SinoDataset::feed::_input', _input.shape)

                    _output = model(_input).squeeze(axis = 1) # (batch_size, 1, n_detectors)
                    output = vgi.toNumpy(_output).astype(self.np_dtype) # (batch_size, 1, n_detectors)
                    outputs[:, sino_k, :] = output 

            return outputs
        # SinoDataSet::feed

#@ SinoDataSet 