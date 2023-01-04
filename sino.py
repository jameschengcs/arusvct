# Creating 720-view projections with a fan-beam-flat-detector scanner
import os
import sys
import glob
import numpy as np
import vgi
from vgi.ct import FanRec

#data_dir = 'D:/Data/MIDRC/'
#ct_dir = data_dir + 'ct/'
#proj_dir = data_dir + 'proj/'
#ct_dir = data_dir + 'ct_test/'
#proj_dir = data_dir + 'proj_test/'

n_args = len(sys.argv)

# Source directory
ct_dir = sys.argv[1] 
# Target directory
proj_dir = sys.argv[2] 
# views
n_angles = 720 if n_args < 4 else int(sys.argv[3])

ct_paths = vgi.getFiles(ct_dir)
n_ct = len(ct_paths)
print('#ct:', n_ct)

if not os.path.exists(proj_dir):
    os.makedirs(proj_dir)

rec_shape = (512, 512)
n_detectors = 768
sino_shape = (n_angles, n_detectors)
det_width = 1.0
source_origin = 512.
origin_det = 512.
ang_range = np.pi * 2
rec_angles = np.linspace(0, ang_range, num = n_angles, endpoint = False)
scanner = FanRec(rec_shape = rec_shape, 
             sino_shape = sino_shape, 
             angles = rec_angles,
             det_width = det_width,
             source_origin = source_origin,
             origin_det = origin_det)

i = 0
for ct_path in ct_paths:
    print(ct_path)
    _, filename, _ = vgi.parsePath(ct_path)
    
    obj = np.load(ct_path)
    n_slices = obj.shape[0]
    print('#slices:', n_slices)
    proj = None

    for j in range(n_slices):
        p = np.expand_dims(scanner.sino(obj[j]), axis = 0)
        #print('p', p.shape, vgi.metric(p))
        #rec_img = fbp.reconstruct(p[0]).astype(np.float32)
        #vgi.showImage(vgi.normalize(rec_img))        
        if proj is None:
            proj = p
        else:
            proj = np.concatenate((proj, p))
    proj_path = proj_dir + filename + '.npy'
    np.save(proj_path, proj)
    print('proj:', proj_path, proj.shape, vgi.metric(proj))

