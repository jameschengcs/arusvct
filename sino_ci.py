import os
import sys
import glob
import numpy as np
import vgi
import json
import time
from scipy.interpolate import CubicSpline


#sino_dir = 'D:/Data/MIDRC/fanflat/sino_test/'
#sino_intp_dir = 'D:/Data/MIDRC/test_ci/'
#smpd = 12

n_args = len(sys.argv)

# Source directory
sino_dir = sys.argv[1] 
# Target directory
sino_intp_dir = sys.argv[2] 
# sampling interval
smpd = int(sys.argv[3])


print('sino_dir:', sino_dir)
print('sino_intp_dir:', sino_intp_dir)
print('sampling interval:', smpd)

paths_init = vgi.getFiles(sino_dir)
paths = []
for path in paths_init:
    _, filename, extname = vgi.parsePath(path)
    if extname == '.npy':
        paths += [path]
n = len(paths)
print('#data:', n)
if n == 0:
    sys.exit(0)
data0 = np.load(paths[0])
data_shape = data0.shape
print('data_shape:', data_shape)
n_slices, n_angles, n_detectors = data0.shape
resolution  = 1 << int(np.log2(n_detectors))
i_slice = n_slices // 2
print('resolution:', resolution)

n_angles_rec = n_angles
n_angles_smp = int(n_angles / smpd)
n_angles_smp_rec = int(n_angles_rec / smpd)

print('View sample:', n_angles, '/', smpd, '=', n_angles_smp)
print('View sample for reconstruction:', n_angles_rec, '/', smpd, '=', n_angles_smp_rec)

if not os.path.exists(sino_intp_dir):
    os.makedirs(sino_intp_dir)
    
# For interpolation
ps = np.arange(n_angles)
p = np.arange(0, n_angles + smpd, smpd)

for path in paths:
    _, filename, extname = vgi.parsePath(path)
    if extname != '.npy':
        continue    
    print(path)
    sino_org = np.load(path).astype(np.float32)   
    print('sino_org', vgi.metric(sino_org), sino_org.shape) #  angles, rows, cols    
    sino_smp = sino_org[:, ::smpd, :]
    print('sino_smp', vgi.metric(sino_smp), sino_smp.shape) #  angles, rows, cols
    
    sino_smp_intp = np.concatenate([sino_smp, np.expand_dims(sino_org[:, 0, :], 1)], axis=1)
    print('sino_smp_intp', vgi.metric(sino_smp_intp), sino_smp_intp.shape) #  angles, rows, cols
    
    sino_intp = np.zeros(data_shape, dtype = np.float32)
    for i_sino in range(n_slices):
        sino_i = sino_smp_intp[i_sino]
        for i_det in range(n_detectors):
            y = sino_i[:, i_det]
            cs = CubicSpline(p, y, bc_type='periodic')
            ys = cs(ps)
            sino_intp[i_sino, :, i_det] = ys
    print('sino_intp', vgi.metric(sino_intp), sino_intp.dtype, sino_intp.shape) #  angles, rows, cols
    
    # save    
    sino_intp_path = sino_intp_dir + filename + '.npy'
    np.save(sino_intp_path, sino_intp)
    print('sinograms saved:', sino_intp_path, sino_intp.dtype)
   