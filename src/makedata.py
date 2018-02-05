import os
import numpy as np
from scipy import misc
import glob
import h5py

def save(path,d):
    f = h5py.File(path,'w')
    f['input']  = d[:10]
    f['output'] = d[10:]
    f.close()

res = glob.glob('/data/denoise/230/*/*/*/*/*/*.png')
res.sort()
base = np.zeros((20,256,256),dtype=np.uint8)

num = 0 
iters = 0
for x in res:
    m = misc.imread(x,'L')
    m = misc.imresize(m,(256,256))
    base[num] = m
    
    if num==19:
        num = 0
        if np.sum(base)/255.0 >5000:
            save('/data/radar_ftp/%d.h5'%iters,base)
            print iters
            iters += 1
        continue
    
    num = num + 1


   
