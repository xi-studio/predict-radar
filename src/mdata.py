import os
import numpy as np
import glob
import random
import cPickle
import gzip


res = glob.glob('/data/radar_ftp/*.h5')
random.shuffle(res)

f = gzip.open('/home/tree/predict-radar/data/idx.pkl.gz','w')
f.write(cPickle.dumps(res[:20000]))
f.write(cPickle.dumps(res[20000:25000]))
f.close()




   
