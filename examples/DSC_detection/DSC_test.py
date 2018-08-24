import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os
import scipy.misc
import time



# Make sure that caffe is on the python path:
caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

EPSILON = 1e-8


def getFileList( p ):
        p = str( p )
        if p=="":
              return [ ]
        #p = p.replace( "/","/")
        if p[ -1] != "/":
             p = p+"/"
        a = os.listdir( p )
        b = [ x   for x in a if os.path.isfile( p + x ) ]
        return b
 
    
data_root = '/home/wpf/data/SBU-shadow/SBU-Test/'
with open('../../data/SBU/test.txt') as f: 
    
#data_root = '/home/xwhu/data/UCF-shadow/test/'
#with open('../../data/UCF/test.txt') as f: 
    test_name = f.readlines()
    
test_lst = [data_root+x.strip() for x in test_name]


#remove the following two lines if testing with cpu
# caffe.set_mode_gpu()
caffe.set_mode_cpu()
# choose which GPU you want to use
caffe.set_device(0)
caffe.SGDSolver.display = 0
# load net
net = caffe.Net('deploy.prototxt', 'snapshot/DSC_iter_12000.caffemodel', caffe.TEST)


#Visualization
def plot_single_scale(scale_lst, name_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    plt.figure()
    for i in range(0, len(scale_lst)):
        s = plt.subplot(1,5,i+1)
        s.set_xlabel(name_lst[i], fontsize=10)
        if name_lst[i] == 'Source':
            plt.imshow(scale_lst[i])
        else:
            plt.imshow(scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()


usedtime = 0;

for idx in range(len(test_lst)):

    # load image
    img = Image.open(test_lst[idx])

    img = img.resize((400,400));
    if img.mode == 'L':
        img_temp = np.zeros((img.size[1], img.size[0], 3))
        img_temp[:,:,0] = img
        img_temp[:,:,1] = img
        img_temp[:,:,2] = img
        img = img_temp
    
    
    img = np.array(img, dtype=np.uint8)
    im = np.array(img, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = im.transpose((2,0,1))

    # load gt
    gt = Image.open(test_lst[idx])

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    # run net and take argmax for prediction
    
    start_time = time.clock()
    net.forward()
    usedtime = usedtime + time.clock() - start_time
    #avgtime = usedtime/(idx+1)
    
    out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
    out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
    out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
    out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
    out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
    out6 = net.blobs['sigmoid-dsn6'].data[0][0,:,:]
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    
    glob = net.blobs['sigmoid-global'].data[0][0,:,:]

    
    res = (glob + fuse) / 2
 
    res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
    name_lst = ['SO1', 'SO2', 'SO3', 'SO4', 'SO5']
    name_lst = ['SO6', 'Fuse', 'Result', 'Source', 'GT']
 
    # save image
    scipy.misc.imsave('../result/' + test_name[idx][13:-5] + '.jpg',res)
    
    if (idx+1)%100 == 0:
        print("%d / %d, time: %f s" % (idx+1,len(test_lst),usedtime/(idx+1)))

print 'done!'


