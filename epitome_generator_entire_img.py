import torch as T
import numpy as np
from matplotlib import pyplot as pt
import random

import models
import util

device = T.device('cpu')
T.set_grad_enabled(True)

img, lc, nlcd = util.load_data()


# initialize model
epitome_size = 299
ep = models.EpitomeModel(epitome_size, 4).to(device)


# train the model (best run on GPU)
# see figure in SI for outputs

n_batches = 10000
batch_size = 256
show_interval = 100

diversify = True # tiny image, no need to 
#mask_threshold = float('1e-8') 
reset_threshold = 0.95

optimizer = T.optim.Adam(ep.parameters(), lr=0.003)

counter = T.zeros((ep.size, ep.size)).to(device)
img2 = img/255.0
img3 = img2.swapaxes(0,1).T

labelsRGB = util.vis_nlcd(nlcd, True)
labelsRGB2 = labelsRGB.swapaxes(1,2).T
#pt.imsave('labels_visualized.png', labelsRGB2)

centerPxls = np.where((nlcd == 41) | (nlcd == 90)) 
centerPxls = list(zip(centerPxls[0], centerPxls[1]))

wmax = 19
centerPxls = [pair for pair in centerPxls if pair[0] < img3.shape[1]-wmax+1 and pair[1] < img3.shape[2]-wmax+1]

for it in range(n_batches):
    w = np.random.randint(5,10)*2+1 # odd number 11 to 19 / patch size from an epitome diversified
    #wmax = 19
    
    #construct the batches
    batch = np.zeros((batch_size, 4, w, w))
    #img is of size 4 by 512 by 512

    xyPrPairs = dict()
    for b in range(batch_size):

        xyPair = random.choice(centerPxls)
        x = xyPair[0]
        y = xyPair[1]

        batch[b] = img3[:,x:x+w,y:y+w] 
        x_pr = int((wmax - w)//2 + (ep.size - wmax + 1) * (x + w//2)/img3.shape[1])
        y_pr = int((wmax - w)//2 + (ep.size - wmax + 1) * (y + w//2)/img3.shape[2])

        xyPrPairs[b] = [x_pr, y_pr]

        
    x = T.from_numpy(batch).to(device, T.float)
    
    
    
    optimizer.zero_grad()
    
    # compute p(x|s)p(s) and smooth
    e = ep(x) / (w/11)**2 
    for idx in list(xyPrPairs.keys()):
        x_pr = xyPrPairs[idx][0]
        y_pr = xyPrPairs[idx][1]
        e[:,idx, x_pr-w//2:x_pr+w//2+1, y_pr-w//2:y_pr+w//2+1] += 70
    
    
    # extract best-modeled quarter of data
    if diversify:
        indices = e.logsumexp((0,2,3)).topk(batch_size//4, largest=False, sorted=False).indices 
        e = e[:,indices]
    
    # compute log likelihood of data over unmasked positions (+const)
    loss = -T.logsumexp(e, (0,2,3))
    loss.mean().backward()
    optimizer.step()
    
    # clamp parameters
    with T.no_grad(): 
        ep.ivar[:].clamp_(min=1, max=10**2)
        ep.prior[:] -= ep.prior.mean()
        ep.prior[:].clamp_(min=-4., max=4.)
        ep.mean[:].clamp_(min = 0, max = 1)

    # show the means
    if it == 9999:
        #pt.imshow(ep.mean.detach().cpu().numpy()[0,:3].T)
        fname = "path" + "forest_patch_1.png"
        pt.imsave(fname, np.flip(np.rot90(ep.mean.detach().cpu().numpy()[0,:3].T), axis = 0))
        #pt.show()
print('all done')
