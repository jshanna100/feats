import torch
from deepeeg import models
import numpy as np
import pickle
from apex.fp16_utils import *

import matplotlib.pyplot as plt
plt.ion()

class Bilder:
    def __init__(self,rects,z_cuda,netGx,axis):
        self.rects = rects
        self.z_cuda = z_cuda
        self.netGx = netGx
        self.axis = axis
    def draw(self):
        z = np.array([r.get_height() for r in self.rects])
        self.z_cuda[0,:,0,0].copy_(torch.from_numpy(z))
        x_hat = self.netGx(z_cuda).detach()
        plt.sca(self.axis)
        plt.imshow(x_hat[0,0,].cpu().numpy().astype(np.float32))

class DragBar:
    def __init__(self,rect,id,bilder):
        self.rect = rect
        self.id = id
        self.on = False
        self.incr = 0.1
        self.bilder = bilder
    def on_click(self, event):
        hit,props = self.rect.contains(event)
        if hit:
            if self.on:
                self.rect.set_color("blue")
                self.on = False
            else:
                self.rect.set_color("red")
                self.on = True
            self.rect.figure.canvas.draw()
    def on_key(self, event):
        if self.on:
            if event.key == "up":
                direction = 1
            elif event.key == "shift+up":
                direction = 10
            elif event.key == "down":
                direction = -1
            elif event.key == "shift+down":
                direction = -10
            else:
                return
        else:
            return
        self.rect.set_height(self.rect.get_height()+self.incr*direction)
        self.rect.figure.canvas.draw()
        self.bilder.draw()

    def connect(self):
        self.cidmouse = self.rect.figure.canvas.mpl_connect(
        "button_press_event", self.on_click)
        self.cidkey = self.rect.figure.canvas.mpl_connect(
        "key_press_event", self.on_key)


net_dir = "/home/jeff/deepeeg/all_b400_z28_res/"
netGxFile = net_dir+"netGx_epoch_0_1.pth"
vecfile = net_dir+"93_A_hand_5ms_ica-raw.fif_Z"

z_cuda = torch.FloatTensor(1,28,1,1)#.cuda()
netGx = models.Generator()
netGx.load_state_dict(torch.load(netGxFile))
#netGx = network_to_half(netGx)
#netGx.cuda()
netGx.eval()

fig, axes = plt.subplots(nrows=1,ncols=2)
plt.sca(axes[0])
plt.axis("off")
z = np.random.normal(loc=0,scale=1,size=28)
plt.sca(axes[1])
barplot = plt.bar(np.array(range(len(z))),z,color="blue")
bilder = Bilder(barplot.patches,z_cuda,netGx,axes[0])
bilder.draw()
dbs = []
for r_idx,r in enumerate(barplot.patches):
    dbs.append(DragBar(r,str(r_idx),bilder))
    dbs[-1].connect()
