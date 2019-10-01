import torch
import mne
from deepeeg import models
import numpy as np
import pickle
from deepeeg.DataManage import EEGVolumeDataSet as EEGVDS

import matplotlib.pyplot as plt
plt.ion()

def sph2cart(sph,radius=1):
    if len(sph)<4:
        raise ValueError("Intended for hyperspheres only.")
    cartvec = np.zeros(len(sph)+1)
    for sph_idx in range(len(sph)):
        to_prod = np.zeros(sph_idx+1)
        for to_sin in range(sph_idx):
            to_prod[to_sin] = np.sin(sph[to_sin])
        to_prod[sph_idx] = np.cos(sph[sph_idx])
        cartvec[sph_idx] = radius * np.prod(to_prod)
    to_prod = np.zeros(sph_idx+1)
    for to_sin in range(len(sph)):
        to_prod[to_sin] = np.sin(sph[to_sin])
    cartvec[-1] = radius * np.prod(to_prod)
    return cartvec


def rowcol_nums(ratio,number):
    # find the row column ratio for a given number of subplots
    a = np.array([[1,1],[1,-1]])
    b = np.array(((np.log(number),np.log(ratio))))
    x = np.exp(np.linalg.solve(a,b))
    r,c = np.ceil(x[0]).astype(int),np.ceil(x[1]).astype(int)
    return r,c


def array_from_bild(batch,pix_pos):
    arr = np.empty((len(pix_pos),batch.shape[0]))
    for p_idx,p in enumerate(pix_pos):
        arr[p_idx,:] = batch[:,0,p[0],p[1]].numpy()
    return arr

class Bilder:
    def __init__(self,rects,buffer,z_cuda,netGx,axis,background):
        self.rects = rects
        self.netGx = netGx
        self.axis = axis
        self.buffer = buffer
        plt.sca(self.axis)
        self.imshow = plt.imshow(buffer)


    def draw(self):
        z = np.array([r.get_height() for r in self.rects])
        self.z_cuda[0,:,0,0].copy_(torch.from_numpy(z))
        x_hat = self.netGx(z_cuda).detach()
        x_hat_nump = x_hat[0,0,].cpu().numpy().astype(np.float32)
        plt.sca(self.axis)
        self.imshow.set_data(x_hat_nump)

class DragBars:
    def __init__(self,rects,ids,bilder,axis,background):
        self.rects = rects
        self.ids = id
        self.ons = np.zeros(len(ids)).astype(np.bool)
        self.incr = 0.1
        self.bilder = bilder
        self.axis = axis
        self.background = background
    def draw(self):
        self.rects[0].figure.canvas.restore_region(self.background)
        for r in self.rects:
            self.axis.draw_artist(r)
        self.rects[0].figure.canvas.blit(self.axis.bbox)
    def on_click(self, event):
        hit = 0
        for r_idx,r in enumerate(self.rects):
            hit,props = r.contains(event)
            if hit:
                if self.ons[r_idx]:
                    r.set_color("blue")
                    self.ons[r_idx] = False
                else:
                    r.set_color("red")
                    self.ons[r_idx] = True
                self.draw()
                break
    def on_key(self, event):
        if event.key == "up":
            direction = 1
        elif event.key == "shift+up":
            direction = 10
        elif event.key == "down":
            direction = -1
        elif event.key == "shift+down":
            direction = -10
        elif event.key == "0":
            direction = 0
        elif event.key == "r":
            for r_idx,r in enumerate(self.rects):
                if self.ons[r_idx]: continue
                self.ons[r_idx] = False
                r.set_color("blue")
                r.set_height(np.random.random()*np.pi
            self.draw()
            self.bilder.draw()
            return
        else:
            return
        for r_idx,r in enumerate(self.rects):
            if self.ons[r_idx]:
                r.set_height(r.get_height()+self.incr*direction)
                if r.get_height() > np.pi:
                    r.set_height(np.pi)
            else:
                if not direction:
                    r.set_height(0)
        self.draw()
        self.bilder.draw()

    def connect(self):
        self.cidmouse = self.rects[0].figure.canvas.mpl_connect(
        "button_press_event", self.on_click)
        self.cidkey = self.rects[0].figure.canvas.mpl_connect(
        "key_press_event", self.on_key)


z_num = 32
res = (64,64)
time_res = 128
ratio = 5/3
r,c = rowcol_nums(ratio,time_res)
bigres = (res[0]*r,res[1]*c)
buffer = np.zeros(bigres)
net_dir = "/home/jeff/deepeeg/prog3d_500ms/"
netGxFile = net_dir+"netGx_0_6.pth"

z_cuda = torch.FloatTensor(1,z_num,1,1,1).cuda().half()
netGx = models.Generator(chan_num=z_num)
netGx.load_state_dict(torch.load(netGxFile))
netGx.cuda()
netGx.half()

bars_fig, bars_axes = plt.subplots(nrows=1,ncols=1)
img_fig, img_axes = plt.subplots(nrows=1,ncols=1)

plt.sca(axes[0])
plt.axis("off")
fig.canvas.draw()
backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]
z = np.random.random(size=z_num-1)*np.pi
plt.sca(axes[1])
barplot = plt.bar(np.array(range(len(z))),z,color="blue")
plt.ylim((-3,3))
if deriv_plot:
    deriv_fig = plt.figure()
    plt.axis("off")
bilder = Bilder(barplot.patches,z_cuda,netGx,axes[0],backgrounds[0])
bilder.draw()
db = DragBars(barplot.patches,[str(x) for x in list(range(z_num-1))],
bilder,axes[1],backgrounds[1])
db.connect()
