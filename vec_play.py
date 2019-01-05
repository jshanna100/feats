import torch
from deepeeg import models
import numpy as np
import pickle
from apex.fp16_utils import *

import matplotlib.pyplot as plt
plt.ion()

class Bilder:
    def __init__(self,rects,z_cuda,netGx,axis,background,deriv_imshow=None):
        self.rects = rects
        self.z_cuda = z_cuda
        self.netGx = netGx
        self.axis = axis
        self.deriv_imshow = deriv_imshow
        x_hat = netGx(z_cuda).detach()
        plt.sca(self.axis)
        self.imshow = plt.imshow(x_hat[0,0,].cpu().numpy().astype(np.float32),
        vmin=-1,vmax=1)
    def draw(self):
        z = np.array([r.get_height() for r in self.rects])
        self.z_cuda[0,:,0,0].copy_(torch.from_numpy(z))
        x_hat = self.netGx(z_cuda).detach()
        x_hat_nump = x_hat[0,0,].cpu().numpy().astype(np.float32)
        if self.deriv_imshow:
            self.deriv_imshow.set_data(abs(self.imshow.get_array()-x_hat_nump))
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
                self.ons[r_idx] = False
                r.set_color("blue")
                r.set_height(np.random.normal(loc=0,scale=1))
            self.draw()
            self.bilder.draw()
            return
        else:
            return
        for r_idx,r in enumerate(self.rects):
            if self.ons[r_idx]:
                r.set_height(r.get_height()+self.incr*direction)
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


net_dir = "/home/jeff/deepeeg/all_b256_z16_res/"
netGxFile = net_dir+"netGx_epoch_0_2.pth"
vecfile = net_dir+"93_A_hand_5ms_ica-raw.fif_Z"
deriv_plot = True

z_cuda = torch.FloatTensor(1,16,1,1).cuda()
netGx = models.Generator()
netGx.load_state_dict(torch.load(netGxFile))
netGx = network_to_half(netGx)
netGx.cuda()
netGx.eval()

fig, axes = plt.subplots(nrows=1,ncols=2)
plt.sca(axes[0])
plt.axis("off")
fig.canvas.draw()
backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]
z = np.random.normal(loc=0,scale=1,size=16)
plt.sca(axes[1])
barplot = plt.bar(np.array(range(len(z))),z,color="blue")
plt.ylim((-3,3))
if deriv_plot:
    deriv_fig = plt.figure()
    deriv_imshow = plt.imshow(np.zeros((64,64)),vmin=0,vmax=.05)
    plt.axis("off")
    bilder = Bilder(barplot.patches,z_cuda,netGx,axes[0],backgrounds[0],deriv_imshow=deriv_imshow)
else:
    bilder = Bilder(barplot.patches,z_cuda,netGx,axes[0],backgrounds[0],deriv_imshow=None)
bilder.draw()
db = DragBars(barplot.patches,[str(x) for x in list(range(16))],
bilder,axes[1],backgrounds[1])
db.connect()
