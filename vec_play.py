import torch
import mne
from deepeeg import models
import numpy as np
import pickle
from deepeeg.DataManage import EEGVolumeDataSet as EEGVDS
from apex.fp16_utils import *

import matplotlib.pyplot as plt
plt.ion()

def array_from_bild(batch,pix_pos):
    arr = np.empty((len(pix_pos),batch.shape[0]))
    for p_idx,p in enumerate(pix_pos):
        arr[p_idx,:] = batch[:,0,p[0],p[1]].numpy()
    return arr

class Bilder:
    def __init__(self,rects,z_cuda,netGx,axis,background,deriv_plot=False,source_plot=False):
        self.rects = rects
        self.z_cuda = z_cuda
        self.netGx = netGx
        self.axis = axis
        self.deriv_plot = deriv_plot
        if self.deriv_plot:
            self.deriv_imshow = plt.imshow(np.zeros((64,64)),vmin=0,vmax=.05)
        self.source_plot = source_plot
        if self.source_plot:
            rawfile = "/home/jeff/deepeeg/proc/93_A_5ms_hand_ica-raw.fif"
            self.raw = mne.io.Raw(rawfile,preload=True).pick_types(eeg=True)
            self.raw = mne.set_eeg_reference(self.raw,projection=True)[0]
            self.pix_pos = np.load("/home/jeff/deepeeg/proc/pix_pos.npy")
            self.std_thresh = np.load("/home/jeff/deepeeg/proc/std_thresh.npy")
            self.mu = np.load("/home/jeff/deepeeg/proc/mu.npy")
            self.raw.crop(tmax=0)
            self.raw[:] = np.zeros(self.raw.get_data().shape)
            fwd = mne.read_forward_solution("/home/jeff/deepeeg/proc/deepeeg_generic-fwd.fif")
            cov = mne.read_cov("/home/jeff/deepeeg/proc/deepeeg_generic-cov.fif")
            self.inv = mne.minimum_norm.make_inverse_operator(self.raw.info,fwd,cov,depth=0.1)
            self.stc = mne.minimum_norm.apply_inverse_raw(self.raw,self.inv,1)
            self.stc_plot = self.stc.plot(subject="deepeeg_generic",subjects_dir="/home/jeff/freesurfer/subjects", hemi="split")
        x_hat = netGx(z_cuda).detach()
        plt.sca(self.axis)
        self.imshow = plt.imshow(x_hat[0,0,].cpu().numpy().astype(np.float32),
        vmin=-1,vmax=1)
    def draw(self):
        z = np.array([r.get_height() for r in self.rects])
        self.z_cuda[0,:,0,0].copy_(torch.from_numpy(z))
        x_hat = self.netGx(z_cuda).detach()
        x_hat_nump = x_hat[0,0,].cpu().numpy().astype(np.float32)
        if self.deriv_plot:
            self.deriv_imshow.set_data(abs(self.imshow.get_array()-x_hat_nump))
        if self.source_plot:
            temp = array_from_bild(x_hat.cpu(),self.pix_pos)
            self.raw[:] = (np.arctan(temp.T)*self.std_thresh+self.mu).T*1e-6
            self.stc = mne.minimum_norm.apply_inverse_raw(self.raw,self.inv,3)
            self.stc_plot.remove_data()
            self.stc_plot.add_data(getattr(self.stc, "lh"+"_data"),hemi="lh",vertices=self.stc.vertices[0],smoothing_steps=8)
            self.stc_plot.add_data(getattr(self.stc, "rh"+"_data"),hemi="rh",vertices=self.stc.vertices[1],smoothing_steps=8)
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


z_num = 24
net_dir = "/home/jeff/deepeeg/all_b712_z{}_res/".format(z_num)
netGxFile = net_dir+"netGx_epoch_0_4.pth"
vecfile = net_dir+"93_A_hand_5ms_ica-raw.fif_Z"
deriv_plot = False
source_plot = True

z_cuda = torch.FloatTensor(1,z_num,1,1).cuda()
netGx = models.Generator(chan_num=z_num)
netGx.load_state_dict(torch.load(netGxFile))
netGx = network_to_half(netGx)
netGx.cuda()
netGx.eval()

fig, axes = plt.subplots(nrows=1,ncols=2)
plt.sca(axes[0])
plt.axis("off")
fig.canvas.draw()
backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes]
z = np.random.normal(loc=0,scale=1,size=z_num)
plt.sca(axes[1])
barplot = plt.bar(np.array(range(len(z))),z,color="blue")
plt.ylim((-3,3))
if deriv_plot:
    deriv_fig = plt.figure()
    plt.axis("off")
bilder = Bilder(barplot.patches,z_cuda,netGx,axes[0],backgrounds[0],deriv_plot=deriv_plot,source_plot=source_plot)
bilder.draw()
db = DragBars(barplot.patches,[str(x) for x in list(range(z_num))],
bilder,axes[1],backgrounds[1])
db.connect()
