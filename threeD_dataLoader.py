# threeD_dataLoader.py: Chuẩn bị train và val dataset
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import pickle
import glob
from utils import normalize
from heatmap_from_keypoint3D import heatmap_from_keypoint

# Trả về 1 đoạn input signal có size là 2 * window
def window_select(data,timestep,window):
    if window ==0:
        return data[timestep : timestep + 1, :, :]
    max_len = data.shape[0]
    l = max(0,timestep-window) 
    u = min(max_len,timestep+window)
    if l == 0:
        return (data[:2*window,:,:]) # Nếu không đủ data để lùi timestep thì lấy từ đầu x2 windows
    elif u == max_len:
        return (data[-2*window:,:,:]) # Nếu không đủ data để tiến thì lấy từ cuối x2 windows
    else:
        return(data[l:u,:,:]) # Nếu đủ thì lấy x2 window với vị trí giữa là timestep


def get_subsample(touch, subsample): # Tính trung bình theo từng cụm subsample * subsample size theo chiều đầu tiên và thay thế 
    for x in range(0, touch.shape[1], subsample):
        for y in range(0, touch.shape[2], subsample):
            v = np.mean(touch[:, x:x+subsample, y:y+subsample], (1, 2))
            touch[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)

    return touch

class sample_data_diffTask_2(Dataset):
    def __init__(self, path, window, subsample, mode):
        self.path = path
        self.files = glob.glob(os.path.join(path, mode, "*.p"))
        self.subsample = subsample
        self.window = window

    def __len__(self):
        # return self.length
        return len(self.files) # Lấy timestamps của camera làm độ dài dataset

    def __getitem__(self, idx): #idx là iterator
        with open(self.files[idx], "rb") as f:
            sample_batched = pickle.load(f)
        tactileU = torch.squeeze(sample_batched[0], 0) # Frame of tactiles
        heatmapU = torch.squeeze(sample_batched[1], 0) # Headmap
        keypointU = torch.squeeze(sample_batched[2], 0) # Keypoint
        tactile_frameU = torch.squeeze(sample_batched[3], 0) # Middle Frame

        if self.subsample > 1:
            tactileU = get_subsample(tactileU, self.subsample) # Nếu có chia theo subsample thì tính trung bình cacs pixel theo giá trị subsample

        return tactileU, heatmapU, keypointU, tactile_frameU # Lấy M frames xung quanh 1 middle frame + heatmap + keypoint của middle frame
