## author: xin luo
## create: 2025.6.18
## des: dataset and dataloader for deep learning tasks in remote sensing

import random
import torch
import rasterio as rio
import numpy as np

## create related functions
## - crop scene to patches
class crop:
    '''randomly crop corresponding to specific patch size'''
    def __init__(self, size=(256,256)):
        self.size = size
    def __call__(self, image, truth):
        '''size: (height, width)'''
        start_h = random.randint(0, truth.shape[0]-self.size[0])
        start_w = random.randint(0, truth.shape[1]-self.size[1])
        patch = image[:,start_h:start_h+self.size[0],start_w:start_w+self.size[1]]
        truth = truth[start_h:start_h+self.size[0], start_w:start_w+self.size[1]]
        return patch, truth

### - Dataset definition
class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths_scene, paths_truth):
        self.paths_scene = paths_scene
        self.paths_truth = paths_truth

    def __getitem__(self, idx):
        # load pairwise scene and truth images
        scene_path = self.paths_scene[idx]
        truth_path = self.paths_truth[idx]
        with rio.open(scene_path) as src, rio.open(truth_path) as truth_src:
            scene_arr = src.read().transpose((1, 2, 0))  # (H, W, C)
            truth_arr = truth_src.read(1)  # (H, W)
        ## Pre-processing
        scene_arr = scene_arr/10000  # normalization
        scene_arr = scene_arr.astype(np.float32).transpose((2, 0, 1))
        patch, truth = crop(size=(256,256))(scene_arr, truth_arr)
        truth = truth[np.newaxis,:].astype(np.float32)
        patch = torch.from_numpy(patch).float()
        truth = torch.from_numpy(truth).float()
        return patch, truth
    def __len__(self):
        return len(self.paths_scene)


