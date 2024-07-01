import random
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
from imgaug import augmenters as iaa
import os

def set_gpu(args, distributed=False, rank=0):
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if distributed and isinstance(args, torch.nn.Module):
		return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
	else:
		return args.cuda()
		
def set_device(args, distributed=False, rank=0):
	""" set parameter to gpu or cpu """
	if torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item, distributed, rank) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key], distributed, rank) for key in args}
		else:
			args = set_gpu(args, distributed, rank)
	return args

def count_files(directory_path):
    for subdir, _, _ in os.walk(directory_path):
        if subdir != directory_path: 
            return len([file for _, _, files in os.walk(subdir) for file in files])
    return 0

def draw_mesh_on_warp(warp, f_local, grid_h, grid_w):
    height = warp.shape[0]
    width = warp.shape[1]
    
    min_w = np.minimum(np.min(f_local[:,:,0]), 0).astype(np.int32)
    max_w = np.maximum(np.max(f_local[:,:,0]), width).astype(np.int32)
    min_h = np.minimum(np.min(f_local[:,:,1]), 0).astype(np.int32)
    max_h = np.maximum(np.max(f_local[:,:,1]), height).astype(np.int32)
    cw = max_w - min_w
    ch = max_h - min_h
    
    pic = np.ones([ch+10, cw+10, 3], np.int32)*255
    pic[0-min_h+5:0-min_h+height+5, 0-min_w+5:0-min_w+width+5, :] = warp
    
    warp = pic
    f_local[:,:,0] = f_local[:,:,0] - min_w+5
    f_local[:,:,1] = f_local[:,:,1] - min_h+5
    
    point_color = (0, 255, 0)
    thickness = 2
    lineType = 8
    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
              
    return warp

def data_aug(img, gt):
    oplist = []
    if random.random() > 0.5:
        oplist.append(iaa.GaussianBlur(sigma=(0.0, 1.0)))
    elif random.random() > 0.5:
        oplist.append(iaa.WithChannels(0, iaa.Add((1, 15))))
    elif random.random() > 0.5:
        oplist.append(iaa.WithChannels(1, iaa.Add((1, 15))))
    elif random.random() > 0.5:
        oplist.append(iaa.WithChannels(2, iaa.Add((1, 15))))
    elif random.random() > 0.5:
        oplist.append(iaa.AdditiveGaussianNoise(scale=(0, 10)))
    elif random.random() > 0.5:
        oplist.append(iaa.Sharpen(alpha=0.15))

    seq = iaa.Sequential(oplist)
    images_aug = seq.augment_images([img])
    gt_aug = seq.augment_images([gt])
    return images_aug[0], gt_aug[0]

def get_weight_mask(mask, gt, pred, weight=10):
    mask = (mask * (weight - 1)) + 1
    gt = gt.mul(mask)
    pred = pred.mul(mask)
    return gt, pred

def adjust_weight(epoch, total_epoch, weight):
    return (1 - 0.9 * (epoch / total_epoch)) * weight

def flow2list(flow):
    h, w, c = flow.shape
    dirs_grid = []
    for i in range(h):
        dirs_row = []
        for j in range(w):
            dx, dy = flow[i, j, :]
            dx = np.round(dx)
            dy = np.round(dy)
            dirs_row.append([-int(dx), -int(dy)])
        dirs_grid.append(dirs_row)
    return dirs_grid