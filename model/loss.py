import torch
import torch.nn as nn

def get_vgg19_FeatureMap(vgg_model, input_tensor, layer_index):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    std = torch.tensor([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    if torch.cuda.is_available():
        mean = mean.cuda()
        std = std.cuda()
        
    vgg_input = (input_tensor - mean) / std
    for i in range(0, layer_index+1):
        if i == 0:
            x = vgg_model.module.features[0](vgg_input)
        else:
            x = vgg_model.module.features[i](x)
    return x

def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))

def mask_flow_loss(flow, gt, task_ids, target_id=5):
    batch_size = flow.size(0)
    target_ids = torch.full((batch_size, 1), target_id, dtype=task_ids.dtype, device=task_ids.device)
    mask = task_ids == target_ids.squeeze(-1)
    flow_mask = flow * mask.view(batch_size, 1, 1, 1)
    gt_mask = gt * mask.view(batch_size, 1, 1, 1)
    return l_num_loss(flow_mask, gt_mask, 1)
        
def cal_appearance_loss_sum(warp_list, gt, weights, eps=1e-7):
    num = len(warp_list)
    loss = []
    for i in range(num):
        warp = warp_list[i]
        loss.append(l_num_loss(warp, gt, 1) + eps)
    
    return torch.sum(torch.stack([w * v for w, v in zip(weights, loss)]))

def cal_perception_loss_sum(vgg_model, warp_list, gt, weights):
    num = len(warp_list)
    loss = []
    for i in range(num):
        warp = warp_list[i]
        warp_feature = get_vgg19_FeatureMap(vgg_model, warp, 24)
        gt_feature = get_vgg19_FeatureMap(vgg_model, gt, 24)
        
        loss.append(l_num_loss(warp_feature, gt_feature, 2))
    
    return torch.sum(torch.stack([w * v for w, v in zip(weights, loss)]))

def cal_point_loss(pre, gt):
    criterion = nn.CrossEntropyLoss()
    return criterion(pre, gt.long().cuda())

def cal_inter_grid_loss_sum(mesh_list, tps_points, weights, eps=1e-8):
    num = len(mesh_list)
    loss = []
    for i in range(num):
        mesh = mesh_list[i]
        grid_w = tps_points[i]-1
        grid_h = tps_points[i]-1
        w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:]

        w_norm1 = torch.sqrt(torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,0:grid_w-1,:], 3) + eps)
        w_norm2 = torch.sqrt(torch.sum(w_edges[:,:,1:grid_w,:] * w_edges[:,:,1:grid_w,:], 3) + eps)
        cos_w = torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:], 3) / (w_norm1 * w_norm2)
        delta_w_angle = 1 - cos_w
        
        h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]

        h_norm1 = torch.sqrt(torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,0:grid_h-1,:,:], 3) + eps)
        h_norm2 = torch.sqrt(torch.sum(h_edges[:,1:grid_h,:,:] * h_edges[:,1:grid_h,:,:], 3) + eps)
        cos_h = torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:], 3) / (h_norm1 * h_norm2)
        delta_h_angle = 1 - cos_h
        
        loss.append(torch.mean(delta_w_angle) + torch.mean(delta_h_angle))
    
    return torch.sum(torch.stack([w * v for w, v in zip(weights, loss)]))
    