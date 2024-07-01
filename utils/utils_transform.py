import torch
import torch.nn.functional as F
import numpy as np
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps_upsample as torch_tps_upsample

def get_rigid_mesh(batch_size, height, width, grid_w, grid_h):
    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()
    
    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2)
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt
    
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) 
    
    return norm_mesh.reshape([batch_size, -1, 2]) 

def transform_tps_fea(offset, input_tensor, grid_w, grid_h, dim, h, w):
    input_tensor = input_tensor.permute(0,2,1).view(-1, dim, h, w)
    batch_size, _, img_h, img_w = input_tensor.size()
    
    mesh_motion = offset.reshape(-1, grid_h+1, grid_w+1, 2)
    
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w, grid_w, grid_h)
    ori_mesh = rigid_mesh + mesh_motion
    
    clamped_x = torch.clamp(ori_mesh[..., 0], min=0, max=img_h - 1)
    clamped_y = torch.clamp(ori_mesh[..., 1], min=0, max=img_w - 1)
    ori_mesh = torch.stack((clamped_x, clamped_y), dim=-1)

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_ori_mesh = get_norm_mesh(ori_mesh, img_h, img_w)
    
    output_tps = torch_tps_transform.transformer(input_tensor, norm_rigid_mesh, norm_ori_mesh, (img_h, img_w))
    output_tps = output_tps.view(-1, dim, h*w).permute(0,2,1)
        
    return output_tps
    
def upsample_tps(offset, grid_w, grid_h, out_h, out_w):
    if(grid_w+1 == out_w):
        return offset
    
    else:
        batch_size, *_ = offset.size()
        mesh_motion = offset.reshape(-1, grid_h+1, grid_w+1, 2)
        
        rigid_mesh = get_rigid_mesh(batch_size, out_h, out_w, grid_w, grid_h)
        ori_mesh = rigid_mesh + mesh_motion

        norm_rigid_mesh = get_norm_mesh(rigid_mesh, out_h, out_w)
        norm_ori_mesh = get_norm_mesh(ori_mesh, out_h, out_w)
        
        up_points = torch_tps_upsample.transformer(norm_rigid_mesh, norm_ori_mesh, (out_h, out_w))
        out = up_points.permute(0, 2, 3, 1).view(-1, out_h*out_w, 2)
            
        return out
    
def get_coordinate(shape, det_uv):
    b, _, w, h = shape
    uv_d = np.zeros([w, h, 2], np.float32)

    for i in range(0, w):
        for j in range(0, h):
            uv_d[i, j, 0] = j
            uv_d[i, j, 1] = i

    uv_d = np.expand_dims(uv_d.swapaxes(2, 1).swapaxes(1, 0), 0)
    uv_d = torch.from_numpy(uv_d).cuda()
    uv_d = uv_d.repeat(b, 1, 1, 1)

    det_uv = uv_d + det_uv
    return det_uv

def uniform(shape, img_uv):
    b, _, w, h = shape
    x0 = (w - 1) / 2. 

    img_nor = (img_uv - x0)/x0             
    img_nor = img_nor.permute(0, 2, 3, 1)
    return img_nor

def resample_image(feature, flow):
    img_uv = get_coordinate(feature.shape, flow)
    grid = uniform(feature.shape, img_uv)
    target_image = F.grid_sample(feature, grid)
    return target_image

def get_coordinate_xy(shape, det_uv):
    b, _, h, w = shape
    uv_d = np.zeros([h, w, 2], np.float32)

    for j in range(0, h):
        for i in range(0, w):
            uv_d[j, i, 0] = i
            uv_d[j, i, 1] = j

    uv_d = np.expand_dims(uv_d.swapaxes(2, 1).swapaxes(1, 0), 0)
    uv_d = torch.from_numpy(uv_d).cuda()
    uv_d = uv_d.repeat(b, 1, 1, 1)
    det_uv = uv_d + det_uv
    return det_uv

def uniform_xy(shape, uv):
    b, _, h, w = shape
    y0 = (h - 1) / 2.
    x0 = (w - 1) / 2.

    nor = uv.clone()
    nor[:, 0, :, :] = (uv[:, 0, :, :] - x0) / x0 
    nor[:, 1, :, :] = (uv[:, 1, :, :] - y0) / y0
    nor = nor.permute(0, 2, 3, 1)  # b w h 2

    return nor

def resample_image_xy(feature, flow):
    uv = get_coordinate_xy(feature.shape, flow)
    grid = uniform_xy(feature.shape, uv)
    target_image = F.grid_sample(feature, grid)
    return target_image