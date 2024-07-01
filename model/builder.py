import torch
import utils.torch_tps_upsample as torch_tps_upsample
from utils.utils_transform import *
import torch.nn.functional as F

def build_model(net, input_tensor1, input_tensor2, mask_tensor, tps_points):
    """
    input_tensor1: source image with original resolution
    input_tensor2: resized image with fixed resolution (256x256)
    .. input_tensor1 == input_tensor2 in training
    """
    batch_size, _, img_h, img_w = input_tensor1.size()
    batch_size, _, input_size, input_size = input_tensor2.size()
    
    offset, flow, point_cls = net(input_tensor2, mask_tensor)
    head_num = len(offset)
    norm_rigid_mesh_list = []
    norm_ori_mesh_list = []
    output_tps_list = []
    ori_mesh_list = []
    tps2flow_list = []
    
    for i in range(head_num):
        mesh_motion = offset[i].reshape(-1, tps_points[i], tps_points[i], 2)
        rigid_mesh = get_rigid_mesh(batch_size, input_size, input_size, tps_points[i]-1, tps_points[i]-1)
        ori_mesh = rigid_mesh + mesh_motion
        clamped_x = torch.clamp(ori_mesh[..., 0], min=0, max=input_size - 1)
        clamped_y = torch.clamp(ori_mesh[..., 1], min=0, max=input_size - 1)
        ori_mesh = torch.stack((clamped_x, clamped_y), dim=-1)
        
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, input_size, input_size)
        norm_ori_mesh = get_norm_mesh(ori_mesh, input_size, input_size)
        tps2flow = torch_tps_upsample.transformer(norm_rigid_mesh, norm_ori_mesh, (img_h, img_w))
        output_tps = resample_image(input_tensor1, tps2flow)
        
        norm_rigid_mesh_list.append(norm_rigid_mesh)
        norm_ori_mesh_list.append(norm_ori_mesh)
        output_tps_list.append(output_tps)
        tps2flow_list.append(tps2flow)
        ori_mesh_list.append(ori_mesh)
    
    tps_flow = tps2flow_list[-1]
    final_flow = flow + tps_flow
    output_flow = resample_image(output_tps_list[-1], flow)
    
    out_dict = {}
    out_dict.update(warp_tps=output_tps_list, warp_flow=output_flow, mesh=ori_mesh_list,
                    flow1=flow, flow2=tps_flow, flow3=final_flow, point_cls=point_cls)
    return out_dict


def build_model_test(net, input_tensor1, input_tensor2, mask_tensor, tps_points, resize_flow=False):
    """
    input_tensor1: source image with original resolution
    input_tensor2: resized image with fixed resolution (256x256)
    .. input_tensor1 = input_tensor2 in training
    """
    batch_size, _, img_h, img_w = input_tensor1.size()
    batch_size, _, input_size, input_size = input_tensor2.size()
    
    offset, flow, point_cls = net(input_tensor2, mask_tensor)
    head_num = len(offset)
    norm_rigid_mesh_list = []
    norm_ori_mesh_list = []
    output_tps_list = []
    ori_mesh_list = []
    tps2flow_list = []
    for i in range(head_num):
        mesh_motion = offset[i].reshape(-1, tps_points[i], tps_points[i], 2)
        rigid_mesh = get_rigid_mesh(batch_size, input_size, input_size, tps_points[i]-1, tps_points[i]-1)
        ori_mesh = rigid_mesh + mesh_motion
        clamped_x = torch.clamp(ori_mesh[..., 0], min=0, max=input_size - 1)
        clamped_y = torch.clamp(ori_mesh[..., 1], min=0, max=input_size - 1)
        ori_mesh = torch.stack((clamped_x, clamped_y), dim=-1)
        
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, input_size, input_size)
        norm_ori_mesh = get_norm_mesh(ori_mesh, input_size, input_size)
        tps2flow = torch_tps_upsample.transformer(norm_rigid_mesh, norm_ori_mesh, (img_h, img_w))
        output_tps = resample_image_xy(input_tensor1, tps2flow)
        norm_rigid_mesh_list.append(norm_rigid_mesh)
        norm_ori_mesh_list.append(norm_ori_mesh)
        output_tps_list.append(output_tps)
        ori_mesh_list.append(ori_mesh)
        tps2flow_list.append(tps2flow)
    
    tps_flow = tps2flow_list[-1]
    if(resize_flow):
        flow = F.interpolate(flow, size=(img_h, img_w), mode='bilinear', align_corners=True)
        scale_H, scale_W = img_h / input_size, img_w / input_size
        flow[:, 0, :, :] *= scale_W
        flow[:, 1, :, :] *= scale_H
    
    final_flow = flow + tps_flow
    output_flow = resample_image_xy(output_tps_list[-1], flow)
    out_dict = {}
    out_dict.update(warp_tps=output_tps_list, warp_flow=output_flow, mesh=ori_mesh_list,
                    flow1=flow, flow2=tps_flow, flow3=final_flow, point_cls=point_cls)
    return out_dict
