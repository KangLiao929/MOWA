import argparse
import torch
from torch.utils.data import DataLoader
from model.builder import *
from dataset_loaders import TestDatasetMask
import os
import glob
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model.network import MOWA
from collections import OrderedDict
from utils import flow_viz
from utils.utils_op import *

def test(args):
    
    path = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.join(path, 'results/', args.method, 'img/')
    MESH_DIR = os.path.join(path, 'results/', args.method, 'mesh/')
    RES_DIR = os.path.join(path, 'results/', args.method, 'res/')
    IMG_DIR_LIST = [os.path.join(IMG_DIR, str(i)) for i in range(len(args.test_path))]
    MESH_DIR_LIST = [os.path.join(MESH_DIR, str(i)) for i in range(len(args.test_path))]
    RES_DIR_LIST = [os.path.join(RES_DIR, str(i)) for i in range(len(args.test_path))]

    for i in range(len(IMG_DIR_LIST)):
        path = IMG_DIR_LIST[i]
        if not os.path.exists(path):
            os.makedirs(path)
    for i in range(len(MESH_DIR_LIST)):
        path = MESH_DIR_LIST[i]
        if not os.path.exists(path):
            os.makedirs(path)
    for i in range(len(RES_DIR_LIST)):
        path = RES_DIR_LIST[i]
        if not os.path.exists(path):
            os.makedirs(path)

    test_loader_list = [DataLoader(dataset=TestDatasetMask(test_path, i), batch_size=1, num_workers=4, shuffle=False, drop_last=False) \
                            for i, test_path in enumerate(args.test_path)]
    
    '''define the network'''
    net = MOWA(img_size=args.input_size, tps_points=args.tps_points, embed_dim=args.embed_dim, win_size=args.win_size, 
                token_projection=args.token_projection, token_mlp=args.token_mlp, depths=args.depths, 
                prompt=args.prompt, task_classes=args.task_classes, head_num=args.head_num, shared_head=args.shared_head)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
        net = net.to(device)

    '''load the existing models'''
    MODEL_DIR = args.model_path
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        
        net.load_state_dict(new_state_dict)
        print('load model from {}!'.format(model_path))
    else:
        raise FileNotFoundError(f'No checkpoint found in directory {MODEL_DIR}!')
    
    print("##################start testing#######################")
    net.eval()
    test_num = len(test_loader_list)
    
    for index in range(test_num):
        print("Task ID:", index)
        NUM = count_files(args.test_path[index])
        acc_temp = 0
        psnr_img = 0
        ssim_img = 0
        path_img = str(IMG_DIR_LIST[index]) + '/'
        path_grid = str(MESH_DIR_LIST[index]) + '/'
        path_res = str(RES_DIR_LIST[index]) + '/'
        
        with torch.no_grad():
            net.eval()
            test_loader = test_loader_list[index]
            for i, outputs in enumerate(test_loader):

                input1_tensor = outputs['input1_tensor'].float()
                input2_tensor = outputs['input2_tensor'].float()
                gt1_tensor = outputs['gt1_tensor'].float()
                gt2_tensor = outputs['gt2_tensor'].float()
                mask_tensor = outputs['mask_tensor'].float()
                task_id_tensor = outputs['task_id_tensor'].float()
                file_name = outputs['file_name'][0]

                if torch.cuda.is_available():
                    input1_tensor = input1_tensor.cuda()
                    input2_tensor = input2_tensor.cuda()
                    gt1_tensor = gt1_tensor.cuda()
                    gt2_tensor = gt2_tensor.cuda()
                    mask_tensor = mask_tensor.cuda()
                    task_id_tensor = task_id_tensor.cuda()

                ''' parsing the output '''
                batch_out = build_model_test(net, input1_tensor, input2_tensor, mask_tensor, args.tps_points, resize_flow=True)
                warp_tps, warp_flow, mesh, flow1, flow2, flow3, point_cls = \
                                                [batch_out[key] for key in ['warp_tps', 'warp_flow', 'mesh', 'flow1', 'flow2', 'flow3', 'point_cls']]

                ''' tensor to numpy and post-processing '''
                _, c, ori_h, ori_w = input1_tensor.shape
                input_np2 = ((input2_tensor[0])*255.0).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
                gt_np1 = ((gt1_tensor[0])*255.0).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
                gt_np2 = ((gt2_tensor[0])*255.0).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
                gt_np2 = cv2.resize(gt_np2, (ori_w, ori_h))
                
                warp_flow_np = ((warp_flow[0])*255.0).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
                flow1 = (flow1[0]).cpu().detach().numpy().transpose(1,2,0)
                flow2 = (flow2[0]).cpu().detach().numpy().transpose(1,2,0)
                flow3 = (flow3[0]).cpu().detach().numpy().transpose(1,2,0)

                flow1 = flow_viz.flow_to_image(flow1)
                flow2 = flow_viz.flow_to_image(flow2)
                flow3 = flow_viz.flow_to_image(flow3)

                _, point_cls = torch.max(point_cls[0], 0)
                acc_temp += (point_cls == task_id_tensor[0]).float().mean().item()
                warp_tps_np = ((warp_tps[-1][0])*255.0).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
                mesh_np = mesh[-1][0].cpu().detach().numpy()
                cv2.imwrite(path_img + file_name + "_mesh" + ".jpg", warp_tps_np)                            
                input_with_mesh = draw_mesh_on_warp(input_np2, mesh_np, args.tps_points[-1]-1, args.tps_points[-1]-1)
                cv2.imwrite(path_grid + file_name + "_mesh" + ".jpg", input_with_mesh)
                
                ''' calculate metrics '''
                psnr_img += psnr(warp_flow_np, gt_np1, data_range=255)
                ssim_img += ssim(warp_flow_np, gt_np1, data_range=255, channel_axis=2)    
                cv2.imwrite(path_img + file_name + "_flow1.jpg", flow1)
                cv2.imwrite(path_img + file_name + "_flow2.jpg", flow2)
                cv2.imwrite(path_img + file_name + "_flow3.jpg", flow3)
                cv2.imwrite(path_res + file_name + ".jpg", warp_flow_np)
                cv2.imwrite(path_img + file_name + "_flow.jpg", warp_flow_np)
        
        print(f"Validation PSNR: {round(psnr_img / NUM, 4)}, Validation SSIM: {round(ssim_img / NUM, 4)}, Validation Acc: {round(acc_temp / NUM, 4)}")



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    
    '''Implementation details'''
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='model/')
    parser.add_argument('--method', type=str, default='method')
    
    '''Network details'''
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--depths', nargs='+', type=int, default=[2, 2, 2, 2, 2, 2, 2, 2, 2], help='depths for transformer layers')
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--win_size', type=int, default=8)
    parser.add_argument('--token_projection', type=str, default='linear')
    parser.add_argument('--token_mlp', type=str, default='leff')
    parser.add_argument('--prompt', type=bool, default=True)
    parser.add_argument('--task_classes', type=int, default=6)
    parser.add_argument('--tps_points', nargs='+', type=int, default=[10, 12, 14, 16], help='tps points for regression heads')
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--shared_head', type=bool, default=False)
    
    '''Dataset settings'''                                                     
    parser.add_argument('--test_path', type=str, default=['/stitch/test/', '/wide-angle/test/', '/RS_Rec/test/', '/Rotation/test/', '/fisheye/test/'])
    
    print('<==================== Testing ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)