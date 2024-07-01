import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from model.builder import *
from model.network import MOWA
from dataset_loaders import TrainDataset
import glob
from model.loss import *
import torchvision.models as models
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
import torch.distributed as dist
from warmup_scheduler import GradualWarmupScheduler
from utils.utils_op import *
torch.backends.cudnn.enabled = True

def train(gpu, ngpus_per_node, args):
    
    """ threads running on each GPU """
    if args.distributed:
        torch.cuda.set_device(int(gpu))
        print('using GPU {} for training'.format(int(gpu)))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = 'tcp://127.0.0.1:' + args.port,
            world_size = ngpus_per_node, 
            rank = gpu,
            group_name='mtorch'
        )
    
    ''' folder settings'''
    path = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(path, 'checkpoint/', args.method)
    SUMMARY_DIR = os.path.join(path, 'summary/', args.method)
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=SUMMARY_DIR)
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if not os.path.exists(SUMMARY_DIR):
            os.makedirs(SUMMARY_DIR)
    
    ''' dataloader settings '''        
    train_dataset = TrainDataset(args.train_path)
    train_sampler = DistributedSampler(train_dataset, num_replicas=ngpus_per_node, rank=gpu)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=4, sampler=train_sampler, drop_last=True)

    ''' define the network and training scheduler'''
    net = MOWA(img_size=args.input_size, tps_points=args.tps_points, embed_dim=args.embed_dim, win_size=args.win_size, 
                token_projection=args.token_projection, token_mlp=args.token_mlp, depths=args.depths, 
                prompt=args.prompt, task_classes=args.task_classes, head_num=args.head_num, shared_head=args.shared_head)
    
    vgg_model = models.vgg19(pretrained=True)
    net = set_device(net, distributed=args.distributed)
    vgg_model = set_device(vgg_model, distributed=args.distributed)
    vgg_model.eval()

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    if args.warmup:
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch-args.warmup_epochs, eta_min=args.eta_min)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

    ''' resume training or not'''
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')
    
    ''' network training'''
    for epoch in range(start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)
        net.train()
        total_loss_sigma = 0.
        appearance_loss_sigma = 0.
        perception_loss_sigma = 0.
        inter_grid_loss_sigma = 0.
        point_loss_sigma = 0.
        flow_loss_sigma = 0.

        for i, batch_value in enumerate(train_loader):
            input_tesnor = batch_value[0].float()
            gt_tesnor = batch_value[1].float()
            mask_tensor = batch_value[2].float()
            task_id_tensor = batch_value[3].float()
            flow_tensor = batch_value[4].float()
            face_mask = batch_value[5].float()
            face_weight = batch_value[6].float()

            if torch.cuda.is_available():
                input_tesnor = set_device(input_tesnor, distributed=args.distributed)
                gt_tesnor = set_device(gt_tesnor, distributed=args.distributed)
                mask_tensor = set_device(mask_tensor, distributed=args.distributed)
                task_id_tensor = set_device(task_id_tensor, distributed=args.distributed)
                flow_tensor = set_device(flow_tensor, distributed=args.distributed)
                face_mask = set_device(face_mask, distributed=args.distributed)
                face_weight = set_device(face_weight, distributed=args.distributed)
            
            optimizer.zero_grad()
            with autocast():
                batch_out = build_model(net.module, input_tesnor, input_tesnor, mask_tensor, args.tps_points)
                warp_tps, mesh, warp_flow, flow, point_cls = \
                    [batch_out[key] for key in ['warp_tps', 'mesh', 'warp_flow', 'flow3', 'point_cls']]

                ''' calculate losses'''
                inter_grid_loss = cal_inter_grid_loss_sum(mesh, args.tps_points, [1.0/args.head_num for _ in range(args.head_num)])
                point_loss = cal_point_loss(point_cls, task_id_tensor) * 0.1
                face_weight_ad = adjust_weight(epoch, args.max_epoch, face_weight)
                flow_tensor, flow = get_weight_mask(face_mask, flow_tensor, flow, weight=face_weight_ad)
                flow_loss = mask_flow_loss(flow, flow_tensor, task_id_tensor) * 0.1
                
                if(epoch <= 10):
                    appearance_loss = cal_appearance_loss_sum(warp_tps, gt_tesnor, args.img_weight1)
                    perception_loss = cal_perception_loss_sum(vgg_model, warp_tps, gt_tesnor, args.img_weight1)
                    total_loss = appearance_loss + perception_loss + inter_grid_loss + point_loss
                else:
                    appearance_loss = cal_appearance_loss_sum(warp_tps+[warp_flow], gt_tesnor, args.img_weight1+args.img_weight2)
                    perception_loss = cal_perception_loss_sum(vgg_model, warp_tps+[warp_flow], gt_tesnor, args.img_weight1+args.img_weight2)
                    total_loss = appearance_loss + perception_loss + inter_grid_loss + flow_loss + point_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_sigma += total_loss.item()
            appearance_loss_sigma += appearance_loss.item()
            perception_loss_sigma += perception_loss.item()
            inter_grid_loss_sigma += inter_grid_loss.item()
            point_loss_sigma += point_loss.item()
            flow_loss_sigma += flow_loss.item()
            
            ''' writting training logs '''
            if i % args.print_interval == 0 and i != 0:
                if dist.get_rank() == 0:
                    total_loss_average = total_loss_sigma / args.print_interval
                    appearance_loss_average = appearance_loss_sigma/ args.print_interval
                    perception_loss_average = perception_loss_sigma/ args.print_interval
                    inter_grid_loss_average = inter_grid_loss_sigma/ args.print_interval
                    point_loss_average = point_loss_sigma/ args.print_interval
                    flow_loss_average = flow_loss_sigma/ args.print_interval
                    
                    total_loss_sigma = 0.
                    appearance_loss_sigma = 0.
                    perception_loss_sigma = 0.
                    inter_grid_loss_sigma = 0.
                    point_loss_sigma = 0.
                    flow_loss_sigma = 0.
                    
                    print(f"Training: Epoch[{epoch + 1:0>3}/{args.max_epoch:0>3}] "
                          f"Iteration[{i + 1:0>3}/{len(train_loader):0>3}] "
                          f"Total Loss: {total_loss_average:.4f}  "
                          f"Appearance Loss: {appearance_loss_average:.4f}  "
                          f"Perception Loss: {perception_loss_average:.4f} "
                          f"Point Loss: {point_loss_average:.4f} "
                          f"Flow Loss: {flow_loss_average:.4f} "
                          f"Inter-Grid Loss: {inter_grid_loss_average:.4f} "
                          f"lr={optimizer.state_dict()['param_groups'][0]['lr']:.8f}")

                    writer.add_image("input", (input_tesnor[0]), glob_iter)
                    writer.add_image("rectangling", (warp_flow[0]), glob_iter)
                    writer.add_image("gt", (gt_tesnor[0]), glob_iter)                
                    writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                    writer.add_scalar('total loss', total_loss_average, glob_iter)
                    writer.add_scalar('appearance loss', appearance_loss_average, glob_iter)
                    writer.add_scalar('perception loss', perception_loss_average, glob_iter)
                    writer.add_scalar('inter-grid loss', inter_grid_loss_average, glob_iter)
                    writer.add_scalar('point loss', point_loss_average, glob_iter)
                    writer.add_scalar('flow loss', flow_loss_average, glob_iter)
            
            glob_iter += 1
        
        ''' save models '''      
        if ((epoch+1) % args.eva_interval == 0 or (epoch+1)==args.max_epoch):
            if dist.get_rank() == 0:
                filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
                model_save_path = os.path.join(MODEL_DIR, filename)
                state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
                torch.save(state, model_save_path)
        
        scheduler.step()        


if __name__=="__main__":
    
    print('<==================== setting arguments ===================>\n')
    
    parser = argparse.ArgumentParser()
    '''Implementation details'''
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('-m', '--method', type=str, default='mowa')
    parser.add_argument('-P', '--port', default='21016', type=str)
    parser.add_argument('-d', '--distributed', type=bool, default=True)
    parser.add_argument('-w', '--warmup', type=bool, default=True)
    parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')
    parser.add_argument('--print_interval', type=int, default=160)
    parser.add_argument('--eva_interval', type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4, help="start learning rate")
    parser.add_argument("--eta_min", type=float, default=1e-7, help="final learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay of the optimizer")
    parser.add_argument('--img_weight1', nargs='+', type=float, default=[0.1, 0.1, 0.2, 0.5], help='weights for img loss (stage1)')
    parser.add_argument('--img_weight2', nargs='+', type=float, default=[0.5], help='weights for img loss (stage2)')

    '''Network details'''
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--depths', nargs='+', type=int, default=[2, 2, 2, 2, 2, 2, 2, 2, 2], help='depths for transformer layers')
    parser.add_argument('--tps_points', nargs='+', type=int, default=[10, 12, 14, 16], help='tps points for regression heads')
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--win_size', type=int, default=8)
    parser.add_argument('--token_projection', type=str, default='linear')
    parser.add_argument('--token_mlp', type=str, default='leff')
    parser.add_argument('--prompt', type=bool, default=True)
    parser.add_argument('--task_classes', type=int, default=6)
    parser.add_argument('--head_num', type=int, default=4)
    parser.add_argument('--shared_head', type=bool, default=False)
    
    '''Dataset settings'''
    parser.add_argument('--train_path', type=str, default=['/Dataset/pano-rectangling/train/', '/Dataset/wide-angle_rectangling/train/', 
                                                           '/Dataset/RS_Rec/RS_Rec/train/', '/Dataset/Rotation/train/',
                                                           '/Dataset/fisheye/train/', '/Dataset/FaceRec/train/'])
    
    args = parser.parse_args()
    print(args)
    
    gpu_str = args.gpu_ids
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))
    opt=0
    print('<==================== start training ===================>\n')
    mp.spawn(train, nprocs=num_gpus, args=(num_gpus, args))
    
    print("################## end training #######################")