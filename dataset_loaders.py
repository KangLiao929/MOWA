from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
from utils.utils_op import data_aug
    
class TrainDataset(Dataset):
    def __init__(self, paths):

        self.width = 256
        self.height = 256
        self.prob = 0.5
        self.input_images = []
        self.gt_images = []
        self.masks = []
        self.flows = []
        self.task_id = []
        for index, path in enumerate(paths):
            inputs = glob.glob(os.path.join(path, 'input/', '*.*'))
            gts = glob.glob(os.path.join(path, 'gt/', '*.*'))
            masks = glob.glob(os.path.join(path, 'mask/', '*.*'))
            flows = glob.glob(os.path.join(path, 'flow_npy/', '*.*'))
            inputs.sort()
            gts.sort()
            masks.sort()
            flows.sort()
            
            lens = len(inputs)
            index_array = [index] * lens
            self.task_id.extend(index_array)
            self.input_images.extend(inputs)
            self.gt_images.extend(gts)
            self.masks.extend(masks)
            self.flows.extend(flows)
        
        print("total dataset num: ", len(self.input_images))
            
    def __getitem__(self, index):
        
        '''load images'''
        task_id = self.task_id[index]
        input_src = cv2.imread(self.input_images[index])
        input_resized = cv2.resize(input_src, (self.width, self.height))
        gt = cv2.imread(self.gt_images[index])
        gt = cv2.resize(gt, (self.width, self.height))
        input_resized, gt = data_aug(input_resized, gt)
        
        input_resized = input_resized.astype(dtype=np.float32)
        input_resized = input_resized / 255.0
        input_resized = np.transpose(input_resized, [2, 0, 1])
        gt = gt.astype(dtype=np.float32)
        gt = gt / 255.0
        gt = np.transpose(gt, [2, 0, 1])
        
        '''load mask'''
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.width, self.height))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(dtype=np.float32)
        mask = mask / 255.0
        mask = np.transpose(mask, [2, 0, 1])
        mask_tensor = torch.tensor(mask)
        
        input_tensor = torch.tensor(input_resized)
        gt_tensor = torch.tensor(gt)
        task_id_tensor = torch.tensor(task_id, dtype=torch.int64)  

        '''load flow and face mask for the portrait task'''
        if(task_id == 5):
            flow = np.load(self.flows[index])
            flow = flow.astype(dtype=np.float32)
            flow = np.transpose(flow, [2, 0, 1])
            flow_tensor = torch.tensor(flow)
            
            face_mask_path = self.input_images[index].replace('/input/', '/mask_face/')
            facemask = cv2.imread(face_mask_path, 0)
            facemask = facemask.astype(dtype=np.float32)
            facemask = (facemask / 255.0)
            facemask = np.expand_dims(facemask, axis=-1)
            facemask = np.transpose(facemask, [2, 0, 1])
            face_mask = torch.tensor(facemask)
            mask_sum = torch.sum(face_mask)
            weight = self.width * self.height / mask_sum - 1
            weight = torch.max(weight / 3, torch.ones(1))
            face_weight = weight.unsqueeze(-1).unsqueeze(-1)
        else:
            flow_tensor = torch.zeros_like(input_tensor[0:2, :, :])
            face_mask = torch.zeros_like(mask_tensor)
            face_weight = torch.mean(face_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     
        
        return (input_tensor, gt_tensor, mask_tensor, task_id_tensor, flow_tensor, face_mask, face_weight)
        
    def __len__(self):
        return len(self.input_images)
    
class TestDatasetMask(Dataset):
    def __init__(self, data_path, task_id):
        self.width = 256
        self.height = 256
        self.test_path = data_path
        self.datas = OrderedDict()
        self.task_id = task_id
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'gt' or data_name == 'mask':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.*'))
                self.datas[data_name]['image'].sort()

    def __getitem__(self, index):
        
        input = cv2.imread(self.datas['input']['image'][index])
        input1 = input.astype(dtype=np.float32)
        input1 = input1 / 255.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        input2 = cv2.resize(input, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = input2 / 255.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        gt = cv2.imread(self.datas['gt']['image'][index])
        gt1 = gt.astype(dtype=np.float32)
        gt1 = gt1 / 255.0
        gt1 = np.transpose(gt1, [2, 0, 1])
        
        gt2 = cv2.resize(gt, (self.width, self.height))
        gt2 = gt2.astype(dtype=np.float32)
        gt2 = gt2 / 255.0
        gt2 = np.transpose(gt2, [2, 0, 1])
        
        mask = cv2.imread(self.datas['mask']['image'][index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.width, self.height))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(dtype=np.float32)
        mask = mask / 255.0
        mask = np.transpose(mask, [2, 0, 1])
        
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        gt1_tensor = torch.tensor(gt1)
        gt2_tensor = torch.tensor(gt2)
        mask_tensor = torch.tensor(mask)
        task_id_tensor = torch.tensor(self.task_id, dtype=torch.int64)
        file_name = os.path.basename(self.datas['input']['image'][index])
        file_name, _ = os.path.splitext(file_name)
        
        out_dict = {}
        out_dict.update(input1_tensor=input1_tensor, input2_tensor=input2_tensor, gt1_tensor=gt1_tensor,  
                            gt2_tensor=gt2_tensor, mask_tensor=mask_tensor, task_id_tensor=task_id_tensor, file_name=file_name)

        return out_dict

    def __len__(self):
        return len(self.datas['input']['image'])