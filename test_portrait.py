import argparse
import json
import torch
from model.builder import *
import os
import glob
import numpy as np
import cv2
from model.network import MOWA
from collections import OrderedDict
from utils.utils_op import *
from tqdm import tqdm

eps = 1e-6

def estimation_flowmap(model, img, device, args):
    model.eval()
    img = cv2.resize(img, (256, 256))
    img = img.astype(dtype=np.float32)
    img = img / 255.0
    img = np.transpose(img, [2, 0, 1])
    img = torch.tensor(img)
    img = img.unsqueeze(0)
    mask = np.ones((256, 256), dtype=np.uint8) * 255
    mask = np.expand_dims(mask, axis=-1)
    mask = mask.astype(dtype=np.float32)
    mask = mask / 255.0
    mask = np.transpose(mask, [2, 0, 1])
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(0)
    with torch.no_grad():
        img = img.to(device)
        mask = mask.to(device)
        batch_out = build_model_test(model, img, img, mask, args.tps_points)
        output = batch_out['flow3']
    output = output.detach().cpu().squeeze(0).numpy()
    return output


# ----------------------The computation process of face metric ---------------------------------------
def compute_cosin_similarity(preds, gts):
    people_num = gts.shape[0]
    points_num = gts.shape[1]
    similarity_list = []
    preds = preds.astype(np.float32)
    gts = gts.astype(np.float32)
    for people_index in range(people_num):
        # the index 63 of lmk is the center point of the face, that is, the tip of the nose
        pred_center = preds[people_index, 63, :]
        pred = preds[people_index, :, :]
        pred = pred - pred_center[None, :]
        gt_center = gts[people_index, 63, :]
        gt = gts[people_index, :, :]
        gt = gt - gt_center[None, :]

        dot = np.sum((pred * gt), axis=1)
        pred = np.sqrt(np.sum(pred * pred, axis=1))
        gt = np.sqrt(np.sum(gt * gt, axis=1))

        similarity_list_tmp = []
        for i in range(points_num):
            if i != 63:
                similarity = (dot[i] / (pred[i] * gt[i] + eps))
                similarity_list_tmp.append(similarity)

        similarity_list.append(np.mean(similarity_list_tmp))

    return np.mean(similarity_list)


# --------------------The normalization function -----------------------------------------------------
def normalization(x):
    return [(float(i) - min(x)) / float(max(x) - min(x) + eps) for i in x]


# -------------------The computation process of line metric-------------------------------------------
def compute_line_slope_difference(pred_line, gt_k):
    scores = []
    for i in range(pred_line.shape[0] - 1):
        pk = (pred_line[i + 1, 1] - pred_line[i, 1]) / (pred_line[i + 1, 0] - pred_line[i, 0] + eps)
        score = np.abs(pk - gt_k)
        scores.append(score)
    scores_norm = normalization(scores)
    score = np.mean(scores_norm)
    score = 1 - score
    return score


# -------------------------------Compute the out put flow map -------------------------------------------------
def compute_ori2shape_face_line_metric(model, oriimg_paths, device, args):
    line_all_sum_pred = []
    face_all_sum_pred = []

    for oriimg_path in tqdm(oriimg_paths):
        # Get the [Source image]
        ori_img = cv2.imread(oriimg_path)  # Read the oriinal image
        ori_height, ori_width, _ = ori_img.shape  # get the size of the oriinal image
        input = ori_img.copy()  # get the image as the input of our model

        # Get the [flow map]"""
        pred = estimation_flowmap(model, input, device, args)
        pflow = pred.transpose(1, 2, 0)
        predflow_x, predflow_y = pflow[:, :, 0], pflow[:, :, 1]

        scale_x = ori_width / predflow_x.shape[1]
        scale_y = ori_height / predflow_x.shape[0]
        predflow_x = cv2.resize(predflow_x, (ori_width, ori_height)) * scale_x
        predflow_y = cv2.resize(predflow_y, (ori_width, ori_height)) * scale_y

        # Get the [predicted image]"""
        ys, xs = np.mgrid[:ori_height, :ori_width]
        mesh_x = predflow_x.astype("float32") + xs.astype("float32")
        mesh_y = predflow_y.astype("float32") + ys.astype("float32")
        pred_out = cv2.remap(input, mesh_x, mesh_y, cv2.INTER_LINEAR)
        cv2.imwrite(oriimg_path.replace(".jpg", "_pred.jpg"), pred_out)

        # Get the landmarks from the [gt image]
        stereo_lmk_file = open(oriimg_path.replace(".jpg", "_stereo_landmark.json"))
        stereo_lmk = np.array(json.load(stereo_lmk_file), dtype="float32")

        # Get the landmarks from the [source image]
        ori_lmk_file = open(oriimg_path.replace(".jpg", "_landmark.json"))
        ori_lmk = np.array(json.load(ori_lmk_file), dtype="float32")

        # Get the landmarks from the the pred out
        out_lmk = np.zeros_like(ori_lmk)
        for i in range(ori_lmk.shape[0]):
            for j in range(ori_lmk.shape[1]):
                x = ori_lmk[i, j, 0]
                y = ori_lmk[i, j, 1]
                if y < predflow_y.shape[0] and x < predflow_y.shape[1]:
                    out_lmk[i, j, 0] = x - predflow_x[int(y), int(x)]
                    out_lmk[i, j, 1] = y - predflow_y[int(y), int(x)]
                else:
                    out_lmk[i, j, 0] = x
                    out_lmk[i, j, 1] = y

        # Compute the face metric
        face_pred_sim = compute_cosin_similarity(out_lmk, stereo_lmk)
        face_all_sum_pred.append(face_pred_sim)
        stereo_lmk_file.close()
        ori_lmk_file.close()

        # Get the line from the [gt image]
        gt_line_file = oriimg_path.replace(".jpg", "_line_lines.json")
        lines = json.load(open(gt_line_file))

        # Get the line from the [source image]
        ori_line_file = oriimg_path.replace(".jpg", "_lines.json")
        ori_lines = json.load(open(ori_line_file))

        # Get the line from the pred out
        pred_ori2shape_lines = []
        for index, ori_line in enumerate(ori_lines):
            ori_line = np.array(ori_line, dtype="float32")
            pred_ori2shape = np.zeros_like(ori_line)
            for i in range(ori_line.shape[0]):
                x = ori_line[i, 0]
                y = ori_line[i, 1]
                pred_ori2shape[i, 0] = x - predflow_x[int(y), int(x)]
                pred_ori2shape[i, 1] = y - predflow_y[int(y), int(x)]
            pred_ori2shape = pred_ori2shape.tolist()
            pred_ori2shape_lines.append(pred_ori2shape)

        # Compute the lines score
        line_pred_ori2shape_sum = []
        for index, line in enumerate(lines):
            gt_line = np.array(line, dtype="float32")
            pred_ori2shape = np.array(pred_ori2shape_lines[index], dtype="float32")
            gt_k = (gt_line[1, 1] - gt_line[0, 1]) / (gt_line[1, 0] - gt_line[0, 0] + eps)
            pred_ori2shape_score = compute_line_slope_difference(pred_ori2shape, gt_k)
            line_pred_ori2shape_sum.append(pred_ori2shape_score)
        line_all_sum_pred.append(np.mean(line_pred_ori2shape_sum))

    return np.mean(line_all_sum_pred) * 100, np.mean(face_all_sum_pred) * 100

def test(args):
    
    net = MOWA(img_size=args.input_size, tps_points=args.tps_points, embed_dim=args.embed_dim, win_size=args.win_size, 
                token_projection=args.token_projection, token_mlp=args.token_mlp, depths=args.depths, 
                prompt=args.prompt, task_classes=args.task_classes, head_num=args.head_num, shared_head=args.shared_head)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
        net = net.to(device)

    MODEL_DIR = args.model_path
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        print(model_path)
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
    
    oriimg_paths = []
    for root, _, files in os.walk(args.test_path):
        for file_name in files:
            if file_name.endswith(".jpg"):
                if "line" not in file_name and "stereo" not in file_name and "pred" not in file_name:
                    oriimg_paths.append(os.path.join(root, file_name))
    
    print("The number of images: :", len(oriimg_paths))
    
    line_score, face_score = compute_ori2shape_face_line_metric(net, oriimg_paths, device, args)
    print("Line_score = {:.3f}, Face_score = {:.3f} ".format(line_score, face_score))


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
    parser.add_argument('--test_path', type=str, default="/Dataset/FaceRec/test_4_3_all/")

    print('<==================== Testing ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
