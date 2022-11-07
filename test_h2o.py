import numpy as np
import torch
import cv2
from torch._C import device
import torch.nn as nn
import argparse
from tqdm import tqdm
import os
import os.path as osp
import glob
from network.full_model import InterShape
from config import cfg

from h2o_dataset import H2ODataset
from torch.utils.data import DataLoader

from my_utils.metrics import eucl_dist_torch, mean_joint_error, PCK, transform_compare
from my_utils.utils import compute_k_value, project_back
from tqdm import tqdm

import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--batch_size', type=int, default='12')
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--heat_xy', action='store_true', default=False)
    parser.add_argument('--render_result', type=str, default=0)
    args = parser.parse_args()
    return args

def selective_load_model(model, folder = cfg.model_dir_w_depth):
    model_file_list = glob.glob(osp.join(folder,'*.pth.tar'))
    cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
    model_path = osp.join(folder, 'snapshot_' + str(cur_epoch) + '.pth.tar')
    print('----> Loading snapshot from {}'.format(model_path))
    para_dict=torch.load("{}".format(model_path))['network']
    for k in model.state_dict().keys():
        if k in para_dict:
            model.state_dict()[k].copy_(para_dict[k])
    return model

def compute_metrics_dict(metrics_dict, l_result, r_result, l_gt, r_gt):
    # Compute metrics
    metrics_dict['l_mje'] += mean_joint_error(l_result, l_gt).mean().detach().numpy()
    metrics_dict['r_mje'] += mean_joint_error(r_result, r_gt).mean().detach().numpy()
    metrics_dict['l_pck_15'] += PCK(l_result, l_gt, threshold=15).mean().detach().numpy()
    metrics_dict['r_pck_15'] += PCK(r_result, r_gt, threshold=15).mean().detach().numpy()
    metrics_dict['l_pck_30'] += PCK(l_result, l_gt, threshold=30).mean().detach().numpy()
    metrics_dict['r_pck_30'] += PCK(r_result, r_gt, threshold=30).mean().detach().numpy()
    return metrics_dict

def compute_root_metrics_dict(metrics_dict, l_root, r_root, l_gt_root, r_gt_root):
    metrics_dict['l_x'] += torch.abs(l_root[:,0]-l_gt_root[:,0]).mean().detach().cpu().numpy()
    metrics_dict['l_y'] += torch.abs(l_root[:,1]-l_gt_root[:,1]).mean().detach().cpu().numpy()
    metrics_dict['l_z'] += torch.abs(l_root[:,2]-l_gt_root[:,2]).mean().detach().cpu().numpy()
    metrics_dict['l_mre'] += eucl_dist_torch(l_root, l_gt_root).mean().detach().cpu().numpy()
    metrics_dict['r_x'] += torch.abs(r_root[:,0]-r_gt_root[:,0]).mean().detach().cpu().numpy()
    metrics_dict['r_y'] += torch.abs(r_root[:,1]-r_gt_root[:,1]).mean().detach().cpu().numpy()
    metrics_dict['r_z'] += torch.abs(r_root[:,2]-r_gt_root[:,2]).mean().detach().cpu().numpy()
    metrics_dict['r_mre'] += eucl_dist_torch(r_root, r_gt_root).mean().detach().cpu().numpy()
    return metrics_dict

def normalize_dict_results(metrics_dict, factor):
    for key in metrics_dict.keys():
        metrics_dict[key] /=factor
    return metrics_dict



def main():
    args = parse_args()

    list_dir = os.listdir(args.model_path)
    device_run=torch.device('cuda:%d'%(args.gpu))
    INPUT_SIZE=256

    model=InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
                            cascaded_num=3,cascaded_input='double',heatmap_attention=True, use_heat_xy = args.heat_xy)

    #test_list = np.loadtxt('h2o_trainval/pose_train.txt', dtype='object')[0:55742:100]
    test_list = np.loadtxt('h2o_trainval/pose_test.txt', dtype='object')[0:23391:100]
    #test_list = np.loadtxt('h2o_trainval/pose_val.txt', dtype='object')[0:9939:100] 
    #test_list = np.loadtxt('h2o_trainval/pose_train.txt', dtype='object')[0:55742:100] # 10%
    test_data = H2ODataset('/media/data/alberto/h2o_dataset_updated', test_list, input_size=INPUT_SIZE, device=device_run, \
            n_jobs=10, mode='bbox', cache='test_k_1')
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    with torch.no_grad():
        for model_dir in list_dir:
            model_folder = osp.join(args.model_path, model_dir)
            if not osp.isdir(model_folder) or osp.basename(model_folder) == 'runs':
                continue

            model = selective_load_model(model, folder=model_folder)
            model.to(device_run)
            model.eval()
            print('load success')

            scores = {'l_mje' : 0. ,'r_mje' : 0. ,'l_pck_15' : 0. ,'r_pck_15' : 0. ,'l_pck_30' : 0. ,'r_pck_30' : 0.}
            scores_centered = {'l_mje' : 0. ,'r_mje' : 0. ,'l_pck_15' : 0. ,'r_pck_15' : 0. ,'l_pck_30' : 0. ,'r_pck_30' : 0.}
            scores_2D = {'l_mje' : 0. ,'r_mje' : 0. ,'l_pck_15' : 0. ,'r_pck_15' : 0. ,'l_pck_30' : 0. ,'r_pck_30' : 0.}
            scores_2D_centered = {'l_mje' : 0. ,'r_mje' : 0. ,'l_pck_15' : 0. ,'r_pck_15' : 0. ,'l_pck_30' : 0. ,'r_pck_30' : 0.}
            root_scores = {'l_x' : 0., 'l_y' : 0., 'l_z' : 0., 'l_mre' : 0., 'r_x' : 0., 'r_y' : 0., 'r_z' : 0., 'r_mre' : 0.}

            with tqdm(test_loader, unit='batch') as tepoch:
                for i, data in enumerate(tepoch):

                    input_tensor, gt, metadata = data

                    k_value = {
                        'l': compute_k_value(metadata['l_area_img'], metadata['l_area_real'], test_data.cam_f),
                        'r': compute_k_value(metadata['r_area_img'], metadata['r_area_real'], test_data.cam_f)
                        }

                    #img_name = test_list[i]
                    result=model(input_tensor, k_value)[0]

                    for idx in range(len(result)):
                        for side in ['l', 'r']:
                            result[idx][side]['trans'] = project_back(result[idx][side]['trans'].unsqueeze(1), test_data.cam_f, test_data.cam_c, metadata['bbox'].cuda()).squeeze()
                            result[idx][side]['T_joints3d'] = (result[idx][side]['joints3d']) + result[idx][side]['trans'][:,None,:]
                            result[idx][side]['T_verts3d'] = (result[idx][side]['verts3d']) + result[idx][side]['trans'][:,None,:]

                    l_c, r_c, l_gt_c, r_gt_c = transform_compare(result[-1], gt, metadata)
                    l, r, l_gt, r_gt = transform_compare(result[-1], gt, metadata, center=False)

                    root_scores = compute_root_metrics_dict(root_scores, l[:,9], r[:,9], l_gt[:,9], r_gt[:,9])

                    scores = compute_metrics_dict(scores, l, r, l_gt, r_gt)
                    scores_centered = compute_metrics_dict(scores_centered, l_c, r_c, l_gt_c, r_gt_c)
                    scores_2D = compute_metrics_dict(scores_2D, l[:,:,:2], r[:,:,:2], l_gt[:,:,:2], r_gt[:,:,:2])
                    scores_2D_centered = compute_metrics_dict(scores_2D_centered, l_c[:,:,:2], r_c[:,:,:2], l_gt_c[:,:,:2], r_gt_c[:,:,:2])

                    tepoch.set_postfix({'Left, Right' : [mean_joint_error(l, l_gt).mean().item(), mean_joint_error(r, r_gt).mean().item()]})

            root_scores = normalize_dict_results(root_scores, i+1)
            scores = normalize_dict_results(scores, i+1)
            scores_centered = normalize_dict_results(scores_centered, i+1)
            scores_2D = normalize_dict_results(scores_2D, i+1)
            scores_2D_centered = normalize_dict_results(scores_2D_centered, i+1)

            print(f'\nMJE in mm:\n Left:{scores["l_mje"].item()}, Right:{scores["r_mje"].item()}')
            print(f'\nPCK 15:\n Left:{scores["l_pck_15"].item()}, Right:{scores["r_pck_15"].item()}')
            print(f'\nPCK 30:\n Left:{scores["l_pck_30"].item()}, Right:{scores["r_pck_30"].item()}')

            final_scores = {'Relative' : scores_centered}#, 'Absolute' : scores, 'Absolute 2D' : scores_2D, 'Relative 2D' : scores_2D_centered, 'root' : root_scores}

            filename = osp.join(model_folder,'Test_metrics.json')

            file = open(filename, 'w')
            json.dump(final_scores, file)
            file.close()
        


if __name__=='__main__':
    main()