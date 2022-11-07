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

#######
import matplotlib.pyplot as plt

from h2o_dataset import H2ODataset

from torch.utils.data import DataLoader

from my_utils.utils import get_att_map, render_gaussian_heatmap, render_gaussian_heatmap_1d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model_path', type=str, default='model')
    parser.add_argument('--test_folder', type=str, default='test_data')
    parser.add_argument('--out_path', type=str, default='results')
    parser.add_argument('--render_result', type=str, default=0)
    args = parser.parse_args()
    return args


def selective_load_model(model, folder = cfg.model_dir_w_depth):
    model_file_list = glob.glob(osp.join(folder,'*.pth.tar'))
    cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
    model_path = osp.join(folder, 'snapshot_' + str(cur_epoch) + '.pth.tar')
    para_dict=torch.load("{}".format(model_path))['network']
    for k in model.state_dict().keys():
        if k in para_dict:
            model.state_dict()[k].copy_(para_dict[k])
    return model

def heat_to_joints(heatmap):
    val_z, idx_z = torch.max(heatmap,2)
    val_zy, idx_zy = torch.max(val_z,2)
    val_zyx, joint_x = torch.max(val_zy,2)
    joint_x = joint_x[:,:,None]
    joint_y = torch.gather(idx_zy, 2, joint_x)
    xyc=torch.cat((joint_x, joint_y, val_zyx[:,:,None]),2).float()
    return xyc

def eucl_dist_torch(output, gt):
    return (output - gt).pow(2).sum(-1).sqrt()


def main():
    args = parse_args()
    side = None#'updown'
    model_new=InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
                        cascaded_num=3,cascaded_input='double',heatmap_attention=True)
    model_old=InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
                        cascaded_num=3,cascaded_input='double',heatmap_attention=True)
    device_run=torch.device('cuda:%d'%(args.gpu))
    #############################################################################
    # Load New
    model_new = selective_load_model(model_new, folder = args.model_folder)
    # Load Old
    para_dict=torch.load("{}".format(args.model_path), map_location=device_run)
    for k in model_old.state_dict().keys():
        if k in para_dict:
            model_old.state_dict()[k].copy_(para_dict[k])
    #############################################################################
    model_old.to(device_run)
    model_old.eval()
    model_new.to(device_run)
    model_new.eval()
    print('load success')
    INPUT_SIZE=256

    test_list = np.loadtxt('h2o_trainval/pose_test.txt', dtype='object')#[0:1000:100]
    test_data = H2ODataset('/media/data/alberto/h2o_dataset_updated', test_list, input_size=INPUT_SIZE, device=device_run, \
            n_jobs=10, mode='bbox')
    test_loader = DataLoader(test_data, batch_size=1)

    from network.InterHand.loss import JointHeatmapLoss

    loss_fn = JointHeatmapLoss()

    old_dist = 0.
    new_dist = 0.

    for i, data in enumerate(test_loader):

        input_tensor, gt, metadata = data
        img_name = test_list[i]
        output_file_name=img_name.split('.')[0]

        ###################################################

        l_joints = gt['l_bbox'][0]#/256*64
        r_joints = gt['r_bbox'][0]#/256*64

        gt_joints = torch.cat([r_joints/256*64, l_joints/256*64], axis=0).unsqueeze(0).cuda()#.detach().cpu()

        gt_heatmap = render_gaussian_heatmap(gt_joints)
        gt_heat_joints = heat_to_joints(gt_heatmap)

        gt_attention = get_att_map(gt_heatmap, side=side)

        att = gt_attention[0,0]+gt_attention[0,1]

        plt.figure(figsize=(15,10))
        plt.subplot(1,3,1)
        plt.matshow(att.cpu().detach(), fignum=False)
        plt.title("Flat distribution:\n {:.3f}".format((loss_fn(torch.zeros_like(att), att).sum()/255).item()), fontsize=16)

        ###################################################

        old_heatmap, _ = model_old(input_tensor, intershape=False)
        old_joints = heat_to_joints(old_heatmap)
        old_att_sep = get_att_map(old_heatmap, side=side)
        old_attention = old_att_sep[0,0]+old_att_sep[0,1]        

        new_heatmap, root_depth = model_new(input_tensor, intershape=False)
        new_joints = heat_to_joints(new_heatmap)
        new_att_sep = get_att_map(new_heatmap, side=side)
        new_attention = new_att_sep[0,0]+new_att_sep[0,1]

        # Compute joint distance
        old_dist += eucl_dist_torch(old_joints[0,:,:2], gt_joints[0,:,:2]).mean()
        new_dist += eucl_dist_torch(new_joints[0,:,:2], gt_joints[0,:,:2]).mean()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
#
        #fig = plt.figure()
        plt.subplot(1,3,2)
        plt.matshow(new_attention.cpu().detach(), fignum=False)
        plt.title("Finetuned Score: \n{:.3f}".format((loss_fn(new_attention, att).sum()/255).item()), fontsize=16)

        #fig = plt.figure()
        plt.subplot(1,3,3)
        plt.matshow(old_attention.cpu().detach(), fignum=False)
        plt.title("Pretrained Score: \n{:.3f}".format((loss_fn(old_attention, att).sum()/255).item()), fontsize=16)
        
        out_path = args.out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print('Saving', img_name)
        plt.savefig(os.path.join(out_path,os.path.basename(output_file_name)+'_attention_.png'))
        plt.close()

        print(f"{img_name} old_loss: {loss_fn(old_heatmap, gt_heatmap).mean()} \n new_loss: {loss_fn(new_heatmap, gt_heatmap).mean()}")

    old_dist /= i+1
    new_dist /= i+1
    print(f'-----> {old_dist}, {new_dist}')


if __name__=='__main__':
    main()
