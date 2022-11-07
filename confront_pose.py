from json import load
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

from my_utils.visualization import plot_3d_skeleton, plot_2d_skeleton, plot_sample, plot_multistage_sample

from my_utils.metrics import mean_joint_error, PCK, transform_compare

from my_utils.utils import compute_k_value, project_back

from my_utils.utils import render_gaussian_heatmap, get_att_map, render_gaussian_heatmap_1d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model1_path', type=str, default='model')
    parser.add_argument('--model2_path', type=str, default='model')
    parser.add_argument('--test_folder', type=str, default='test_data/h2o_train')
    parser.add_argument('--out_path', type=str, default='results/valid')
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


def main():
    args = parse_args()
    model_fine=InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
                        cascaded_num=3,cascaded_input='double',heatmap_attention=True)
    #model_new=InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
    #                    cascaded_num=3,cascaded_input='double',heatmap_attention=True)
    model_old=InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
                        cascaded_num=3,cascaded_input='double',heatmap_attention=True)#, rescale_pose=True)
    device_run=torch.device('cuda:%d'%(args.gpu))
    #############################################################################
    # Load Fine
    model_fine= selective_load_model(model_fine, folder = args.model1_path)
    model_old= selective_load_model(model_old, folder = args.model2_path)
    model_old.to(device_run)
    model_old.eval()
    model_fine.to(device_run)
    model_fine.eval()
    print('load success')
    INPUT_SIZE=256

    test_list = np.loadtxt('h2o_trainval/pose_val.txt', dtype='object')[0:1000:100]
    test_data = H2ODataset('/media/data/alberto/h2o_dataset_updated', test_list, input_size=INPUT_SIZE, device=device_run, \
            n_jobs=10, mode='bbox')
    test_loader = DataLoader(test_data, batch_size=1)

    for i, data in enumerate(test_loader):

        input_tensor, gt, metadata = data

        # compte k_value
        k_value = {
            'l': compute_k_value(metadata['l_area_img'], metadata['l_area_real'], test_data.cam_f),
            'r': compute_k_value(metadata['r_area_img'], metadata['r_area_real'], test_data.cam_f)
            }

        img_name = test_list[i]
        output_file_name=img_name.split('.')[0]

        plt.figure(figsize=(15,10))

        ######################## GT  ############################
        joints_gt = torch.cat([gt['l_bbox'][0], gt['r_bbox'][0]]).squeeze()#.detach().cpu()
        plt.subplot(2,3,1)
        plot_sample(input_tensor, joints_gt, skeleton=test_data.skeleton)
        plt.title('GT', fontsize=15)

        ######################## GT  ############################

        ######################## NEW ############################
        result_fine, _, attention_fine=model_fine(input_tensor, k_value=k_value)
        for idx in range(len(result_fine)):
            for side in ['l', 'r']:
                result_fine[idx][side]['trans'] = project_back(result_fine[idx][side]['trans'].unsqueeze(1), test_data.cam_f, test_data.cam_c, metadata['bbox'].cuda()).squeeze()
                result_fine[idx][side]['T_joints3d'] = (result_fine[idx][side]['joints3d']) + result_fine[idx][side]['trans'][None,None,:]
                result_fine[idx][side]['T_verts3d'] = (result_fine[idx][side]['verts3d']) + result_fine[idx][side]['trans'][None,None,:]

        joints_fine = torch.cat([result_fine[-1]['l']['jointsimg'].squeeze(0).detach().cpu(), result_fine[-1]['r']['jointsimg'].squeeze(0).detach().cpu()])

        l_result_fine, r_result_fine, l_gt, r_gt = result_fine[-1]['l']['joints3d']*1000, result_fine[-1]['r']['joints3d']*1000, gt['l_mano_joints']*1000, gt['r_mano_joints']*1000#transform_compare(result_fine[-1], gt, metadata)
        l_both_fine, r_both_fine, l_both_gt, r_both_gt = result_fine[-1]['l']['T_joints3d']*1000, result_fine[-1]['r']['T_joints3d']*1000, gt['l_joints']*1000, gt['r_joints']*1000#transform_compare(result_fine[-1], gt, metadata, center=False)

        plt.subplot(2,3,2)
        plot_sample(input_tensor, joints_fine, skeleton=test_data.skeleton)
        plt.title(f'Trained\nL_c: {mean_joint_error(l_result_fine, l_gt).mean().item():.2f}\nR_c: {mean_joint_error(r_result_fine, r_gt).mean().item():.2f}', fontsize=15)#\nL: {mean_joint_error(l_both_fine, l_both_gt).mean().item():.5f}\nR: {mean_joint_error(r_both_fine, r_both_gt).mean().item():.5f}\nL_2D: {mean_joint_error(l_both_fine[:,:,:2], l_both_gt[:,:,:2]).mean().item():.5f}')
        ######################## NEW ############################

        ######################## OLD ############################
        result_old, _, attention_old=model_old(input_tensor, k_value=k_value)
        for idx in range(len(result_old)):
            for side in ['l', 'r']:
                result_old[idx][side]['trans'] = project_back(result_old[idx][side]['trans'].unsqueeze(1), test_data.cam_f, test_data.cam_c, metadata['bbox'].cuda()).squeeze()
                result_old[idx][side]['T_joints3d'] = (result_old[idx][side]['joints3d']) + result_old[idx][side]['trans'][None,None,:]
                result_old[idx][side]['T_verts3d'] = (result_old[idx][side]['verts3d']) + result_old[idx][side]['trans'][None,None,:]

        joints_old = torch.cat([result_old[-1]['l']['jointsimg'].squeeze(0).detach().cpu(), result_old[-1]['r']['jointsimg'].squeeze(0).detach().cpu()])

        l_result_old, r_result_old, l_gt, r_gt = result_old[-1]['l']['joints3d']*1000, result_old[-1]['r']['joints3d']*1000, gt['l_mano_joints']*1000, gt['r_mano_joints']*1000#transform_compare(result_old[-1], gt, metadata)
        l_both_old, r_both_old, l_both_gt, r_both_gt = result_old[-1]['l']['T_joints3d']*1000, result_old[-1]['r']['T_joints3d']*1000, gt['l_joints']*1000, gt['r_joints']*1000#transform_compare(result_old[-1], gt, metadata, center=False)

        #plot_multistage_sample(input_tensor, left_xy, right_xy)
        plt.subplot(2,3,3)
        plot_sample(input_tensor, joints_old, skeleton=test_data.skeleton)
        plt.title(f'Pretrained\nL_c: {mean_joint_error(l_result_old, l_gt).mean().item():.5f}\nR_c: {mean_joint_error(r_result_old, r_gt).mean().item():.5f}\nL: {mean_joint_error(l_both_old, l_both_gt).mean().item():.5f}\nR: {mean_joint_error(r_both_old, r_both_gt).mean().item():.5f}\nL_2D: {mean_joint_error(l_both_old[:,:,:2], l_both_gt[:,:,:2]).mean().item():.5f}')
        ######################## OLD ############################

        ######################## GT  ############################
        plt.subplot(2,3,4)
        gt_heatmap = render_gaussian_heatmap((joints_gt/256*64).unsqueeze(0))
        attention = get_att_map(gt_heatmap)
        gt_attention = attention[0,0] + attention[0,1]
        plt.matshow(gt_attention.detach().cpu(), fignum=False)
        ######################## GT  ############################

        ######################## NEW ############################
        plt.subplot(2,3,5)
        fine_attention = attention_fine[0,0] + attention_fine[0,1]
        plt.matshow(fine_attention.detach().cpu(), fignum=False)
        ######################## NEW ############################

        ######################## OLD ############################
        plt.subplot(2,3,6)
        old_attention = attention_old[0,0] + attention_old[0,1]
        plt.matshow(old_attention.detach().cpu(), fignum=False)
        ######################## OLD ############################


        out_path = args.out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print('Saving', img_name)
        plt.savefig(os.path.join(out_path,os.path.basename(output_file_name)+'_confr_joints_.png'))
        plt.close()




        #########################################################
        from my_utils.visualization import plot_3d_skeleton
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,2,1, projection='3d')
        plot_3d_skeleton(torch.cat([l_result_fine.squeeze().detach().cpu(), l_gt.squeeze().detach().cpu()]), test_data.skeleton, ax)
        ax.set_title('Left Hand')
        ax1 = fig.add_subplot(1,2,2, projection='3d')
        #gt_joints = torch.cat([result_fine[-1]['r']['joints3d'].squeeze(), gt['r_mano_joints'].squeeze()])
        plot_3d_skeleton(torch.cat([r_result_fine.squeeze().detach().cpu(), r_gt.squeeze().detach().cpu()]), test_data.skeleton, ax1)
        ax1.set_title('Right Hand')
        plt.savefig(os.path.join(out_path,os.path.basename(output_file_name)+'_3d_joints_centered_.png'))
        plt.close()
        #########################################################

        #########################################################
        from my_utils.visualization import plot_3d_skeleton
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,2,1, projection='3d')
        plot_3d_skeleton(torch.cat([l_both_fine.squeeze().detach().cpu(), l_both_gt.squeeze().detach().cpu()]), test_data.skeleton, ax)
        ax.set_title('Left Hand')
        ax1 = fig.add_subplot(1,2,2, projection='3d')
        #gt_joints = torch.cat([result_fine[-1]['r']['joints3d'].squeeze(), gt['r_mano_joints'].squeeze()])
        #plot_3d_skeleton(torch.cat([r_both_fine.squeeze().detach().cpu(), gt['r_joints'][0].squeeze().detach().cpu()*1000]), test_data.skeleton, ax1)
        plot_3d_skeleton(torch.cat([r_both_fine.squeeze().detach().cpu(), r_both_gt.squeeze().detach().cpu()]), test_data.skeleton, ax1)
        ax1.set_title('Right Hand')
        plt.savefig(os.path.join(out_path,os.path.basename(output_file_name)+'_3d_joints_.png'))
        plt.close()
        #########################################################

        #########################################################
        # Confront 2d cam pose projection with gt
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(2,2,1)
        plot_2d_skeleton(torch.cat([l_both_fine.squeeze().detach().cpu(), gt['l_joints'][0].squeeze().detach().cpu()*1000]), test_data.skeleton, ax)
        ax.set_title('Left Hand')
        ax1 = fig.add_subplot(2,2,2)
        plot_2d_skeleton(torch.cat([r_both_fine.squeeze().detach().cpu(), gt['r_joints'][0].squeeze().detach().cpu()*1000]), test_data.skeleton, ax1)
        #joints_gt = torch.cat([gt['l_bbox'][0], gt['r_bbox'][0]]).squeeze().detach().cpu()
        #plot_2d_skeleton(joints_gt[:21], test_data.skeleton, ax1)
        ax1.set_title('Right Hand')

        ax3 = fig.add_subplot(2,2,3)
        plot_2d_skeleton(torch.cat([l_result_fine.squeeze().detach().cpu(), l_gt.squeeze().detach().cpu()]), test_data.skeleton, ax3)
        ax3.set_title('Left Hand Centered')
        ax4 = fig.add_subplot(2,2,4)
        plot_2d_skeleton(torch.cat([r_result_fine.squeeze().detach().cpu(), r_gt.squeeze().detach().cpu()]), test_data.skeleton, ax4)
        #joints_gt = torch.cat([gt['l_bbox'][0], gt['r_bbox'][0]]).squeeze().detach().cpu()
        #plot_2d_skeleton(joints_gt[:21], test_data.skeleton, ax1)
        ax4.set_title('Right Hand Centered')

        plt.savefig(os.path.join(out_path,os.path.basename(output_file_name)+'_2d_joints_.png'))
        plt.close()
        #########################################################

        #########################################################
        # Confront 2d cam pose projection with gt
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(2,2,1)
        plot_2d_skeleton(torch.cat([l_both_fine.squeeze().detach().cpu()[:, 1:], gt['l_joints'][0].squeeze().detach().cpu()[:, 1:]*1000]), test_data.skeleton, ax)
        ax.set_title('Left Hand')
        ax1 = fig.add_subplot(2,2,2)
        plot_2d_skeleton(torch.cat([r_both_fine.squeeze().detach().cpu()[:, 1:], gt['r_joints'][0].squeeze().detach().cpu()[:, 1:]*1000]), test_data.skeleton, ax1)
        #joints_gt = torch.cat([gt['l_bbox'][0], gt['r_bbox'][0]]).squeeze().detach().cpu()
        #plot_2d_skeleton(joints_gt[:21], test_data.skeleton, ax1)
        ax1.set_title('Right Hand')

        ax3 = fig.add_subplot(2,2,3)
        plot_2d_skeleton(torch.cat([l_result_fine.squeeze().detach().cpu()[:, 1:], l_gt.squeeze().detach().cpu()[:, 1:]]), test_data.skeleton, ax3)
        ax3.set_title('Left Hand Centered')
        ax4 = fig.add_subplot(2,2,4)
        plot_2d_skeleton(torch.cat([r_result_fine.squeeze().detach().cpu()[:, 1:], r_gt.squeeze().detach().cpu()[:, 1:]]), test_data.skeleton, ax4)
        #joints_gt = torch.cat([gt['l_bbox'][0], gt['r_bbox'][0]]).squeeze().detach().cpu()
        #plot_2d_skeleton(joints_gt[:21], test_data.skeleton, ax1)
        ax4.set_title('Right Hand Centered')
        plt.savefig(os.path.join(out_path,os.path.basename(output_file_name)+'_2d_joints_upside_.png'))
        plt.close()
        #########################################################



if __name__=='__main__':
    main()
