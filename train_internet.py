from syslog import LOG_SYSLOG
import os.path as osp
import glob
import numpy as np
import torch
import cv2
from torch._C import device
import torch.nn as nn
import argparse
from tqdm import tqdm
import os
from network.full_model import InterShape

from h2o_dataset import H2ODataset
#from my_utils.losses import JointLoss
from network.InterHand.loss import JointHeatmapLoss, RelRootDepthLoss
import torchvision.transforms as T
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import cfg

from my_utils.utils import render_gaussian_heatmap, render_gaussian_heatmap_1d, get_att_map
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model_path', type=str, default='model/model.pts')
    parser.add_argument('--data_folder', type=str, default='/media/data/alberto/h2o_dataset_updated')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--render_result', type=str, default=0)
    args = parser.parse_args()
    return args

def save_model(state, epoch):
    file_path = osp.join(cfg.model_dir_w_depth,'snapshot_{}.pth.tar'.format(str(epoch)))
    torch.save(state, file_path)
    #self.logger.info("Write snapshot into {}".format(file_path))


def train_one_epoch(epoch_number, train_loader, model, loss_fn, optimizer, tb_writer):#, epoch_index):
    running_loss = 0.
    running_heat = 0.
    running_depth = 0.
    avg_loss = 0.
    i = 0
    
    #for i, data in enumerate(loader):
    with tqdm(train_loader, unit='batch') as tepoch:
        for data in tepoch:
            input_tensor, gt, metadata = data

            optimizer.zero_grad()
            #result, heatmap, hand_type = model(input_tensor)
            heatmap, root_depth_out = model(input_tensor, intershape=False)

            gt_joints = torch.cat([gt['r_bbox']/256*64, gt['l_bbox']/256*64], axis=1)

            gt_heatmap = render_gaussian_heatmap(gt_joints)

            gt_1d_heatmap = render_gaussian_heatmap_1d(gt_joints)

            loss_heat = loss_fn['heatmap'](heatmap, gt_heatmap).mean()
            loss_depth = loss_fn['depth'](root_depth_out, gt_1d_heatmap).mean()

            loss = loss_heat + loss_depth

            loss.backward()

            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            running_heat += loss_heat.item()
            running_depth += loss_depth.item()


            tepoch.set_description(f"Batch [{i}/{len(train_loader)}]")
            tepoch.set_postfix({'Heat Loss' : loss_heat.item(), 'Depth Loss' : loss_depth.item() })

            i += 1

            if i % 100 == 99:
                avg_loss = running_loss / 100 # loss per batch
                avg_heat = running_heat / 100
                avg_depth = running_depth / 100
                #print('  batch {} loss: {}'.format(i + 1, avg_loss))
                tb_x = epoch_number * len(train_loader) + i + 1
                #tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
                tb_writer.add_scalars('Loss',
                    { 'Heat Loss' : avg_heat, 'Depth Loss' : avg_depth, 'Total Loss' : avg_loss },
                    tb_x)
                running_loss = 0.
                running_heat = 0.
                running_depth = 0.


    return avg_loss

def main():

    args = parse_args()
    writer = SummaryWriter(log_dir='runs/w_depth')
    model = InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
                        cascaded_num=3,cascaded_input='double',heatmap_attention=True)
    device_run = torch.device('cuda:%d'%(args.gpu))
    torch.cuda.empty_cache()
    para_dict=torch.load("{}".format(args.model_path), map_location=device_run)
    for k in model.state_dict().keys():
        if k in para_dict:
            model.state_dict()[k].copy_(para_dict[k])
    model.to(device_run)

    print('load success')
    INPUT_SIZE=256
    #right_face=model.mesh_reg.mano_layer['r'].th_faces
    #left_face=model.mesh_reg.mano_layer['l'].th_faces

    # Load Data
    train_list = np.loadtxt('h2o_trainval/pose_train.txt', dtype='object')
    train_data = H2ODataset('/media/data/alberto/h2o_dataset_updated', train_list, j_from_mano=True, input_size=INPUT_SIZE, device=device_run, \
            n_jobs=10, mode='bbox', cache='train_new')
    train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
    val_list = np.loadtxt('h2o_trainval/pose_val.txt', dtype='object')#[0:1100:10]
    val_data = H2ODataset('/media/data/alberto/h2o_dataset_updated', val_list, j_from_mano=True, input_size=INPUT_SIZE, device=device_run, \
            n_jobs=10, mode='bbox', cache='val_new')
    val_loader = DataLoader(val_data, batch_size=12, shuffle=True)

    # Loss Function
    loss_fn = {}
    loss_fn['heatmap'] = JointHeatmapLoss()
    loss_fn['depth'] = RelRootDepthLoss()
    optimizer = torch.optim.Adam(model.heatmap_predictor.parameters()) # train only heatmap_predictor

    if not osp.exists(cfg.model_dir_w_depth):
        os.makedirs(cfg.model_dir_w_depth)

    best_vloss = 1_000_000.

    # Train cycle
    for epoch in range(20):
        
        print('EPOCH {}:'.format(epoch + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch, train_loader, model, loss_fn, optimizer, writer)
        model.train(False)

        running_vloss = 0.
        i=0
        with tqdm(val_loader, unit='batch') as vepoch:
            for vdata in vepoch:
                vinput, vgt, _ = vdata

                vheatmap, vroot_depth_out= model(vinput, intershape=False)
                vgt_joints = torch.cat([vgt['r_bbox']/256*64, vgt['l_bbox']/256*64], axis=1)
                vgt_heatmap = render_gaussian_heatmap(vgt_joints)
                vgt_1d_heatmap = render_gaussian_heatmap_1d(vgt_joints)

                #voutputs, _, _ = model(vinput)
                vloss_heat = loss_fn['heatmap'](vheatmap, vgt_heatmap).mean()
                vloss_depth = loss_fn['depth'](vroot_depth_out, vgt_1d_heatmap).mean()
                vloss = vloss_heat + vloss_depth
                #sep_vloss = loss_fn(voutputs[-1], vgt)
                running_vloss += vloss.item()
                vepoch.set_description(f"Validation Batch [{i}/{len(val_loader)}]")
                vepoch.set_postfix({'Loss' : vloss.item()})
                i+=1

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)

        writer.flush()

        #if avg_vloss < best_vloss:
        #    best_vloss = avg_vloss
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            state = {'epoch': epoch, 'network': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_model(state, epoch)
            print('\n\t---> Model saved!\n')

    return


if __name__ == '__main__':

    main()