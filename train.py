
from syslog import LOG_SYSLOG
import numpy as np
import torch
import argparse
from tqdm import tqdm
import os
from network.full_model import InterShape

from h2o_dataset import H2ODataset
from my_utils.losses import InterShapeLoss
from torch.utils.data import DataLoader

from tqdm import tqdm

import os.path as osp
import glob
from config import cfg
import json
import sys #Save terminal output

from torch.utils.tensorboard import SummaryWriter

import collections

from my_utils.utils import compute_k_value, project_results, project_back


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model_path', type=str, default=cfg.model_dir_w_depth)#'model/model.pts')
    parser.add_argument('--data_folder', type=str, default='/media/data/alberto/h2o_dataset_updated')
    parser.add_argument('--output_folder', type=str, default='model/finetune_relative_jl2_rot_usegt_2_resc')
    parser.add_argument('--submodel_folder', type=str, default='finetune')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--use_gt', action='store_true', default=False)
    parser.add_argument('--absolute', action='store_true', default=False)
    parser.add_argument('--rescale', action='store_true', default=False)
    parser.add_argument('--root', action='store_true', default=False)
    parser.add_argument('--root', action='store_true', default=False)
    #parser.add_argument('--render_result', type=str, default=0)
    parser.add_argument('--patience', type=int, default='7')
    args = parser.parse_args()
    return args

def save_model(state, epoch, output_folder):
    file_path = osp.join(output_folder,'snapshot_{}.pth.tar'.format(str(epoch)))
    torch.save(state, file_path)
    print("Write snapshot into {}".format(file_path))

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


def train_one_epoch(epoch_number, train_loader, model, loss_fn, optimizer, tb_writer, cam_f = [1., 1.], cam_c = [0., 0.], mem=100, use_gt=False, rescale=False):#, epoch_index):
    running_loss = 0.
    tot_loss = 0.
    avg_loss = 0.
    i = 0
    
    #for i, data in enumerate(loader):
    with tqdm(train_loader, unit='batch', position=0, leave=True) as tepoch:
        for data in tepoch:

            input_tensor, gt, meta = data

            # compte k_value
            k_value = {
                'l': compute_k_value(meta['l_area_img'], meta['l_area_real'], cam_f),
                'r': compute_k_value(meta['r_area_img'], meta['r_area_real'], cam_f)
                }

            optimizer.zero_grad()
            output, _, _ = model(input_tensor, k_value=k_value)
            #result = output

            ######### Project results to 2d space using cam_intrinsics ###########
            for idx in range(len(output)):
                for side in ['l', 'r']:
                    output[idx][side]['trans'] = project_back(output[idx][side]['trans'].unsqueeze(1), cam_f, cam_c, meta['bbox'].cuda()).squeeze()
                    output[idx][side]['T_joints3d'] = (output[idx][side]['joints3d']) + output[idx][side]['trans'][:,None,:]
                    output[idx][side]['T_verts3d'] = (output[idx][side]['verts3d']) + output[idx][side]['trans'][:,None,:]
                    if use_gt:
                        if rescale:
                            joints = output[idx][side]['joints3d']*meta[side+'_mano_length'][:,None,None]
                        else:
                            joints = output[idx][side]['joints3d']
                        output[idx][side]['T_joints3d'][meta[side+'_joints'] == 1] = joints[meta[side+'_joints'] == 1] + gt[side+'_joints'][meta[side+'_joints'] == 1][:,None,9]
                    output[idx][side]['joints2d'] = project_results(output[idx][side]['T_joints3d'], cam_f, cam_c, meta['bbox'].cuda())
            ######### ################################################ ###########

            ######################################################
            ######################################################
            counter = collections.Counter()
            for idx in range(len(output)):
                counter.update(loss_fn(output[idx], gt, meta))
            loss = dict(counter)
            tloss = sum(loss[k] for k in loss)
            tloss.backward()
            ######################################################
            ######################################################

            optimizer.step()

            # Gather data and report
            #running_loss += tloss
            tot_loss += tloss.item()
            #running_loss += loss.item()

            tepoch.set_description(f"Training Batch [{i}/{len(train_loader)}]")
            tepoch.set_postfix({'Loss' : tloss.item()})

            i += 1

            if i % mem == mem-1:
                #avg_loss = running_loss / mem # loss per batch
                #print('  batch {} loss: {}'.format(i + 1, avg_loss))
                tb_x = epoch_number * len(train_loader) + i + 1
                #for idx in range(len(output)-1):
                #    tb_writer.add_scalars(str(idx)+'stage_Loss', loss[idx], tb_x)
                tb_writer.add_scalars('Training Losses', loss, tb_x)
                #tb_writer.add_scalar('Loss', avg_loss, tb_x)
                running_loss = 0.

        tot_loss /= (i+1)

    return tot_loss

def main():

    args = parse_args()

    ## Fix random seeds
    #seed = 123
    #torch.backends.cudnn.deterministic = True
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #np.random.seed(seed)

    root = None

    loss_params = { 'l_offset' : 1, 
                    'l_consistency' : 0.01, 
                    'l_joint' : 10,
                    #l_joint2d=100, 
                    'l_length' : 100,
                    'l_shape' : 0.1,#0.1,
                    'l_reg' : 0.1,
                    'l_center' : 10,
                    #'l_depth' : 100,
                    'l_g_rot' : 100 }

    if args.finetune:
        if args.root:
            root = 'only'
            args.model_path = args.output_folder
            with open(osp.join(args.model_path, 'loss_params.json')) as f:
                loss_params = json.load(f)
            args.output_folder = osp.join(args.output_folder, 'root_only')
        else:
            args.model_path = args.output_folder
            args.output_folder = osp.join(args.output_folder, args.submodel_folder)

    if args.absolute:
        args.output_folder = args.output_folder+'absolute'
        if args.root:
            root = True
            args.output_folder = args.output_folder+'_root'
    else:
        args.output_folder = args.output_folder+'relative'
        #if args.root:
        #    root = True
        #    args.output_folder = args.output_folder+'_root'
    if args.use_gt:
        args.output_folder = args.output_folder+'_usegt'
    if args.rescale:
        args.output_folder = args.output_folder+'_resc'

    # Create output dir
    if not osp.exists(args.output_folder):
        print('\n -----> Creating folder {}'.format(args.output_folder))
        os.makedirs(args.output_folder)

    # Init log
    if args.log:
        sys_log = open(osp.join(args.output_folder,"log.txt"), "w")
        sys.stdout = sys_log

    # Save Loss params
    file = open(osp.join(args.output_folder, 'loss_params.json'), 'w')
    json.dump(loss_params, file)
    file.close()

    # Save input args log
    file = open(osp.join(args.output_folder, 'input_args.json'), 'w')
    json.dump(args.__dict__, file)
    file.close()

    #writer = SummaryWriter(log_dir='runs/'+osp.basename(args.output_folder))
    writer = SummaryWriter(log_dir=osp.join(args.output_folder, 'runs/'))
    model = InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
                        cascaded_num=3,cascaded_input='double',heatmap_attention=True, rescale_pose=args.rescale)
    device_run = torch.device('cuda:%d'%(args.gpu))
    torch.cuda.empty_cache()
    model = selective_load_model(model, folder = args.model_path)
    model.to(device_run)

    print('load success')
    INPUT_SIZE=256

    # Load Data
    train_list = np.loadtxt('h2o_trainval/pose_train.txt', dtype='object')#[0:55742:100] # 10%
    train_data = H2ODataset('/media/data/alberto/h2o_dataset_updated', train_list, input_size=INPUT_SIZE, device=device_run, \
            n_jobs=10, mode='bbox', cache='train_k', rescale_pose=args.rescale)
    train_loader = DataLoader(train_data, batch_size=12, shuffle=True)

    val_list = np.loadtxt('h2o_trainval/pose_val.txt', dtype='object')#[0:9939:100]
    val_data = H2ODataset('/media/data/alberto/h2o_dataset_updated', val_list, input_size=INPUT_SIZE, device=device_run, \
            n_jobs=10, mode='bbox', cache='val_k', rescale_pose=args.rescale)
    val_loader = DataLoader(val_data, batch_size=12, shuffle=True)

    # Loss Function
    if args.finetune or args.absolute:
        loss_fn = InterShapeLoss(l_offset = loss_params['l_offset'], 
                                l_consistency = loss_params['l_consistency'] , 
                                l_joint = loss_params['l_joint'],
                                l_length = loss_params['l_length'],
                                l_shape = loss_params['l_shape'],
                                l_reg = loss_params['l_reg'],
                                l_center = loss_params['l_center'],
                                #l_depth = loss_params['l_depth'],
                                l_g_rot = loss_params['l_g_rot'] , absolute=True, root=root)
    else:
        loss_fn = InterShapeLoss(l_offset = loss_params['l_offset'], 
                                l_consistency = loss_params['l_consistency'] , 
                                l_joint = loss_params['l_joint'],
                                l_length = loss_params['l_length'],
                                l_shape = loss_params['l_shape'],
                                l_reg = loss_params['l_reg'],
                                l_center = loss_params['l_center'],
                                #l_depth = loss_params['l_depth'],
                                l_g_rot = loss_params['l_g_rot'], root=root)

    if root=='only':
        optimizer = torch.optim.Adam(model.mesh_reg.root_net.parameters())
    else:
        optimizer = torch.optim.Adam(model.mesh_reg.parameters())
    #optimizer = torch.optim.Adam(model.parameters())
    
    best_vloss = 1_000_000.
    patience = args.patience #Early stopping patience
    trigger_times = 0
    
    # Train cycle
    for epoch in range(30):
        
        print('EPOCH {}:'.format(epoch + 1))
    
        model.train(True)
        avg_loss = train_one_epoch(epoch, train_loader, model, loss_fn, optimizer, writer, train_data.cam_f, train_data.cam_c, mem=1, use_gt=args.use_gt, rescale=args.rescale)
        model.train(False)
    
        running_vloss = 0.
        i=0
        with tqdm(val_loader, unit='batch') as vepoch:
            for vdata in vepoch:
                vinput, vgt, vmeta = vdata
                # compte k_value
                vk_value = {
                    'l': compute_k_value(vmeta['l_area_img'], vmeta['l_area_real'], val_data.cam_f),
                    'r': compute_k_value(vmeta['r_area_img'], vmeta['r_area_real'], val_data.cam_f)
                    }
                voutput, _, _ = model(vinput, k_value=vk_value)
                ######### Project results to 2d space using cam_intrinsics ###########
                for idx in range(len(voutput)):
                    for side in ['l', 'r']:
                        voutput[idx][side]['trans'] = project_back(voutput[idx][side]['trans'].unsqueeze(1), val_data.cam_f, val_data.cam_c, vmeta['bbox'].cuda()).squeeze()
                        voutput[idx][side]['T_joints3d'] = (voutput[idx][side]['joints3d']) + voutput[idx][side]['trans'][:,None,:]
                        voutput[idx][side]['T_verts3d'] = (voutput[idx][side]['verts3d']) + voutput[idx][side]['trans'][:,None,:]
                        if args.use_gt:
                            if args.rescale:
                                vjoints = voutput[idx][side]['joints3d']*vmeta[side+'_mano_length'][:,None,None]
                            else:
                                vjoints = voutput[idx][side]['joints3d']
                            voutput[idx][side]['T_joints3d'][vmeta[side+'_joints'] == 1] = vjoints[vmeta[side+'_joints'] == 1] + vgt[side+'_joints'][vmeta[side+'_joints'] == 1][:,None,9]
                        voutput[idx][side]['joints2d'] = project_results(voutput[idx][side]['T_joints3d'], val_data.cam_f, val_data.cam_c, vmeta['bbox'].cuda())
                ######### ################################################ ###########
                sep_vloss = loss_fn(voutput[-1], vgt, vmeta)
                vloss = sum(sep_vloss[k] for k in sep_vloss).item()
                running_vloss += vloss
                vepoch.set_description(f"Validation Batch [{i}/{len(val_loader)}]")
                vepoch.set_postfix({'Loss' : vloss})
                i+=1
    
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        trigger_times += 1
    
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
    
        writer.flush()
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            state = {'epoch': epoch, 'network': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_model(state, epoch, args.output_folder)
            print('\n\t---> Model saved!\n')
            trigger_times = 0
        
        # Check early stopping criterion
        if trigger_times>patience:
            print('---> Early Stopping!')
            break
    
    if args.log:
        sys_log.close()

    return


if __name__ == '__main__':

    main()