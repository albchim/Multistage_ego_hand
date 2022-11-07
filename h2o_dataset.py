
#%%
import os.path as osp
import numpy as np
import cv2
from torch.utils.data import Dataset
from manopth.manolayer import ManoLayer

import torch
import joblib

import time

from my_utils.transforms import cam2pixel, compute_rel_trans
from my_utils.preprocessing import get_bbox, process_bbox, generate_patch_image, compute_hand_area


# Dataset class
class H2ODataset(Dataset):

    N_JOINTS = 21
    N_POSE_TRANS = 3
    N_POSE = 48
    N_SHAPE = 10
    skeleton = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

    def __init__(self, h2o_folder, f_list, device='cpu', input_size=256, n_jobs=10, mode='full', cache=None, rescale_pose=False):
        self.input_size = input_size
        self.device = device
        self.mode = mode # set to 'full', 'bbox', 'double' bboxes
        self.rescale = rescale_pose
        self.mano_right = ManoLayer(center_idx=9, side='right', mano_root='mano/models', use_pca=False,
                             root_rot_mode='axisang', flat_hand_mean=True).to(device)
        self.mano_left = ManoLayer(center_idx=9, side='left', mano_root='mano/models', use_pca=False,
                             root_rot_mode='axisang', flat_hand_mean=True).to(device)


        self.rgb_path = np.asarray([osp.join(h2o_folder, file) for file in f_list])

        # Load cam intrinsics
        cam_int = osp.join(osp.dirname(osp.dirname(self.rgb_path[0])), 'cam_intrinsics.txt')
        cam_int = torch.tensor(np.loadtxt(cam_int),device=self.device,dtype=torch.float32).float()
        self.cam_f = cam_int[:2]; self.cam_c = cam_int[2:4]; self.orig_size = cam_int[4:]

        parallel = joblib.Parallel(n_jobs=n_jobs, prefer="threads")

        print('Loading training data...')

        if cache is not None and osp.exists(osp.join(h2o_folder, cache+'_annot_cache.pth')):
            print('Found cached data...')
            tic = time.time()
            self.gt, self.metadata = torch.load(osp.join(h2o_folder, cache+'_annot_cache.pth'))
            toc = time.time()
            print('loaded data from cache file in {}'.format(toc-tic))

        else:
            tic = time.time()
            # Parallelize annotations loading
            hp = torch.tensor(parallel(
                        self.__load_annot_from_framepath__(self.rgb_path[idx], 'hand_pose')
                        for idx in range(len(self.rgb_path))
                        ), device=self.device,dtype=torch.float32)
            hp_mano = torch.tensor(parallel(
                        self.__load_annot_from_framepath__(self.rgb_path[idx], 'hand_pose_mano')
                        for idx in range(len(self.rgb_path))
                        ), device=self.device,dtype=torch.float32)
            toc = time.time()
            print('loaded data in {}'.format(toc-tic))

            # transform annotations to dictionary labels
            annot = np.asarray(parallel(
                        self.__pose_to_dict__(hp[idx], hp_mano[idx])
                        for idx in range(len(hp))
                            ))

            self.gt = annot[:,0]
            self.metadata = annot[:,1]

            tic = time.time()
            print('transform into dict in {}'.format(tic-toc))

            ## compute relative translation
            #parallel(self.__get_rel_trans(idx)
            #            for idx in range(len(self.gt)))

            # compute MANO joint projections
            parallel(self.__get_MANO_proj(idx)
                        for idx in range(len(self.gt)))

            ## normalize MANO joint projections by [0,9] bone
            #parallel(self.__normalize_MANO(idx)
            #            for idx in range(len(self.gt)))

            # project annot to image space
            parallel(self.__cam2pix(idx)
                        for idx in range(len(self.gt)))
            
            toc = time.time()
            print('projected to img in {}'.format(toc-tic))
            
            # get bboxes from annotations
            parallel(self.__get_bboxes(idx)
                        for idx in range(len(self.gt)))

            # get bboxes from annotations
            parallel(self.__get_hand_areas(idx)
                        for idx in range(len(self.gt)))
            
            tic = time.time()
            print('got bbox coords in {}'.format(tic-toc))############### Longest
            
            # project annot to image space
            parallel(self.__pix2bbox(idx)
                        for idx in range(len(self.gt)))
            
            toc = time.time()
            print('projected to bbox in {}'.format(toc-tic))################ Second longest
            
            if cache is not None:
                print('Saving to cache file...')
                torch.save(np.stack([self.gt, self.metadata]), osp.join(h2o_folder, cache+'_annot_cache.pth'))


    @joblib.delayed
    def __load_annot_from_framepath__(self, fpath, annot_name):
        '''
        Creates file lists for img and annotations  
        '''
        annot = osp.splitext(fpath)[0] + '.txt'

        annot = annot.replace('rgb', annot_name)
        annot = np.loadtxt(annot)
        return annot

    @joblib.delayed
    def __pose_to_dict__(self, hp, hp_mano):
        meta = {'l_joints' : hp[0], 'r_joints' : hp[64], 'l_mano' : hp_mano[0], 'r_mano' : hp_mano[62]}
        
        gt = {}
        shift = {'l': 0, 'r' : 62}
        joints_shift = {'l': 0, 'r' : 64}
        for side in ['l', 'r']:
            if meta[side+'_joints'] == 1:
                gt[side+'_joints'] = hp[1+joints_shift[side]:64+joints_shift[side]].reshape(-1,3)
            else:
                gt[side+'_joints'] = torch.zeros_like(hp[1:64].reshape(-1,3))
            if meta[side+'_mano'] == 1:
                gt[side+'_pose_trans'] = hp_mano[1+shift[side]:1+3+shift[side]].reshape(self.N_POSE_TRANS)
                gt[side+'_pose'] = hp_mano[1+3+shift[side]:52+shift[side]].reshape(self.N_POSE)
                gt[side+'_shape'] = hp_mano[52+shift[side]:62+shift[side]].reshape(self.N_SHAPE)
            else:
                gt[side+'_pose_trans'] = torch.zeros_like(hp_mano[1+shift[side]:1+3+shift[side]].reshape(self.N_POSE_TRANS))
                gt[side+'_pose'] = torch.zeros_like(hp_mano[1+3+shift[side]:52+shift[side]].reshape(self.N_POSE))
                gt[side+'_shape'] = torch.zeros_like(hp_mano[52+shift[side]:62+shift[side]].reshape(self.N_SHAPE))
        return gt, meta

    #@joblib.delayed
    #def __get_rel_trans(self, idx):
    #    self.metadata[idx]['l_joint_length'] = (self.gt[idx]['l_joints'][9]-self.gt[idx]['l_joints'][0]).norm()
    #    self.metadata[idx]['r_joint_length'] = (self.gt[idx]['r_joints'][9]-self.gt[idx]['r_joints'][0]).norm()
    #    if self.metadata[idx]['r_joint_length'] != 0. :
    #        self.gt[idx]['r_joints_norm'] = (self.gt[idx]['r_joints'] - self.gt[idx]['r_joints'][9])/self.metadata[idx]['r_joint_length']
    #    else:
    #        self.gt[idx]['r_joints_norm'] = (self.gt[idx]['r_joints'] - self.gt[idx]['r_joints'][9])
    #    #if self.metadata[idx]['l_joint_length'] != 0. :
    #    #    self.gt[idx]['l_joints_norm'] = (self.gt[idx]['l_joints'] - self.gt[idx]['r_joints'][9])/self.metadata[idx]['l_joint_length']
    #    #else:
    #    #    self.gt[idx]['l_joints_norm'] = (self.gt[idx]['l_joints'] - self.gt[idx]['r_joints'][9])
    #    #self.gt[idx]['rel_trans'] = self.gt[idx]['l_joints_norm'][9] - self.gt[idx]['r_joints_norm'][9]
    #    if self.metadata[idx]['l_joint_length'] != 0. :
    #        self.gt[idx]['l_joints_norm'] = (self.gt[idx]['l_joints'] - self.gt[idx]['l_joints'][9])/self.metadata[idx]['l_joint_length']
    #    else:
    #        self.gt[idx]['l_joints_norm'] = (self.gt[idx]['l_joints'] - self.gt[idx]['l_joints'][9])
    #    #self.gt[idx]['rel_trans'] = self.gt[idx]['l_joints_norm'][9] - self.gt[idx]['r_joints_norm'][9]
    #
    #    return

    @joblib.delayed
    def __get_MANO_proj(self, idx):
        if self.metadata[idx]['l_mano'] == 1:
            _, self.gt[idx]['l_mano_joints'], _ = self.mano_left(self.gt[idx]['l_pose'].unsqueeze(0), th_betas=self.gt[idx]['l_shape'].unsqueeze(0))
            self.gt[idx]['l_mano_joints'] = self.gt[idx]['l_mano_joints'].squeeze()/1000
        else:
            self.gt[idx]['l_mano_joints'] = torch.zeros_like(self.gt[idx]['l_joints'])
        if self.metadata[idx]['r_mano'] == 1:
            _, self.gt[idx]['r_mano_joints'], _ = self.mano_right(self.gt[idx]['r_pose'].unsqueeze(0), th_betas=self.gt[idx]['r_shape'].unsqueeze(0))
            self.gt[idx]['r_mano_joints'] = self.gt[idx]['r_mano_joints'].squeeze()/1000
        else:
            self.gt[idx]['r_mano_joints'] = torch.zeros_like(self.gt[idx]['r_joints'])
        return

    #@joblib.delayed
    def __normalize_MANO(self, idx):
        if 'l_mano_length' not in self.metadata[idx].keys():
            self.metadata[idx]['l_mano_length'] = (self.gt[idx]['l_mano_joints'][9]-self.gt[idx]['l_mano_joints'][0]).norm()
            if 'l_rescaled' not in self.metadata[idx].keys():
                self.gt[idx]['l_mano_joints'] /= self.metadata[idx]['l_mano_length']
                self.metadata[idx]['l_rescaled'] = True
            #self.gt[idx]['l_mano_joints'] += self.gt[idx]['rel_trans']
        if 'r_mano_length' not in self.metadata[idx].keys():
            self.metadata[idx]['r_mano_length'] = (self.gt[idx]['r_mano_joints'][9]-self.gt[idx]['r_mano_joints'][0]).norm()
            if 'r_rescaled' not in self.metadata[idx].keys():
                self.gt[idx]['r_mano_joints'] /= self.metadata[idx]['r_mano_length']
                self.metadata[idx]['r_rescaled'] = True

        return 

    @joblib.delayed
    def __cam2pix(self, idx):
        for side in ['l', 'r']:
            #if self.metadata[idx][side+'_joints'] != 1:
            #    print(f"{side} - hand not present in frame {self.rgb_path[idx]}")
            self.gt[idx][side+'_img'] = cam2pixel(self.gt[idx][side+'_joints'], self.cam_f, self.cam_c).squeeze()
            self.gt[idx][side+'_bbox'] = torch.clone(self.gt[idx][side+'_img'])
            self.gt[idx][side+'_img'][:,:2] = self.gt[idx][side+'_img'][:,:2]/self.orig_size*256
            self.gt[idx][side+'_img'][:,2] = self.gt[idx][side+'_img'][:,2]*256 ###################

        return


    @joblib.delayed
    def __get_bboxes(self, idx):
        if self.mode == 'bbox':
            bbox = get_bbox(torch.cat([self.gt[idx]['l_bbox'], self.gt[idx]['r_bbox']]), 
                                    torch.cat([self.metadata[idx]['l_joints'].repeat(self.gt[idx]['l_joints'].shape[0]),
                                                self.metadata[idx]['r_joints'].repeat(self.gt[idx]['r_joints'].shape[0])]))
            self.metadata[idx]['bbox'] = process_bbox(bbox).to(self.device)

        elif self.mode == 'double':
            for side in ['l', 'r']:
                #if self.metadata[idx][side+'_joints'] == 1:
                bbox = get_bbox(self.gt[idx][side+'_bbox'], self.metadata[idx][side+'_joints'].repeat(self.gt[idx][side+'_joints'].shape[0]))
                self.metadata[idx][side+'_bbox'] = process_bbox(bbox).to(self.device)
        return


    @joblib.delayed
    def __get_hand_areas(self, idx):
        #bbox = {}
        #bbox['l'] = get_bbox(self.gt[idx]['l_bbox'], self.metadata[idx]['l_joints'].repeat(self.gt[idx]['l_joints'].shape[0]))
        #bbox['r'] = get_bbox(self.gt[idx]['r_bbox'], self.metadata[idx]['r_joints'].repeat(self.gt[idx]['r_joints'].shape[0]))
        for side in ['l', 'r']:    
            if self.metadata[idx][side+'_mano'] == 1:
                bbox = get_bbox(self.gt[idx][side+'_bbox'], self.metadata[idx][side+'_joints'].repeat(self.gt[idx][side+'_joints'].shape[0]))
                self.metadata[idx][side+'_area_img'] = torch.tensor(bbox[2] * bbox[3], device=self.device, dtype=torch.float32)
                self.metadata[idx][side+'_area_real'] = compute_hand_area(self.gt[idx][side+'_mano_joints'])*1000**2
            else:
                self.metadata[idx][side+'_area_img'] = torch.tensor(1e-10, device=self.device, dtype=torch.float32)#None
                self.metadata[idx][side+'_area_real'] = torch.tensor(1e-10, device=self.device, dtype=torch.float32)#None
        return


    @joblib.delayed
    def __pix2bbox(self, idx):
        if self.mode == 'bbox':
            for side in ['l', 'r']:
                if self.metadata[idx][side+'_joints'] == 1:
                    self.gt[idx][side+'_bbox'][:,0] = (self.gt[idx][side+'_bbox'][:,0] - self.metadata[idx]['bbox'][0])/self.metadata[idx]['bbox'][2]*self.input_size
                    self.gt[idx][side+'_bbox'][:,1] = (self.gt[idx][side+'_bbox'][:,1] - self.metadata[idx]['bbox'][1])/self.metadata[idx]['bbox'][3]*self.input_size
                    self.gt[idx][side+'_bbox'][:,2] *= self.input_size ###################


        elif self.mode == 'double':
            for side in ['l', 'r']:
                if self.metadata[idx][side+'_joints'] == 1:
                    self.gt[idx][side+'_bbox'][:,0] = (self.gt[idx][side+'_bbox'][:,0] - self.metadata[idx][side+'_bbox'][0])/self.metadata[idx][side+'_bbox'][2]*self.input_size/2
                    self.gt[idx][side+'_bbox'][:,1] = (self.gt[idx][side+'_bbox'][:,1] - self.metadata[idx][side+'_bbox'][1])/self.metadata[idx][side+'_bbox'][3]*self.input_size
                    self.gt[idx][side+'_bbox'][:,2] *= self.input_size ###################

                    # translate right joints to stiched img coords
                    if side == 'r':
                        self.gt[idx]['r_bbox'][:,0] += 256/2
        return
    

    def __len__(self):
        return len(self.rgb_path)

    def __getitem__(self, idx):
        orig_img = cv2.imread(self.rgb_path[idx])
        self.metadata[idx]['ratio'] = 1/self.orig_size*self.input_size
        #self.metadata[idx]['ratio'] = self.input_size/max(*orig_img.shape[:2])

        if self.rescale:
            self.__normalize_MANO(idx)

        # Process and resize input image
        if self.mode == 'full':
            #M=np.array([[self.metadata[idx]['ratio'],0,0],[0,self.metadata[idx]['ratio'],0]],dtype=np.float32)
            M=np.array([[self.metadata[idx]['ratio'][0],0,0],[0,self.metadata[idx]['ratio'][1],0]],dtype=np.float32)
            img = orig_img
            img = cv2.warpAffine(orig_img,M,(self.input_size,self.input_size),flags=cv2.INTER_LINEAR,borderValue=[0,0,0])
        
        elif self.mode == 'bbox':
            img, trans, inv_trans = generate_patch_image(orig_img, self.metadata[idx]['bbox'], do_flip=False, 
                                    scale=1, rot=0, out_shape=[self.input_size, self.input_size])
            
            self.metadata[idx]['trans'] = trans
            self.metadata[idx]['inv_trans'] = inv_trans

        elif self.mode == 'double':
            l_img_patch, trans, inv_trans = generate_patch_image(orig_img, self.metadata[idx]['l_bbox'], do_flip=False, 
                                    scale=1, rot=0, out_shape=[self.input_size, self.input_size/2])
            r_img_patch, trans, inv_trans = generate_patch_image(orig_img, self.metadata[idx]['r_bbox'], do_flip=False, 
                                    scale=1, rot=0, out_shape=[self.input_size, self.input_size/2])
            img = cv2.hconcat([l_img_patch, r_img_patch])

        img=img[:,:,::-1].astype(np.float32)/255-0.5
        img=torch.tensor(img.transpose(2,0,1),device=self.device,dtype=torch.float32)#.float()

        return img, self.gt[idx], self.metadata[idx]



def main():

    from my_utils.visualization import plot_sample, plot_3d_skeleton, plot_2d_skeleton
    import matplotlib.pyplot as plt

    f_list = np.loadtxt('h2o_trainval/pose_train.txt', dtype='object')

    data = H2ODataset('/media/data/alberto/h2o_dataset_updated', f_list[10000:10100:10], n_jobs=10, mode = 'bbox')#, cache='test')

    img, gt, meta = data[0]

    plot_sample(img, torch.cat([gt['l_bbox'], gt['r_bbox']]))#, ratio=meta['ratio'])
    mano_joints = torch.cat([gt['l_mano_joints'], gt['r_mano_joints']])/1000

    print('Camera center:', data.cam_c)
    print('\n\nTrans:', meta['trans'], '\n\n')
    print('areas:', meta['l_area_img'], meta['l_area_real'])

    plt.show()
    ax1 = plot_3d_skeleton(mano_joints, data.skeleton, get_axes=True)
    ax1.scatter3D(mano_joints[21+9,0], mano_joints[21+9,1], mano_joints[21+9,2], color='red', marker='^')
    plt.show()
    #plot_2d_skeleton(img_joints, data.skeleton)
    plt.show()
    plot_2d_skeleton(torch.cat([gt['l_img'], gt['r_img']])/256, data.skeleton)
    plt.show()

    ############################################
    ax2 = plot_3d_skeleton((gt['l_joints']-gt['l_joints'][9]), data.skeleton, get_axes=True)
    ax2.scatter3D(0,0,0, color='red', marker='^')
    plt.show()
    ax3 = plot_3d_skeleton((gt['r_joints']-gt['r_joints'][9]), data.skeleton, get_axes=True, color='C0')
    ax3.scatter3D(0,0,0, color='red', marker='^')
    plt.show()
    ############################################

    joints_gt = torch.cat([gt['l_bbox'], gt['r_bbox']]).squeeze().detach().cpu()
    plt.scatter(joints_gt[21:,0], joints_gt[21:,1])
    plt.close()
    #print(img_joints[:,:2].detach().cpu().numpy().mean(axis=0))
    print((torch.cat([gt['l_img'], gt['r_img']])[:,:2]/256).detach().cpu().numpy().mean(axis=0))


    ####################################
    # Process Img coords to match provided labels coordinate system
    img_j = torch.cat([gt['l_img'], gt['r_img']])
    img_j -= 128    
    img_j /= 256    
    plot_2d_skeleton(img_j, data.skeleton)
    plt.show()
    print(np.sqrt(torch.sum((img_j[0]-img_j[9])**2))*100)
    #print(np.sqrt(torch.sum((img_joints[0]-img_joints[9])**2))*100)
    ####################################

    ####################################
    real_joints = torch.cat([gt['l_joints'], gt['r_joints']])
    #mano_joints[:21] *= meta['l_joint_length'][None,None]
    #mano_joints[21:] *= meta['r_joint_length'][None,None]
    mano_joints += real_joints[None, 21+9]
    print(torch.any(real_joints.squeeze()-mano_joints.squeeze()>1e-7))

    ####################################


if __name__ == '__main__':
    main()