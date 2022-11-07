
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

########### Two-hand loss #############

class JointOffsetLoss(nn.Module):
    def __init__(self):
        super(JointOffsetLoss, self).__init__()

    def forward(self, l_joints_out, r_joints_out, l_joints_gt, r_joints_gt):
        #pdist = nn.PairwiseDistance(p=2) # L2 distance
        loss_fn = torch.nn.MSELoss(reduction='none')
        # check shape of the inputs

        ####
        out = r_joints_out - l_joints_out
        gt = r_joints_gt - l_joints_gt
        if l_joints_out.shape[-1] == l_joints_gt.shape[-1] == 3:
            loss = torch.sum(loss_fn(out.swapaxes(-1,-2), gt.swapaxes(-1,-2)), axis=1)
        else:
            loss = torch.sum(loss_fn(out, gt), axis=1)
        return loss

class ShapeConsistencyLoss(nn.Module):
    def __init__(self):
        super(ShapeConsistencyLoss, self).__init__()

    def forward(self, l_shape, r_shape):
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss = torch.sum(loss_fn(l_shape, r_shape), axis=-1)#.sum(-1)
        #pdist = nn.PairwiseDistance(p=2) # L2 distance
        #loss = pdist(r_shape, l_shape)
        return loss

########## Single-hand loss ###########

class JointLoss(nn.Module):
    def __init__(self, loss='l2'):
        super(JointLoss, self).__init__()
        if loss == 'l2':
            self.loss_fn = torch.nn.MSELoss(reduction='none')
        elif loss == 'l1':
            self.loss_fn = torch.nn.L1Loss(reduction='none')

    def forward(self, joint_out, joint_gt):
        loss = torch.sum(self.loss_fn(joint_out, joint_gt), axis=[-1, -2])#.sum(-1)
        return loss

class BoneLengthLoss(nn.Module):
    def __init__(self):
        super(BoneLengthLoss, self).__init__()

    def forward(self, joint_out, joint_gt, skeleton):
        loss_fn = torch.nn.MSELoss(reduction='none')
        bones_out = torch.zeros([joint_gt.shape[0], len(skeleton)])
        bones_gt = torch.zeros([joint_gt.shape[0], len(skeleton)])
        for i, bone in enumerate(skeleton):
            bones_out[:,i] = torch.sqrt(torch.sum((joint_out[:,bone[0]] - joint_out[:,bone[1]])**2, axis=-1))
            bones_gt[:,i] = torch.sqrt(torch.sum((joint_gt[:,bone[0]] - joint_gt[:,bone[1]])**2, axis=-1))
            pass
        loss = torch.sum(loss_fn(bones_out, bones_gt), axis=[-1])#.sum(-1)
        #loss = torch.sum(torch.abs(joint_out - joint_gt), axis=[-1, -2])#.sum(-1)
        return loss

class ShapeLoss(nn.Module):
    def __init__(self):
        super(ShapeLoss, self).__init__()

    def forward(self, shape_out, shape_gt):
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss = torch.sum(loss_fn(shape_out, shape_gt), axis=-1)#.sum(-1)
        #pdist = nn.PairwiseDistance(p=2)
        #loss = pdist(shape_out, shape_gt)
        return loss

class MANORegLoss(nn.Module):
    def __init__(self, lmbda=0.1):
        super(MANORegLoss, self).__init__()
        self.lmbda = lmbda

    def forward(self, pose, shape):
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss = self.lmbda * torch.sum(loss_fn(shape, torch.zeros_like(shape)), axis=-1) + torch.sum(loss_fn(pose, torch.zeros_like(pose)), axis=-1)
        #pdist = nn.PairwiseDistance(p=2)
        #loss = self.lmbda * pdist(shape,torch.zeros_like(shape)) + pdist(pose,torch.zeros_like(pose))
        return loss

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, out, gt):
        loss_fn = torch.nn.L1Loss(reduction='none')
        loss = loss_fn(out, gt)
        if len(loss.shape) > 1:
            loss = torch.sum(loss, axis=-1)
        return loss

class L2RegLoss(nn.Module):
    def __init__(self, lmbda=1):
        super(L2RegLoss, self).__init__()
        self.lmbda = lmbda
    
    def forward(self, inpt):
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss = self.lmbda * loss_fn(inpt, torch.zeros_like(inpt))
        return loss


########## Hand Rotation loss ###########

class GlobalRotLoss(nn.Module):
    def __init__(self):
        super(GlobalRotLoss, self).__init__()

    def forward(self, joints, gt):
        j = torch.clone(joints[:,9] - joints[:,0])
        g = torch.clone(gt[:,9] - gt[:,0])
        dot = (j*g).sum(-1)#Batched dot prod
        j_norm = (j*j).sum(-1).pow(0.5)
        g_norm = (g*g).sum(-1).pow(0.5)
        loss = 1 - (dot/(j_norm*g_norm)) # cos similarity
        return loss

class RootLoss(nn.Module):
    def __init__(self, loss='l1'):
        super(RootLoss, self).__init__()
        if loss == 'l2':
            self.loss_fn = torch.nn.MSELoss(reduction='none')
        elif loss == 'l1':
            self.loss_fn = torch.nn.L1Loss(reduction='none')

    def forward(self, out, gt):
        loss = self.loss_fn(out, gt)
        if len(loss.shape) > 1:
            loss = torch.sum(loss, axis=-1)
        return loss

## Weakly supervised loss

class InterShapeLoss(nn.Module):

    skeleton = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

    def __init__(self, l_offset=1, 
                    l_consistency=0.1, 
                    l_joint=100,
                    #l_joint2d=100, 
                    l_length=100,
                    l_shape=1,#0.1,
                    l_reg=0.1,
                    l_center=10,
                    #l_depth=100,
                    l_scale_reg=1,
                    l_g_rot=100,
                    absolute=False,
                    root=None):
        super(InterShapeLoss, self).__init__()

        self.absolute = absolute
        self.root = root

        if self.absolute:
            self.pose = 'T_joints3d'
            self.gt_pose = '_joints'
            print('-----> Supervising Absolute Pose')
        else:
            self.pose = 'joints3d'
            self.gt_pose = '_mano_joints'

        self.l_offset = l_offset 
        self.l_consistency = l_consistency
        self.l_joint = l_joint
        #self.l_joint2d = l_joint2d
        self.l_length = l_length
        self.l_shape = l_shape
        self.l_reg = l_reg
        self.l_center = l_center
        #self.l_depth = l_depth
        self.l_scale_reg = l_scale_reg

        self.l_g_rot = l_g_rot

        self.offset = JointOffsetLoss()
        self.consistency = ShapeConsistencyLoss()
        self.joint = JointLoss(loss = 'l1')
        self.joint2d = JointLoss(loss = 'l1')
        self.length = BoneLengthLoss()
        self.shape = ShapeLoss()
        ###########################
        self.center = RootLoss(loss = 'l2')
        ###########################
        ###########################
        #self.depth = L1()
        #self.trans = ShapeLoss()
        ###########################
        self.reg = MANORegLoss()
        self.param_reg = L2RegLoss()

        self.g_rot = GlobalRotLoss()

    def forward(self, result, gt, meta):#, mode='bbox', reproj=False):#, side):
        loss = 0
        #### Add boolean disclaimer for hand side to speed up computation and get rid of some if statements
        loss = {}
        ## Single-hand losses
        for side in ['l', 'r']:
            
                # Root Loss
            if self.root is not None:
                if 'center' not in loss.keys():
                    #gt_center = torch.cat([gt[side+'_bbox'][meta[side+'_joints']==1][:,9][:,:2]/256, gt[side+'_joints'][meta[side+'_joints'] == 1][:,9][:,-1].unsqueeze(-1)], axis=-1)
                    #loss['center'] = self.l_center * self.center(result[side]['trans'][meta[side+'_joints'] == 1], gt_center).mean()
                    loss['center'] = self.l_center * self.center(result[side]['trans'][meta[side+'_joints'] == 1], gt[side+'_joints'][meta[side+'_joints'] == 1][:,9]).mean()
                else:
                    #gt_center = torch.cat([gt[side+'_bbox'][meta[side+'_joints']==1][:,9][:,:2]/256, gt[side+'_joints'][meta[side+'_joints'] == 1][:,9][:,-1].unsqueeze(-1)], axis=-1)
                    #loss['center'] += self.l_center * self.center(result[side]['trans'][meta[side+'_joints'] == 1], gt_center).mean()
                    loss['center'] += self.l_center * self.center(result[side]['trans'][meta[side+'_joints'] == 1], gt[side+'_joints'][meta[side+'_joints'] == 1][:,9]).mean()
                if self.root == 'only':
                    continue

            # Joint loss

            if side+'_mano_joints' in gt.keys() and side in result.keys():
            #if meta[side+'_joints'] == 1 and side in result.keys():
                if 'l1_joints' not in loss.keys():
                    loss['l2_joints'] = self.l_joint * self.joint(result[side][self.pose][meta[side+'_joints'] == 1], gt[side+self.gt_pose][meta[side+'_joints'] == 1]).mean()
                else:
                    loss['l2_joints'] += self.l_joint * self.joint(result[side][self.pose][meta[side+'_joints'] == 1], gt[side+self.gt_pose][meta[side+'_joints'] == 1]).mean()

                #if 'joints_cent' not in loss.keys():
                #    loss['joints_cent'] = self.l_joint * self.joint(result[side]['joints3d'], gt[side+'_joints_norm']).mean()
                #else:
                #    loss['joints_cent'] += self.l_joint * self.joint(result[side]['joints3d'], gt[side+'_joints_norm']).mean()

                ###########################################################################
                ####################### CENTER REGRESSION #################################
                ###########################################################################

                #if self.absolute:
                #    if 'center' not in loss.keys():
                #        #loss['trans'] = self.l_trans * self.consistency(result[side]['trans'][meta[side+'_joints'] == 1], gt[side+'_joints'][meta[side+'_joints'] == 1][:,9]).mean()
                #        loss['center'] = self.l_trans * self.depth(result[side]['trans'][meta[side+'_joints'] == 1][:,:2], gt[side+'_joints'][meta[side+'_joints'] == 1][:,9][:,:2]).mean()
                #        #loss['trans_reg'] = self.l_scale_reg * self.param_reg(result[side]['trans']).mean()
                #    else:
                #        #loss['trans'] += self.l_trans * self.consistency(result[side]['trans'][meta[side+'_joints'] == 1], gt[side+'_joints'][meta[side+'_joints'] == 1][:,9]).mean()
                #        loss['center'] += self.l_trans * self.depth(result[side]['trans'][meta[side+'_joints'] == 1][:,:2], gt[side+'_joints'][meta[side+'_joints'] == 1][:,9][:,:2]).mean()
                #        #loss['trans_reg'] += self.l_scale_reg * self.param_reg(result[side]['trans']).mean()
                #
                #    if 'depth' not in loss.keys():
                #        loss['depth'] = self.l_depth * self.depth(result[side]['trans'][meta[side+'_joints'] == 1][:,-1], gt[side+'_joints'][meta[side+'_joints'] == 1][:,9,-1]).mean()
                #    else:
                #        loss['depth'] += self.l_depth * self.depth(result[side]['trans'][meta[side+'_joints'] == 1][:,-1], gt[side+'_joints'][meta[side+'_joints'] == 1][:,9,-1]).mean()

                if 'global_rot' not in loss.keys():
                    loss['global_rot'] = self.l_g_rot * (self.g_rot(result[side]['joints3d'][meta[side+'_joints'] == 1], gt[side+'_mano_joints'][meta[side+'_joints'] == 1]).mean())# + 1e-8)
                    #if torch.isnan(loss['global_rot']):
                    #    print(self.g_rot(result[side]['joints3d'], gt[side+'_mano_joints']).mean()) 
                else:
                    loss['global_rot'] += self.l_g_rot * (self.g_rot(result[side]['joints3d'][meta[side+'_joints'] == 1], gt[side+'_mano_joints'][meta[side+'_joints'] == 1]).mean())# + 1e-8)
                    #if torch.isnan(loss['global_rot']):
                    #    print(self.g_rot(result[side]['joints3d'], gt[side+'_mano_joints']).mean()) 

                #if reproj:
                #    if 'joints2d' in result[side].keys():
                #        if 'joints2d' not in loss.keys():
                #            loss['joints2d'] = self.joint(result[side]['joints2d'][meta[side+'_joints'] == 1][:,:,:2], gt[side+'_bbox'][meta[side+'_joints'] == 1][:,:,:2]/256).mean()
                #        if 'joints2d' not in loss.keys():
                #            loss['joints2d'] += self.joint(result[side]['joints2d'][meta[side+'_joints'] == 1][:,:,:2], gt[side+'_bbox'][meta[side+'_joints'] == 1][:,:,:2]/256).mean()

                #if 'scale' not in loss.keys():
                #    loss['scale'] = self.l_scale *torch.abs(result[side]['scale'] - meta[side+'_joint_length']).mean()
                #    loss['scale_reg'] = self.l_scale_reg * self.param_reg(result[side]['scale']).mean()
                #else:
                #    loss['scale'] += self.l_scale *torch.abs(result[side]['scale'] - meta[side+'_joint_length']).mean()
                #    loss['scale_reg'] += self.l_scale_reg * self.param_reg(result[side]['scale']).mean()

                ## joints 2d loss
                #if 'joints2d' not in loss.keys():
                #    loss['joints2d'] = self.l_joint2d * self.joint2d(result[side]['jointsimg']/256, gt[side+'_'+mode][:,:,:2]/256).mean()
                #else:
                #    loss['joints2d'] += self.l_joint2d * self.joint2d(result[side]['jointsimg']/256, gt[side+'_'+mode][:,:,:2]/256).mean()

                # Bone length loss
                if 'bone_len' not in loss.keys():
                    loss['bone_len'] = self.l_length * self.length(result[side]['joints3d'][meta[side+'_joints'] == 1], gt[side+'_mano_joints'][meta[side+'_joints'] == 1], self.skeleton).mean()
                else:
                    loss['bone_len'] += self.l_length * self.length(result[side]['joints3d'][meta[side+'_joints'] == 1], gt[side+'_mano_joints'][meta[side+'_joints'] == 1], self.skeleton).mean()
                #loss += self.l_joint * self.joint(result[side+'_joints']['joints3d'], gt[side+'_mano_joints'])#.mean()

            # Shape/Pose reg
            if side+'_shape' in gt.keys() and side in result.keys():
            #if meta[side+'_mano'] == 1 and side in result.keys():
                if 'shape' not in loss.keys():
                    loss['shape'] = self.l_shape * self.consistency(result[side]['shape'][meta[side+'_mano'] == 1], gt[side+'_shape'][meta[side+'_mano'] == 1]).mean()
                else:
                    loss['shape'] += self.l_shape * self.consistency(result[side]['shape'][meta[side+'_mano'] == 1], gt[side+'_shape'][meta[side+'_mano'] == 1]).mean()
                #loss += self.l_shape * self.consistency(result[side+'_shape'], gt[side+'_shape'])#.mean()

                # Mano regularizers
                #if side+'_pose' in gt.keys() and side in result.keys():
                if 'MANO_reg' not in loss.keys():
                    loss['MANO_reg'] = self.l_reg * self.reg(result[side]['pose'][meta[side+'_mano'] == 1], result[side]['shape'][meta[side+'_mano'] == 1]).mean()
                else:    
                    loss['MANO_reg'] += self.l_reg * self.reg(result[side]['pose'][meta[side+'_mano'] == 1], result[side]['shape'][meta[side+'_mano'] == 1]).mean()
                #loss += self.l_reg * self.reg(result[side+'_pose'], result[side+'_shape'])#.mean()

        if self.root == 'only':
            return loss

        # Two-hand losses
        # Shape Consistency
        if 'l' in result.keys() and 'r' in result.keys():
            loss['two_shape'] = self.l_consistency * self.consistency(result['l']['shape'], result['r']['shape']).mean()
            #loss += self.l_consistency * self.consistency(result['l_shape'], result['r_shape'])#.mean()
            loss['offset'] = self.l_offset * self.offset(result['l']['T_joints3d'], result['r']['T_joints3d'], gt['l_joints'], gt['r_joints']).mean(1).mean()


        #if ('l' in result.keys() and 'r' in result.keys()):# and ('l_shape' in gt.keys() and 'r_shape' in gt.keys()): #### To BE FIXED
        #    loss['offset'] = self.l_offset * self.offset(result['l']['joints3d'], result['r']['joints3d'], gt['l_mano_joints'], gt['r_mano_joints']).mean(1).mean()
            #loss += self.l_offset * self.offset(result['l_joints']['joints3d'], result['r_joints']['joints3d'], gt['l_mano_joints'], gt['r_mano_joints']).mean(1)

        #loss['scale2d_reg'] = self.l_scale_reg * torch.abs(result['scale_2d']).mean()#self.scale_reg(result['scale_2d']).mean()
        
        return loss



# torch.sum(torch.sqrt(torch.sum((result[side+'_joints']['joints3d'] - gt[side+'_joints']) ** 2, axis=-1)), axis=-1)

# torch.sum((result[side+'_joints']['joints3d'] - gt[side+'_joints']) ** 2, axis=-1).sum(-1)

# torch.sum(F.mse_loss(result[side+'_joints']['joints3d'], gt[side+'_joints'], reduction = 'none'), axis=-1).sum(-1)