
import torch
import sys, os
sys.path.append('~/hand_pose/my_intershape/')

from .transforms import cam2pixel, pixel2cam


def render_gaussian_heatmap(joint_coord):
    x = torch.arange(64)#cfg.output_hm_shape[2])
    y = torch.arange(64)#cfg.output_hm_shape[1])
    z = torch.arange(64)#cfg.output_hm_shape[0])
    zz,yy,xx = torch.meshgrid(z,y,x)
    xx = xx[None,None,:,:,:].cuda().float(); yy = yy[None,None,:,:,:].cuda().float(); zz = zz[None,None,:,:,:].cuda().float()
    
    x = joint_coord[:,:,0,None,None,None]; y = joint_coord[:,:,1,None,None,None]; z = joint_coord[:,:,2,None,None,None]
    heatmap = torch.exp(-(((xx-x)/2.5)**2)/2 -(((yy-y)/2.5)**2)/2 - (((zz-z)/2.5)**2)/2)
    heatmap = heatmap * 255
    return heatmap

def render_gaussian_heatmap_1d(joint_coord):
    zz = torch.arange(64)
    zz = zz[None,None,:].cuda().float()
    z = (joint_coord[:,None,9,2,None]-joint_coord[:,None,21+9,2,None])
    heatmap_1d = torch.exp(-(((zz-z)/2.5)**2)/2)
    heatmap_1d *= 255
    return heatmap_1d.squeeze(0)


def get_att_map(heatmap, side=None):
    if side == 'updown':
        heatmap = heatmap.permute(0,1,3,2,4)
    elif side == 'side':
        heatmap = heatmap.permute(0,1,4,3,2)
    attention_map=heatmap.reshape(heatmap.shape[0],-1,heatmap.shape[3],heatmap.shape[4])
    right_attention_map,_=attention_map[:,:(21*64),:,:].max(dim=1,keepdim=True)
    left_attention_map,_=attention_map[:,(21*64):,:,:].max(dim=1,keepdim=True)
    attention=torch.cat([right_attention_map,left_attention_map],dim=1)
    return attention


def compute_k_value(area_img, area_real, f):
    k_value = torch.sqrt(area_real*f[0]*f[1]/area_img)
    #k_value = torch.tensor(torch.sqrt(area_img*f[0]*f[1]/area_real), dtype=torch.float32)
    return k_value/1000


def project_results(joints, cam_f, cam_c, bbox):
    proj_joints = cam2pixel(joints, cam_f.unsqueeze(0), cam_c.unsqueeze(0)).squeeze()
    # account for bbox
    proj_joints[:,:,0] = (proj_joints[:,:,0] - bbox[:,None,0])/bbox[:,None,2]
    proj_joints[:,:,1] = (proj_joints[:,:,1] - bbox[:,None,1])/bbox[:,None,3]
    return proj_joints

def project_back(joints, cam_f, cam_c, bbox):
    img_joints = joints.clone()
    img_joints[:,:,0] = (img_joints[:,:,0] * bbox[:,None,2] + bbox[:,None,0])# / bbox[:,None,2]
    img_joints[:,:,1] = (img_joints[:,:,1] * bbox[:,None,3] + bbox[:,None,1])# / bbox[:,None,3]
    reproj_joints = pixel2cam(img_joints, cam_f.unsqueeze(0), cam_c.unsqueeze(0)).squeeze()
    return reproj_joints
