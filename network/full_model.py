import torch
import torch.nn as nn
from network.meshreg import MeshRegNet
from network.InterHand.module import BackboneNet, PoseNet

class InferenceModel(nn.Module):
    def __init__(self, backbone_net, pose_net):
        super(InferenceModel, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.pose_net = pose_net
          
    def forward(self, inputs):
        input_img = inputs
        batch_size = input_img.shape[0]
        img_feat = self.backbone_net(input_img)
        joint_heatmap_out, rel_root_depth_out, hand_type = self.pose_net(img_feat)
        return joint_heatmap_out,rel_root_depth_out, hand_type

class InterShape(nn.Module):
    def __init__(self,input_size,resnet_version,mano_use_pca,mano_neurons,\
                    cascaded_num,cascaded_input,heatmap_attention,rescale_pose=False, use_heat_xy=False):
        super(InterShape, self).__init__()

        self.depth_downsample_factor=4
        self.spatil_downsample_factor=2
        self.mesh_reg=MeshRegNet(input_size=input_size, resnet_version=resnet_version, mano_use_pca=mano_use_pca, mano_neurons=mano_neurons,\
                                    addition_channels=42*64//self.depth_downsample_factor,\
                                    cascaded_num=cascaded_num, cascaded_input=cascaded_input, rescale_pose=rescale_pose, use_heat_xy=use_heat_xy)
        backbone_net = BackboneNet()
        pose_net = PoseNet(21) #initial heatmap estimation from InerNet
        self.heatmap_predictor = InferenceModel(backbone_net, pose_net)
        self.heatmap_attention = heatmap_attention
    def forward(self,x, k_value=None, intershape=True, gt_heatmap=None):
        heatmap,root_depth_out,hand_type=self.heatmap_predictor(x+0.5) #########
        if intershape:
            downsampled_heatmap=heatmap[:,:,::self.depth_downsample_factor,::self.spatil_downsample_factor,::self.spatil_downsample_factor]
            B,K,D,H,W=downsampled_heatmap.shape
            if self.heatmap_attention:
                if gt_heatmap is not None:
                    heatmap = gt_heatmap
                attention_map=heatmap.reshape(heatmap.shape[0],-1,heatmap.shape[3],heatmap.shape[4])
                right_attention_map,_=attention_map[:,:(21*64),:,:].max(dim=1,keepdim=True)
                left_attention_map,_=attention_map[:,(21*64):,:,:].max(dim=1,keepdim=True)
                attention=torch.cat([right_attention_map,left_attention_map],dim=1)
            else:
                attention=None

            val_z, idx_z = torch.max(heatmap,2)
            val_zy, idx_zy = torch.max(val_z,2)
            val_zyx, joint_x = torch.max(val_zy,2)
            joint_x = joint_x[:,:,None]
            joint_y = torch.gather(idx_zy, 2, joint_x)
            xyc=torch.cat((joint_x, joint_y, val_zyx[:,:,None]),2).float()

            output=self.mesh_reg(x,downsampled_heatmap.reshape(B,K*D,H,W),xyc,attention, k_value)
            return output, xyc, attention
        else:
            return heatmap, root_depth_out