"""
This part reuses code from https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/meshregnet.py
Thanks to Yana Hasson for the excellent work.
"""
import torch
from torch import nn

from network import manobranch
from network import resnet_anydim as resnet
from network import absolutebranch
from manopth.manolayer import ManoLayer

from network.RootNet.model import get_root_net

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, outplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MeshRegNet(nn.Module):
    def __init__(
        self,
        fc_dropout=0,
        resnet_version=50,
        mano_neurons=[512, 512],
        mano_comps=45, ################################################################################### 45
        mano_use_pca=True,
        trans_dim=3,#4
        input_size=3,
        addition_channels=42*64,
        cascaded_num=1,
        cascaded_input='double',
        mano_use_fhm=True,
        rescale_pose=False,
        use_heat_xy=False,
        #scale2d=True, ######################################################################################
    ):
        super(MeshRegNet, self).__init__()
        if int(resnet_version) == 18:
            img_feature_size = 512
            base_net = resnet.resnet18(input_size=input_size,pretrained=False,addition_channel=addition_channels,return_inter=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            base_net = resnet.resnet50(input_size=input_size,pretrained=False,addition_channel=addition_channels,return_inter=True)
        elif int(resnet_version) == 101:
            img_feature_size = 2048
            base_net = resnet.resnet101(input_size=input_size,pretrained=False,addition_channel=addition_channels,return_inter=True)

        self.rescale = rescale_pose
        self.use_heat_xy = use_heat_xy

        mano_base_neurons = [img_feature_size] + mano_neurons
        self.base_net = base_net
        self.mano_layer=nn.ModuleDict({
            'r':ManoLayer(
                    ncomps=mano_comps, # only used if use_pca is set to True
                    center_idx=9,
                    side='right',
                    mano_root='mano/models',
                    use_pca=mano_use_pca,
                    flat_hand_mean=mano_use_fhm,
                    root_rot_mode='axisang',
                ),
            'l':ManoLayer(
                    ncomps=mano_comps, # only used if use_pca is set to True
                    center_idx=9,
                    side='left',
                    mano_root='mano/models',
                    use_pca=mano_use_pca,
                    flat_hand_mean=mano_use_fhm,
                    root_rot_mode='axisang',
                )
        })

        self.mano_layer['l'].th_shapedirs[:,0,:] *= -1
        # Predict left hand
        self.left_mano_branch = manobranch.ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            dropout=fc_dropout,
            mano_pose_coeff=1,
            use_shape=True,
            use_pca=mano_use_pca,
        )
        

        # Predict right hand
        self.right_mano_branch = manobranch.ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            dropout=fc_dropout,
            mano_pose_coeff=1,
            use_shape=True,
            use_pca=mano_use_pca,
        )

        #self.trans_branch = absolutebranch.AbsoluteBranch(
        #    base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=trans_dim
        #)

        ###################################################################################
        #
        #self.center_branch = nn.ModuleDict({
        #    'r' : absolutebranch.AbsoluteBranch(
        #        base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=trans_dim
        #        ),
        #    'l' : absolutebranch.AbsoluteBranch(
        #        base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=trans_dim
        #        ),
        #})
        #
        #self.scale_branch = nn.ModuleDict({
        #    'r' : absolutebranch.AbsoluteBranch(
        #        base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=1
        #        ),
        #    'l' : absolutebranch.AbsoluteBranch(
        #        base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=1
        #        ),
        #})
        ###################################################################################

        ##################################################################################

        ####self.center_branch = nn.ModuleDict({
        ####    'r' : absolutebranch.AbsoluteBranch(
        ####        base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=2
        ####        ),
        ####    'l' : absolutebranch.AbsoluteBranch(
        ####        base_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=2
        ####        ),
        ####})
        ####
        ####self.depth_branch = nn.ModuleDict({
        ####    'r' : RootNet(
        ####        depth_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=1
        ####        ),
        ####    'l' : RootNet(
        ####        depth_neurons=[img_feature_size, int(img_feature_size / 2)], out_dim=1
        ####        ),
        ####})
        ##################################################################################


        ##################################################################################
        ####################################### RootNet ##################################
        self.root_net = nn.ModuleDict({
            'r' : get_root_net(img_feature_size, 256),
            'l' : get_root_net(img_feature_size, 256),
        })
        ##################################################################################


        self.MANO_cascaded=cascaded_num>0
        self.cascaded_num=cascaded_num

        if self.MANO_cascaded:
            self.cascaded_left_mano_branch_list=nn.ModuleList([])
            self.cascaded_right_mano_branch_list=nn.ModuleList([])
            if cascaded_input=='double':
                mano_base_neurons_channels = [img_feature_size+(10+48)*2+4]+mano_neurons
                #mano_base_neurons_channels = [img_feature_size+(10+48)*2+3]+mano_neurons
                self.cascaded_input_single_hand=2
            elif cascaded_input=='no':
                mano_base_neurons_channels = [img_feature_size]+mano_neurons
                self.cascaded_input_single_hand=3
            else:
                mano_base_neurons_channels = [img_feature_size+10+48+4]+mano_neurons
                #mano_base_neurons_channels = [img_feature_size+10+48+3]+mano_neurons
                self.cascaded_input_single_hand=0 if cascaded_input=='single' else 1
            for i in range(cascaded_num):
                self.cascaded_left_mano_branch_list.append(
                    manobranch.ManoBranch(
                        ncomps=mano_comps,
                        base_neurons=mano_base_neurons_channels,
                        dropout=fc_dropout,
                        mano_pose_coeff=1,
                        use_shape=True,
                        use_pca=mano_use_pca,
                    ))
                
                self.cascaded_right_mano_branch_list.append(
                    manobranch.ManoBranch(
                        ncomps=mano_comps,
                        base_neurons=mano_base_neurons_channels,
                        dropout=fc_dropout,
                        mano_pose_coeff=1,
                        use_shape=True,
                        use_pca=mano_use_pca,
                    ))

            feature_channel=[256,512,1024,2048]
            self.cascaded_left_feature_extractor_list=nn.ModuleList([])
            self.cascaded_right_feature_extractor_list=nn.ModuleList([])
            for i in range(cascaded_num):
                C=feature_channel[-(i+1)]
                input_channel=C
                stride=2**i
                downsample_right = nn.Conv2d(input_channel, feature_channel[-1], kernel_size=1, stride=stride, bias=False)
                downsample_left = nn.Conv2d(input_channel, feature_channel[-1], kernel_size=1, stride=stride, bias=False)
                self.cascaded_right_feature_extractor_list.append(Bottleneck(input_channel,C//2,feature_channel[-1],stride=stride,downsample=downsample_right))
                self.cascaded_left_feature_extractor_list.append(Bottleneck(input_channel,C//2,feature_channel[-1],stride=stride,downsample=downsample_left))

    def generate_batch_heatmap(self,batch_points,image_size,sigma):
        x_value=torch.arange(image_size,device=batch_points.device,dtype=torch.float32)
        x_value=batch_points[:,:,0].unsqueeze(2)-x_value[None,None,:]
        y_value=torch.arange(image_size,device=batch_points.device,dtype=torch.float32)
        y_value=batch_points[:,:,1].unsqueeze(2)-y_value[None,None,:]
        heatmap=torch.exp(-(x_value.unsqueeze(2)**2+y_value.unsqueeze(3)**2)/sigma)
        return heatmap

    def register_heatmap(self,xyc:torch.Tensor,J:torch.Tensor,origin_size:int,output_size:int):
        # xyc is the xy location of the joints in feature space (64x64) and confidence parameter for the joint
        device_run=xyc.device
        batch_size=xyc.shape[0]
        #### Project regressed joints on image space
        M=torch.cat([J[:,:,:2,None],torch.eye(2,device=device_run)[None,None,:,:].repeat(batch_size,J.shape[1],1,1)],dim=-1) # shape [B,21,2,3]
        # xyc[:,:,2,None,None] is the confidence for each joint
        wM=xyc[:,:,2,None,None]*M # shape [B,21,2,3]
        wB=xyc[:,:,2,None,None]*xyc[:,:,:2,None] # shape [B,21,2,1]
        wM=wM.reshape(batch_size,-1,3) # shape [B,42,3]
        wB=wB.reshape(batch_size,-1,1) # shape [B,42,1]
        # batched matrix multiplication
        MTM=torch.bmm(wM.transpose(2,1),wM) # shape [B,3,3]
        MTB=torch.bmm(wM.transpose(2,1),wB) # shape [B,3,1]
        sT=torch.bmm(torch.inverse(MTM),MTB)[:,:,0].detach() # shape [1,3,1] ---> [1,3]
        ratio=output_size/origin_size
        projected_xy=(J[:,:,:2]*sT[:,None,0,None]+sT[:,None,1:])*ratio
        #### Generate heatmap
        heatmap=self.generate_batch_heatmap(projected_xy,output_size,3)
        output_heatmap=1-torch.prod(1-heatmap,dim=1)
        ####
        return output_heatmap

    def projection_2D(self,xyc:torch.Tensor,J:torch.Tensor,origin_size:int,output_size:int, get_weak=False):
        device_run=xyc.device
        batch_size=xyc.shape[0]
        M=torch.cat([J[:,:,:2,None],torch.eye(2,device=device_run)[None,None,:,:].repeat(batch_size,J.shape[1],1,1)],dim=-1)
        wM=xyc[:,:,2,None,None]*M+1e-8
        wB=xyc[:,:,2,None,None]*xyc[:,:,:2,None]+1e-8
        wM=wM.reshape(batch_size,-1,3)
        wB=wB.reshape(batch_size,-1,1)
        MTM=torch.bmm(wM.transpose(2,1),wM)
        MTB=torch.bmm(wM.transpose(2,1),wB)
        sT=torch.bmm(torch.inverse(MTM),MTB)[:,:,0].detach()
        ratio=output_size/origin_size
        projected_xy=(J[:,:,:2]*sT[:,None,0,None]+sT[:,None,1:])*ratio
        #if get_weak:
        #    return torch.cat([projected_xy, J[:,:,2,None]], axis=-1), sT#############################
        #else:
        return torch.cat([projected_xy, J[:,:,2,None]], axis=-1)#, sT#############################

    #def get_abs_rel_trans(self, joints:torch.Tensor):
    #
    #    # translate right hand center
    #    joints = joints - joints[:,21+9]
    #    # normalize over the respective [0,9] lengths
    #    ####l_norm = ((joints[:,:21])[:,9] - (joints[:,:21])[:,0]).norm(dim=1)
    #    r_norm = (joints[:,21+9] - joints[:,21]).norm(dim=1)
    #    joints /= r_norm[:,None,None]
    #    # compute distance components for 9th joints
    #    dist = joints[:,9]
    #    return dist

    def normalize_pose_coords(self, mano_para):
        mano_para['norm_length'] = (mano_para['joints3d'][:,9] - mano_para['joints3d'][:,0]).norm(dim=1)
        mano_para['joints3d'] /= mano_para['norm_length'][:,None,None]
        mano_para['verts3d'] /= mano_para['norm_length'][:,None,None]
        return mano_para

    def apply_rel_transcale(self, mano_para, trans, scale=None):
        if scale is not None:
            mano_para['T_joints3d'] = (mano_para['joints3d'] * scale[:,0,None,None]) + trans[:,:].view(-1,1,3)
            mano_para['T_verts3d'] = (mano_para['verts3d'] * scale[:,0,None,None]) + trans[:,:].view(-1,1,3)
        else:
            mano_para['T_joints3d'] = (mano_para['joints3d']) + trans[:,:].view(-1,1,3)
            mano_para['T_verts3d'] = (mano_para['verts3d']) + trans[:,:].view(-1,1,3)
        return mano_para

    def create_result_dict(self, left_mano_para, right_mano_para, l_trans, r_trans, l_scale=None, r_scale=None):
        result = {}
        result['l'] = left_mano_para
        result['r'] = right_mano_para
        result['l']['trans'] = l_trans
        result['r']['trans'] = r_trans
        if l_scale is not None:
            result['l']['scale'] = l_scale
        if r_scale is not None:
            result['r']['scale'] = r_scale
        return result

    def pose_shape2vertsjoints(self,pose,shape,side):
        verts, joints, T_joints=self.mano_layer[side](pose,th_betas=shape)
        results = {"verts3d": verts/1000, "joints3d": joints/1000, "T_joints3d": T_joints, "shape": shape, "pose": pose}
        return results

    def forward(self, input, heatmap, xyc=None, attention=None, k_value=None):
        #xyc:(Batch,42,3) left,right, x,y,confidence
        features, intermediates = self.base_net(input,heatmap)
        pool = nn.AvgPool2d(2)
        if not attention is None:
            right_attention_map=attention[:,0,None,:,:]
            left_attention_map=attention[:,1,None,:,:]
            right_features=intermediates['res_layer4']*nn.functional.interpolate(right_attention_map,size=[8,8]) # size [1,2048,8,8]
            left_features=intermediates['res_layer4']*nn.functional.interpolate(left_attention_map,size=[8,8])
            right_features_center=pool(right_features) # size [1,2048, 4, 4]
            left_features_center=pool(left_features)
            right_features=right_features.mean([2,3]) # size [1,2048]
            left_features=left_features.mean([2,3])

        else:
            right_attention_map=None
            left_attention_map=None
            right_features=features
            left_features=features

        pose,shape=self.left_mano_branch(left_features)
        left_mano_para=self.pose_shape2vertsjoints(pose,shape,'l')
        pose,shape=self.right_mano_branch(right_features)
        right_mano_para=self.pose_shape2vertsjoints(pose,shape,'r')
        #trans = self.trans_branch(features) #########################################

        ####################################################################
        #left_center = self.center_branch['l'](left_features)
        #right_center = self.center_branch['r'](right_features)
        #left_scale = self.scale_branch['l'](left_features)
        #right_scale = self.scale_branch['r'](right_features)
        ####################################################################
        ###################################################################
        ####left_center = self.center_branch['l'](left_features)
        ####right_center = self.center_branch['r'](right_features)
        ####left_center = torch.cat([ left_center, self.depth_branch['l'](left_features, k_value = k_value['l'])], dim=-1)
        ####right_center = torch.cat([ right_center, self.depth_branch['r'](right_features, k_value = k_value['r'])], dim=-1)
        ###################################################################

        ###################################################################
        ############################ RootNet ##############################
        left_center = self.root_net['l'](left_features_center, k_value = k_value['l'])
        right_center = self.root_net['r'](right_features_center, k_value = k_value['r'])
        ###################################################################


        # Normalize regressed poses
        #left_mano_para = self.normalize_pose_coords(left_mano_para)
        #right_mano_para = self.normalize_pose_coords(right_mano_para)############## Normalize or not?
        ## Apply relatve translation to left hand
        #left_mano_para = self.apply_rel_transcale(left_mano_para, left_center, left_scale)
        #right_mano_para = self.apply_rel_transcale(right_mano_para, right_center, right_scale)

        left_mano_para = self.apply_rel_transcale(left_mano_para, left_center)
        right_mano_para = self.apply_rel_transcale(right_mano_para, right_center)

        if self.rescale:
            left_mano_para = self.normalize_pose_coords(left_mano_para)
            right_mano_para = self.normalize_pose_coords(right_mano_para)

        # Project to image
        left_mano_para['jointsimg'] = self.projection_2D(xyc[:,21:,:],left_mano_para['joints3d'],64,256)
        right_mano_para['jointsimg'] = self.projection_2D(xyc[:,:21,:],right_mano_para['joints3d'],64,256)
        if self.use_heat_xy:
            left_center[:,:2] = left_mano_para['jointsimg'][:,9,:2]/256
            right_center[:,:2] = right_mano_para['jointsimg'][:,9,:2]/256
        #left_mano_para['jointsimg'], left_mano_para['sT'] = self.projection_2D(xyc[:,21:,:],left_mano_para['joints3d'],64,256)
        #right_mano_para['jointsimg'], right_mano_para['sT'] = self.projection_2D(xyc[:,:21,:],right_mano_para['joints3d'],64,256)

        #result = self.create_result_dict(left_mano_para,right_mano_para,left_center, right_center, left_scale, right_scale)
        result = self.create_result_dict(left_mano_para,right_mano_para,left_center, right_center)

        mano_para_list=[result]
        if self.MANO_cascaded:
            for i in range(self.cascaded_num):
                lower_left_mano_para=mano_para_list[-1]['l']
                lower_right_mano_para=mano_para_list[-1]['r']
                #lower_trans=mano_para_list[-1]['trans']
                left_lower_trans = mano_para_list[-1]['l']['trans']
                #left_lower_scale = mano_para_list[-1]['l']['scale']
                right_lower_trans = mano_para_list[-1]['r']['trans']
                #right_lower_scale = mano_para_list[-1]['r']['scale']


                feature_name='res_layer%d'%(4-i)
                left_heatmap=self.register_heatmap(xyc[:,21:,:],lower_left_mano_para['joints3d'],\
                                                    64,intermediates[feature_name].shape[2])
                right_heatmap=self.register_heatmap(xyc[:,:21,:],lower_right_mano_para['joints3d'],\
                                                    64,intermediates[feature_name].shape[2])
                right_features=intermediates[feature_name]*right_heatmap[:,None,:,:]
                left_features=intermediates[feature_name]*left_heatmap[:,None,:,:]
                right_features=self.cascaded_right_feature_extractor_list[i](right_features)
                left_features=self.cascaded_left_feature_extractor_list[i](left_features)
                right_features_center=pool(right_features) # size [1,2048, 4, 4]
                left_features_center=pool(left_features)
                right_features=right_features.mean([2,3])
                left_features=left_features.mean([2,3])

                ####################################################################
                #left_center = self.center_branch['l'](left_features)
                #right_center = self.center_branch['r'](right_features)
                #left_scale = self.scale_branch['l'](left_features)
                #right_scale = self.scale_branch['r'](right_features)
                ####################################################################

                ###################################################################
                ####left_center = self.center_branch['l'](left_features)
                ####right_center = self.center_branch['r'](right_features)
                ####left_center = torch.cat([ left_center, self.depth_branch['l'](left_features, k_value = k_value['l'])], dim=-1)
                ####right_center = torch.cat([ right_center, self.depth_branch['r'](right_features, k_value = k_value['r'])], dim=-1)
                ###################################################################

                ###################################################################
                ############################ RootNet ##############################
                left_center = self.root_net['l'](left_features_center, k_value = k_value['l'])
                right_center = self.root_net['r'](right_features_center, k_value = k_value['r'])
                ###################################################################


                if self.cascaded_input_single_hand==1:
                    #left_pose,left_shape=self.cascaded_left_mano_branch_list[i](torch.cat([lower_right_mano_para["shape"],lower_right_mano_para["pose"],left_features,torch.cat([left_lower_trans, left_lower_scale], axis = -1),],dim=1))
                    #right_pose,right_shape=self.cascaded_right_mano_branch_list[i](torch.cat([lower_left_mano_para["shape"],lower_left_mano_para["pose"],right_features,torch.cat([right_lower_trans, right_lower_scale], axis = -1),],dim=1))
                    left_pose,left_shape=self.cascaded_left_mano_branch_list[i](torch.cat([lower_right_mano_para["shape"],lower_right_mano_para["pose"],left_features,torch.cat([left_lower_trans, torch.ones_like(left_lower_trans[:,0])[:, None]], axis = -1),],dim=1))
                    right_pose,right_shape=self.cascaded_right_mano_branch_list[i](torch.cat([lower_left_mano_para["shape"],lower_left_mano_para["pose"],right_features,torch.cat([right_lower_trans, torch.ones_like(right_lower_trans[:,0])[:, None]], axis = -1),],dim=1))
                elif self.cascaded_input_single_hand==0:
                    left_pose,left_shape=self.cascaded_left_mano_branch_list[i](torch.cat([lower_left_mano_para["shape"],lower_left_mano_para["pose"],left_features,torch.cat([left_lower_trans, torch.ones_like(left_lower_trans[:,0])[:, None]], axis = -1),],dim=1))
                    right_pose,right_shape=self.cascaded_right_mano_branch_list[i](torch.cat([lower_right_mano_para["shape"],lower_right_mano_para["pose"],right_features,torch.cat([right_lower_trans, torch.ones_like(right_lower_trans[:,0])[:, None]], axis = -1),],dim=1))
                elif self.cascaded_input_single_hand==2:
                    #left_pose,left_shape=self.cascaded_left_mano_branch_list[i](torch.cat([lower_right_mano_para["shape"],lower_right_mano_para["pose"],lower_left_mano_para["shape"],lower_left_mano_para["pose"],left_features,torch.cat([left_lower_trans, left_lower_scale], axis = -1)],dim=1))
                    #right_pose,right_shape=self.cascaded_right_mano_branch_list[i](torch.cat([lower_left_mano_para["shape"],lower_left_mano_para["pose"],lower_right_mano_para["shape"],lower_right_mano_para["pose"],right_features,torch.cat([right_lower_trans, right_lower_scale], axis = -1)],dim=1))
                    left_pose,left_shape=self.cascaded_left_mano_branch_list[i](torch.cat([lower_right_mano_para["shape"],lower_right_mano_para["pose"],lower_left_mano_para["shape"],lower_left_mano_para["pose"],left_features,torch.cat([left_lower_trans, torch.ones_like(left_lower_trans[:,0])[:, None]], axis = -1),],dim=1))
                    right_pose,right_shape=self.cascaded_right_mano_branch_list[i](torch.cat([lower_left_mano_para["shape"],lower_left_mano_para["pose"],lower_right_mano_para["shape"],lower_right_mano_para["pose"],right_features,torch.cat([right_lower_trans, torch.ones_like(left_lower_trans[:,0])[:, None]], axis = -1),],dim=1))
                else:
                    left_pose,left_shape=self.cascaded_left_mano_branch_list[i](left_features)
                    right_pose,right_shape=self.cascaded_right_mano_branch_list[i](right_features)
                cascaded_left_para=self.pose_shape2vertsjoints(left_pose,left_shape,'l')
                cascaded_right_para=self.pose_shape2vertsjoints(right_pose,right_shape,'r')
                ## Normalize regressed poses
                #cascaded_left_para = self.normalize_pose_coords(cascaded_left_para)
                #cascaded_right_para = self.normalize_pose_coords(cascaded_right_para)
                ## Apply relatve translation to left hand
                #cascaded_left_para = self.apply_rel_trans(cascaded_left_para, lower_trans)

                ## Normalize regressed poses
                #cascaded_left_para = self.normalize_pose_coords(cascaded_left_para)
                #cascaded_right_para = self.normalize_pose_coords(cascaded_right_para)############## Normalize or
                ## Apply relatve translation to left hand
                #cascaded_left_para = self.apply_rel_transcale(cascaded_left_para, left_center, left_scale)
                #cascaded_right_para = self.apply_rel_transcale(cascaded_right_para, right_center, right_scale)
                cascaded_left_para = self.apply_rel_transcale(cascaded_left_para, left_center)
                cascaded_right_para = self.apply_rel_transcale(cascaded_right_para, right_center)

                if self.rescale:
                    cascaded_left_para = self.normalize_pose_coords(cascaded_left_para)
                    cascaded_right_para = self.normalize_pose_coords(cascaded_right_para)


                # Project to image
                cascaded_left_para['jointsimg'] = self.projection_2D(xyc[:,21:,:],cascaded_left_para['joints3d'],64,256)
                cascaded_right_para['jointsimg'] = self.projection_2D(xyc[:,:21,:],cascaded_right_para['joints3d'],64,256)#, get_weak=True)
                if self.use_heat_xy:
                    left_center[:,:2] = left_mano_para['jointsimg'][:,9,:2]/256
                    right_center[:,:2] = right_mano_para['jointsimg'][:,9,:2]/256

                ####################################################################
                #reproj_joints = torch.cat([cascaded_left_para['jointsimg']/256, cascaded_right_para['jointsimg']/256], axis=1)
                #reproj_joints[:,:,:2] = reproj_joints[:,:,:2]/weak_params[:,None,0,None]-weak_params[:,None,1:]
                #img_dist = self.get_abs_rel_trans(reproj_joints)
                #r_dist = self.get_abs_rel_trans(torch.cat([cascaded_left_para['joints3d'], cascaded_right_para['joints3d']], axis=1))
                #cascaded_left_para['joints3d'][:,:,2:] += (img_dist-r_dist)[:,2:]
                ####################################################################


                #cascaded_result = self.create_result_dict(cascaded_left_para, cascaded_right_para, left_center, right_center, left_scale, right_scale)
                cascaded_result = self.create_result_dict(cascaded_left_para, cascaded_right_para, left_center, right_center)

                mano_para_list.append(cascaded_result)

        return mano_para_list

        #left_mano_para=left_mano_para_list[-1]
        #right_mano_para=right_mano_para_list[-1]
        #trans=trans_list[-1]

        #result = {}

        ## Normalize joints by [0,9] bone length
        #########################################
        #result['l_length'] = (left_mano_para['joints3d'][:,9] - left_mano_para['joints3d'][:,0]).norm(dim=1)
        #result['r_length'] = (right_mano_para['joints3d'][:,9] - right_mano_para['joints3d'][:,0]).norm(dim=1)
        #left_mano_para['joints3d'] /= result['l_length'][:,None,None]
        #left_mano_para['joints3d'] = (left_mano_para['joints3d'] + trans[:,1:].view(-1,1,3))*torch.exp(trans[:,0,None,None])
        #left_mano_para['verts3d'] /= result['l_length'][:,None,None]
        #left_mano_para['verts3d'] = (left_mano_para['verts3d'] + trans[:,1:].view(-1,1,3))*torch.exp(trans[:,0,None,None])
        #right_mano_para['joints3d'] /= result['r_length'][:,None,None]
        #right_mano_para['verts3d'] /= result['r_length'][:,None,None]
        #########################################

        ########################################
        # Perform 2d image projections with weak-perspective model
        
        #for i in range(len(left_mano_para_list)):
        #if self.scale2d:
        #    left_xy=self.projection_2D(xyc[:,:21,:],left_mano_para['joints3d']*scale_2d[:,None,:],64,256)####################
        #    right_xy=self.projection_2D(xyc[:,21:,:],right_mano_para['joints3d']*scale_2d[:,None,:],64,256)####################
        #else:
        #left_xy=self.projection_2D(xyc[:,21:,:],left_mano_para['joints3d'],64,256)
        #right_xy=self.projection_2D(xyc[:,:21,:],right_mano_para['joints3d'],64,256)
        #
        ## Save 2d rescaled projections
        #left_mano_para['jointsimg'] = left_xy#[-1]
        #right_mano_para['jointsimg'] = right_xy#[-1]
        ##
        #########################################
        #
        #########################################
        #result['l_joints'] = left_mano_para
        #result['r_joints'] = right_mano_para
        #
        #result['l_pose'] = left_pose
        #result['l_shape'] = left_shape
        #result['r_pose'] = right_pose
        #result['r_shape'] = right_shape
        #result['trans'] = trans
        #################################
        ##if self.scale2d:
        ##    result['scale_2d'] = scale_2d
        #################################
        ##########################################
                
        #return result, left_xy, right_xy
