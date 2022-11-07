import torch

def eucl_dist_torch(output, gt):
    return (output - gt).pow(2).sum(-1).sqrt()

def mean_joint_error(output, gt):
    dist = eucl_dist_torch(output, gt)
    return dist.mean([-1,-2]).detach().cpu()

def PCK(output, gt, threshold=15):
    dist = eucl_dist_torch(output, gt)
    # Loop over batches
    out = []
    for i, batch_d in enumerate(dist):
        correct_mask = batch_d < threshold
        out.append(len(correct_mask[correct_mask])/len(correct_mask))
    return torch.tensor(out)


def transform_compare_gt(output, gt, metadata, center=True):
    # Rescale relative pose to actual bone length also converting to millimiters
    r_scale, l_scale = metadata['r_mano_length'], metadata['l_mano_length'] # in mm, use 'side_joint_length' for meters
    l_gt = gt['l_mano_joints'] * l_scale[:,None,None]
    r_gt = gt['r_mano_joints'] * r_scale[:,None,None]
    l_result = output['l']['joints3d'] * l_scale[:,None,None]
    r_result = output['r']['joints3d'] * r_scale[:,None,None]
    if center:
        # Center left hand on 9th joint
        l_gt = l_gt - l_gt[:,9][:,None,:]
        l_result = l_result - l_result[:,9][:,None,:]

    return l_result, r_result, l_gt, r_gt


def transform_compare_old(output, gt, metadata, center=True):
    # Rescale relative pose to actual bone length also converting to millimiters
    r_scale, l_scale = metadata['r_joint_length'], metadata['l_joint_length'] # in mm, use 'side_joint_length' for meters
    l_result = (output['l']['joints3d'] * l_scale[:,None,None])+gt['r_joints'][:,9][:,None,:]
    r_result = (output['r']['joints3d'] * r_scale[:,None,None])+gt['r_joints'][:,9][:,None,:]
    l_gt, r_gt = gt['l_joints'], gt['r_joints']
    if center:
        # Center left hand on 9th joint
        l_gt = l_gt - l_gt[:,9][:,None,:]
        l_result = l_result - l_result[:,9][:,None,:]
        r_gt = r_gt - r_gt[:,9][:,None,:]
        r_result = r_result - r_result[:,9][:,None,:]

    return l_result*1000, r_result*1000, l_gt*1000, r_gt*1000

def transform_compare(output, gt, metadata, center=True):
    # Rescale relative pose to actual bone length also converting to millimiters
    #r_scale, l_scale = metadata['r_joint_length'], metadata['l_joint_length'] # in mm, use 'side_joint_length' for meters
    l_gt, r_gt = gt['l_joints'], gt['r_joints']
    if not center:
        #l_result = (output['l']['joints3d'] * output['l']['scale'][None, :])+output['l']['trans'][:,None,:]
        l_result = output['l']['T_joints3d']# * output['l']['scale'][None, :])+output['l']['trans'][:,None,:]
        #r_result = (output['r']['joints3d'] * output['r']['scale'][None, :])+output['r']['trans'][:,None,:]
        r_result = output['r']['T_joints3d']# * output['r']['scale'][None, :])+output['r']['trans'][:,None,:]
    else:
        #l_result, r_result = output['l']['joints3d']* output['l']['scale'][:, None], output['r']['joints3d']* output['r']['scale'][:, None]
        l_result, r_result = output['l']['joints3d'], output['r']['joints3d']
    #    # Center left hand on 9th joint
        l_gt = l_gt - l_gt[:,9,:][:,None,:]
    #    l_result = l_result - l_result[:,9][:,None,:]
        r_gt = r_gt - r_gt[:,9,:][:,None,:]
    #    r_result = r_result - r_result[:,9][:,None,:]


    return l_result*1000, r_result*1000, l_gt*1000, r_gt*1000