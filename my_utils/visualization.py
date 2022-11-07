
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def plot_sample(img, joints_in=None, skeleton=None):
    #if torch.is_tensor(img):
    out = img.detach().cpu().numpy().squeeze()#.astype(np.uint8)
    if out.shape[0] == 3:
        out = out.transpose(1,2,0)
    #out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    #plt.figure(figsize=(10,10))
    plt.imshow(out+0.5, vmin=-1, vmax=1)
    if joints_in is not None:
        #for i in range(len(joints[:,0])):
        joints = joints_in.clone().detach().cpu()
        plt.scatter(joints[:,0], joints[:,1], color='C0', cmap='jet', s=8)#, label=i)
        if skeleton is not None:
            for bone in skeleton:
                plt.plot(joints[bone[:], 0], joints[bone[:], 1], color = 'C1')
                if joints.shape[0]>21:
                    input_joints = joints[21:]
                    plt.plot(input_joints[bone[:], 0], input_joints[bone[:], 1], color = 'C2')
    plt.show()

def plot_multistage_sample(img, input_joints=None, add_joints=None, skeleton=None):
    #if torch.is_tensor(img):
    out = img.detach().cpu().numpy().squeeze()#.astype(np.uint8)
    if out.shape[0] == 3:
        out = out.transpose(1,2,0)
    plt.imshow(out+0.5, vmin=-1, vmax=1)
    if input_joints is not None:
        for i in range(len(input_joints)):
        #out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            l_joints = input_joints[i].squeeze(0).detach().cpu()
            if add_joints is not None:
                r_joints = add_joints[i].squeeze(0).detach().cpu()
                joints = torch.cat([l_joints, r_joints])
                plt.scatter(joints[:,0], joints[:,1], color='C0', cmap='jet', s=4)
            else:
                plt.scatter(l_joints[:,0], l_joints[:,1], color='C0', cmap='jet', s=4)
    plt.show()

def plot_3d_skeleton(in_joints, skeleton, axes=None, get_axes=False, color='C1'):
    input_joints = in_joints.detach().cpu().numpy()#.squeeze(0)
    #fig = plt.figure()
    #plt.title('Image projected pose')
    #ax = Axes3D(fig, auto_add_to_figure=False)
    if axes == None:
        ax = plt.axes(projection='3d')
    else:
        ax = axes#plt.axes(projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    #fig.add_axes(ax)
    for bone in skeleton:
        ax.plot3D(input_joints[bone[:], 0], input_joints[bone[:], 1], input_joints[bone[:], 2], color = color)
        if input_joints.shape[0]>21:
            joints = input_joints[21:]
            ax.plot3D(joints[bone[:], 0], joints[bone[:], 1], joints[bone[:], 2], color = 'C0')
    ax.scatter3D(input_joints[:,0], input_joints[:,1], input_joints[:,2], color = color)
    if get_axes:
        return ax

def plot_2d_skeleton(in_joints, skeleton, axes=None):
    input_joints = in_joints.detach().cpu().numpy()#.squeeze(0)
    #fig = plt.figure()
    #plt.title('Image projected pose')
    #ax = Axes3D(fig, auto_add_to_figure=False)
    if axes == None:
        ax = plt.axes()
    else:
        ax = axes#plt.axes(projection='3d')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    #fig.add_axes(ax)
    for bone in skeleton:
        ax.plot(input_joints[bone[:], 0], input_joints[bone[:], 1], color = 'C1')
        if input_joints.shape[0]>21:
            joints = input_joints[21:]
            ax.plot(joints[bone[:], 0], joints[bone[:], 1], color = 'C2')
    ax.scatter(input_joints[:,0], input_joints[:,1], color = 'C0')