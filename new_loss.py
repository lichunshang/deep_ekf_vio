import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from liegroups.torch import SO3

def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s

def geo_loss(Ps, Gs, do_scale = True, gamma = 0.9):
    n = Ps.shape[0]
    for i in range(n):
        w = gamma ** (n - i - 1)
        
        if do_scale:
            s = fit_scale(Ps, Gs)
    return 

def scale_pose(est,gt):
    # (BxSx4x4)
    angle_est = est[:,:,:3,:3]
    trans_est = est[:,:,:3,3]
    trans_gt_norm = torch.norm(gt[:,:,:3,3], dim=2).unsqueeze(2)
    trans_est_norm = torch.norm(trans_est, dim=2).unsqueeze(2)
    trans_est = trans_est / trans_est_norm * trans_gt_norm
    norm_est_pose = torch.eye(4,4).repeat(est.size(0),est.size(1),1,1)
    norm_est_pose[:,:,:3,:3] = angle_est
    norm_est_pose[:,:,:3,3] = trans_est
    return norm_est_pose

def rescale_pose(est, gt):
    angle_est = est[:,:,:3,:3]
    trans_gt = gt[:,:,:3,3]
    trans_gt_norm = torch.norm(trans_gt, dim=-1).unsqueeze(-1)
    trans_est = est[:,:,:3,3]
    trans_est_norm = torch.norm(trans_est, dim=-1).unsqueeze(-1)
    new_trans_est = trans_est / trans_est_norm * trans_gt_norm
    new_est = torch.eye(4,4).repeat(est.size(0),gt.size(1),1,1)
    new_est[:,:,:3,:3] = angle_est
    new_est[:,:,:3,3] = new_trans_est
    return new_est

def euler_to_matrix(data):
    data = data.detach().cpu().numpy()
    mat = np.concatenate([[np.eye(4,4)]] * data.shape[0], axis=0)
    print(mat.shape)
    rot = R.from_euler('xyz',(data[-1,0],data[-1,1],data[-1,2])).as_matrix()
    mat[:,:3,:3] = rot
    mat[:,:3,3] = (data[:,3],data[-1:,4],data[-1:,5])
    return mat

def matrix_to_euler(mat):
    r = mat[:3,:3]
    t = mat[:3,3]
    rot = R.from_matrix(r).as_euler('xyz')
    return np.concatenate([rot,t])

def euler_to_matrix_torch(data):
    roll = data[:,0]
    pitch = data[:,1]
    yaw = data[:,2]
    x = data[:,3]
    y = data[:,4]
    z = data[:,5]
    mat = torch.eye(4,4).repeat(data.shape[0],1,1)
    rot = R.from_euler('xyz',(roll,pitch,yaw)).as_matrix()
    mat[:3,:3] = rot
    mat[:3,3] = (x,y,z)
    return mat

if __name__ == '__main__':
    a = torch.rand(12,1,4,4)
    b = scale_pose(a)
    c = [ 9.4073580e-04,  1.3585356e-03,  1.7236921e-04,  1.3084458e+00, -2.3419287e-03 , 8.3899405e-03]
    d = np.array([[ 9.99999062e-01 ,-1.71091112e-04 , 1.35869672e-03 , 1.30844581e+00],
                    [ 1.72369051e-04  ,9.99999543e-01, -9.40501479e-04, -2.34192866e-03],
                    [-1.35853519e-03 , 9.40734794e-04 , 9.99998635e-01,  8.38994049e-03],
                    [ 0.00000000e+00,  0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    
    out = matrix_to_euler(d)
    c = torch.tensor([[9.4073580e-04,  1.3585356e-03,  1.7236921e-04,  1.3084458e+00, -2.3419287e-03 , 8.3899405e-03],
                      [9.4073580e-04,  1.3585356e-03,  1.7236921e-04,  1.3084458e+00, -2.3419287e-03 , 8.3899405e-03]])
    # print(euler_to_matrix(c))
    
    c9 = SO3.from_rpy(c[:,:3])
    print(torch.Tensor(c9))