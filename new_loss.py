import torch

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