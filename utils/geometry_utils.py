import torch
import numpy as np
from pyquaternion import Quaternion

import cv2

EPS = 1e-4

def best_matches(sim, topk=8000, T=0.3, nn=1):
    ''' Find the best matches for a given NxN matrix.
        Optionally, pass in the actual indices corresponding to this matrix
        (cond1, cond2) and update the matches according to these indices.
    '''
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]

    ids1 = torch.arange(0, sim.shape[0]).to(nn12.device)
    mask = (ids1 == nn21[nn12])

    matches = torch.stack([ids1[mask], nn12[mask]])

    preds = sim[ids1[mask], nn12[mask]]
    res, ids = preds.sort()
    ids = ids[res > T]

    matches = matches[:,ids[-topk:]]

    return matches.t()


def normalizePts(kp):
    B = kp.size(0)

    # First make homogeneous
    B, P, C = kp.size()
    kp_ones = torch.ones((B,P,1)).to(kp.device)
    kp = torch.cat((kp, kp_ones), 2)

    mean = kp.mean(dim=1, keepdim=True)

    kpT = (kp - mean)
    scale = np.sqrt(C) / kpT.pow(2).sum(dim=2).sqrt().mean(dim=1, keepdim=True).clamp(min=1e-2)

    aff = torch.eye(3).to(kp.device).unsqueeze(0).repeat(kp.size(0),1,1)
    aff[:,0,0] = scale.view(B,); aff[:,1,1] = scale.view(B,)
    aff[:,0:2,2] = - scale * mean[:,0,0:2]

    kp = torch.einsum('bij,bpj->bpi', aff, kp)
    return kp, aff

def null(matrix, eps=1e-15):
    assert(matrix.shape[0] == matrix.shape[1])

    u, s, vh = matrix.svd()
    null_mask = (s <= eps)

    null_space = vh[:,null_mask]

    return null_space.T

def rot_to_quaternion(rot):
    quat = Quaternion(matrix=rot)

    return [quat.w, quat.x, quat.y, quat.z]

def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    vector = vector.reshape(3,)
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])


def random_affine(sx=(0.8,1.0), sy=(0.8,1.0), cx=(-0.5,0.5), cy=(-0.5,0.5),
                  theta=(-np.pi/8,np.pi/8), phi=(-np.pi/8,np.pi/8)):
    A = torch.eye(3)

    sx = torch.rand((1,)) * (sx[1] - sx[0]) + sx[0]
    sy = torch.rand((1,)) * (sy[1] - sy[0]) + sy[0]

    cx = torch.rand((1,)) * (cx[1] - cx[0]) + cx[0]
    cy = torch.rand((1,)) * (cy[1] - cy[0]) + cy[0]

    theta = torch.rand((1,)) * (theta[1] - theta[0]) + theta[0]
    phi = torch.rand((1,)) * (phi[1] - phi[0]) + phi[0]

    # Now set up the rest of it
    A[0,0] = sx * theta.cos()
    A[0,1] = - theta.sin()
    A[1,0] = theta.sin()
    A[1,1] = theta.cos() * sy
    A[0,2] = cx
    A[1,2] = cy

    return A[0:2,:], A.inverse()[0:2,:]

def transform_affine(im, A):
    B, H, W = im.size()
    ys, xs = torch.meshgrid(torch.linspace(-1,1,im.shape[1]), torch.linspace(-1,1,im.shape[2]))

    grid = torch.cat((xs.unsqueeze(2), ys.unsqueeze(2), torch.ones(xs.shape).unsqueeze(2)), 2)

    sampler = torch.einsum('ij,xyj->xyi', A, grid)

    return torch.nn.functional.grid_sample(im.unsqueeze(0), 
                                sampler.unsqueeze(0), padding_mode='border').view(B,H,W)

def homogenize(kp):
    B, P, _ = kp.size()
    kp_ones = torch.ones((B,P,1)).to(kp.device)

    return torch.cat((kp, kp_ones), 2)


def predict_F_mat(kp1, kp2, weights=None):
    assert(kp1.shape[2] == 2)
    assert(kp2.shape[2] == 2)

    B = kp1.size(0)

    kp1, aff1 = normalizePts(kp1)
    kp2, aff2 = normalizePts(kp2)

    # Use 8 pt algorithm
    u = kp1[:,:,0:1]; up = kp2[:,:,0:1]
    v = kp1[:,:,1:2]; vp = kp2[:,:,1:2]

    u_up = u * up; v_up = v * up 
    u_vp = u * vp; v_vp = v * vp

    ones = torch.ones(u_up.shape).to(kp1.device)

    # a u up + b u vp + c u + 
    # d v up + e v vp + f v +
    # e v vp + g up   + 1 = 0

    A = torch.cat((u_up, v_up, up, u_vp, v_vp, vp, u, v, ones), 2)

    if not weights is None:
        A = A * weights.view(B,-1,1) + torch.randn(A.shape).to(A.device) / 100.

    _, _, VT = A.svd(some=False)

    fmatrix = VT[:,:,-1].view(-1,3,3)

    # And now make sure that have 7 DofF
    U, S, VT = fmatrix.svd(some=False)
    S = torch.diag_embed(S)
    S[:,-1,-1] = 0

    fmatrix = U.bmm(S.bmm(VT.permute(0,2,1)))
    fmatrix = aff2.permute(0,2,1).bmm(fmatrix.bmm(aff1))

    fmatrix = fmatrix / (fmatrix.pow(2).view(B,-1,1).sum(dim=1, keepdim=True).sqrt())
    fmatrix = fmatrix * (-1 * ((fmatrix[:,-1:,-1:] < 0).float() * 2 - 1))

    return fmatrix

def predict_F_ransac(kp1, kp2, mask, W=512):
    # For each element in the batch, use opencv
    B = kp1.size(0)
    allF = torch.zeros(B,3,3).to(kp1.device)

    for b in range(0, B):
        pts1 = kp1[b][mask[b,:,0] > 0,:]
        pts2 = kp2[b][mask[b,:,0] > 0,:]

        try:
            F, inliers = cv2.findFundamentalMat(pts1.cpu().numpy(), pts2.cpu().numpy(), ransacReprojThreshold=6./(W / 2.))
        except:
            F, inliers = cv2.findFundamentalMat(pts1.cpu().numpy(), pts2.cpu().numpy(), method=cv2.LMEDS)
        allF[b] = torch.Tensor(F / ((F ** 2).sum() ** 0.5))

    return allF

def predict_Flines(F, xys, samples=256):
    B,H,W,_ = xys.size()
    lines = torch.zeros(xys.shape).to(F.device)
    for b in range(0, B):
        t_lines = cv2.computeCorrespondEpilines(xys[b,:,:,0:2].view(-1,1,2).cpu().data.numpy(), 1, F[b].cpu().data.numpy())
        lines[b] = torch.Tensor(t_lines).view(H,W,3)

    x0 = - torch.ones((B,H,W,1)).to(xys.device)
    y0 = - (lines[:,:,:,2:3] + lines[:,:,:,0:1] * (-1)) / lines[:,:,:,1:2]

    x1 = 1
    y1 = - (lines[:,:,:,2:3] + lines[:,:,:,0:1] * x1) / lines[:,:,:,1:2]

    values = torch.linspace(0,1,samples).to(xys.device)
    values_x = x0 + (x1 - x0) * values[None,None,None,:]
    values_y = y0 + (y1 - y0) * values[None,None,None,:]

    return torch.cat((values_x[:,:,:,:,None], values_y[:,:,:,:,None]), -1) 


def predict_F_dist(fmatrix, kp1, kp2, disType='sampson'):
    kp1 = homogenize(kp1)
    kp2 = homogenize(kp2)

    pfp = kp2.bmm(fmatrix)
    pfp = pfp * kp1

    d = pfp.sum(dim=2).pow(2)
    if disType == 'sampson':
        ep1 = torch.einsum('bij,bpj->bpi', fmatrix, kp1)
        ep2 = torch.einsum('bij,bpj->bpi', fmatrix.permute(0,2,1), kp2)

        d = d / (ep1.pow(2)[:,:,0:2].sum(dim=2) + ep2.pow(2)[:,:,0:2].sum(dim=2))

    return d

if __name__ == '__main__':
    # Testing the FMat prediction

    # Simplest case : just identity
    # match_points1 = torch.Tensor([[
    #     [0.7803, 0.5752], [0.3897, 0.0598], [0.2417, 0.2348], [0.4039, 0.3532], [0.0965, 0.8212], [0.1320, 0.0154], [0.9421, 0.0430], [0.9561, 0.1690]
    # ]])

    # match_points2 = torch.Tensor([[
    #     [0.4868, 0.6443], [0.4359, 0.3786], [0.4468, 0.8116], [0.3063, 0.5328], [8, 8], [4, 4], [2, 2], [9, 9]
    # ]])

    # fMat = predict_F_mat(match_points1.repeat(10,1,1), match_points2.repeat(10,1,1))
    # print(fMat)

    # print(predict_F_dist(fMat[0:1], match_points1, match_points2))


    A = np.array([[2,3,5],[-4,2,3],[0,0,0]])
    print(null(torch.Tensor(A)))
