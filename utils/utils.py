import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os


from utils.geometry_utils import rot_to_quaternion

from sklearn.decomposition import PCA

from models.layers.bn import StandingBN, StandingModel

import cv2
from PIL import Image, ImageDraw, ImageOps

from tqdm import tqdm

def gradClamp(parameters, clip=1):
    for p in parameters:
        if not p.grad is None:
            p.grad.data.clamp_(min=-clip, max=clip)

def reinitialise_parameters(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()

    for child in model.children():
        reinitialise_parameters(child)  

# Modify every BatchNorm to (Instance/Layer)Norm
def convert_batchnorm(model, norm_class='instance_norm'):
    if norm_class == 'batch_norm':
        return; 

    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            if norm_class == 'instance_norm':
                setattr(model, child_name, nn.InstanceNorm2d(num_features=child.num_features, affine=True))
            elif norm_class == 'layer_norm':
                setattr(model, child_name, nn.GroupNorm(1, child.num_features)) ## I don't trust LayerNorm default
            else:
                raise ValueError('Unknown layer normalization %s'%(norm_class))
        elif isinstance(child, nn.BatchNorm1d):
            if norm_class == 'instance_norm':
                setattr(model, child_name, nn.InstanceNorm1d(num_features=child.num_features, affine=True))
            elif norm_class == 'layer_norm':
                setattr(model, child_name, nn.GroupNorm(1,child.num_features))
            else:
                raise ValueError('Unknown layer normalization %s'%(norm_class))
        else:
            convert_batchnorm(child, norm_class)

def resize_tensor(tensor, size, mode='nearest'):
    if len(tensor.size()) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    else:
        tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size, mode=mode)
    tensor = tensor.squeeze()
    
    if len(tensor.size()) == 2:
        tensor = tensor.unsqueeze(0)
    return tensor

def distance(pts1, pts2):
    assert(pts1.size(1) == 2)

    dist = (pts1 - pts2).pow(2).sum(dim=1, keepdim=True).sqrt()

    return dist

def compute_pca(feats, other_feats):
    B,C,H,W = feats.size()

    other_pca = []
    for other_feat in other_feats:
        _,_,H2,W2 = other_feat.size()
        t_pca = []
        t_other_pca = []
        for b in range(0, feats.size(0)):
            pca = PCA(3)
            t_feats = feats[b].view(C,-1).permute(1,0).cpu()
            transform_pca = torch.Tensor(pca.fit_transform(t_feats)).view(1,H,W,3).permute(0,3,1,2)
            t_pca.append(transform_pca / transform_pca.abs().max() * 0.5 + 0.5)

            t_feats = other_feat[b].view(C,-1).permute(1,0).cpu()
            t_other_pca.append(torch.Tensor(pca.transform(t_feats)).view(1,H2,W2,3).permute(0,3,1,2) / transform_pca.abs().max() * 0.5 + 0.5)

        other_pca.append(torch.cat(t_other_pca, 0))

    print(other_pca[0].min(), other_pca[0].max())

    return torch.cat(t_pca, 0), other_pca

# Visualise correspondences
def visualise_correspondences(xs_im1, xs_im2, im1, im2, conf=None, N=3):

    # Now draw on the keypoints weighted with alpha by conf
    n=0
    # First concatenate the images along the width dimension
    B,_,H,W = im1.size()
    im_cat = torch.zeros((B,3,H,W*2)).to(im1.device)

    # Convert to PIL image
    for b in range(0, xs_im1.shape[0]):
        im_pil = torch.cat((im1[b], im2[b]), 2).permute(1,2,0).data.cpu().numpy() * 128 + 128
        im_pil = Image.fromarray(im_pil.astype(np.uint8))
        draw = ImageDraw.Draw(im_pil)

        # And now draw the keypoints and lines
        xs_1 = (xs_im1[b,:,:] * (W // 2) + (W // 2)).long()
        xs_2 = (xs_im2[b,:,:] * (W // 2) + (W // 2)).long()
        for kp in range(0, min(2000,xs_1.shape[0]), 4):
            draw.ellipse([xs_1[kp,0] - 4, xs_1[kp,1] - 4, xs_1[kp,0] + 4, xs_1[kp,1] + 4], fill=(255,0,0,255))
            draw.ellipse([xs_2[kp,0] - 4 + W, xs_2[kp,1] - 4, xs_2[kp,0] + 4 + W, xs_2[kp,1] + 4], fill=(255,0,0,255))
            draw.line([xs_1[kp,0], xs_1[kp,1], xs_2[kp,0] + W, xs_2[kp,1]], fill=(0,255,0,255))

        im_cat[b,:,:,:] = torch.Tensor(np.array(im_pil)).to(im1.device).permute(2,0,1) / 255.

    return im_cat

def set_accumulate(model):
    '''
    Set running mean and variance to 0 and use the moving average setup.
    '''
    model.accumulate_images = True
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, StandingBN(bn=child))

        elif isinstance(child, nn.BatchNorm1d):
            setattr(model, child_name, StandingBN(bn=child))

        set_accumulate(child)

def reset_accumulate(model):
    '''
    Set running mean and variance to 0 and use the moving average setup.
    '''
    model.accumulate_images = False

def set_batchnorm(model):
    '''
    Set running mean and variance to 0 and use the moving average setup.
    '''
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, StandingBN(bn=child, accumulate_images=True))

        elif isinstance(child, nn.BatchNorm1d):
            setattr(model, child_name, StandingBN(bn=child, accumulate_images=True))

        set_batchnorm(child)

def reset_batchnorm(model):
    '''
    Set running mean and variance to 0 and use the moving average setup.
    '''
    for child_name, child in model.named_children():
        if isinstance(child, StandingBN):
            child.accumulate_images = False

        reset_batchnorm(child)


def accumulate_standing_stats(model, dataloader, num_accumulations=20):
    model.train()
    # First set everything to 0 and update the batch norms to a normal weighted average
    set_accumulate(model)

    # Then iterate over the data loader to accumulate batch norm statistics
    print('Accumulating values...')
    with torch.no_grad():
        for i in tqdm(range(0, num_accumulations)):
            model.step(dataloader, val=True) 

    # reset_accumulate(model)
    model.eval()


# Taken from the deepvoxels codebase (util.py)
def parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
    with open(filepath, 'r') as file:
        f, cx, cy = list(map(float, file.readline().split()))[:3]

        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        near_plane = float(file.readline())
        scale = float(file.readline())

        height, width = map(float, file.readline().split())

        cx = cx / width * trgt_sidelength
        cy = cy / width * trgt_sidelength
        f = trgt_sidelength / height * f

        fx = f
        if invert_y:
            fy = -f
        else:
            fy = f

        full_intrinsic = np.array([[fx, 0., cx, 0.],
                                   [0., fy, cy, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

        print(full_intrinsic, near_plane)

        return full_intrinsic, grid_barycenter, scale, near_plane


def image_loader(path, W, resize=True):
    if not os.path.exists(path):
        print(path)
        print(1+'1')
    image = cv2.imread(path)
    image = np.float32(image) / 255.0
    image = cv2.resize(image, (W, W))
    if resize:
        image = cv2.copyMakeBorder(image, 
                    top=80,
                    bottom=30,
                    left=20,
                    right=20, 
                    borderType=cv2.BORDER_CONSTANT,
                    value = [0, 0, 0]
                    )

        h, w, _ = image.shape
        image = image[(h - 170) // 2:(h+170)//2, (w - 170) // 2:(w+170)//2,:]
        image = cv2.resize(image, (256, 256))
    return image

def rand_trans(image, rng):
    # Given an image, randomly transform it in 2D and return the 
    # new image and transformation.
    W, H, _ = image.shape
    assert(W == H)

    R = (np.random.randint(360) - 180) / 180. * np.pi
    s = np.random.rand() * 0.4 + 0.8
    centerW = np.random.randint(W // 2) + W // 4 
    centerH = np.random.randint(H // 2) + H // 4 
    centerW = 0
    centerH = 0

    M = np.eye(3)
    M[0,0] = np.cos(R) * s
    M[0,1] = - np.sin(R) * s
    M[1,0] = np.sin(R) * s
    M[1,1] = np.cos(R) * s

    M[0,2] = centerW
    M[1,2] = centerH
    M = M[0:2,:]

    rotMat = cv2.getRotationMatrix2D((W // 2, H // 2), R * 180 / np.pi, s)
    rotMat[0,2] = rotMat[0,2] + centerW
    rotMat[1,2] = rotMat[1,2] + centerH

    sampled_image = cv2.warpAffine(image, 
            rotMat, 
            (W, H),
            borderValue=255)
    confidence_image = cv2.warpAffine(np.ones((W,H)),
            rotMat,
            (W, H),
            borderValue=255)

    R = np.eye(3)
    R[0:2,0:2] = M[0:2,0:2] / s


    R = torch.Tensor(rot_to_quaternion(np.linalg.inv(R.T)))
    T = torch.Tensor([- centerW / float(W) * 2, 
                      - centerH / float(H) * 2, 0])
    s = torch.Tensor([1 / s, 1 / s, 1])

    return sampled_image, confidence_image, [R, T, s]


def rgb_preprocess(image):
    if len(image.shape) == 2:
      image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image

def resize(image):
    if 'dtype' in dir(image) and image.dtype is torch.float32:
        max_W = max(image.size(1), image.size(2))
        new_image = torch.zeros((1, max_W, max_W, 2)) - 21

        c_W = max_W // 2; c_H = max_W // 2
        s_W = image.size(1) // 2; s_H = image.size(2) // 2

        new_image[:,c_W-s_W:c_W-s_W+image.size(1),c_H-s_H:c_H-s_H+image.size(2),:] = image

        return new_image

    else:
        desired_size = max(image.size[0], image.size[1])
        d_W = desired_size - image.size[0]
        d_H = desired_size - image.size[1]

        padding = (d_W//2, d_H//2, d_W - (d_W//2), d_H - (d_H//2))

        new_image = ImageOps.expand(image, padding)
        return new_image
        