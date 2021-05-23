# Script to preprocess the megadepth file into a format that is what the v3d code expects.

# This requires the following:
# For each set of 'dense' views, finding pairs of images that overlap within some threshold T
# These pairs are then written to a file list.

import os
import h5py
from pyquaternion import Quaternion
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms

import torch
import torch.nn.functional as F

import utils.utils as utils
import utils.geometry_utils as geometry_utils
from utils.patch_utils import PatchMaker

class RGBDManipulator():
    ''' A class for performing operations on an RGBD image. Uses torch inorder to perform the operations
    on the GPU'''
    def __init__(self, H, W, EPS=1e-1):
        
        # Now check projection
        self.H = H; self.W = W
        self.EPS = EPS

        xs, ys = np.meshgrid(np.linspace(0,1, W), np.linspace(0,1, H))
        xs = xs.reshape(1,H,W)
        ys = ys.reshape(1,H,W)
        np.set_printoptions(precision=3, suppress=True)
        self.xys = np.vstack((xs, ys, np.ones(xs.shape), np.ones(xs.shape)))

    def get_overlap(self, depth1, cam_1, cam_2, T=0.55, M=1e-3):
        '''
        Given the two depth images and the cameras corresponding to the images, transforms one depth
        map into another and says that the images overlap if > T % of the depths are within some margin M.
        '''

        depth1 = torch.Tensor(depth1)

        # Extract the cameras
        K_inv, P1_inv = torch.Tensor(cam_1['Kinv']), torch.Tensor(cam_1['RTinv'])
        K, P2 = torch.Tensor(cam_2['K']), torch.Tensor(cam_2['RT'])

        RT = P2.mm(P1_inv)

        xys = torch.Tensor(self.xys) * depth1
        xys[-1] = 1

        xys = xys.view(4, -1)

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.mm(xys)

        # RT = torch.eye(4).unsqueeze(0).repeat(cam1_X.size(0), 1, 1).to(depth.device)
        cam2_X = RT.mm(cam1_X)

        # And intrinsics

        xy_proj = K.mm(cam2_X)

        sampler = xy_proj[0:2,:] / xy_proj[2:3,:]
        sampler[sampler != sampler] = -10
        sampler[sampler.abs() > 2] = -10
        sampler = sampler.view(1,2,self.H,self.W).permute(0,2,3,1)

        sampler[depth1.unsqueeze(0).unsqueeze(3).repeat(1,1,1,2) <= 0] = -10 
        
        return sampler * 2 - 1

def load_depth_img(directory, dense_id, im_name, base_path=os.environ['MEGADEPTH']):
    hdf5_file_read = h5py.File(base_path + 'phoenix/S6/zl548/MegaDepth_v1/%s/dense%s/depths/%s.h5' % (directory, dense_id, im_name),'r')
    depth1 = hdf5_file_read.get('/depth')
    depth1 = np.array(depth1)
    hdf5_file_read.close()

    depth1[depth1 == 0] = -10000

    return depth1

def get_intrinsics(idx, camera_ids):
    intrinsics = np.eye(4)
    W = camera_ids[idx,2].astype(np.float32)
    H = camera_ids[idx,3].astype(np.float32)
    
    intrinsics[0,0] = camera_ids[idx,4].astype(np.float32)
    intrinsics[1,1] = camera_ids[idx,4].astype(np.float32)
    intrinsics[0,2] = camera_ids[idx,5].astype(np.float32)
    intrinsics[1,2] = camera_ids[idx,6].astype(np.float32)
    
    # Now move the intrinsics such that is [0,1] in x/y by dividing by height/width
    intrinsics[0,:] = intrinsics[0,:] / W
    intrinsics[1,:] = intrinsics[1,:] / H
    
    return intrinsics

def get_extrinsics(idx, camera_ids):
    extrinsics = np.eye(4)
    
    # Load rotation as quaternion and then convert to rotation matrix
    quat = Quaternion(*camera_ids[idx,1:5].astype(np.float32))
    rot_matrix = quat.rotation_matrix
    extrinsics[0:3,0:3] = rot_matrix
    
    extrinsics[0:3,3] = camera_ids[idx,5:8].astype(np.float32)
    return extrinsics
    
def get_camera_parameters(image_ids, camera_ids, img):
    idx = np.where(image_ids[:,-1] == img)[0][0]
    camera_id = int(image_ids[idx,-2])
    
    cam_idx = np.where(camera_ids[:,0].astype(np.int32) == camera_id)[0]
    
    if cam_idx.size == 0:
        return None, None

    
    cam_idx = cam_idx[0]
    
    intrinsics = get_intrinsics(cam_idx, camera_ids)
    extrinsics = get_extrinsics(idx, image_ids)
    
    return intrinsics, extrinsics
    
def get_preprocessed_cameras(info, Ks, RTs, img):
    idx = np.where(info[:,-1] == img)[0][0]

    K = Ks[idx]
    RT = RTs[idx]

    return K, RT

def load_images_to_ids(directory, dense_id, base_path=os.environ['MEGADEPTH']):
    file_name = base_path + 'MegaDepth/MegaDepth_v1_SfM/%s/sparse/manhattan/%s/images.txt' % (directory, dense_id)
    
    cameras = []

    with open(file_name, 'r') as f:
        # Read the header
        f.readline()
        f.readline()
        f.readline()
        f.readline()

        next_line = f.readline()
        while next_line:
            camera_params = next_line.split()
            cameras += [camera_params]

            f.readline()
            next_line = f.readline()

    camera_params = np.array(cameras)

    return camera_params

class MegaDepth():
    def __init__(self, opts, split, seed=0, file_path=os.environ['MEGADEPTH']):
        self.base_path = file_path
        self.opts = opts
        self.hard_multiplier = 513

        if self.opts.W == -1:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])

        else:
            if 'jitter' in self.opts and self.opts.jitter:
                self.transform = transforms.Compose([
                    transforms.Resize((self.opts.W,self.opts.W)),
                    transforms.ColorJitter(0.1,0.1,0.1,0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                ])

            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.opts.W,self.opts.W)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                ])

        self.patchmaker = PatchMaker(opts=opts, W=8)

        directories = []
        if split == 'train':
            npzfilestxt = np.loadtxt('./dataset/files/train_megadepth.txt')
            npzfiles = []
            for i in npzfilestxt:
                dense_id = '0'
                npzfiles += ['%04d.%s.npz' % (i, dense_id)]
        else:
            npzfilestxt = np.loadtxt('./dataset/files/valid_megadepth.txt')
            npzfiles = []
            for i in npzfilestxt:
                dense_id = '0'
                npzfiles += ['%04d.%s.npz' % (i, dense_id)]

        print("Loading dataset...")
        for npzfile in tqdm(npzfiles):
            directory = npzfile.split('.')[0]
            denseid = npzfile.split('.')[1]

            if not os.path.exists('%s/MegaDepth/scene_info/%s.%s.npz' % 
                            (self.base_path, directory, denseid)):
                print("Warning: can't find %s.%s.npz" % (directory, denseid))
                continue

            data = np.load('%s/MegaDepth/scene_info/%s.%s.npz' % 
                            (self.base_path, directory, denseid), allow_pickle=True)

            xs, ys = np.where(data['overlap_matrix'] > 0.2)
            images = [im.split('/')[-1] if not im is None else im for im in data['image_paths']]
        
            iminfo = np.load('%s/MegaDepth_v1_SfM/%s/sparse/manhattan/%s/processed_cams.txt.npz' 
                            % (self.base_path, directory, denseid), allow_pickle=True)
            info = iminfo['info']
            Ks = iminfo['Ks']
            RTs = iminfo['RTs']

            directories += [
                {'directory' : directory, 'denseid' : denseid, 'xs' : xs, 'ys' : ys,
                 'images' : images, 'info' : info, 'Ks' : Ks, 'RTs' : RTs}]

        print("Loaded dataset...")

        self.directories = directories
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return 50000

    def get_specific_match(self, img_name1, img_name2, folder):
        directories = [self.directories[i]['directory'] for i in range(0, len(self.directories))]
        index = directories.index(folder)

        image_idx1 = self.directories[index]['images'].index(img_name1)
        image_idx2 = self.directories[index]['images'].index(img_name2)

        idx1 = np.where(self.directories[index]['xs'] == image_idx1)[0][0] 
        idx2 = np.where(self.directories[index]['ys'] == image_idx2)[0][0]

        return self.__getitem__(index, idx1, idx2)

    def __getitem__(self, index, pair_idx1=None, pair_idx2=None):
        # index = 5; self.rng = np.random.RandomState(0)
        index = index % len(self.directories)

        directory = self.directories[index]['directory']
        denseid = self.directories[index]['denseid']
        xs = self.directories[index]['xs']
        ys = self.directories[index]['ys']
        images = self.directories[index]['images']
        info = self.directories[index]['info']
        Ks = self.directories[index]['Ks']
        RTs = self.directories[index]['RTs']

        if pair_idx1 is None:
            pair_idx1 = self.rng.randint(xs.shape[0])
            pair_idx2 = pair_idx1

        im1_name = images[xs[pair_idx1]]
        im2_name = images[ys[pair_idx2]]

        depth_im1 = load_depth_img(directory, denseid, im1_name.split('.')[0], self.base_path)
        depth_im2 = load_depth_img(directory, denseid, im2_name.split('.')[0], self.base_path)

        K1, RT1 = get_preprocessed_cameras(info, Ks, RTs, im1_name)
        K2, RT2 = get_preprocessed_cameras(info, Ks, RTs, im2_name)


        manipulator = RGBDManipulator(W=depth_im1.shape[1], H=depth_im1.shape[0])
        sample12 = manipulator.get_overlap(
                        depth_im1, 
                        {'Kinv' : np.linalg.inv(K1), 'RTinv' : np.linalg.inv(RT1)},
                        {'K' : K2, 'RT' : RT2}).squeeze()

        # Now get the epipolar geometry
        offset = np.array([
            [2, 0, -1, 0],
            [0, 2, -1, 0],
            [0, 0,  1, 0],
            [0, 0, 0, 1]
        ])

        K1 = offset @ K1; K2 = offset @ K2

        P = RT2 @ np.linalg.inv(RT1)
        t = P[0:3,3:]
        R = P[0:3,0:3]

        E = geometry_utils.skew(t) @ R
        Fmat = geometry_utils.skew(K2[0:3,0:3] @ t) @ K2[0:3,0:3] @ R @ np.linalg.inv(K1[0:3,0:3])
        # Now normalise
        Fmat = Fmat / max(np.sum(Fmat ** 2) ** 0.5, 0.001)
        if Fmat[-1,-1] < 0:
            Fmat = - Fmat


        im1 = Image.open(self.base_path + 'MegaDepth_v1_SfM/%s/images/' % (directory) + im1_name).convert('RGB')
        im2 = Image.open(self.base_path + 'MegaDepth_v1_SfM/%s/images/' % (directory) + im2_name).convert('RGB')
        im1_size = im1.size
        im2_size = im2.size

        # Resize sample12 to 256 x 256 to make better / faster / multi resolution
        sample12 = sample12.unsqueeze(0)

        if 'keep_aspect' in self.opts and self.opts.keep_aspect:
            # Update the samplers
            W = im2.size[0]; H = im2.size[1]
            max_W = max(W, H)

            mask = sample12 < -1

            sample12[:,:,:,0] = sample12[:,:,:,0] * W / max_W
            sample12[:,:,:,1] = sample12[:,:,:,1] * H / max_W

            sample12[mask] = -21

            sample12 = utils.resize(sample12)
            im1 = utils.resize(im1)
            im2 = utils.resize(im2)

        if self.opts.W > 0:
            sample12 = F.upsample(sample12.permute(0,3,1,2), 
                                size=(self.opts.W,self.opts.W), mode='nearest')
            sample12 = sample12.permute(0,2,3,1)

        im1 = self.transform(im1)

        mask1 = (torch.Tensor(sample12) >= -1).float() * (torch.Tensor(sample12) <= 1).float() 
        mask1 = mask1.min(dim=3)[0]

        transformed_img = F.grid_sample(transforms.ToTensor()(im2).unsqueeze(0), sample12).squeeze()
        
        transformed_img = transforms.ToPILImage()(transformed_img)
        img12 = self.transform(transformed_img)

        im2 = self.transform(im2)

        vPatchesIm1, vPatchesIm2, vMatches, index_prev = self.patchmaker.create_matches(im1, im2, sample12, mask1,
                                                img12.size(1), img12.size(2),
                                                samples=self.opts.hard_samples, valid=True,
                                                return_sampler=True)
        invPatchesIm1, invPatchesIm2, invMatches, _ = self.patchmaker.create_matches(im1, im2, sample12, mask1,
                                                img12.size(1), img12.size(2),
                                                samples=self.opts.hard_samples, valid=False,
                                                return_sampler=True, index_prev=index_prev)

        
        # Complete there are less points to sample from than samples requested
        if vPatchesIm1.size(0) < self.opts.hard_samples:
            complete = torch.zeros((self.opts.hard_samples-vPatchesIm1.size(0), vPatchesIm1.size(1)), dtype=torch.float32, device=vPatchesIm1.device)
            vPatchesIm1 = torch.cat((vPatchesIm1,complete),0)
            vPatchesIm2 = torch.cat((vPatchesIm2,complete),0)
            vMatches = torch.cat((vMatches,complete[:,:1]),0)
            vPoint = torch.cat((torch.ones(vPatchesIm1.size(0), dtype=torch.float32, device=vPatchesIm1.device),\
                             torch.zeros(self.opts.hard_samples-vPatchesIm1.size(0), dtype=torch.float32, device=vPatchesIm1.device)),0)

            invcomplete = torch.zeros((self.hard_multiplier*self.opts.hard_samples-invPatchesIm1.size(0), invPatchesIm1.size(1)), dtype=torch.float32, device=invPatchesIm1.device)
            invPatchesIm1 = torch.cat((invPatchesIm1,invcomplete),0)
            invPatchesIm2 = torch.cat((invPatchesIm2,invcomplete),0)
            invMatches = torch.cat((invMatches,invcomplete[:,:1]),0)
            invPoint = torch.cat((torch.ones(invPatchesIm1.size(0), dtype=torch.float32, device=invPatchesIm1.device),\
                             torch.zeros(self.hard_multiplier*self.opts.hard_samples-invPatchesIm1.size(0), dtype=torch.float32, device=invPatchesIm1.device)),0)
        else:
            vPoint = torch.ones(self.opts.hard_samples, dtype=torch.float32, device=vPatchesIm1.device)
            invPoint = torch.ones(self.hard_multiplier*self.opts.hard_samples, dtype=torch.float32, device=vPatchesIm1.device)


        depth_im1 = torch.Tensor(depth_im1).unsqueeze(0).unsqueeze(0)
        depth_im2 = torch.Tensor(depth_im2).unsqueeze(0).unsqueeze(0)

        depth_im1 = F.upsample(size=(self.opts.W, self.opts.W), input=depth_im1, mode='nearest')
        depth_im1 = depth_im1.view(1,self.opts.W, self.opts.W)

        depth_im2 = F.upsample(size=(self.opts.W, self.opts.W), input=depth_im2, mode='nearest')
        depth_im2 = depth_im2.view(1,self.opts.W, self.opts.W)

        if 'jitter' in self.opts and self.opts.jitter:
            # Jitter the images and an affine transformation
            A1, A1inv = geometry_utils.random_affine()
            A2, A2inv = geometry_utils.random_affine()

            vPatchesIm1 = torch.cat((vPatchesIm1, torch.ones(self.opts.hard_samples, 1)), 1)
            vPatchesIm2 = torch.cat((vPatchesIm2, torch.ones(self.opts.hard_samples, 1)), 1)
            vPatchesIm1 = torch.einsum('ij,pj->pi', A1inv, vPatchesIm1)
            vPatchesIm2 = torch.einsum('ij,pj->pi', A2inv, vPatchesIm2)

            im1 = geometry_utils.transform_affine(im1, A1)
            sample12 = sample12.squeeze().permute(2,0,1)
            sampler12 = geometry_utils.transform_affine(sample12, A1)
            mask1 = geometry_utils.transform_affine(mask1, A1)

            sampler12 = torch.einsum('ij,jxy->ixy', A2inv, 
                            torch.cat((sampler12, torch.ones((1,self.opts.W,self.opts.W))), 0))

            im2 = geometry_utils.transform_affine(im2, A2)

            # Update valid based on whether they are within an epsilon or not
            vMatches = self.patchmaker.check(vPatchesIm1, vPatchesIm2, sampler12, W=img12.size(2))
            invMatches = self.patchmaker.check(invPatchesIm1, invPatchesIm2, sampler12, W=img12.size(2))

            return {'Im1' : im1, 'Im2' : im2, 
                    'vPatchesIm1' : vPatchesIm1, 'vPatchesIm2' : vPatchesIm2,
                    'invPatchesIm1' : invPatchesIm1, 'invPatchesIm2' : invPatchesIm2,
                    'vMatches' : vMatches, 'invMatches' : invMatches,
                    'vvalidPoint' : vPoint, 'invvalidPoint' : invPoint,
                    'mask12' : mask1, 'sampler12' : sample12}


        return {'Im1' : im1, 'Im2' : im2, 'Im12' : img12,
                'E' : E, 'F' : Fmat, 'K1' : K1, 'K2' : K2, 'R1' : RT1, 'R2' : RT2,
                'vvalidPoint':vPoint, 'invvalidPoint': invPoint,
                'DepthIm1' : depth_im1, 'DepthIm2' : depth_im2,
                'Im1Size' : list(im1_size), 'Im2Size' : list(im2_size), 
                'directory' : directory, 'im1_name' : im1_name, 'im2_name' : im2_name,
                'sampler12' : sample12.permute(0,3,1,2).squeeze(), 'mask12' : mask1, 
                'vPatchesIm1' : vPatchesIm1, 'vPatchesIm2' : vPatchesIm2, 'vMatches' : vMatches,
                'invPatchesIm1' : invPatchesIm1, 'invPatchesIm2' : invPatchesIm2, 'invMatches' : invMatches}

