import os, time 
from pathlib import Path

import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import resnet34, resnet18

from utils import scheduler
from utils import utils as utils
import utils.geometry_utils as geometry_utils

import models.architectures as architectures
import models.losses as losses

import segmentation_models_pytorch as smp

import numpy as np

# Keep running statistic for ResNet
def conv2d_init(m):
    if isinstance(m,nn.Conv2d):
        m.padding_mode = 'circular'

class MultiResolutionFPNNoMixConf(nn.Module):
    def __init__(self, opts=None, input_nc=3, output_nc=3, nf=32):
        super().__init__()

        self.opts = opts

        # Use a pretrained ResNet for the initial features
        if opts.network == 'resnet50':
            self.features = smp.Unet(opts.network, decoder_channels=(256,256,128,128,64), 
                                        encoder_depth=5, classes=64, encoder_weights='imagenet', full=True)
        else:
            self.features = smp.Unet('resnet18', decoder_channels=(256,128,128,64,64), 
                                        encoder_depth=5, classes=64, encoder_weights='imagenet')
        # self.features = smp.FPN('resnet18', decoder_segmentation_channels=128, encoder_depth=5, classes=128, decoder_dropout=0., encoder_weights='imagenet')

        def create_mlp(input_nc, nf, nl=2):
            layers = []
            for i in range(0, nl):
                if i == 0:
                    in_nc = input_nc
                else:
                    in_nc = nf
                
                out_nc = nf

                layers += [nn.Conv2d(in_nc, out_nc, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(out_nc)]

            layers += [nn.Sigmoid()]

            return nn.Sequential(*layers)

        nf = 64
        self.regress_conf = create_mlp(nf, 1, nl=3)
        self.binary_class = create_mlp(nf, 1, nl=3)

        self.fp1L = nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0,bias=True)
        self.fp2L = nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0,bias=True)
        self.fp1S = nn.Conv2d(1024,128,kernel_size=1,stride=1,padding=0,bias=True)
        self.fp2S = nn.Conv2d(1024,128,kernel_size=1,stride=1,padding=0,bias=True)

        # Switch to instance norm to see if this fixes things
        if opts.norm_class == 'instance_norm':
            utils.convert_batchnorm(self, opts.norm_class)

        # Optimizer
        self.optimizerG = self.get_optimizer()
        self.plateauscheduler = scheduler.Scheduler(opts, self.optimizerG)
        self.loss = losses.HingeLoss(self.opts.hinge_samples)
        self.stats = losses.Stats()

        self.accumulate_images = False
        self.accumulate_batches = {}

        # Grid
        self.W = 6; self.H = 6
        ys, xs = torch.meshgrid(torch.linspace(-1,1,self.H), torch.linspace(-1,1,self.W))
        self.register_buffer('grid', torch.cat((xs.unsqueeze(2), ys.unsqueeze(2)), 2).unsqueeze(0))

        
        train_layers = ['layer2', 'layer3', 'layer4']
        for name, param in self.features.encoder.named_parameters():
            if not('train_encoder' in self.opts) or (not self.opts.train_encoder) or (not name.split('.')[0] in train_layers):
                param.requires_grad = False


    def get_optimizer(self):
        optimizer = [
            {'params' :   [prm for prm in self.parameters() if prm.requires_grad]}
        ]

        return optim.Adam(optimizer, lr=self.opts.lr, weight_decay=0.0005)

    def load_checkpoint(self, opts):
        if opts.resume:
            if os.path.isdir(opts.model_epoch_path[:-19]) and len(os.listdir(opts.model_epoch_path[:-19]))>0:
                model_path = Path(opts.model_epoch_path[:-19])
                model_path = [x[0] for x in sorted([(fn, os.stat(fn)) for fn in model_path.iterdir()], key = lambda x: x[1].st_ctime)][0]
                c_epoch = torch.load(model_path)['epoch']+1
            else:
                return 0, 0
        else:
            c_epoch = opts.continue_epoch
        print("Continuing from epoch %d..." % c_epoch)

        past_state = torch.load(opts.model_epoch_path % str(c_epoch - 1))
        self.load_state_dict(
            torch.load(opts.model_epoch_path % str(c_epoch - 1))['state_dict'])
        self.optimizerG.load_state_dict(
            torch.load(opts.model_epoch_path % str(c_epoch - 1))['optimizer'])
        self.plateauscheduler.load_state_dict(
            torch.load(opts.model_epoch_path % str(c_epoch - 1))['scheduler'])

        opts = past_state['opts']
        opts.continue_epoch = c_epoch
        opts.epoch = c_epoch
        self.plateauscheduler.update_opts(opts)
        return opts, c_epoch

    def step_plateau(self, loss):
        return
        # self.plateauscheduler.step(loss)

    def step(self, loader, visualise=False, val=False):
        if val:
            loss, imgs = self.forward(loader.next(), visualise)
            return loss, imgs

        t_loss = 0
        self.optimizerG.zero_grad()
        for i in range(0, 1):
            loss, imgs = self.forward(loader.next(), visualise)
            t_loss += loss['Total Loss']
            (loss['Total Loss'] / 1.).backward()

        self.optimizerG.step()
        loss['Total Loss'] = t_loss / float(i+1)
        return loss, imgs

    def get_checkpoint(self, epoch):
        return {'state_dict' : self.state_dict(),
                'optimizer' : self.optimizerG.state_dict(),
                'epoch' : epoch,
                'opts' : self.opts,
                'scheduler' : self.plateauscheduler.state_dict()}

    def obtain_descriptors(self, cond1, cond2):
        cond = cond1[:]

        cond[-1] = torch.cat((cond1[-1][:,:,:,:], cond1[-1][:,:,:,:]), 1)
        cond[-2] = torch.cat((cond1[-2][:,:,:,:], cond1[-2][:,:,:,:]), 1)

        imFeats = self.features.segmentation_head(self.features.decoder(*cond))
        return imFeats

    def sample_features(self, feat, sampler):
        B,C,H,W = feat.size()
        B,S,_ = sampler.size()
        sampler = sampler.unsqueeze(1)

        if 'train_end2end' in self.opts and self.opts.train_end2end:
            conf_feat = feat
        else:
            conf_feat = feat.detach()
        confidence = self.regress_conf(conf_feat)
        confidence = F.grid_sample(confidence, sampler)

        sFeat = F.grid_sample(feat, sampler)
        sFeat = architectures.NormBlock(dim=1)(sFeat).view(B,-1,S)

        return sFeat, confidence.squeeze()


    def decision(self, feat1, feat2, compAll=False):
        if compAll:
            feat1 = feat1.unsqueeze(3)
            feat2 = feat2.unsqueeze(2)

        if self.opts.losses == 'hinge':
            # Compare all features to all others -- looking at different scales
            dist = (feat1[:, :] - feat2[:,:]).pow(2).sum(dim=1)

            return dist, 0

        elif self.opts.losses == 'bce':
            dist = (feat1 * feat2).sum(dim=1) * 0.5 + 0.5
            return dist, 0

    def run_feature(self, im1, im2, resolution='medium', 
                            keypoints=None, sz1=(128,128), sz2=(128,128), 
                            r_im1=(1., 1.),r_im2=(1., 1.),
                            factor=1,
                            MAX=4096, use_conf=True, T=0.0,
                            RETURN_ALLKEYPOINTS=False, RETURN_FEATURES=False):
        # raise Exception("Not implemented")
        # Obtain both the keypoint locations and corresponding features
        image1, jitImage1 = im1
        image2, jitImage2 = im2

        def sampler_values(size, rx, ry):
            _, _, H, W = size
            # Now create a grid of locations from which to obtain a bunch of patches
            ys, xs = torch.meshgrid(torch.linspace(-1,1,H) * ry, torch.linspace(-1,1,W) * rx)
            origSampler = torch.cat((xs.unsqueeze(2), 
                                        ys.unsqueeze(2)), 2).unsqueeze(0)
            origSampler = origSampler.view(1,-1,2)

            return origSampler

        # Get features
        cond1 = self.features.encoder(image1)
        cond2 = self.features.encoder(image2)

        featsim1 = self.obtain_descriptors(cond1, cond2)
        featsim2 = self.obtain_descriptors(cond2, cond1)

        confim1 = self.regress_conf(featsim1)
        confim2 = self.regress_conf(featsim2)

        maskim1 = self.binary_class(featsim1)
        maskim2 = self.binary_class(featsim2)
        

        if not(keypoints is None):
            B,C,_,_ = featsim1.size()
            feats1, conf1 = self.sample_features(featsim1, keypoints[0].view(B,-1,2))
            feats2, conf2 = self.sample_features(featsim2, keypoints[1].view(B,-1,2))

            kp1 = keypoints[0]; kp2 = keypoints[1]

            
        else:
            B = featsim1.size(0)

            origSample1 = sampler_values((1,1,sz1[1],sz1[0]), r_im1[0], r_im1[1])
            origSample1 = origSample1.to(featsim1.device)

            origSample2 = sampler_values((1,1,sz2[1],sz2[0]), r_im2[0], r_im2[1])
            origSample2 = origSample2.to(featsim1.device)

            feats1, conf1 = self.sample_features(featsim1, origSample1.repeat(B,1,1))
            feats2, conf2 = self.sample_features(featsim2, origSample2.repeat(B,1,1))

            if factor > 1:
                mask = torch.zeros(conf1.size()).to(conf1.device)
                mask.view(1,sz1[1],sz1[0])[:,0::factor,0::factor] = 1
                conf1 = conf1 * mask

            kp1 = origSample1.clone().repeat(B,1,1)
            kp2 = origSample2.clone().repeat(B,1,1)

        m_s1 = F.grid_sample(maskim1, kp1.unsqueeze(1)).squeeze()
        m_s2 = F.grid_sample(maskim2, kp2.unsqueeze(1)).squeeze()

        # conf1 = conf1 * (m_s1 > 0.5).float()
        # conf2 = conf2 * (m_s2 > 0.5).float()

        # Only keep MAX most likely keypoints in each image
        conf1, conf1_inds = conf1.view(B,-1).sort(descending=True, dim=1)
        conf2, conf2_inds = conf2.view(B,-1).sort(descending=True, dim=1)

        if not RETURN_ALLKEYPOINTS:
            kp1 = kp1.gather(index=conf1_inds[:,0:MAX].unsqueeze(2).expand(-1,-1,2), dim=1).squeeze()
            kp2 = kp2.gather(index=conf2_inds[:,0:MAX].unsqueeze(2).expand(-1,-1,2), dim=1).squeeze()
        else:
            kp1 = kp1.squeeze()
            kp2 = kp2.squeeze()

        feats1 = feats1.gather(index=conf1_inds[:,0:MAX].unsqueeze(1).expand(-1,64,-1), dim=2).squeeze()
        feats2 = feats2.gather(index=conf2_inds[:,0:MAX].unsqueeze(1).expand(-1,64,-1), dim=2).squeeze()

        conf1 = conf1[:,0:MAX]
        conf2 = conf2[:,0:MAX]

        # Compute dense comparison
        if use_conf:
            match = torch.einsum('bci,bcj->bij', \
                    feats1 * conf1.unsqueeze(1), \
                    feats2 * conf2.unsqueeze(1)).squeeze()
        else:
            match = torch.einsum('bci,bcj->bij', feats1.unsqueeze(0), feats2.unsqueeze(0)).squeeze()

        if RETURN_ALLKEYPOINTS:
            dict_to_ret = {'match' : match, 'kp1' : kp1, 'kp2' : kp2, 
                    'conf1' : conf1_inds[:,0:MAX].squeeze(), 'conf2' : conf2_inds[:,0:MAX].squeeze()}
        else:
            dict_to_ret = {'match' : match, 'kp1' : kp1, 'kp2' : kp2, 
                    'conf1' : confim1, 'conf2' : confim2, 
                    'feats1' : feats1, 'feats2' : feats2}

        if RETURN_FEATURES:
            dict_to_ret['featsim1'] = featsim1
            dict_to_ret['featsim2'] = featsim2

        return dict_to_ret
    
    def compare_features(self, pt1, pt2, precomputed_values):
        B = pt1.size(0)
        feats1, conf1 = self.sample_features(precomputed_values['featsim1'], pt1.reshape(B,-1,2))
        feats2, conf2 = self.sample_features(precomputed_values['featsim2'], pt2.reshape(B,-1,2))

        _, C, P = feats1.size()

        feats1 = feats1.view(B,C,P,1)
        feats2 = feats2.view(B,C,P,-1)

        match = (feats1 - feats2).pow(2).sum(dim=1)
        return match

    def forward(self, batch, visualise=False):
        im1 = batch['Im1'].cuda(); im2 = batch['Im2'].cuda()
        vPIm1 = batch['vPatchesIm1'].cuda(); vPIm2 = batch['vPatchesIm2'].cuda()
        invPIm1 = batch['invPatchesIm1'].cuda(); invPIm2 = batch['invPatchesIm2'].cuda()
        vMatch = batch['vMatches'].cuda().squeeze(); invMatch = batch['invMatches'].cuda().squeeze()
        vPoint = batch['vvalidPoint'].cuda().squeeze(); invPoint = batch['invvalidPoint'].cuda().squeeze()

        b, s, _ = vPIm1.size()

        # Get features
        cond1 = self.features.encoder(im1)
        cond2 = self.features.encoder(im2)

        feats1 = self.obtain_descriptors(cond1, cond2)
        feats2 = self.obtain_descriptors(cond2, cond1)

        # Now compare and say that positive examples should match, negative ones shouldn't
        trueFeats1, conf1 = self.sample_features(feats1, vPIm1)
        trueFeats2, conf2 = self.sample_features(feats2, vPIm2)

        negFeats1, negconf1 = self.sample_features(feats1, invPIm1)
        negFeats2, negconf2 = self.sample_features(feats2, invPIm2)

        # Now the loss: 
        trueMatch, _ = self.decision(trueFeats1, trueFeats2)
        negMatch, _ = self.decision(negFeats1, negFeats2)

        N1 = 32; N2 = 512*32
        if not('train_end2end' in self.opts and self.opts.train_end2end):
            with torch.no_grad():
                # Now compare to features in the first image to see how often they match
                predMatch = self.decision(trueFeats1[:,:,:N1], negFeats2[:,:,:N2], compAll=True)
        else:
            # Now compare to features in the first image to see how often they match
            predMatch = self.decision(trueFeats1[:,:,:N1], negFeats2[:,:,:N2], compAll=True)

        confNeg = negconf1 * negconf2
        confPos = conf1 * conf2

        trueLoss = self.loss(trueMatch, vMatch, vPoint, conf=1, complete=True)
        negLoss = self.loss(negMatch, invMatch, invPoint, conf=1, M = 1+trueMatch.detach()[:,None,:], negative=True)
        completeLoss = self.loss(negMatch, invMatch, invPoint, conf=1, complete=True, negative=True)

        regConfLoss = self.loss.compute_prob_errs(conf1[:,:N1], predMatch[0])
        sampler_mask = batch['mask12'].cuda()

        # Get predicted confidence
        if self.opts.train_end2end:
            conf1Im = self.regress_conf(feats1.detach())
            conf2Im = self.regress_conf(feats2.detach())
            match1Im = self.binary_class(feats1)
        else:
            conf1Im = self.regress_conf(feats1.detach())
            conf2Im = self.regress_conf(feats2.detach())
            match1Im = self.binary_class(feats1.detach())

        bceConfLoss = nn.BCELoss()(match1Im, sampler_mask)

        top1, topk = self.stats(negMatch, trueMatch, invMatch, invPoint)

        loss = {'Total Loss' : trueLoss + negLoss + 1*completeLoss + (regConfLoss  + bceConfLoss) * 0.1, 
                'TrueLoss' : trueLoss, 'NegLoss' : negLoss, 'ConfLoss' : regConfLoss, 'BCEConfLoss' : bceConfLoss,
                'ComLoss':completeLoss, 'Top1': top1, 'Top5':topk}

        
            # Visualise the predicted keypoints in order
        if os.environ['DEBUG'] and visualise:
            imcat = utils.visualise_correspondences(vPIm1, vPIm2, batch['Im1'], batch['Im2'])
            torchvision.utils.save_image(imcat, './temp/truecorr.png')
            torchvision.utils.save_image(batch['Im1'] * 0.5 + 0.5, './temp/im1.png')
            torchvision.utils.save_image(batch['Im2'] * 0.5 + 0.5, './temp/im2.png')
            if 'Im12' in batch.keys():
                torchvision.utils.save_image(batch['Im12'] * 0.5 + 0.5, './temp/transImg.png')
            torchvision.utils.save_image(batch['mask12'], './temp/mask.png')

        if visualise:
            images = {
                'OrigImage' : im1 * 0.5 + 0.5, 'OtherImage' : im2 * 0.5 + 0.5,
                'Conf1' : conf1Im, 'Conf2' : conf2Im,
                'MatchIm1' : match1Im, 
            }
            if 'Im12' in batch.keys():
                images['TransImg12'] = batch['Im12']
            return loss, images

        return loss, {}

class MultiResolutionFPNMixConfClassBroadcast(nn.Module):
    def __init__(self, opts=None, input_nc=3, output_nc=3, nf=32):
        super().__init__()

        self.opts = opts

        # Use a pretrained ResNet for the initial features
        if opts.network == 'resnet50' or True:
            self.features = smp.Unet(opts.network, decoder_channels=(256,256,128,128,64), 
                                        encoder_depth=5, classes=64, encoder_weights='imagenet', full=True)
        else:
            self.features = smp.Unet('resnet18', decoder_channels=(256,128,128,64,64), 
                                        encoder_depth=5, classes=64, encoder_weights='imagenet')
        # self.features = smp.FPN('resnet18', decoder_segmentation_channels=128, encoder_depth=5, classes=128, decoder_dropout=0., encoder_weights='imagenet')

        def create_mlp(input_nc, nf, nl=2):
            layers = []
            for i in range(0, nl):
                if i == 0:
                    in_nc = input_nc
                else:
                    in_nc = nf
                
                out_nc = nf

                layers += [nn.Conv2d(in_nc, out_nc, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(out_nc)]

            layers += [nn.Sigmoid()]

            return nn.Sequential(*layers)

        nf = 64
        self.regress_conf = create_mlp(nf, 1, nl=3)
        self.binary_class = create_mlp(nf, 1, nl=3)

        if opts.network == 'resnet50':
            self.fp1L = nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0,bias=True)
            self.fp2L = nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0,bias=True)
            self.fp1S = nn.Conv2d(1024,128,kernel_size=1,stride=1,padding=0,bias=True)
            self.fp2S = nn.Conv2d(1024,128,kernel_size=1,stride=1,padding=0,bias=True)
        else:
            self.fp1L = nn.Conv2d(320,256,kernel_size=1,stride=1,padding=0,bias=True)
            self.fp2L = nn.Conv2d(320,256,kernel_size=1,stride=1,padding=0,bias=True)
            self.fp1S = nn.Conv2d(112,128,kernel_size=1,stride=1,padding=0,bias=True)
            self.fp2S = nn.Conv2d(112,128,kernel_size=1,stride=1,padding=0,bias=True)

        # Switch to instance norm to see if this fixes things
        if opts.norm_class == 'instance_norm':
            utils.convert_batchnorm(self, opts.norm_class)

        # Optimizer
        self.optimizerG = self.get_optimizer()
        self.plateauscheduler = scheduler.Scheduler(opts, self.optimizerG)
        self.loss = losses.HingeLoss(self.opts.hinge_samples)
        self.stats = losses.Stats()

        self.accumulate_images = False
        self.accumulate_batches = {}

        # Grid
        self.W = 6; self.H = 6
        ys, xs = torch.meshgrid(torch.linspace(-1,1,self.H), torch.linspace(-1,1,self.W))
        self.register_buffer('grid', torch.cat((xs.unsqueeze(2), ys.unsqueeze(2)), 2).unsqueeze(0))

        
        train_layers = ['layer2', 'layer3', 'layer4']
        for name, param in self.features.encoder.named_parameters():
            if not('train_encoder' in self.opts) or (not self.opts.train_encoder) or (not name.split('.')[0] in train_layers):
                param.requires_grad = False


    def get_optimizer(self):
        optimizer = [
            {'params' :   [prm for prm in self.parameters() if prm.requires_grad]}
        ]

        return optim.Adam(optimizer, lr=self.opts.lr, weight_decay=0.0005)

    def load_checkpoint(self, opts):
        if opts.resume:
            if os.path.isdir(opts.model_epoch_path[:-19]) and len(os.listdir(opts.model_epoch_path[:-19]))>0:
                model_path = Path(opts.model_epoch_path[:-19])
                model_path = [x[0] for x in sorted([(fn, os.stat(fn)) for fn in model_path.iterdir()], key = lambda x: x[1].st_ctime)][0]
                c_epoch = torch.load(model_path)['epoch']+1
            else:
                return 0, 0
        else:
            c_epoch = opts.continue_epoch
        print("Continuing from epoch %d..." % c_epoch)

        past_state = torch.load(opts.model_epoch_path % str(c_epoch - 1))
        self.load_state_dict(
            torch.load(opts.model_epoch_path % str(c_epoch - 1))['state_dict'])
        self.optimizerG.load_state_dict(
            torch.load(opts.model_epoch_path % str(c_epoch - 1))['optimizer'])
        self.plateauscheduler.load_state_dict(
            torch.load(opts.model_epoch_path % str(c_epoch - 1))['scheduler'])

        opts = past_state['opts']
        opts.continue_epoch = c_epoch
        opts.epoch = c_epoch
        self.plateauscheduler.update_opts(opts)
        return opts, c_epoch

    def step_plateau(self, loss):
        return
        # self.plateauscheduler.step(loss)

    def step(self, loader, visualise=False, val=False):
        if val:
            loss, imgs = self.forward(loader.next(), visualise)
            return loss, imgs

        t_loss = 0
        self.optimizerG.zero_grad()
        for i in range(0, 1):
            loss, imgs = self.forward(loader.next(), visualise)
            t_loss += loss['Total Loss']
            (loss['Total Loss'] / 1.).backward()

        self.optimizerG.step()
        loss['Total Loss'] = t_loss / float(i+1)
        return loss, imgs

    def get_checkpoint(self, epoch):
        return {'state_dict' : self.state_dict(),
                'optimizer' : self.optimizerG.state_dict(),
                'epoch' : epoch,
                'opts' : self.opts,
                'scheduler' : self.plateauscheduler.state_dict()}

    def obtain_descriptors(self, cond1, cond2):
        cond = cond1[:]

        cond_fp1L = self.fp1L(cond1[-1]); cond_fp2L = self.fp2L(cond2[-1])
        cond_fp1S = self.fp1S(cond1[-2]); cond_fp2S = self.fp2S(cond2[-2])

        attention_map = torch.einsum('bcij,bckl->bijkl', cond_fp1L, cond_fp2L)
        B,W1,H1,W2,H2 = attention_map.shape
        attention_mapL = nn.Softmax(dim=3)(attention_map.view(B,W1,H1,W2*H2))

        attention_map = torch.einsum('bcij,bckl->bijkl', cond_fp1S, cond_fp2S)
        B,W1,H1,W2,H2 = attention_map.shape
        attention_mapS = nn.Softmax(dim=3)(attention_map.view(B,W1,H1,W2*H2))

        # assert(self.opts.network == 'resnet50')
        B,C,W,H = cond2[-1].shape
        cond2_transL = torch.einsum('bijp,bcp->bcij', attention_mapL, cond2[-1].view(B,C,-1))
        cond[-1] = torch.cat((cond1[-1][:,:,:,:], cond2_transL),1)

        B,C,W,H = cond2[-2].shape
        cond2_transS = torch.einsum('bijp,bcp->bcij', attention_mapS, cond2[-2].view(B,C,-1))
        cond[-2] = torch.cat((cond1[-2][:,:,:,:], cond2_transS),1)

        imFeats = self.features.segmentation_head(self.features.decoder(*cond))
        return imFeats

    def sample_features(self, feat, sampler):
        B,C,H,W = feat.size()
        B,S,_ = sampler.size()
        sampler = sampler.unsqueeze(1)

        if 'train_end2end' in self.opts and self.opts.train_end2end:
            conf_feat = feat
        else:
            conf_feat = feat.detach()
        confidence = self.regress_conf(conf_feat)
        confidence = F.grid_sample(confidence, sampler)

        sFeat = F.grid_sample(feat, sampler)
        sFeat = architectures.NormBlock(dim=1)(sFeat).view(B,-1,S)

        return sFeat, confidence.squeeze()


    def decision(self, feat1, feat2, compAll=False):
        if compAll:
            feat1 = feat1.unsqueeze(3)
            feat2 = feat2.unsqueeze(2)

        if self.opts.losses == 'hinge':
            # Compare all features to all others -- looking at different scales
            dist = (feat1[:, :] - feat2[:,:]).pow(2).sum(dim=1)

            return dist, 0

        elif self.opts.losses == 'bce':
            dist = (feat1 * feat2).sum(dim=1) * 0.5 + 0.5
            return dist, 0

    def run_feature(self, im1, im2, resolution='medium', 
                            keypoints=None, sz1=(128,128), sz2=(128,128), 
                            r_im1=(1., 1.),r_im2=(1., 1.),
                            factor=1,
                            MAX=4096, use_conf=True, T=0.0,
                            return_4dtensor=False,
                            RETURN_ALLKEYPOINTS=False, RETURN_FEATURES=False,):
        # raise Exception("Not implemented")
        # Obtain both the keypoint locations and corresponding features
        image1, jitImage1 = im1
        image2, jitImage2 = im2

        def sampler_values(size, rx, ry):
            _, _, H, W = size
            # Now create a grid of locations from which to obtain a bunch of patches
            ys, xs = torch.meshgrid(torch.linspace(-1,1,H) * ry, torch.linspace(-1,1,W) * rx)
            origSampler = torch.cat((xs.unsqueeze(2), 
                                        ys.unsqueeze(2)), 2).unsqueeze(0)
            origSampler = origSampler.view(1,-1,2)

            return origSampler

        # Get features
        cond1 = self.features.encoder(image1)
        cond2 = self.features.encoder(image2)

        featsim1 = self.obtain_descriptors(cond1, cond2)
        featsim2 = self.obtain_descriptors(cond2, cond1)

        confim1 = self.regress_conf(featsim1)
        confim2 = self.regress_conf(featsim2)

        maskim1 = self.binary_class(featsim1)
        maskim2 = self.binary_class(featsim2)
        

        if not(keypoints is None):
            B,C,_,_ = featsim1.size()
            feats1, conf1 = self.sample_features(featsim1, keypoints[0].view(B,-1,2))
            feats2, conf2 = self.sample_features(featsim2, keypoints[1].view(B,-1,2))

            kp1 = keypoints[0]; kp2 = keypoints[1]

            
        else:
            B = featsim1.size(0)

            origSample1 = sampler_values((1,1,sz1[1],sz1[0]), r_im1[0], r_im1[1])
            origSample1 = origSample1.to(featsim1.device)

            origSample2 = sampler_values((1,1,sz2[1],sz2[0]), r_im2[0], r_im2[1])
            origSample2 = origSample2.to(featsim1.device)

            feats1, conf1 = self.sample_features(featsim1, origSample1.repeat(B,1,1))
            feats2, conf2 = self.sample_features(featsim2, origSample2.repeat(B,1,1))

            if factor > 1:
                mask = torch.zeros(conf1.size()).to(conf1.device)
                mask.view(1,sz1[1],sz1[0])[:,0::factor,0::factor] = 1
                conf1 = conf1 * mask

            kp1 = origSample1.clone().repeat(B,1,1)
            kp2 = origSample2.clone().repeat(B,1,1)

        m_s1 = F.grid_sample(maskim1, kp1.unsqueeze(1)).squeeze()
        m_s2 = F.grid_sample(maskim2, kp2.unsqueeze(1)).squeeze()

        # conf1 = conf1 * (m_s1 > 0.5).float()
        # conf2 = conf2 * (m_s2 > 0.5).float()

        # Only keep MAX most likely keypoints in each image
        if not return_4dtensor:
            conf1, conf1_inds = conf1.view(B,-1).sort(descending=True, dim=1)
            conf2, conf2_inds = conf2.view(B,-1).sort(descending=True, dim=1)

        else:
            conf1 = conf1.view(B, -1)
            conf2 = conf2.view(B, -1)

        if not RETURN_ALLKEYPOINTS and not return_4dtensor:
            kp1 = kp1.gather(index=conf1_inds[:,0:MAX].unsqueeze(2).expand(-1,-1,2), dim=1).squeeze()
            kp2 = kp2.gather(index=conf2_inds[:,0:MAX].unsqueeze(2).expand(-1,-1,2), dim=1).squeeze()
        else:
            kp1 = kp1.squeeze()
            kp2 = kp2.squeeze()

        if not return_4dtensor:
            feats1 = feats1.gather(index=conf1_inds[:,0:MAX].unsqueeze(1).expand(-1,64,-1), dim=2).squeeze()
            feats2 = feats2.gather(index=conf2_inds[:,0:MAX].unsqueeze(1).expand(-1,64,-1), dim=2).squeeze()

            conf1 = conf1[:,0:MAX]
            conf2 = conf2[:,0:MAX]
        else:
            feats1 = feats1.squeeze()
            feats2 = feats2.squeeze()

        # Compute dense comparison
        if use_conf:
            match = torch.einsum('bci,bcj->bij', \
                    feats1 * conf1.unsqueeze(1), \
                    feats2 * conf2.unsqueeze(1)).squeeze()
        else:
            print("Warning: not using conf!")
            match = torch.einsum('bci,bcj->bij', feats1.unsqueeze(0), feats2.unsqueeze(0)).squeeze()

        if return_4dtensor:
            H1, W1 = sz1
            H2, W2 = sz2
            match = match.view(H1,W1,H2,W2)

        if RETURN_ALLKEYPOINTS and not return_4dtensor:
            dict_to_ret = {'match' : match, 'kp1' : kp1, 'kp2' : kp2, 
                    'conf1' : conf1_inds[:,0:MAX].squeeze(), 'conf2' : conf2_inds[:,0:MAX].squeeze()}
        elif return_4dtensor:
            dict_to_ret = {'match' : match, 'kp1' : kp1, 'kp2' : kp2, 'conf1' : None, 'conf2' : None}

        else:
            dict_to_ret = {'match' : match, 'kp1' : kp1, 'kp2' : kp2, 
                    'conf1' : confim1, 'conf2' : confim2, 
                    'feats1' : feats1, 'feats2' : feats2}

        if RETURN_FEATURES:
            dict_to_ret['featsim1'] = featsim1
            dict_to_ret['featsim2'] = featsim2

        return dict_to_ret
    
    def compare_features(self, pt1, pt2, precomputed_values):
        B = pt1.size(0)
        feats1, conf1 = self.sample_features(precomputed_values['featsim1'], pt1.reshape(B,-1,2))
        feats2, conf2 = self.sample_features(precomputed_values['featsim2'], pt2.reshape(B,-1,2))

        _, C, P = feats1.size()

        feats1 = feats1.view(B,C,P,1)
        feats2 = feats2.view(B,C,P,-1)

        match = (feats1 - feats2).pow(2).sum(dim=1)
        return match

    def forward(self, batch, visualise=False):
        im1 = batch['Im1'].cuda(); im2 = batch['Im2'].cuda()
        vPIm1 = batch['vPatchesIm1'].cuda(); vPIm2 = batch['vPatchesIm2'].cuda()
        invPIm1 = batch['invPatchesIm1'].cuda(); invPIm2 = batch['invPatchesIm2'].cuda()
        vMatch = batch['vMatches'].cuda().squeeze(); invMatch = batch['invMatches'].cuda().squeeze()
        vPoint = batch['vvalidPoint'].cuda().squeeze(); invPoint = batch['invvalidPoint'].cuda().squeeze()

        b, s, _ = vPIm1.size()

        # Get features
        cond1 = self.features.encoder(im1)
        cond2 = self.features.encoder(im2)

        feats1 = self.obtain_descriptors(cond1, cond2)
        feats2 = self.obtain_descriptors(cond2, cond1)

        # Now compare and say that positive examples should match, negative ones shouldn't
        trueFeats1, conf1 = self.sample_features(feats1, vPIm1)
        trueFeats2, conf2 = self.sample_features(feats2, vPIm2)

        negFeats1, negconf1 = self.sample_features(feats1, invPIm1)
        negFeats2, negconf2 = self.sample_features(feats2, invPIm2)

        # Now the loss: 
        trueMatch, _ = self.decision(trueFeats1, trueFeats2)
        negMatch, _ = self.decision(negFeats1, negFeats2)

        N1 = 32; N2 = 512*32
        if not('train_end2end' in self.opts and self.opts.train_end2end):
            with torch.no_grad():
                # Now compare to features in the first image to see how often they match
                predMatch = self.decision(trueFeats1[:,:,:N1], negFeats2[:,:,:N2], compAll=True)
        else:
            # Now compare to features in the first image to see how often they match
            predMatch = self.decision(trueFeats1[:,:,:N1], negFeats2[:,:,:N2], compAll=True)

        confNeg = negconf1 * negconf2
        confPos = conf1 * conf2

        trueLoss = self.loss(trueMatch, vMatch, vPoint, conf=1, complete=True)
        negLoss = self.loss(negMatch, invMatch, invPoint, conf=1, M = 1+trueMatch.detach()[:,None,:], negative=True)
        completeLoss = self.loss(negMatch, invMatch, invPoint, conf=1, complete=True, negative=True)

        regConfLoss = self.loss.compute_prob_errs(conf1[:,:N1], predMatch[0])
        sampler_mask = batch['mask12'].cuda()

        # Get predicted confidence
        if self.opts.train_end2end:
            conf1Im = self.regress_conf(feats1.detach())
            conf2Im = self.regress_conf(feats2.detach())
            match1Im = self.binary_class(feats1)
        else:
            conf1Im = self.regress_conf(feats1.detach())
            conf2Im = self.regress_conf(feats2.detach())
            match1Im = self.binary_class(feats1.detach())

        bceConfLoss = nn.BCELoss()(match1Im, sampler_mask)

        top1, topk = self.stats(negMatch, trueMatch, invMatch, invPoint)

        loss = {'Total Loss' : trueLoss + negLoss + 1*completeLoss + (regConfLoss  + bceConfLoss) * 0.1, 
                'TrueLoss' : trueLoss, 'NegLoss' : negLoss, 'ConfLoss' : regConfLoss, 'BCEConfLoss' : bceConfLoss,
                'ComLoss':completeLoss, 'Top1': top1, 'Top5':topk}

        
            # Visualise the predicted keypoints in order
        if os.environ['DEBUG'] and visualise:
            imcat = utils.visualise_correspondences(vPIm1, vPIm2, batch['Im1'], batch['Im2'])
            torchvision.utils.save_image(imcat, './temp/truecorr.png')
            torchvision.utils.save_image(batch['Im1'] * 0.5 + 0.5, './temp/im1.png')
            torchvision.utils.save_image(batch['Im2'] * 0.5 + 0.5, './temp/im2.png')
            if 'Im12' in batch.keys():
                torchvision.utils.save_image(batch['Im12'] * 0.5 + 0.5, './temp/transImg.png')
            torchvision.utils.save_image(batch['mask12'], './temp/mask.png')

        if visualise:
            images = {
                'OrigImage' : im1 * 0.5 + 0.5, 'OtherImage' : im2 * 0.5 + 0.5,
                'Conf1' : conf1Im, 'Conf2' : conf2Im,
                'MatchIm1' : match1Im, 
            }
            if 'Im12' in batch.keys():
                images['TransImg12'] = batch['Im12']
            return loss, images

        return loss, {}