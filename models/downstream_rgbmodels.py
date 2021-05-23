import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import torch.optim as optim

from utils import scheduler

import glob

import models.unet_model as unet_model
import models.architectures as architectures
import models.multi_resolution as multi_resolution
import models.losses as losses

import os

class StylizeImageConcatenateInputs(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.opts = opts
        self.topK = opts.topk

        feature_model = torch.load(opts.feature_model)
        self.initial_model = multi_resolution.MultiResolutionFPNMixConfClass(opts=feature_model['opts'])
        self.initial_model.load_state_dict(feature_model['state_dict'])

        if opts.use_weights:
            self.params = 4
        else:
            self.params = 3
            
        if opts.use_image:
            num_inputs = self.params * self.topK + 3
        else:
            num_inputs = self.params * self.topK

        if opts.network == 'resnetblocks':
            channels = [num_inputs, opts.ngf, opts.ngf//2,
                        opts.ngf//2, opts.ngf//2, 
                        opts.ngf//2, 3]
            norms = ['identity', 'upsample', 
                     'identity', 'upsample', 
                     'identity', 'upsample']
            self.refine_rgb = architectures.ResNetNoiseBlocks(channels, norms)
        else:
            self.refine_rgb = unet_model.define_G(num_inputs, 3, 32, opts.network)

        if self.opts.use_discriminator:
            self.optimizerG = self.get_optimizer(betas=(opts.beta1, opts.beta2))
            self.netD = losses.DiscriminatorLoss(opts)
            self.optimizerD = torch.optim.Adam(list(self.netD.parameters()), 
                                                lr=opts.lr_d, betas=(opts.beta1, opts.beta2))
        else:
            self.optimizerG = self.get_optimizer()
        self.plateauscheduler = scheduler.Scheduler(opts, self.optimizerG)

    def get_optimizer(self, betas=(0.9,0.999)):
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
            loss, imgs, _ = self.forward(loader.next(), visualise)
            return loss, imgs

        self.optimizerG.zero_grad()
        if self.opts.use_discriminator:
            t_loss, imgs, batch = self.forward(loader.next(), visualise=True)
            if self.opts.use_mask:
                print("Using mask")
                mask = (batch['sampler12'].to(imgs['Im1'].device).min(dim=1, keepdim=True)[0] < -1).float()
                g_loss = self.netD.run_generator_one_step(imgs['PredIm'] * mask, 
                                                      imgs['Im1'] * mask)
            else:
                g_loss = self.netD.run_generator_one_step(imgs['PredIm'], 
                                                      imgs['Im1'])

            (g_loss['Total Loss'] + t_loss['Total Loss']).mean().backward()
            self.optimizerG.step()

            self.optimizerD.zero_grad()
            if self.opts.use_mask:
                d_loss = self.netD.run_discriminator_one_step(imgs['PredIm'] * mask,
                                                          imgs['Im1'] * mask)
            else:
                d_loss = self.netD.run_discriminator_one_step(imgs['PredIm'],
                                                          imgs['Im1'])
            (d_loss['Total Loss']).mean().backward()
            self.optimizerD.step()

            g_loss.pop("Total Loss")
            d_loss.pop("Total Loss")
            t_loss.update(g_loss)
            t_loss.update(d_loss)
        else:
            t_loss, imgs, _ = self.forward(loader.next(), visualise)
            (t_loss['Total Loss'] / 1.).backward()

            self.optimizerG.step()

        if visualise:
            return t_loss, imgs
        else:
            return t_loss, {}

    def get_checkpoint(self, epoch):
        return {'state_dict' : self.state_dict(),
                'optimizer' : self.optimizerG.state_dict(),
                'epoch' : epoch,
                'opts' : self.opts,
                'scheduler' : self.plateauscheduler.state_dict()}

    def best_matches(self, sim):
        nn12 = torch.max(sim, dim=1)[1]
        nn21 = torch.max(sim, dim=0)[1]

        ids1 = torch.arange(0, sim.shape[0]).to(nn12.device)
        mask = (ids1 == nn21[nn12])
        matches = torch.stack([ids1[mask], nn12[mask]])

        preds = sim[ids1[mask], nn12[mask]]
        res, ids = preds.sort()

        matches = matches[:,ids]
        return matches.t()

    def obtain_matches(self, im1, im2):
        B = im1.size(0)
        results = self.initial_model.run_feature((im1, None), (im2, None), use_conf=True,
                                sz1=(self.opts.WCoarse,self.opts.WCoarse),
                                sz2=(self.opts.WCoarse,self.opts.WCoarse), 
                                MAX=self.opts.WCoarse * self.opts.WCoarse)

        kp1 = results['kp1']
        kp2 = results['kp2']
        matches = results['match']

        im1_small = F.upsample(size=(self.opts.WCoarse, self.opts.WCoarse), input=im1)
        im2_small = F.upsample(size=(self.opts.WCoarse, self.opts.WCoarse), input=im2)

        gen_images = torch.zeros((B,3*self.topK,self.opts.WCoarse,self.opts.WCoarse)).to(im1.device)
        gen_weights = torch.zeros((B,self.topK,self.opts.WCoarse,self.opts.WCoarse)).to(im1.device)

        weights, index_im2 = matches.topk(dim=2, k=self.topK)
        
        xys1 = kp1 / 2 + 0.5
        xys2 = kp2.gather(index=index_im2.view(B,-1,1).repeat(1,1,2), dim=1).view(B,-1,self.topK,2) / 2 + 0.5

        xys1 = (xys1 * (self.opts.WCoarse)).long().clamp(min=0, max=self.opts.WCoarse-1)
        xys2 = (xys2 * (self.opts.WCoarse)).long().clamp(min=0, max=self.opts.WCoarse-1)

        for b in range(0, B):
            for k in range(0, self.topK):
                gen_images[b,3*k:3*(k+1),xys1[b,:,1],xys1[b,:,0]] = \
                        im2_small[b,:,xys2[b,:,k,1],xys2[b,:,k,0]]

                gen_weights[b,k,xys1[b,:,1],xys1[b,:,0]] = weights[b,:,k]

        if not(self.opts.network == 'resnetblocks'):
            gen_images = F.upsample(size=(self.opts.W,self.opts.W), input=gen_images)
            gen_weights = F.upsample(size=(self.opts.W,self.opts.W), input=gen_weights)

        return gen_images, gen_weights

    def forward(self, batch, visualise=False):
        im1 = batch['Im1'].cuda(); im2 = batch['Im2'].cuda(); sampler12 = batch['sampler12'].cuda()
        B = im1.size(0)

        # First find most similar points in first image to second, along with weights and corresponding
        # colour. Concatenate to btain inputs
        with torch.no_grad():
            im_sampled, gen_weights = self.obtain_matches(im1, im2)

        if self.opts.use_weights:
            print("Using weights")
            inputs = torch.cat((im_sampled, gen_weights),1)
        else:
            inputs = im_sampled

        if self.opts.use_image:
            inputs = torch.cat((im1, inputs), 1)

        pred_image = F.tanh(self.refine_rgb(inputs))

        if self.opts.network == 'resnetblocks':
            pred_image = F.upsample(size=(self.opts.W,self.opts.W), mode='nearest', input=pred_image)

        # Then sample in order to find best matches
        gt_pred_image = F.grid_sample(im2, sampler12.permute(0,2,3,1))

        # And use resampled as GT in order to find loss
        mask = (sampler12.abs().max(dim=1, keepdim=True)[0] < 1).float()
        err = losses.L1Mask()(pred_image, gt_pred_image, mask=mask)

        alllosses = {'L1' : err, 'Total Loss' : err * self.opts.lambda1}
        images = {}
        if visualise:
            if self.topK > 1:
                _,_,HS,WS = im_sampled.size()
                im_sampled = (im_sampled.view(B,self.topK,3,HS,WS)) * gen_weights.unsqueeze(2)
                im_sampled = im_sampled.sum(dim=1) / gen_weights.unsqueeze(2).sum(dim=1)
            images['Im1'] = im1 * 0.5 + 0.5; images['PredIm'] = pred_image * 0.5 + 0.5
            images['Im2'] = im2 * 0.5 + 0.5; images['GTIm'] = gt_pred_image * 0.5 + 0.5

            images['Im2Sampled'] = im_sampled * 0.5 + 0.5

        return alllosses, images, batch

class StylizeImageConcatenateInputs2Step(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.opts = opts
        self.topK = opts.topk

        feature_model = torch.load(glob.glob(opts.feature_model)[0])
        self.initial_model = multi_resolution.MultiResolutionFPNMixConfClassBroadcast(opts=feature_model['opts'])
        self.initial_model.load_state_dict(feature_model['state_dict'])

        if opts.use_weights:
            self.params = 4
        else:
            self.params = 3
            
        if opts.use_image:
            num_inputs = self.params * self.topK + 3
        else:
            num_inputs = self.params * self.topK

        if opts.network == 'resnetblocks':
            channels = [num_inputs, opts.ngf, opts.ngf//2,
                        opts.ngf//2, opts.ngf//2, 
                        opts.ngf//2, 3]
            norms = ['identity', 'upsample', 
                     'identity', 'upsample', 
                     'identity', 'upsample']
            self.refine_rgb_l1 = architectures.ResNetNoiseBlocks(channels, norms)

            channels[0] = 3+1
            self.refine_rgb_disc = architectures.ResNetNoiseBlocks(channels, norms)
        else:
            self.refine_rgb_l1 = unet_model.define_G(num_inputs, 3, 32, opts.network)

            channels = [num_inputs, opts.ngf, opts.ngf//2,
                        opts.ngf//2, opts.ngf//2, 
                        opts.ngf//2, 3]
            norms = ['downsample', 'identity', 
                     'downsample', 'upsample', 
                     'identity', 'upsample']

            channels[0] = 3+1
            self.refine_rgb_disc = architectures.ResNetNoiseBlocks(channels, norms)

        if self.opts.use_discriminator:
            self.optimizerG = self.get_optimizer(betas=(opts.beta1, opts.beta2), lr=opts.lr/2)
            self.netD = losses.DiscriminatorLoss(opts)
            self.optimizerD = torch.optim.Adam(list(self.netD.parameters()), 
                                                lr=opts.lr*2, betas=(opts.beta1, opts.beta2))
        else:
            self.optimizerG = self.get_optimizer()
        self.plateauscheduler = scheduler.Scheduler(opts, self.optimizerG)

    def get_optimizer(self, betas=(0.9,0.999), lr=1e-4):
        optimizer = [
            {'params' :   [prm for prm in self.parameters() if prm.requires_grad]}
        ]

        return optim.Adam(optimizer, lr=lr)

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
            loss, imgs, _ = self.forward(loader.next(), visualise)
            return loss, imgs

        self.optimizerG.zero_grad()
        if self.opts.use_discriminator:
            t_loss, imgs, batch = self.forward(loader.next(), visualise=True)
            if self.opts.use_mask:
                print("Using mask")
                mask = (batch['sampler12'].to(imgs['Im1'].device).min(dim=1, keepdim=True)[0] < -1).float()
                g_loss = self.netD.run_generator_one_step(imgs['PredIm'] * mask, 
                                                      imgs['Im1'] * mask)
            else:
                g_loss = self.netD.run_generator_one_step(imgs['PredIm'], 
                                                      imgs['Im1'])

            (g_loss['Total Loss'] * self.opts.lambda2 + t_loss['Total Loss']).mean().backward()
            self.optimizerG.step()

            self.optimizerD.zero_grad()
            if self.opts.use_mask:
                d_loss = self.netD.run_discriminator_one_step(imgs['PredIm'] * mask,
                                                          imgs['Im1'] * mask)
            else:
                d_loss = self.netD.run_discriminator_one_step(imgs['PredIm'],
                                                          imgs['Im1'])
            (d_loss['Total Loss']).mean().backward()
            self.optimizerD.step()

            g_loss.pop("Total Loss")
            d_loss.pop("Total Loss")
            t_loss.update(g_loss)
            t_loss.update(d_loss)
        else:
            t_loss, imgs, _ = self.forward(loader.next(), visualise)
            (t_loss['Total Loss'] / 1.).backward()

            self.optimizerG.step()

        if visualise:
            return t_loss, imgs
        else:
            return t_loss, {}

    def get_checkpoint(self, epoch):
        return {'state_dict' : self.state_dict(),
                'optimizer' : self.optimizerG.state_dict(),
                'epoch' : epoch,
                'opts' : self.opts,
                'scheduler' : self.plateauscheduler.state_dict()}

    def best_matches(self, sim):
        nn12 = torch.max(sim, dim=1)[1]
        nn21 = torch.max(sim, dim=0)[1]

        ids1 = torch.arange(0, sim.shape[0]).to(nn12.device)
        mask = (ids1 == nn21[nn12])
        matches = torch.stack([ids1[mask], nn12[mask]])

        preds = sim[ids1[mask], nn12[mask]]
        res, ids = preds.sort()

        matches = matches[:,ids]
        return matches.t()

    def obtain_matches(self, im1, im2):
        B = im1.size(0)
        results = self.initial_model.run_feature((im1, None), (im2, None), use_conf=True,
                                sz1=(self.opts.WCoarse,self.opts.WCoarse),
                                sz2=(self.opts.WCoarse,self.opts.WCoarse), 
                                MAX=self.opts.WCoarse * self.opts.WCoarse)

        kp1 = results['kp1']
        kp2 = results['kp2']
        matches = results['match']

        if len(kp1.shape) < 3:
            kp1 = kp1.unsqueeze(0)
            kp2 = kp2.unsqueeze(0)
            matches = matches.unsqueeze(0)

        im1_small = F.upsample(size=(self.opts.WCoarse, self.opts.WCoarse), input=im1)
        im2_small = F.upsample(size=(self.opts.WCoarse, self.opts.WCoarse), input=im2)

        gen_images = torch.zeros((B,3*self.topK,self.opts.WCoarse,self.opts.WCoarse)).to(im1.device)
        gen_weights = torch.zeros((B,self.topK,self.opts.WCoarse,self.opts.WCoarse)).to(im1.device)

        weights, index_im2 = matches.topk(dim=2, k=self.topK)
        
        xys1 = kp1 / 2 + 0.5
        xys2 = kp2.gather(index=index_im2.view(B,-1,1).repeat(1,1,2), dim=1).view(B,-1,self.topK,2) / 2 + 0.5

        xys1 = (xys1 * (self.opts.WCoarse)).long().clamp(min=0, max=self.opts.WCoarse-1)
        xys2 = (xys2 * (self.opts.WCoarse)).long().clamp(min=0, max=self.opts.WCoarse-1)

        for b in range(0, B):
            for k in range(0, self.topK):
                gen_images[b,3*k:3*(k+1),xys1[b,:,1],xys1[b,:,0]] = \
                        im2_small[b,:,xys2[b,:,k,1],xys2[b,:,k,0]]

                gen_weights[b,k,xys1[b,:,1],xys1[b,:,0]] = weights[b,:,k]

        if not(self.opts.network == 'resnetblocks'):
            gen_images = F.upsample(size=(self.opts.W,self.opts.W), input=gen_images)
            gen_weights = F.upsample(size=(self.opts.W,self.opts.W), input=gen_weights)

        return gen_images, gen_weights

    def forward(self, batch, visualise=False):
        im1 = batch['Im1'].cuda(); im2 = batch['Im2'].cuda(); sampler12 = batch['sampler12'].cuda()
        B = im1.size(0)

        # First find most similar points in first image to second, along with weights and corresponding
        # colour. Concatenate to btain inputs
        with torch.no_grad():
            im_sampled, gen_weights = self.obtain_matches(im1, im2)

        if self.opts.use_weights:
            print("Using weights")
            inputs = torch.cat((im_sampled, gen_weights),1)
        else:
            inputs = im_sampled

        if self.opts.use_image:
            inputs = torch.cat((im1, inputs), 1)

        pred_image_l1 = F.tanh(self.refine_rgb_l1(inputs))

        if self.opts.network == 'resnetblocks':
            pred_image_l1 = F.upsample(size=(self.opts.W,self.opts.W), mode='nearest', input=pred_image_l1)

        # Then sample in order to find best matches
        gt_pred_image = F.grid_sample(im2, sampler12.permute(0,2,3,1))

        # And use resampled as GT in order to find loss
        mask = (sampler12.abs().max(dim=1, keepdim=True)[0] < 1).float()
        err_l1 = losses.L1Mask()(pred_image_l1, gt_pred_image, mask=mask)

        # Now refine with a discriminator
        inputs = torch.cat((pred_image_l1.detach(), gen_weights), 1)
        pred_image_disc = F.tanh(self.refine_rgb_disc(inputs))
        err_disc = losses.L1Mask()(pred_image_disc, gt_pred_image, mask=mask)

        alllosses = {'L1' : err_l1, 'L1 Disc' : err_disc, 'Total Loss' : err_l1  + err_disc }
        images = {}
        if visualise:
            if self.topK > 1:
                _,_,HS,WS = im_sampled.size()
                im_sampled = (im_sampled.view(B,self.topK,3,HS,WS)) * gen_weights.unsqueeze(2)
                im_sampled = im_sampled.sum(dim=1) / gen_weights.unsqueeze(2).sum(dim=1)
            images['Im1'] = im1 * 0.5 + 0.5; images['PredImL1'] = pred_image_l1 * 0.5 + 0.5
            images['Im2'] = im2 * 0.5 + 0.5; images['GTIm'] = gt_pred_image * 0.5 + 0.5
            images['PredIm'] = pred_image_disc * 0.5 + 0.5

            images['Im2Sampled'] = im_sampled * 0.5 + 0.5

        return alllosses, images, batch
