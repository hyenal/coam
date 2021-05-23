import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import scheduler

import torch.optim as optim
import torchvision.models as models

import models.architectures as architectures
import models.losses as losses

from models.unetstyle_model import UnetStyleGenerator
from models.pix2pix_unetmodel import UnetGenerator

class StyleModel(nn.Module):

    def __init__(self, opts=None, input_nc=3, output_nc=3, nf=32):
        super(StyleModel, self).__init__()

        self.opts = opts

        if self.opts.use_depth:
            self.blocks = UnetStyleGenerator(input_nc + 1, output_nc, 5, ngf=nf)
        else:
            self.blocks = UnetStyleGenerator(input_nc, output_nc, 5, ngf=nf)
        self.image_encoder = models.vgg16(pretrained=False, num_classes=32)

        self.optimizerG = self.get_optimizer()
        self.plateauscheduler = scheduler.Scheduler(opts, self.optimizerG)

    def get_optimizer(self):
        optimizer = [
            {'params' : list(self.blocks.parameters()) + list(self.image_encoder.parameters())}
        ]

        return optim.Adam(optimizer, lr=self.opts.lr)

    def step_plateau(self, loss):
        self.plateauscheduler.step(loss)

    def step(self, batch, visualise=False, val=False):
        if val:
            loss, imgs = self.forward(batch, visualise)
            return loss, imgs

        self.optimizerG.zero_grad()
        loss, imgs = self.forward(batch, visualise)

        loss['Total Loss'].backward()

        self.optimizerG.step()
        return loss, imgs

    def get_checkpoint(self, epoch):
        return {'state_dict' : self.state_dict(), 
                'optimizer' : self.optimizerG.state_dict(), 
                'epoch' : epoch,
                'opts' : self.opts, 
                'scheduler' : self.plateauscheduler.state_dict()}

    def load_checkpoint(self, opts):
        c_epoch = opts.continue_epoch
        print("Continuing from epoch %d..." % opts.continue_epoch)

        past_state = torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))
        self.model.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['state_dict'])
        self.optimizerG.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['optimizer'])
        self.plateauscheduler.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['scheduler'])

        opts = past_state['opts']
        opts.continue_epoch = c_epoch
        opts.epoch = c_epoch
        self.plateauscheduler.update_opts(opts)
        return opts

    def forward(self, batch, visualise=False):
        inpImg = batch['TransformedImg'].cuda()
        jitFeatImg = batch['Jit1'].cuda()
        featImg = batch['Im1'].cuda()
        mask = batch['Mask'].cuda()

        feat = self.image_encoder(jitFeatImg)

        if self.opts.use_depth:
            depth = batch['Depth1'].cuda()
            pred_img = F.tanh(self.blocks(torch.cat((inpImg, depth), 1), feat))
        else:
            pred_img = F.tanh(self.blocks(inpImg, feat))

        if self.opts.use_mask:
            loss = losses.L1Mask()(pred_img, featImg, mask)
        else:
            loss = nn.L1Loss()(pred_img, featImg)

        if visualise:
            images =  {
                'TransformedImg' : inpImg * 0.5 + 0.5,
                'PredImg' : pred_img *0.5 + 0.5,
                'FeatImg' : featImg * 0.5 + 0.5,
                'JitFeatImg' : jitFeatImg * 0.5 + 0.5,
                'Mask' : mask,
                'OutputImg' : batch['Im2'] * 0.5 + 0.5
            }

            if self.opts.use_depth:
                images['Depth'] = depth

            return {'Total Loss' : loss}, images

        return {'Total Loss' : loss}, {}

class ConcatStyleModel(nn.Module):

    def __init__(self, opts=None, input_nc=3, output_nc=3, nf=16):
        super(ConcatStyleModel, self).__init__()

        self.opts = opts
        if self.opts.use_depth:
            self.blocks = UnetGenerator(input_nc + 32 + 1, output_nc, 5, ngf=nf)
        else:
            self.blocks = UnetGenerator(input_nc + 32, output_nc, 5, ngf=nf)
        self.image_encoder = models.vgg11(pretrained=False, num_classes=32)

        self.optimizerG = self.get_optimizer()
        self.plateauscheduler = scheduler.Scheduler(opts, self.optimizerG)

    def get_optimizer(self):
        optimizer = [
            {'params' : list(self.blocks.parameters()) + list(self.image_encoder.parameters())}
        ]

        return optim.Adam(optimizer, lr=self.opts.lr)

    def step_plateau(self, loss):
        self.plateauscheduler.step(loss)

    def step(self, batch, visualise=False, val=False):
        if val:
            loss, imgs = self.forward(batch, visualise)
            return loss, imgs

        self.optimizerG.zero_grad()
        loss, imgs = self.forward(batch, visualise)

        loss['Total Loss'].backward()

        self.optimizerG.step()
        return loss, imgs

    def get_checkpoint(self, epoch):
        return {'state_dict' : self.state_dict(), 
                'optimizer' : self.optimizerG.state_dict(), 
                'epoch' : epoch,
                'opts' : self.opts, 
                'scheduler' : self.plateauscheduler.state_dict()}

    def load_checkpoint(self, opts):
        c_epoch = opts.continue_epoch
        print("Continuing from epoch %d..." % opts.continue_epoch)

        past_state = torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))
        self.model.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['state_dict'])
        self.optimizerG.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['optimizer'])
        self.plateauscheduler.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['scheduler'])

        opts = past_state['opts']
        opts.continue_epoch = c_epoch
        opts.epoch = c_epoch
        self.plateauscheduler.update_opts(opts)
        return opts

    def forward(self, batch, visualise=False):
        inpImg = batch['TransformedImg'].cuda()
        featImg = batch['Im1'].cuda()
        jitFeatImg = batch['Jit1'].cuda()
        mask = batch['Mask'].cuda()

        feat = self.image_encoder(jitFeatImg)

        _, _, H, W = inpImg.size()

        feat = feat.unsqueeze(2).unsqueeze(2)
        feat = feat.repeat(1,1,H,W)

        if self.opts.use_depth:
            depth = batch['Depth1'].cuda()
            pred_img = F.tanh(self.blocks(torch.cat((inpImg, feat, depth), 1)))
        else:
            pred_img = F.tanh(self.blocks(torch.cat((inpImg, feat), 1)))

        if self.opts.use_mask:
            loss = losses.L1Mask()(pred_img, featImg, mask)
        else:
            loss = nn.L1Loss()(pred_img, featImg)

        if visualise:
            images = {
                'TransformedImg' : inpImg * 0.5 + 0.5,
                'PredImg' : pred_img *0.5 + 0.5,
                'FeatImg' : featImg * 0.5 + 0.5,
                'JitFeatImg' : jitFeatImg * 0.5 + 0.5,
                'Mask' : mask,
                'OutputImg' : batch['Im2'] * 0.5 + 0.5
            }

            if self.opts.use_depth:
                images['Depth'] = depth

            return {'Total Loss' : loss}, images

        return {'Total Loss' : loss}, {}

class Stylize3DModel(nn.Module):
    """Stylize 3D Model: basically the idea is that can we swap out features
    between images of the same structure and recreate the original image using
    style techniques.
    """
    def __init__(self, opts=None, input_nc=3, output_nc=3, nf=16):
        super(Stylize3DModel, self).__init__()

        self.opts = opts

        # Model to obtain the features
        self.features3D = nn.Sequential(
            architectures.ResNetBlocks(
                [input_nc, nf, nf, 2*nf, 2*nf, 4*nf, 4*nf, 8*nf, 8*nf],
                ['downsample', 'downsample', 'downsample', 'downsample', 
                'upsample', 'upsample', 'upsample', 'upsample']
            ),
            architectures.NormBlock()
        )

        # Model to stylize the given images
        self.stylize_features = nn.Sequential(
            architectures.ResNetBlocks(
                [8*nf, 8*nf, 4*nf, 4*nf, 2*nf, 2*nf, nf, nf],
                ['downsample', 'downsample', 'downsample', 'identity',
                 'upsample', 'upsample', 'upsample']
            ),
            nn.Conv2d(nf, output_nc, stride=1, padding=1, kernel_size=3),
            nn.Tanh()
        )

        # Now model to obtain the style feature -- use a pretty shallow network plus 
        # a MLP in order to obtain style feature
        convolutional_layers = nn.Sequential(
            architectures.ConvBlock(input_nc, nf), # 128
            architectures.ConvBlock(nf, nf), # 64
            architectures.ConvBlock(nf, nf*2), # 32
            architectures.ConvBlock(nf*2, nf*2), # 16
            architectures.ConvBlock(nf*2, nf*4), # 8 
            architectures.ConvBlock(nf*4, nf*4), # 4
            nn.AdaptiveAvgPool2d((1,1))
        )
        nc = nf * 4
        resize = architectures.ResizeBlock()
        mlp = nn.Sequential(
            architectures.MLPBlock(nc, 128),
            architectures.MLPBlock(128, 128),
            architectures.MLPBlock(128, 128),
            architectures.MLPBlock(128, 128)
        )
        self.style_feature = nn.Sequential(
            convolutional_layers,
            resize,
            mlp
        )

        # Optimizer
        self.optimizerG = self.get_optimizer()
        self.plateauscheduler = scheduler.Scheduler(opts, self.optimizerG)

    def get_checkpoint(self, epoch):
        return {'state_dict' : self.state_dict(),
                'optimizer' : self.optimizerG.state_dict(),
                'epoch' : epoch,
                'opts' : self.opts,
                'scheduler' : self.plateauscheduler.state_dict()}

    def load_checkpoint(self, opts):
        c_epoch = opts.continue_epoch
        print("Continuing from epoch %d..." % opts.continue_epoch)

        past_state = torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))
        self.model.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['state_dict'])
        self.optimizerG.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['optimizer'])
        self.plateauscheduler.load_state_dict(
            torch.load(opts.model_epoch_path % str(opts.continue_epoch - 1))['scheduler'])

        opts = past_state['opts']
        opts.continue_epoch = c_epoch
        opts.epoch = c_epoch
        self.plateauscheduler.update_opts(opts)
        return opts

    def get_optimizer(self):
        optimizer = [
            {'params' : list(self.features3D.parameters()) + list(self.stylize_features.parameters()) + \
                        list(self.style_feature.parameters())}
        ]

        return optim.Adam(optimizer, lr=self.opts.lr)

    def step_plateau(self, loss):
        self.plateauscheduler.step(loss)

    def step(self, batch, visualise=False, val=False):
        if val:
            loss, imgs = self.forward(batch, visualise)
            return loss, imgs

        self.optimizerG.zero_grad()
        loss, imgs = self.forward(batch, visualise)

        loss['Total Loss'].backward()

        self.optimizerG.step()
        return loss, imgs

    def feature_loss(self, feat1, feat2, sample12, mask1):
        pred_feat = F.grid_sample(feat2, sample12)
        err = losses.L1Mask()(feat1, pred_feat, mask1)

        return err

    def forward(self, batch, visualise):
        im1 = batch['Im1'].cuda()
        im2 = batch['Im2'].cuda()
        jit1 = batch['Jit1'].cuda()
        jit2 = batch['Jit2'].cuda()
        mask1 = batch['Mask1'].cuda()
        mask2 = batch['Mask2'].cuda()
        sample12 = batch['Sample12'].cuda()
        sample21 = batch['Sample21'].cuda()

        style_feature1 = self.style_feature(jit1)
        style_feature2 = self.style_feature(jit2)
        print(style_feature1.min(), style_feature1.max())
        print(style_feature2.min(), style_feature2.max())

        features3D1 = self.features3D(im1)
        features3D2 = self.features3D(im2)
        print(features3D1.min(), features3D1.max())
        print(features3D2.min(), features3D2.max())

        # And enforce that the features should match at similar points
        lossFeat12 = self.feature_loss(features3D1, features3D2, sample12, mask1)
        lossFeat21 = self.feature_loss(features3D2, features3D1, sample21, mask2)
        print(lossFeat12.min(), lossFeat12.max())
        print(lossFeat21.min(), lossFeat21.max())

        # Now swap features around based on matching
        warp12 = sample12 # TODO: self.determine_warp(features3D1, features3D2, sample12, mask1)
        warp21 = sample21 # TODO: self.determine_warp(features3D2, features3D1, sample21, mask2)
        swapFeatures1 = F.grid_sample(features3D2, warp12) # Swap ground truth ones; 
        swapFeatures2 = F.grid_sample(features3D1, warp21) # Swap ground truth ones

        print(swapFeatures1.min(), swapFeatures1.max())
        print(swapFeatures2.min(), swapFeatures2.max())

        # Now generate the final images
        genIm1 = self.stylize_features(swapFeatures1)
        genIm2 = self.stylize_features(swapFeatures2)
        print(genIm1.min(), genIm1.max())
        print(genIm2.min(), genIm2.max())

        # And the loss is that these images should match the true one
        lossGenIm1 = losses.L1Mask()(genIm1, im1, mask1)
        lossGenIm2 = losses.L1Mask()(genIm2, im2, mask2)

        tLoss = lossGenIm1 + lossGenIm2
        if not tLoss == tLoss:
            # Where did those nans come from??
            import pdb; pdb.set_trace()

        loss = {
            'Total Loss'  : lossGenIm1 + lossGenIm2,
            'Loss GenIm1' : lossGenIm1,
            'Loss GenIm2' : lossGenIm2,
            'Feat Loss12' : lossFeat12,
            'Feat Loss21' : lossFeat21
        }
        if not visualise:
            return loss, {}

        images = {
            'Im1' : im1 * 0.5 + 0.5, 'Im2' : im2 * 0.5 + 0.5,
            'GenIm1' : genIm1 * 0.5 + 0.5, 
            'GenIm2' : genIm2 * 0.5 + 0.5,
            'Dep1' : batch['Depth1'], 'Dep2' : batch['Depth2'],
            'TransImg12' : batch['TransImg12'] * 0.5 + 0.5, 
            'TransImg21' : batch['TransImg21'] * 0.5 + 0.5
        }

        return loss, images




