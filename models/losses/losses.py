import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architectures import VGG19

import numpy as np

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake

def loss_hinge_gen(fake):
    return - torch.mean(fake)

num_samples = 513

class NCELoss(nn.Module):

    def compute_prob_errs(self, confQuery, matches, M=0.5, tau=0.25):
        ''' Computes how well probs predicts the true distribution of distractors.
        Probs predicts 1 / NUM_CONFUSERS^tau. If low, should be ignored, if high
        should be used.

        confQuery: the confidence of the point
        matches: number of distractors
        '''
        B = confQuery.size(0)

        cos_matches = matches > M
        numMatches = (cos_matches).float().sum(dim=2).clamp(min=1)
        err = nn.L1Loss()(confQuery, 1 / numMatches.pow(tau) )

        return err
        

    def forward(self, pos, negs, pos_y, neg_y, pos_v=1, neg_v=1, M=0.5, s=4):
        neg_y = (neg_y < 0.5).float()
        pos_y = (1 - (pos_y < 0.5).float())

        # Enforce margin
        cos_p_M = s * (pos - M)
        cos_n = s * negs

        B, P = cos_p_M.size()
        cos_n = cos_n.view(B,-1,P)
        neg_y = neg_y.view(B,-1,P)
        neg_v = neg_v.view(B,-1,P)

        # Compute loss: -1/N sum_i=1...N log e^(s theta_true + m) / e^(s theta_true + m) + sum_negs e^(s * theta_neg)
        L_nce = - cos_p_M + (cos_p_M.exp() + (cos_n.exp() * neg_y * neg_v).sum(dim=1)).log()
        L_nce = (L_nce * pos_y * pos_v).sum(dim=1) / (pos_y * pos_v).sum(dim=1).clamp(min=1)

        return L_nce.mean()

class HingeLoss(nn.Module):
    def __init__(self, n_samples=1):
        super(HingeLoss, self).__init__()
        self.n_samples = n_samples

    def compute_prob_errs(self, confQuery, matches, M=1, tau=0.25):
        ''' Computes how well probs predicts the true distribution of distractors.
        Probs predicts 1 / NUM_CONFUSERS^tau. If low, should be ignored, if high
        should be used.

        confQuery: the confidence of the point
        matches: number of distractors
        '''
        B = confQuery.size(0)

        numMatches = (matches < M).float().sum(dim=2).clamp(min=1)
        err = nn.L1Loss()(confQuery, 1 / numMatches.pow(tau) )

        return err
        
    def forward(self, x, y, valid=1, conf=1, M=1., negative=False, complete=False):
        
        negs = (y < 0.5).float()
        
        if self.n_samples > 0 and not complete:
            if negative:
                x = x.view(x.size(0), num_samples, -1)
                valid = valid.view(valid.size(0), num_samples,-1)
                negs = negs.view(negs.size(0), num_samples,-1)
                if isinstance(conf, torch.Tensor):
                    conf = conf.view(conf.size(0), num_samples,-1)

                negLoss = torch.topk(((M - x)*negs*valid*conf).clamp(min=0), self.n_samples, largest=True, dim=1)[0].sum() 
                #posLoss = torch.topk((x * (1 - negs)*valid), self.n_samples, largest=True, dim=1)[0].sum((1,2)).sum() / valid.sum(1,2)
                posLoss = 0
                negLoss = negLoss * float(negs.shape[0]) / (self.n_samples*valid[:,0,:]).sum()
            else:
                negLoss = 0
                posLoss = torch.topk((x * (1 - negs)*valid*conf), self.n_samples, largest=True, dim=-1)[0].mean(1).sum()
        else:
            if negative:
                negLoss = ((M - x) * negs*valid*conf).clamp(min=0).sum() * float(negs.shape[0]) / (valid*negs).sum()
                posLoss = 0
            else:
                posLoss = ((x * (1 - negs)*valid*conf).sum(1) / valid.sum(1)).sum()
                negLoss = 0 

        return (negLoss + posLoss) / float(negs.shape[0])


# Laplacian loss to encode uncertainty
class LaplacianLoss(nn.Module):
    def forward(self, err, conf, mask=None):
        B = err.size(0)
        
        conf = conf.clamp(min=1e-3) # Stop this from giving NaNs

        loss = (np.sqrt(2) / (2. * conf)).log() + (- np.sqrt(2) * err / conf)
        loss = - loss

        if not (mask is None):
            loss = loss * mask
            return (loss.view(B,-1).sum(dim=1) / mask.view(B,-1).sum(dim=1).clamp(min=1)).mean()

        return loss.view(B,-1).mean(dim=1).mean()

class Stats(nn.Module):
    def __init__(self, topk=5):
        super(Stats, self).__init__()
        self.topk = topk

    def forward(self, x, y, negs, valid):
        x = x.view(x.size(0), num_samples, -1)
        valid = valid.view(valid.size(0),num_samples,-1)
        
        # Compute false negative and remove them from statistics
        negs = (negs < 0.5).float()
        negs = negs.view(negs.size(0),num_samples,-1)
        x = x * negs + 5 * (1-negs)

        # Compute top1 and topk valid points
        stats = torch.topk((x - y[:,None,:])*valid, self.topk, largest=False, dim=1)[0]
        top1 = (stats[:,0,:] > 0).float().sum(1) / valid[:,0,:].sum(1)
        topk = (stats[:,-1,:] > 0).float().sum(1) / valid[:,0,:].sum(1)

        return top1.mean(), topk.mean()

# Reweighted BCE Loss
class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        if weights is not None:
            assert len(weights) == 2

        # weights: [neg pos]
        self.weights = weights

    def forward(self, output, target, mask=None):
        print(output.min().item(), output.max().item(), 
              target.min().item(), target.max().item(),
              self.weights[0], self.weights[1])
        if mask is None:
            loss = self.weights[1] * (target * torch.log(output)) + \
                self.weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = self.weights[1] * mask * (target * torch.log(output)) + \
                self.weights[0] * mask * ((1 - target) * torch.log(1 - output))

        B = output.size(0)
        return - loss.view(B,-1).mean(dim=1).mean()

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class L1Mask(nn.Module):
    def forward(self, img1, img2, mask=1):
        b = img1.size(0)

        err = (img1 - img2).abs().mean(dim=1, keepdim=True) * mask
        
        if isinstance(mask, int):
            err = err.view(b, -1).mean(dim=1)
            count = mask
        else:
            err = err.view(b, -1).sum(dim=1)
            count = mask.view(b, -1).sum(dim=1).clamp(min=1)

        return (err / count).mean()
        
