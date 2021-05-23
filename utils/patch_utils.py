import torch
import torch.nn.functional as F

import utils.utils as utils

class PatchMaker():
    def __init__(self, opts, W):
        # Buffer
        self.W = W; self.H = W
        ys, xs = torch.meshgrid(torch.linspace(-1,1,self.H), torch.linspace(-1,1,self.W))
        self.grid = torch.cat((xs.unsqueeze(2), ys.unsqueeze(2)), 2)

    def create_patch(self, img, gridDimensions, offsets, bI=None):
        # Resize to be correct shape
        gridDimensions = gridDimensions.unsqueeze(0)
        offsets = offsets.unsqueeze(1).unsqueeze(1)

        cG, wG, hG = self.grid.size()
        if bI is None:
            _, cI, wI, hI = img.size()
            bI = offsets.size(0)
        else:
            cI, wI, hI = img.size()

        sampler = self.grid.expand(bI,cG,wG,hG).to(img.device) * gridDimensions + offsets
        patch = F.grid_sample(img.expand(bI,cI,wI,hI), sampler)
        return patch

    def check(self, sampleIm1, trueSample, sampler12, W):
        sampler = sampler12.permute(1,2,0).unsqueeze(0)
        otherSample = F.grid_sample(sampler12.unsqueeze(0), sampleIm1.view(1,1,-1,2), padding_mode='border')
        otherSample = otherSample.squeeze().permute(1,0)

        areMatches = utils.distance(trueSample, otherSample) < self.W / float(W)
        validIm1 = (sampleIm1.abs().max(dim=1, keepdim=True)[0] < 1).float()
        validIm2 = (otherSample.abs().max(dim=1, keepdim=True)[0] < 1).float()
        validTrue = (trueSample.abs().max(dim=1, keepdim=True)[0] < 1).float()
        return areMatches.float() * validIm1.float() * validIm2.float() * validTrue.float()
        
    def create_matches(self, origImg, otherImg, sampler, mask, W, H, index_prev=None, return_sampler=False, valid=True, samples=2):
        gridDimensions = torch.Tensor([[self.W / float(W), self.H / float(H)]]).to(origImg.device)

        # Randomly choose N values from the sampler within the mask

        # Initialise grid
        ys, xs = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W))
        origSampler = torch.cat((xs.unsqueeze(2).to(origImg.device), 
                                ys.unsqueeze(2).to(origImg.device)), 2).unsqueeze(0)

        validMask = mask.view(-1,1).squeeze() > 0
        samplerIdxs = sampler.squeeze().view(-1,2)[validMask > 0,:]
        allSamplers = sampler.squeeze().view(-1,2)
        origIdxs = origSampler.squeeze().view(-1,2)[validMask > 0,:]


        if samples > samplerIdxs.size(0):
            samples = samplerIdxs.size(0)

        # Randomly select a bunch of indices
        # indices = torch.randperm(samplerIdxs.size(0))[0:samples]
        if index_prev is None:
            # Randomly select a bunch of indices
            indices = torch.randperm(samplerIdxs.size(0))[0:samples]
        else:
            indices = index_prev

        if valid:
            otherIndices = indices
            otherSample = samplerIdxs[otherIndices,:]          
        else:
            num_sample = 512
            indices = indices.repeat(num_sample+1)
            indexes_rot = torch.randperm(origSampler.squeeze().view(-1,2).size(0))
            otherIndices = torch.cat([ torch.cat([indexes_rot[i:],indexes_rot[:i]],0)[:samples] for i in range(num_sample)],0)
            otherSample = origSampler.squeeze().view(-1,2)[otherIndices,:]    

            otherIndices = torch.randperm(samplerIdxs.size(0))[0:samples]
            otherSample = torch.cat((otherSample, samplerIdxs[otherIndices,:]),0)          


        # And sample these positions
        origSample = origIdxs[indices,:]
        trueSample = samplerIdxs[indices,:]
        
        # Take care bad sampling
        if valid:
            areMatches = torch.ones((samples,1)).to(origImg.device)
        else:
            areMatches = utils.distance(trueSample, otherSample) < self.W / float(W)
            areMatches = areMatches.float() 

        if return_sampler:
            return origSample, otherSample, areMatches, indices

        # Then construct N grids from these within the given width
        origPatch = self.create_patch(origImg, gridDimensions, origSample, samples)
        otherPatch = self.create_patch(otherImg, gridDimensions, otherSample, samples)

        return origPatch, otherPatch, areMatches

