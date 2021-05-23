import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class StandingModel(nn.Module):
    def __init__(self, model, accumulate_images=False, B=16):
        super().__init__()
        self.model = model
        self.saved_images = None

        self.B = B
        self.accumulate_images = accumulate_images

    def forward(self, *inputs):
        if self.training:
            if self.accumulate_images:
                if self.saved_images is None:
                    self.saved_images = [list(input.clone().split(dim=0, split_size=1)) for input in inputs]
                else:
                    for i in range(0, len(inputs)):
                        self.saved_images[i].extend(list(inputs[i].clone().split(dim=0, split_size=1)))

            return self.model(*inputs)

        else:
            B = inputs[0].size(0)

            inputs = list(inputs)
            if B < self.B:
                for i in range(0, len(inputs)):
                    inputs[i] = torch.cat([inputs[i]] + self.saved_images[i][:self.B-inputs[i].size(0)], 0)

            result = self.model(*inputs)
            
            if isinstance(result, torch.Tensor):
                return result[:B,:,:,:]

            result = [r[:B,:,:,:] for r in result]

            return result

    def reset(self):
        self.saved_images = []
        self.accumulate_images = False

class StandingBN(nn.Module):
    def __init__(self, bn, eps=1e-5, B=16, accumulate_images=False):
        super().__init__()
        self.bn = bn

        self.bn.track_running_stats = False

        self.eps = eps
        self.B = B

    def forward(self, inputs):
        result = F.batch_norm(inputs, self.bn.running_mean, self.bn.running_var, 
                                    self.bn.weight, self.bn.bias, True, 0.0, self.bn.eps)

        return result




    def reset_stats(self):
        self.saved_images = []
