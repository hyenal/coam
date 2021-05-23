from models.multi_resolution import *
from models.downstream_rgbmodels import *

from dataset.megadepth import MegaDepth

def get_model(opt):
    print(opt)
    if opt.model_type == 'fpn_nocond_conf':
        model = MultiResolutionFPNNoMixConf(opts=opt)
    elif opt.model_type == 'fpn_conf_broad':
        model = MultiResolutionFPNMixConfClassBroadcast(opts=opt)
    elif opt.model_type == 'stylize_rgb_2steps':
        model = StylizeImageConcatenateInputs2Step(opts=opt)
    return model

def get_dataset(opt):
    if opt.dataset == 'megadepth':
        return lambda x, y : MegaDepth(opts=opt, split=x, seed=y, file_path=opt.dataset_base_path)
    else:
        raise Exception()

