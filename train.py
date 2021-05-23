import os
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision

import time, copy
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'

from tensorboardX import SummaryWriter

from options.train_options import ArgumentParser, get_log_path, get_model_path
from options.options import get_model, get_dataset

import utils.utils as utils

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

opts, _ = ArgumentParser().parse()
opts.model_epoch_path = get_model_path(opts)

Dataset = get_dataset(opts)
model = get_model(opts)

model = model.cuda()

log_path = get_log_path(opts)
print(log_path)
writer = SummaryWriter(log_path)

# Temporary folder to visualize correspondances
os.makedirs('./temp/', exist_ok=True)

def train(epoch, model):
    train_set = Dataset('train', epoch)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True, drop_last=True)
    iter_training_data_loader = iter(training_data_loader)
    n_iter_epoch = 500
    losses = {}

    for iteration in range(1, n_iter_epoch):
        t_losses, images = model.step(iter_training_data_loader, visualise=(iteration == 1)) 

        for l in t_losses.keys():
            if l in losses.keys():
                    losses[l] = t_losses[l].cpu().item() + losses[l]
            else:
                    losses[l] = t_losses[l].cpu().item()

        for k in images.keys():
            if 'Vid' in k:
                print(images[k].min(), images[k].max())
                writer.add_video('Vid_train_%s_%d' % (k, iteration), images[k][0:8], epoch)
            else:
                writer.add_image('Image_train_%s_%d' % (k, iteration), torchvision.utils.make_grid(images[k][0:8], normalize=('Vis' in k)), epoch)
        
        str_to_print = "Train: Epoch {}: {}/{} with ".format(epoch, iteration, n_iter_epoch)
        for l in losses.keys():
            str_to_print += ' %s : %0.4f | ' % (l, losses[l] / float(iteration))
        print(str_to_print)

    # if opts.norm_class == 'batch_norm':
        # Run 50 forward passes to update momentum
        # utils.accumulate_standing_stats(model, iter_training_data_loader)
    
    return {l : losses[l] / float(iteration) for l in losses.keys()} 

def val(epoch, model):
    train_set = Dataset('test', 0)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=False, drop_last=True)
    iter_training_data_loader = iter(training_data_loader)
    n_iter_epoch = 101

    losses = {}
    for iteration in range(1, n_iter_epoch):
        t_losses, images = model.step(iter_training_data_loader, visualise=(iteration == 1), val=True) 

        for l in t_losses.keys():
            if l in losses.keys():
                    losses[l] = t_losses[l].cpu().item() + losses[l]
            else:
                    losses[l] = t_losses[l].cpu().item()

        for k in images.keys():
            if 'Vid' in k:
                writer.add_video('Vid_val_%s_%d' % (k, iteration), images[k][0:8], epoch)
            else:
                writer.add_image('Image_val_%s_%d' % (k, iteration), 
                        torchvision.utils.make_grid(images[k][0:8], normalize=('Vis' in k)), epoch)
        
        str_to_print = "Val: Epoch {}: {}/{} with ".format(epoch, iteration, n_iter_epoch)
        for l in losses.keys():
            str_to_print += ' %s : %0.4f | ' % (l, losses[l] / float(iteration))
        print(str_to_print)

    
    return {l : losses[l] / float(iteration) for l in losses.keys()} 

def checkpoint(model, save_path, epoch):
    checkpoint_state = model.get_checkpoint(epoch)

    torch.save(checkpoint_state, save_path)

def run(opts):
    if opts.continue_epoch > 0 or opts.resume:
        _, opts.continue_epoch = model.load_checkpoint(opts)

    for epoch in range(opts.continue_epoch, 10000):
        print(opts)
        print('At epoch %d...' % epoch)
        model.epoch = epoch
        model.train()
        train_loss = train(epoch, model)
        model.eval()
        with torch.no_grad():
                loss = val(epoch, model)

        model.step_plateau(loss['Total Loss'])

        for l in train_loss.keys():
            if l in loss.keys():
                writer.add_scalars('loss_recon_%s/train_val' % l, {'train' : train_loss[l], 'val' : loss[l]}, epoch)
            else:
                writer.add_scalars('loss_recon_%s/train_val' % l, {'train' : train_loss[l]}, epoch)

        if epoch % 1 == 0:
            print(opts.model_epoch_path)
            checkpoint(model, opts.model_epoch_path % str(epoch), epoch)

            for i in range(1,15):
                if os.path.exists(opts.model_epoch_path % str((epoch - i))):
                    os.remove(opts.model_epoch_path % str((epoch - i)))



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_idx)
    if not opts.model_zoo is None:
        os.environ["TORCH_MODEL_ZOO"] = opts.model_zoo
    
    if opts.load_old_model:
        pretrained_dict = (torch.load(opts.old_model)['state_dict'])     
        opts = torch.load(opts.old_model)['opts']
        model.load_state_dict(pretrained_dict)
        run(opts)
    else:
        run(opts)
