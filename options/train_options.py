import argparse
import os

class ArgumentParser():
        def __init__(self):
                self.parser = argparse.ArgumentParser(description='self-supervised learning')
                self.add_data_parameters()
                self.add_train_parameters()
                self.add_model_parameters()

        def add_model_parameters(self):
                model_params = self.parser.add_argument_group('model')
                model_params.add_argument('--old-model', type=str, default='')
                model_params.add_argument('--model-type', type=str, default='')
                model_params.add_argument('--network', type=str, default='resnet50')
                model_params.add_argument('--use_pretrained', type=int, default=0)
                model_params.add_argument('--resume', type=int, default=0)
                model_params.add_argument('--norm_class', type=str, default='batch_norm', help='Choose among (batch_norm|instance_norm|layer_norm) will raise ValueError instead')
                model_params.add_argument('--norm_D', type=str, default='spectralinstance', help='Choose among (batch_norm|instance_norm|layer_norm) will raise ValueError instead')
                model_params.add_argument('--ngf', type=int, default=32)
                model_params.add_argument('--in_vec', type=int, default=256)
                model_params.add_argument('--use_discriminator', type=int, default=0)
                model_params.add_argument('--use_mask', type=int, default=0)
                model_params.add_argument('--use_image', action='store_true', default=True)
                model_params.add_argument('--use_depth', action='store_true', default=False)
                model_params.add_argument('--use_weights', type=int, default=0)
                model_params.add_argument('--topk', type=int, default=1)
                model_params.add_argument('--max_warps', type=int, default=1)
                model_params.add_argument('--feature_model', type=str, default='') 
        
        def add_data_parameters(self):
                dataset_params = self.parser.add_argument_group('data')
                dataset_params.add_argument('--dataset', type=str, default='faces')
                dataset_params.add_argument('--dataset_base_path', type=str, required=True)
                dataset_params.add_argument('--keep_aspect', type=int, default=0) 

        def add_train_parameters(self):
                training = self.parser.add_argument_group('training')
                training.add_argument('--gpu_idx', type=int, default=0)
                training.add_argument('--exp_idx', type=int, default=0)
                training.add_argument('--train_encoder', type=int, default=0)

                training.add_argument('--train_end2end', type=int, default=0)
                training.add_argument('--jitter', type=int, default=0)

                training.add_argument('--num_workers', type=int, default=5)
                training.add_argument('--model_zoo', type=str, default='~/.model_zoo/')
                training.add_argument('--start-epoch', type=int, default=0)
                training.add_argument('--W', type=int, default=256)
                training.add_argument('--WCoarse', type=int, default=256)
                training.add_argument('--lr', type=float, default=0.001)
                training.add_argument('--lambda1', type=float, default=1)
                training.add_argument('--hard_samples', type=int, default=1)
                training.add_argument('--hinge_samples', type=int, default=1)
                training.add_argument('--lambda2', type=float, default=1)
                training.add_argument('--lambda_feat', type=float, default=1)

                training.add_argument('--beta1', type=float, default=0.)
                training.add_argument('--beta2', type=float, default=0.9)
                training.add_argument('--lr_d', type=float, default=1e-3*2)

                training.add_argument('--suffix', type=str, default='')
                training.add_argument('--losses', type=str, default='bce', help='Choose among bce|hinge for selecting loss')
       
                training.add_argument('--load-old-model', action='store_true', default=False) 
        
                training.add_argument('--log-dir', type=str, default='./checkpoints/imp3d/%s/')

                training.add_argument('--batch-size', type=int, default=16)
                training.add_argument('--continue-epoch', type=int, default=0)
                training.add_argument('--model-epoch-path', type=str, default='models/%s_lr%0.5f_bs%d_model%s_loss%s_lambs%0.4f|%0.4f/')
                training.add_argument('--run-dir', type=str, default='runs/%s_lr%0.5f_bs%d_model%s_loss%s_lambs%0.4f|%0.4f/')

        def parse(self, arg_str=None):
                if arg_str is None:
                        args = self.parser.parse_args()
                else:
                        args = self.parser.parse_args(arg_str.split())

                arg_groups = {}
                for group in self.parser._action_groups:
                        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
                        arg_groups[group.title] = group_dict

                return (args, arg_groups)

def get_log_path(opts):
        return opts.log_dir % opts.dataset + opts.run_dir % (opts.suffix, opts.lr, opts.batch_size, opts.model_type, 
                        opts.losses, opts.lambda1, opts.lambda2)

def get_model_path(opts):
        model_path = opts.log_dir % opts.dataset + opts.model_epoch_path % (opts. suffix, opts.lr, opts.batch_size, opts.model_type, 
                        opts.losses, opts.lambda1, opts.lambda2)
        if not os.path.exists(model_path):
                os.makedirs(model_path)

        return model_path + '/model_epoch_%s.pth'
