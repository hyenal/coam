import torch.optim as optim


class Scheduler:
    def __init__(self, opts, optimizer, max_epochs=10):
        # self.plateauscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        self.opts = opts
        self.opts.max_epochs = 50
        self.opts.epoch = -1
        self.reset()

    def reset(self):
        self.opts.track_best_loss = 1000
        self.opts.factor =0.01
        self.opts.best_epoch = self.opts.epoch

    def update_opts(self, opts):
        opts.max_epochs = self.opts.max_epochs
        opts.epoch = self.opts.epoch
        opts.factor = 0.005
        opts.best_epoch = self.opts.best_epoch
        self.opts = opts

    def state_dict(self):
        pass
        # return self.plateauscheduler.state_dict()

    def load_state_dict(self, state_dict):
        pass
        # self.plateauscheduler.load_state_dict(state_dict)

    def step(self, loss):
        print(self.opts.epoch, loss, self.opts.factor)
        self.opts.epoch += 1
        if loss < self.opts.track_best_loss:
            self.opts.best_epoch = self.opts.epoch
            self.opts.track_best_loss = loss

        elif (self.opts.epoch - self.opts.best_epoch) > self.opts.max_warps:
            self.opts.factor = max(1, self.opts.factor * 2)


            


