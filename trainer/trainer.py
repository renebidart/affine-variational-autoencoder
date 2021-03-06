import numpy as np
import pickle
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None,
                 save_rot_loc=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        
        self.save_rot_loc = save_rot_loc
        if self.save_rot_loc is not None:
            self.rot_dict = {}
            self.rot_dict

    def _eval_metrics(self, output, data):
        """for AUTO"""
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, data)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics
    
    def _train_epoch(self, epoch, give_model_label=False):
        torch.multiprocessing.set_sharing_strategy('file_system') 
        """
        ONLY FOR VAES
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        if self.save_rot_loc is not None:
            self.rot_dict[epoch] = {}
            
        for batch_idx, stuff in enumerate(self.data_loader):
            if self.save_rot_loc is not None:
                data, target, rot = stuff
            else:
                data, target = stuff

            data, target = data.to(self.device), target.to(self.device) # ???to longtensor???

            self.optimizer.zero_grad()
            if give_model_label:
                output = self.model(data, target)
                loss = self.loss(output, data, target)
            else:
                output = self.model(data)
                loss = self.loss(output, data)
            if self.save_rot_loc is not None:
                recon_x, mu_logvar, affine_params, _x_affine = self.model(data, return_affine=True)
                affine_params = affine_params.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                rot = rot.detach().numpy()
                self.rot_dict[epoch][batch_idx]=(affine_params, target, rot)
                
                output = (recon_x, mu_logvar)
                loss = self.loss(output, data)
            self.optimizer.zero_grad() # ??? Why didn't they include this???
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            
            if give_model_label:
                total_metrics += self._eval_metrics(output, data, target)
            else:
                total_metrics += self._eval_metrics(output, data)
                
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader.dataset),
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        if self.save_rot_loc is not None:
            pickle.dump(self.rot_dict, open(str(self.save_rot_loc)+
                                                '/'+'the_only_rot_history.pkl', "wb"))

        return log

    def _valid_epoch(self, epoch, give_model_label=False):
        torch.multiprocessing.set_sharing_strategy('file_system') 
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            
            for batch_idx, stuff in enumerate(self.valid_data_loader):
                if self.save_rot_loc is not None:
                    data, target, rot = stuff
                else:
                    data, target = stuff
                data, target = data.to(self.device), target.to(self.device)
                if give_model_label:
                    output = self.model(data, target)
                    loss = self.loss(output, data, target)
                else:
                    output = self.model(data)
                    loss = self.loss(output, data)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
    
                if give_model_label:
                    total_val_metrics += self._eval_metrics(output, data, target)
                else:
                    total_val_metrics += self._eval_metrics(output, data)
                
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
