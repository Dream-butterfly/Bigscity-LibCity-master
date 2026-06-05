import time
import numpy as np
import torch
import os
from GNNTP.models import loss
from functools import partial
from GNNTP.common.traffic_state_executor import TrafficStateExecutor
from GNNTP.utils import tune


class MTGNNExecutor(TrafficStateExecutor):
    """MTGNN 专用执行器，实现节点拆分训练 (node-splitting)。

    MTGNN 的自适应图学习模块依赖全节点对，无法直接用标准 batch 训练。
    因此每个 batch 内将节点打乱并拆分为多个子图 (num_split) 分别前向传播。

    关键参数:
      - num_split: 子图数量
      - step_size2: 节点排列重洗频次 (每 step_size2 个 batch)

    Migrated from Bigscity-LibCity-master/libcity/executor/mtgnn_executor.py
    """

    def __init__(self, config, model, data_feature):
        TrafficStateExecutor.__init__(self, config, model, data_feature)
        self.step_size2 = self.config.get('step_size2', 100)
        self.num_nodes = self.data_feature.get('num_nodes')
        self.num_split = self.config.get('num_split', 1)

    def _build_train_loss(self):
        if self.train_loss.lower() == 'none':
            self._logger.warning(
                'Received none train loss func and will use the loss func defined in the model.')
            return None
        if self.train_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber',
                                             'quantile', 'masked_mae', 'masked_mse', 'masked_rmse',
                                             'masked_mape', 'r2', 'evar']:
            self._logger.warning(
                'Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info(
                'You select `{}` as train loss function.'.format(self.train_loss.lower()))

        def func(batch, idx=None, batches_seen=None):
            if idx is not None:
                idx = torch.tensor(idx).to(self.model.device)
                tx = batch['X'][:, :, idx, :].clone()
                y_true = batch['y'][:, :, idx, :]
                batch_new = {'X': tx}
                y_predicted = self._unwrap_model().predict(batch_new, idx)
            else:
                y_true = batch['y']
                y_predicted = self._unwrap_model().predict(batch)
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

            if self.train_loss.lower() == 'mae':
                lf = loss.masked_mae_torch
            elif self.train_loss.lower() == 'mse':
                lf = loss.masked_mse_torch
            elif self.train_loss.lower() == 'rmse':
                lf = loss.masked_rmse_torch
            elif self.train_loss.lower() == 'mape':
                lf = loss.masked_mape_torch
            elif self.train_loss.lower() == 'logcosh':
                lf = loss.log_cosh_loss
            elif self.train_loss.lower() == 'huber':
                lf = loss.huber_loss
            elif self.train_loss.lower() == 'quantile':
                lf = loss.quantile_loss
            elif self.train_loss.lower() == 'masked_mae':
                lf = partial(loss.masked_mae_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mse':
                lf = partial(loss.masked_mse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_rmse':
                lf = partial(loss.masked_rmse_torch, null_val=0)
            elif self.train_loss.lower() == 'masked_mape':
                lf = partial(loss.masked_mape_torch, null_val=0)
            elif self.train_loss.lower() == 'r2':
                lf = loss.r2_score_torch
            elif self.train_loss.lower() == 'evar':
                lf = loss.explained_variance_score_torch
            else:
                lf = loss.masked_mae_torch

            if self.model.training:
                if batches_seen % self._unwrap_model().step_size == 0 \
                   and self._unwrap_model().task_level < self._unwrap_model().output_window:
                    self._unwrap_model().task_level += 1
                    self._logger.info('Training: task_level increase from {} to {}'.format(
                        self._unwrap_model().task_level - 1,
                        self._unwrap_model().task_level))
                    self._logger.info('Current batches_seen is {}'.format(batches_seen))
                if self._unwrap_model().use_curriculum_learning:
                    return lf(
                        y_predicted[:, :self._unwrap_model().task_level, :, :],
                        y_true[:, :self._unwrap_model().task_level, :, :])
                else:
                    return lf(y_predicted, y_true)
            else:
                return lf(y_predicted, y_true)
        return func

    def train(self, train_dataloader, eval_dataloader):
        if self._is_rank0():
            self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        if self._is_rank0():
            self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num * self.num_split
        for epoch_idx in range(self._epoch_num, self.epochs):
            if self.is_distributed and hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch_idx)

            start_time = time.time()
            losses, batches_seen = self._train_epoch(
                train_dataloader, epoch_idx, batches_seen, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)
            if self._is_rank0():
                self._logger.info("epoch complete!")

            if self._is_rank0():
                self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(
                eval_dataloader, epoch_idx, batches_seen, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if self._is_rank0() and (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, batches_seen, np.mean(losses),
                           val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune and self._is_rank0():
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved and self._is_rank0():
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    if self._is_rank0():
                        self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if self._is_rank0() and len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        self.model.train()
        loss_func = loss_func if loss_func is not None else self._unwrap_model().calculate_loss
        losses = []
        for iter_, batch in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            if iter_ % self.step_size2 == 0:
                perm = np.random.permutation(range(self.num_nodes))
            num_sub = int(self.num_nodes / self.num_split)
            for j in range(self.num_split):
                if j != self.num_split - 1:
                    idx = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    idx = perm[j * num_sub:]
                loss = loss_func(batch, idx=idx, batches_seen=batches_seen)
                self._logger.debug(loss.item())
                losses.append(loss.item())
                batches_seen += 1
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        if self.is_distributed:
            import torch.distributed as dist
            loss_tensor = torch.tensor([np.mean(losses)], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            losses = [loss_tensor.item()] * len(losses)
        return losses, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self._unwrap_model().calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            if self.is_distributed:
                import torch.distributed as dist
                loss_tensor = torch.tensor([mean_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_loss = loss_tensor.item()
            self._writer.add_scalar('eval loss', mean_loss, batches_seen)
            return mean_loss
