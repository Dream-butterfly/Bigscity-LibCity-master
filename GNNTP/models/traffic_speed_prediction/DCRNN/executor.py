"""DCRNN 专属 Executor — 继承 TrafficStateExecutor，增加 curriculum learning 的 batches_seen 追踪。"""
import time
import numpy as np
from functools import partial
from logging import getLogger

from GNNTP.common.traffic_state_executor import TrafficStateExecutor
from GNNTP.models import loss


class DCRNNExecutor(TrafficStateExecutor):
    """DCRNN Executor — 在标准 TrafficStateExecutor 基础上增加:
    - batches_seen 全局步数追踪（curriculum learning 需要）
    - 定制的 _build_train_loss 传递 batches_seen 给模型
    - 定制的 train/_train_epoch/_valid_epoch 维护 batches_seen
    """

    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)

    def _build_train_loss(self):
        """构建传递 batches_seen 的 loss 函数。"""
        if self.train_loss.lower() == 'none':
            self._logger.warning(
                'Received none train loss func and will use the loss func defined in the model.')
            return None
        if self.train_loss.lower() not in [
            'mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile',
            'masked_mae', 'masked_mse', 'masked_rmse', 'masked_mape', 'r2', 'evar',
        ]:
            self._logger.warning(
                'Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info(
                'You select `{}` as train loss function.'.format(self.train_loss.lower()))

        def func(batch, batches_seen=None):
            y_true = batch['y']
            y_predicted = self._unwrap_model().predict(batch, batches_seen)
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
            elif self.train_loss.lower() == 'smape':
                lf = loss.masked_smape_torch
            elif self.train_loss.lower() == 'masked_smape':
                lf = partial(loss.masked_smape_torch, null_val=0)
            elif self.train_loss.lower() == 'r2':
                lf = loss.r2_score_torch
            elif self.train_loss.lower() == 'evar':
                lf = loss.explained_variance_score_torch
            else:
                lf = loss.masked_mae_torch
            return lf(y_predicted, y_true)

        return func

    def train(self, train_dataloader, eval_dataloader):
        """训练流程 — DCRNN 版本（含 batches_seen 追踪 + AMP + DDP）。"""
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        for epoch_idx in range(self._epoch_num, self.epochs):
            if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch_idx)
            start_time = time.time()
            losses, batches_seen = self._train_epoch(
                train_dataloader, epoch_idx, batches_seen, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)
            self._logger.info("epoch complete!")

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

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, batches_seen, np.mean(losses), val_loss,
                           log_lr, (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        """单轮训练 — 含 AMP + batches_seen 追踪。"""
        self.model.train()
        loss_func = loss_func if loss_func is not None else self._unwrap_model().calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            with self._autocast_context():
                loss = loss_func(batch, batches_seen)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            batches_seen += 1
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(loss).backward()
                if self.clip_grad_norm:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        return losses, batches_seen

    def _valid_epoch(self, eval_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        """单轮验证 — 含 AMP + batches_seen 传递。"""
        import torch
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self._unwrap_model().calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                with self._autocast_context():
                    loss = loss_func(batch, batches_seen)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, batches_seen)
            return mean_loss
