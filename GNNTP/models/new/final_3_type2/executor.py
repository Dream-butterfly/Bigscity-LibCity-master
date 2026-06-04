"""final_3_type2 专属 Executor

基于 TrafficStateExecutor，修复 DDP 训练结束时 load_model_with_epoch 竞态条件：
采用 STGformer 模式——最佳模型 state_dict 保存在内存中（所有 rank 各自维护一份），
训练结束后直接从内存恢复，完全不依赖磁盘 I/O，彻底消除竞态。
Rank 0 仍保存到磁盘用于断点续训。
"""

import os
import copy
import time

import numpy as np

from GNNTP.common.traffic_state_executor import TrafficStateExecutor
from GNNTP.utils import tune


class FinalNewExecutor3_Type2(TrafficStateExecutor):
    """final_3_type2 模型专属 Executor"""

    def train(self, train_dataloader, eval_dataloader):
        """训练流程

        与父类 TrafficStateExecutor.train() 的核心区别：
        - 最佳模型 state_dict 在内存中维护（所有 rank），训练后直接 restore
        - 避免 load_model_with_epoch 的磁盘竞态
        - 仅 rank 0 输出训练日志
        """
        if self._is_rank0():
            self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        best_state_dict = None
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        if self._is_rank0():
            self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch_idx)
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            train_loss = float(np.mean(losses))
            train_time.append(time.time() - start_time)
            self._writer.add_scalar('training loss', train_loss, epoch_idx)

            eval_start = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func)
            eval_time.append(time.time() - eval_start)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if self._is_rank0() and (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, train_loss, val_loss, log_lr, (time.time() - start_time))
                self._logger.info(message)

            if self.hyper_tune and self._is_rank0():
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                # 所有 rank 各自 deepcopy 最佳权重到内存，避免后续磁盘加载竞态
                best_state_dict = copy.deepcopy(self._unwrap_model().state_dict())
                if self.saved and self._is_rank0():
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                elif self._is_rank0():
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_val_loss, val_loss))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait >= self.patience and self.use_early_stop:
                    if self._is_rank0():
                        self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

        if self._is_rank0() and len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch and best_state_dict is not None:
            if self._is_rank0():
                self._logger.info("Loading best model state from epoch {}".format(best_epoch))
            self._unwrap_model().load_state_dict(best_state_dict)
        return min_val_loss
