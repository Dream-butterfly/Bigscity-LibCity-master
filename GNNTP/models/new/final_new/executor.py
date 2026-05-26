"""final_new 专属 Executor

基于 TrafficStateExecutor，仅修改 train() 的最佳模型恢复方式：
采用 STGformer 内存模式——最佳模型 state_dict 在内存中维护（所有 rank 各自 deepcopy），
训练结束后直接从内存 load_state_dict，不依赖磁盘 I/O。
避免 DDP 下 load_model_with_epoch 的跨 rank 竞态。

_train_epoch / _valid_epoch 保持与父类一致，不做任何改动。
"""

import os
import copy
import time

import numpy as np

from GNNTP.common.traffic_state_executor import TrafficStateExecutor
from GNNTP.utils import tune


class FinalNewExecutor(TrafficStateExecutor):
    """final_new 模型专属 Executor

    仅覆盖 train()，其余全部继承父类。
    """

    def train(self, train_dataloader, eval_dataloader):
        """训练流程

        与父类唯一区别：
        - val_loss 改善时 deepcopy state_dict 到内存（所有 rank 各自维护）
        - 训练结束后 load_state_dict 从内存恢复，不走磁盘
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        best_state_dict = None
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch_idx)
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune and self._is_rank0():
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                # 所有 rank 各自 deepcopy 最佳权重到内存
                best_state_dict = copy.deepcopy(self._unwrap_model().state_dict())
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
        if self.load_best_epoch and best_state_dict is not None:
            self._unwrap_model().load_state_dict(best_state_dict)
        return min_val_loss
