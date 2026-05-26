"""final_new 专属 Executor

基于 TrafficStateExecutor，修复 DDP 训练结束时 load_model_with_epoch 竞态条件：
在 save_model_with_epoch（仅 rank 0 写磁盘）和 load_model_with_epoch（所有 rank 读磁盘）之间
添加 barrier 同步，确保 rank 0 写完文件后其他 rank 才开始读取。
"""

import os
import numpy as np
import time
from GNNTP.common.traffic_state_executor import TrafficStateExecutor
from GNNTP.utils import tune


class FinalNewExecutor(TrafficStateExecutor):
    """final_new 模型专属 Executor

    继承 TrafficStateExecutor，仅覆盖 train() 方法，
    在训练循环结束后、加载最佳模型前插入 dist.barrier() 同步点。
    """

    def train(self, train_dataloader, eval_dataloader):
        """训练流程（覆盖父类，仅添加 barrier 修复 DDP 竞态）

        Args:
            train_dataloader: 训练数据
            eval_dataloader: 评估数据
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
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
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=val_loss)

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
            # FIX: DDP barrier — 确保 rank 0 完成 save_model_with_epoch 磁盘写入后，
            # 其他 rank 才执行 load_model_with_epoch 读取文件，避免竞态崩溃
            self._barrier()
            self.load_model_with_epoch(best_epoch)
        return min_val_loss
