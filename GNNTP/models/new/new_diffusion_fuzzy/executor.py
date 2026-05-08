"""new_diffusion_fuzzy 专属 Executor.

与 TrafficStateExecutor 的关键区别:
- _train_epoch: loss_func=None 时走 self.model(batch) 触发 DDP forward hook
- _valid_epoch: loss_func=None 时走 calculate_loss（no_grad 下无需 DDP）
- 与基类隔离，修改不影响其他模型
"""

from __future__ import annotations

import numpy as np
import torch

from GNNTP.common.traffic_state_executor import TrafficStateExecutor


class DiffusionTrafficStateExecutor(TrafficStateExecutor):
    """扩散模型专用 Executor，继承自 TrafficStateExecutor。

    仅重写 _train_epoch / _valid_epoch 的 loss_func=None 分支，
    其余逻辑（optimizer、lr_scheduler、early_stop、save/load 等）全部复用基类。
    """

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """完成模型一个轮次的训练。

        loss_func=None 时通过 self.model(batch) 调用 forward(),
        确保 DDP 梯度同步 hook 被触发。
        """
        self.model.train()
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            with self._autocast_context():
                if loss_func is not None:
                    loss = loss_func(batch)
                else:
                    loss = self.model(batch)  # DDP forward hook 同步梯度
            self._logger.debug(loss.item())
            losses.append(loss.item())
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
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        """完成模型一个轮次的评估。

        no_grad 下直接用 calculate_loss，无需 DDP hook。
        """
        with torch.no_grad():
            self.model.eval()
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                with self._autocast_context():
                    if loss_func is not None:
                        loss = loss_func(batch)
                    else:
                        loss = self._unwrap_model().calculate_loss(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            # DDP: all_reduce 求全局平均损失
            if self.is_distributed:
                import torch.distributed as dist
                loss_tensor = torch.tensor([mean_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_loss = loss_tensor.item()
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
