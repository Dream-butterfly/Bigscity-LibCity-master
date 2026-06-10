import os
import time
from contextlib import nullcontext
import numpy as np
import torch
from logging import getLogger
from GNNTP.common.abstract_executor import AbstractExecutor
from GNNTP.utils import get_evaluator, ensure_dir, get_run_dir, get_run_subdir, tune
from GNNTP.models import loss
from functools import partial


class _NoopSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


class TrafficStateExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.data_feature = data_feature
        self.device = self.config.get('device', torch.device('cpu'))
        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)
        self.is_distributed = self.config.get('is_distributed', False)
        # 模型设备迁移: 单卡时在此执行，DDP 时由外部 pipeline 完成
        if not self.is_distributed:
            self.model = model.to(self.device)
        else:
            self.model = model  # 已在 pipeline 中 DDP 包装并移到设备
        self.exp_id = self.config.get('exp_id', None)

        self.run_dir = get_run_dir(self.exp_id)
        self.cache_dir = get_run_subdir(self.exp_id, 'model_cache')
        self.evaluate_res_dir = get_run_subdir(self.exp_id, 'evaluate_cache')

        self._writer = _NoopSummaryWriter()
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        if self._is_rank0():
            self._logger.info(self.model)
            for name, param in self.model.named_parameters():
                self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                                  str(param.device) + '\t' + str(param.requires_grad))
            total_num = sum([param.nelement() for param in self.model.parameters()])
            self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.epochs = self.config.get('max_epoch', 100)
        self.train_loss = self.config.get('train_loss', 'none')
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 0)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.hyper_tune = self.config.get('hyper_tune', False)

        self.output_dim = self.config.get('output_dim', 1)
        self.use_amp = self.config.get('use_amp', True)
        self.amp_dtype = str(self.config.get('amp_dtype', 'float16')).lower()
        if self.amp_dtype in ('fp16', 'float16'):
            self.amp_dtype = 'float16'
            self.amp_torch_dtype = torch.float16
        elif self.amp_dtype in ('bf16', 'bfloat16'):
            self.amp_dtype = 'bfloat16'
            self.amp_torch_dtype = torch.bfloat16
        else:
            self._logger.warning('Unsupported amp_dtype `%s`, fallback to float16.', self.amp_dtype)
            self.amp_dtype = 'float16'
            self.amp_torch_dtype = torch.float16
        if self.amp_dtype == 'bfloat16' and self.device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            self._logger.warning('Current CUDA device does not support bfloat16 AMP, fallback to float16.')
            self.amp_dtype = 'float16'
            self.amp_torch_dtype = torch.float16
        if self.use_amp and self.device.type != 'cuda':
            self._logger.warning('AMP is only enabled on CUDA devices, disable AMP on device `%s`.', self.device)
        self.amp_enabled = self.use_amp and self.device.type == 'cuda'
        self.grad_scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled and self.amp_dtype == 'float16')
        if self.amp_enabled:
            self._logger.info('Enable AMP training (dtype=%s).', self.amp_dtype)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)
        self.loss_func = self._build_train_loss()

    def _unwrap_model(self):
        """获取原始模型（DDP 包装下取 .module）"""
        if self.is_distributed:
            return self.model.module
        return self.model

    def _is_rank0(self):
        """当前进程是否为主进程（rank 0）"""
        if not self.is_distributed:
            return True
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0

    def _barrier(self):
        """所有进程同步点"""
        if self.is_distributed:
            import torch.distributed as dist
            dist.barrier()

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        if not self._is_rank0():
            return
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        model = self._unwrap_model()
        torch.save((model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = torch.load(cache_name, map_location=self.device)
        model = self._unwrap_model()
        model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        if not self._is_rank0():
            return ""
        ensure_dir(self.cache_dir)
        config = dict()
        model = self._unwrap_model()
        config['model_state_dict'] = model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['grad_scaler_state_dict'] = self.grad_scaler.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location=self.device)
        model = self._unwrap_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'grad_scaler_state_dict' in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _autocast_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type=self.device.type, dtype=self.amp_torch_dtype)
        return nullcontext()

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def _build_train_loss(self):
        """
        根据全局参数`train_loss`选择训练过程的loss函数
        如果该参数为none，则需要使用模型自定义的loss函数
        注意，loss函数应该接收`Batch`对象作为输入，返回对应的loss(torch.tensor)
        """
        if self.train_loss.lower() == 'none':
            self._logger.warning('Received none train loss func and will use the loss func defined in the model.')
            return None
        if self.train_loss.lower() not in ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                                           'masked_mse', 'masked_rmse', 'masked_mape', 'r2', 'evar']:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        else:
            self._logger.info('You select `{}` as train loss function.'.format(self.train_loss.lower()))

        def func(batch):
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

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                with self._autocast_context():
                    output = self._unwrap_model().predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])[..., :self.output_dim]
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])[..., :self.output_dim]
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)

            # DDP: 各 rank 只推理自己的数据分片，需 all_gather 汇总
            if self.is_distributed:
                import torch.distributed as dist
                # 确保所有 rank 都完成 for 循环再进入 all_gather
                dist.barrier()
                world_size = dist.get_world_size()
                y_preds_t = torch.from_numpy(y_preds).to(self.device)
                y_truths_t = torch.from_numpy(y_truths).to(self.device)
                # 收集各 rank 样本数
                local_size = torch.tensor([y_preds.shape[0]], dtype=torch.long, device=self.device)
                all_sizes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(world_size)]
                dist.all_gather(all_sizes, local_size)
                sizes = [int(s.item()) for s in all_sizes]
                max_size = max(sizes)
                # 填充到统一大小后 all_gather
                B, T, N, F = y_preds.shape
                pad_pred = torch.zeros(max_size, T, N, F, device=self.device)
                pad_truth = torch.zeros(max_size, T, N, F, device=self.device)
                pad_pred[:B] = y_preds_t
                pad_truth[:B] = y_truths_t
                gathered_preds = [torch.zeros_like(pad_pred) for _ in range(world_size)]
                gathered_truths = [torch.zeros_like(pad_truth) for _ in range(world_size)]
                dist.all_gather(gathered_preds, pad_pred)
                dist.all_gather(gathered_truths, pad_truth)
                y_preds = torch.cat([g[:sizes[i]] for i, g in enumerate(gathered_preds)], dim=0).cpu().numpy()
                y_truths = torch.cat([g[:sizes[i]] for i, g in enumerate(gathered_truths)], dim=0).cpu().numpy()
                # 仅 rank 0 写文件
                if not self._is_rank0():
                    return {}

            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + self.config['model'] + '_' + self.config['dataset'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            test_result = self.evaluator.save_result(self.evaluate_res_dir)
            return test_result

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
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
                # Type-2 diagnostics (if model supports it)
                try:
                    unwrapped = self._unwrap_model()
                    if hasattr(unwrapped, 'get_type2_diagnostics'):
                        diag = unwrapped.get_type2_diagnostics()
                        if diag:
                            parts = []
                            parts2 = []
                            parts3 = []
                            if 'beta' in diag:
                                parts.append('β=[{:.3f},{:.3f},{:.3f}]'.format(*diag['beta']))
                                if 'beta_entropy' in diag:
                                    parts[-1] += ' H={:.3f}'.format(diag['beta_entropy'])
                                if 'logits_std' in diag:
                                    parts[-1] += ' lσ={:.4f}'.format(diag['logits_std'])
                            if 'blend' in diag:
                                parts.append('blend={:.3f}'.format(diag['blend']))
                            if 'cell_blend' in diag:
                                parts.append('cell_b={:.3f}'.format(diag['cell_blend']))
                            if 'fou_mean' in diag:
                                parts2.append('FOU(μ={:.1e},σ={:.1e})'.format(
                                    diag['fou_mean'], diag['fou_std']))
                            if 'sigma_low_mean' in diag:
                                parts2.append('σ_low={:.4f}±{:.4f} σ_high={:.4f}±{:.4f} σ_δ={:.4f}±{:.4f}'.format(
                                    diag['sigma_low_mean'], diag['sigma_low_std'],
                                    diag['sigma_high_mean'], diag['sigma_high_std'],
                                    diag['sigma_high_mean'] - diag['sigma_low_mean'],
                                    diag.get('sigma_high_std', 0)))
                                if 'sigma_ratio' in diag:
                                    parts2[-1] += ' r={:.4f}'.format(diag['sigma_ratio'])
                            if 'loss_mae' in diag:
                                parts2.append('L=[mae={:.4f} ent={:.1e} gap={:.1e} fou={:.1e} consv={:.1e} pn={:.1e} ln={:.1e}]'.format(
                                    diag['loss_mae'],
                                    diag.get('loss_ent', 0),
                                    diag.get('loss_gap', 0),
                                    diag.get('loss_fou', 0),
                                    diag.get('loss_consv', 0),
                                    diag.get('loss_proto_norm', 0),
                                    diag.get('loss_latent_norm', 0)))
                            if 'raw_ent' in diag:
                                parts2.append('raw=[ent={:.3f} gap={:.4f} fou={:.4f}]'.format(
                                    diag['raw_ent'], diag['raw_gap'], diag['raw_fou']))
                            if 'd2_mean' in diag:
                                parts2.append('d²={:.1f}±{:.1f} μ_raw(L={:.3f},H={:.3f}) δ={:.2f} μΔ(m={:.4f},M={:.4f})'.format(
                                    diag['d2_mean'], diag['d2_std'],
                                    diag.get('mu_raw_low', 0), diag.get('mu_raw_high', 0),
                                    diag.get('sigma_delta_mean', 0),
                                    diag.get('mu_diff_mean', 0), diag.get('mu_diff_max', 0)))
                            if 'proto_norm' in diag:
                                parts2.append('|proto|={:.1f} |latent|={:.1f} Δc={:.1f} |W|={:.1f} Δp={:.2e} Δl={:.1f}'.format(
                                    diag['proto_norm'], diag['latent_norm'],
                                    diag['center_dist'],
                                    diag.get('transform_w', 0),
                                    diag.get('proto_up', 0),
                                    diag.get('latent_up', 0)))
                            if 'log_sigma_low_grad' in diag:
                                parts3.append('∇β={:.2e} σ_low={:.2e} δ={:.2e} proto={:.2e}'.format(
                                    diag.get('relation_mix_logits_grad', 0),
                                    diag['log_sigma_low_grad'],
                                    diag.get('log_sigma_delta_grad', 0),
                                    diag['prototype_center_grad']))
                            if 'R_gap' in diag:
                                parts3.append('R_gap={:.2f} w={:.1e}'.format(
                                    diag['R_gap'], diag.get('eff_width', 0)))
                            if 'beta_delta' in diag:
                                parts3.append('Δβ={:.6f}'.format(diag['beta_delta']))
                            if parts:
                                self._logger.info('  [T2] ' + ' | '.join(parts))
                            if parts2:
                                self._logger.info('  [T2σ] ' + ' | '.join(parts2))
                            if parts3:
                                self._logger.info('  [T2∇] ' + ' | '.join(parts3))
                except Exception:
                    pass  # diagnostics should never crash training

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
            self._barrier()  # 等待 rank 0 写完 checkpoint 文件
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            list: 每个batch的损失的数组
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self._unwrap_model().calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            with self._autocast_context():
                loss = loss_func(batch)
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
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 评估数据的平均损失值
        """
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self._unwrap_model().calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                with self._autocast_context():
                    loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            # DDP: all_reduce 求全局平均损失，保证所有 rank 的 val_loss 一致
            if self.is_distributed:
                import torch.distributed as dist
                dist.barrier()
                loss_tensor = torch.tensor([mean_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                mean_loss = loss_tensor.item()
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
