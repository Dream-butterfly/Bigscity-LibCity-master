import json
import os
import time
import copy
from logging import getLogger

import numpy as np
import torch

from libcity.common.traffic_state_executor import TrafficStateExecutor
from libcity.utils import ensure_dir, tune


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def mse_np(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def rmse_np(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def mae_np(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def mape_np(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


class STGformerExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self._logger = getLogger()

    def train(self, train_dataloader, eval_dataloader):
        self._logger.info("Start training ...")
        min_val_loss = float("inf")
        wait = 0
        best_epoch = 0
        best_state_dict = None
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            train_loss = float(np.mean(losses))
            train_time.append(time.time() - start_time)
            self._writer.add_scalar("training loss", train_loss, epoch_idx)

            if self.lr_scheduler is not None and self.lr_scheduler_type.lower() != "reducelronplateau":
                self.lr_scheduler.step()

            eval_start = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx)
            eval_time.append(time.time() - eval_start)

            if self.lr_scheduler is not None and self.lr_scheduler_type.lower() == "reducelronplateau":
                self.lr_scheduler.step(val_loss)

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]["lr"]
                message = "Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s".format(
                    epoch_idx, self.epochs, train_loss, val_loss, log_lr, (time.time() - start_time)
                )
                self._logger.info(message)

            if self.hyper_tune:
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                best_state_dict = copy.deepcopy(self.model.state_dict())
                self._logger.info("Val loss decrease from {:.4f} to {:.4f}".format(min_val_loss, val_loss))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait >= self.patience and self.use_early_stop:
                    self._logger.warning("Early stopping at epoch: %d" % epoch_idx)
                    break

        if len(train_time) > 0:
            self._logger.info(
                "Trained totally {} epochs, average train time is {:.3f}s, average eval time is {:.3f}s".format(
                    len(train_time), sum(train_time) / len(train_time), sum(eval_time) / len(eval_time)
                )
            )
        if self.load_best_epoch and best_state_dict is not None:
            self._logger.info("Loading best model state from epoch {}".format(best_epoch))
            self.model.load_state_dict(best_state_dict)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx):
        with torch.no_grad():
            self.model.eval()
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                y_true = batch["y"][..., : self.output_dim]
                y_predicted = self.model.predict(batch)[..., : self.output_dim]
                y_true = self._scaler.inverse_transform(y_true)
                y_predicted = self._scaler.inverse_transform(y_predicted)
                loss = masked_mae_loss(y_predicted, y_true)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = float(np.mean(losses))
            self._writer.add_scalar("eval loss", mean_loss, epoch_idx)
            return mean_loss

    def evaluate(self, test_dataloader):
        self._logger.info("Start evaluating ...")
        with torch.no_grad():
            self.model.eval()
            y_truths = []
            y_preds = []
            start_time = time.time()
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch["y"][..., : self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., : self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
            end_time = time.time()
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)
            outputs = {"prediction": y_preds, "truth": y_truths}
            filename_prefix = (
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
                + "_"
                + self.config["model"]
                + "_"
                + self.config["dataset"]
            )
            ensure_dir(self.evaluate_res_dir)
            np.savez_compressed(
                os.path.join(self.evaluate_res_dir, f"{filename_prefix}_predictions.npz"), **outputs
            )
            result = self._calc_independent_metrics(y_truths, y_preds)
            result["inference_time"] = float(end_time - start_time)
            with open(
                os.path.join(self.evaluate_res_dir, f"{filename_prefix}_independent_metrics.json"),
                "w",
                encoding="utf-8",
            ) as metrics_file:
                json.dump(result, metrics_file, ensure_ascii=False, indent=2)
            self._logger.info("Independent-style test result is {}".format(json.dumps(result)))
            return result

    @staticmethod
    def _calc_independent_metrics(y_true, y_pred):
        if y_true.ndim == 4 and y_true.shape[-1] == 1:
            y_true = y_true[..., 0]
            y_pred = y_pred[..., 0]

        rmse_all = float(rmse_np(y_true, y_pred))
        mae_all = float(mae_np(y_true, y_pred))
        mape_all = float(mape_np(y_true, y_pred))
        result = {
            "RMSE@all": rmse_all,
            "MAE@all": mae_all,
            "MAPE@all": mape_all,
        }
        out_steps = y_pred.shape[1]
        for i in range(out_steps):
            result[f"RMSE@{i + 1}"] = float(rmse_np(y_true[:, i, :], y_pred[:, i, :]))
            result[f"MAE@{i + 1}"] = float(mae_np(y_true[:, i, :], y_pred[:, i, :]))
            result[f"MAPE@{i + 1}"] = float(mape_np(y_true[:, i, :], y_pred[:, i, :]))
        result["MSE@all"] = float(mse_np(y_true, y_pred))
        return result
