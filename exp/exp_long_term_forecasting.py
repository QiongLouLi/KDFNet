from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
import math
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

class LossA_HeteroLaplaceNLL_Param(nn.Module):
    """
    改进版A：幂律异方差，尺度/幂指数均可学习（每通道 σ_c + 全局 α）
    - 用 Laplace 一致性常数 ln 2 修正 MAD
    - b_{l,c} = softplus(σ_c) * l^{softplus(α)}
    """
    def __init__(self, num_channels: int, alpha_init: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 每通道可学习尺度（对数/软阈参数化，初始先置 0 ~ 1 的数量级）
        self.log_sigma_c = nn.Parameter(torch.zeros(num_channels))   # -> softplus 后约 ~0.693
        # 可学习幂指数，softplus 保证 α >= 0
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))
        self._ln2 = math.log(2.0)

    @torch.no_grad()
    def calibrate_from_gt(self, gt: torch.Tensor):
        """
        可选：用 gt 的 MAD 粗校准每通道初值（更贴近数据量级）
        gt: (B, L, C)
        """
        median_t = gt.median(dim=1, keepdim=True).values         # (B,1,C)
        mad = (gt - median_t).abs().median(dim=1, keepdim=True).values  # (B,1,C)
        # Laplace 下 b ≈ MAD / ln2，按通道平均
        b_init = (mad / self._ln2).mean(dim=(0,1)).clamp_min(self.eps)  # (C,)
        # 反 softplus 映射到参数空间（近似）
        self.log_sigma_c.data = torch.log(torch.expm1(b_init))   # softplus(log_sigma) ≈ b_init

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B, L, C = pred.shape
        e = (pred - gt).abs()

        device, dtype = pred.device, pred.dtype

        l = torch.arange(1, L + 1, device=device, dtype=dtype).view(1, L, 1)  # 保证在 pred.device 上
        sigma_c = F.softplus(self.log_sigma_c).to(device=device, dtype=dtype).view(1, 1, C)
        alpha = F.softplus(self.alpha_raw).to(device=device, dtype=dtype)

        b = sigma_c * torch.pow(l, alpha) + self.eps

        loss = e / b + torch.log(b)
        return loss.mean()

class HeteroLaplace_loss(nn.Module):
    def __init__(self, num_channels: int, alpha_init: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.log_sigma_c = nn.Parameter(torch.zeros(num_channels))
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))
        self._ln2 = math.log(2.0)

    @torch.no_grad()
    def calibrate_from_gt(self, gt: torch.Tensor):

        median_t = gt.median(dim=1, keepdim=True).values  # (B,1,C)
        mad = (gt - median_t).abs().median(dim=1, keepdim=True).values  # (B,1,C)
        b_init = (mad / self._ln2).mean(dim=(0, 1)).clamp_min(self.eps)  # (C,)
        self.log_sigma_c.data = torch.log(torch.expm1(b_init))  # softplus(log_sigma) ≈ b_init

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B, L, C = pred.shape
        e = (pred - gt).abs()

        device, dtype = pred.device, pred.dtype

        l = torch.arange(1, L + 1, device=device, dtype=dtype).view(1, L, 1)
        sigma_c = F.softplus(self.log_sigma_c).to(device=device, dtype=dtype).view(1, 1, C)
        alpha = F.softplus(self.alpha_raw).to(device=device, dtype=dtype)

        b = sigma_c * torch.pow(l, alpha) + self.eps

        loss = e / b + torch.log(b)
        return loss.mean()


class WeightedL1Loss:
    def __init__(self, alpha, loss_mode):
        # alpha: 衰减因子，控制预测步长越远，权重衰减的速度
        # loss_mode: 损失模式，支持 'L1'（MAE）、'L2'（MSE）、'L1L2'（两者结合）
        self.alpha = alpha
        self.loss_mode = loss_mode

        # 定义一个可学习的权重参数 (1,96,1)，初始化为全1
        # 通常96表示预测长度，requires_grad=True意味着这些权重可以在训练中更新
        self.weights = nn.Parameter(torch.ones(1, 96, 1, dtype=torch.float32), requires_grad=True)

        # 根据loss_mode选择损失函数，reduction='none' 保留逐元素损失，方便后续加权
        if self.loss_mode == 'L1':
            self.loss_fun = nn.L1Loss(reduction='none')  # 平均绝对误差（MAE）
        elif self.loss_mode == 'L2':
            self.loss_fun = nn.MSELoss(reduction='none')  # 均方误差（MSE）
        elif self.loss_mode == 'L1L2':
            # 同时定义L1和L2，后面会组合使用
            self.loss_fun1 = nn.L1Loss(reduction='none')
            self.loss_fun2 = nn.MSELoss(reduction='none')

    def __call__(self, pred, gt):
        # pred: 预测值，形状 [batch_size, seq_len, feature_dim]
        # gt:   真实值，形状同pred
        # [b,l,n] 表示 [batch_size, 序列长度, 特征维度]

        # 正常预测任务
        L = pred.shape[1]  # 序列长度（预测步数）
        # 1. 生成时间步索引 [1, 2, ..., L]，用于计算每个时间步的权重
        time_steps = torch.arange(1, L + 1, device=pred.device, dtype=pred.dtype)
        # 2. 计算衰减权重 l^{-alpha}，越远的时间步，权重越小
        decay_weights = time_steps.pow(-self.alpha)
        # 3. 调整权重张量形状为 (1, L, 1)，方便与 (batch_size, L, feature_dim) 广播
        weights = decay_weights.unsqueeze(0).unsqueeze(-1)

        if self.loss_mode in ['L1', 'L2']:
            # 计算逐元素损失（形状和pred相同）
            loss_vec = self.loss_fun(pred, gt)
            # 按时间步加权，最后取所有元素的均值
            weightedLoss = torch.mean(loss_vec * weights)
            # 注意：下面被注释掉的代码，是用可学习的 self.weights 替代固定的衰减权重
            # weightedLoss = torch.mean(loss_vec * self.weights.to(pred.device))

        # 返回最终的加权损失
        return weightedLoss



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.gpu = args.gpu

    # ============================
    def _count_trainable_params(self):
        """
        Count trainable parameters of the current model.
        Return:
            params_num: total trainable parameter number
            params_m: parameter number in millions
        """
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_m = params_num / 1e6
        return params_num, params_m

    def _cuda_sync(self):
        """
        Synchronize CUDA for accurate timing.
        """
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
    # =============================

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # criterion = HeteroLaplace_loss(num_channels=self.args.loss_channels)
        # criterion = WeightedL1Loss(self.args.lossfun_alpha, self.args.loss_mode)

        # ================= Efficiency: parameter count =================
        params_num, params_m = self._count_trainable_params()
        print(f"[Efficiency] Trainable Params: {params_num} ({params_m:.4f} M)")
        # ===============================================================

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time_list = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # =====================================
            epoch_cost = time.time() - epoch_time
            epoch_time_list.append(epoch_cost)
            print("Epoch: {} cost time: {:.4f}".format(epoch + 1, epoch_cost))
            # =====================================

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # 修改为：
        # 先加载到CPU
        checkpoint = torch.load(best_model_path, map_location='cpu')
        # # # 再加载到模型并转移到指定GPU
        self.model.load_state_dict(checkpoint)
        self.model = self.model.cuda(f'cuda:{self.gpu}')  # 使用第一个指定的GPU
        # # print(f"Max Memory (MB): {max_memory}")
        #
        # return self.model

        # ================= Efficiency: average training time =================
        if len(epoch_time_list) > 0:
            avg_train_time = float(np.mean(epoch_time_list))
            print(f"[Efficiency] Avg Train Time: {avg_train_time:.4f} s/epoch "
                  f"over {len(epoch_time_list)} epoch(s)")
        # ====================================================================

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        self.model.eval()
        # ================= Efficiency: inference latency =================
        latency_list = []
        warmup_batches = 5  # 前几个 batch 用于 warm-up，不计入统计
        max_measure_batches = 100  # 最多统计 100 个 batch，避免太慢
        # =================================================================

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # else:
                #     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # ==============================================
                # encoder - decoder
                self._cuda_sync()
                infer_start = time.time()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                self._cuda_sync()
                infer_end = time.time()

                # 只统计模型 forward 时间，不包括数据搬运、反归一化、画图和保存
                if i >= warmup_batches and len(latency_list) < max_measure_batches:
                    latency_list.append((infer_end - infer_start) * 1000.0)  # ms/batch
                # ==============================================

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        # ================= Efficiency: print inference latency =================
        if len(latency_list) > 0:
            avg_latency = float(np.mean(latency_list))
            std_latency = float(np.std(latency_list))
            print(f"[Efficiency] Inference Latency: {avg_latency:.4f} ± {std_latency:.4f} ms/batch "
                  f"(warmup={warmup_batches}, measured={len(latency_list)} batches)")
        else:
            avg_latency = None
            print("[Efficiency] Inference Latency: Not measured.")
        # ======================================================================
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        # ======================================
        # abs_err = np.abs(preds - trues)
        # abs_err = abs_err.reshape(-1)
        # q50 = np.quantile(abs_err, 0.50)
        # q75 = np.quantile(abs_err, 0.75)
        # q90 = np.quantile(abs_err, 0.90)
        # q95 = np.quantile(abs_err, 0.95)
        #
        # print("Q50:", q50)
        # print("Q75:", q75)
        # print("Q90:", q90)
        # print("Q95:", q95)
        # ======================================


        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
