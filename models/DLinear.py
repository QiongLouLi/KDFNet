import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

class HilbertJointSCDK(nn.Module):
    """
    HilbertJointSCDK: 同时使用希尔伯特包络 + 相位特征，
    生成一个样本自适应的时间-通道门控 g(B,D,L)，
    对 x 与 t 做凸组合:
        f = (1 - g) * x + g * t

    输入:
        x: (B, D, L)  实值时间序列
        t: (B, D, L)  实值先验 / learnable prior，形状与 x 相同

    输出:
        f: (B, D, L)  实值，Hilbert 条件融合后的序列
    """

    def __init__(
        self,
        hidden_channels: int = 16,
        kernel_size: int = 5,
        eps: float = 1e-6,
        enc_in: int =  21,
        d_model: int = 512,
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size should be odd, got {kernel_size}.")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}.")

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.eps = float(eps)

        self.tem = torch.nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)

        # 使用 6 个 Hilbert 特征通道:
        # [log A_x, log A_t, cos φ_x, sin φ_x, cos φ_t, sin φ_t]
        in_channels = 6

        self.hilbert_net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # "same" padding
            ),
            nn.GELU(),
            # nn.Conv1d(
            #     in_channels=hidden_channels,
            #     out_channels=1,
            #     kernel_size=kernel_size,
            #     padding=kernel_size // 2,
            # ),
            # nn.GELU(),
            # nn.Conv1d(
            #     in_channels=hidden_channels,
            #     out_channels=1,
            #     kernel_size=kernel_size,
            #     padding=kernel_size // 2,
            # )
        )

    # ------- 内部：希尔伯特解析信号（analytic signal） -------
    @staticmethod
    def _analytic_signal(x: torch.Tensor) -> torch.Tensor:
        """
        使用 FFT 构造解析信号:
            给定实信号 x(B,D,L)，
            返回 z = x + j * H{x} ∈ C^{B×D×L}
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B, D, L), got {x.shape}.")

        B, D, L = x.shape
        device = x.device

        # 全频 FFT
        X = torch.fft.fft(x, dim=-1)  # (B, D, L), complex

        # 构造长度为 L 的 Hilbert 滤波器 H[k]
        H = torch.zeros(L, dtype=X.dtype, device=device)
        H[0] = 1.0
        if L % 2 == 0:
            # 偶长度: [0, 1..L/2-1, L/2, L/2+1..L-1]
            H[1 : L // 2] = 2.0
            H[L // 2] = 1.0
        else:
            # 奇长度: [0, 1..(L-1)/2, (L+1)/2..L-1]
            H[1 : (L + 1) // 2] = 2.0

        H = H.view(1, 1, L)  # 方便广播到 (B,D,L)

        Z = torch.fft.ifft(X * H, dim=-1)  # (B, D, L), complex
        return Z


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, _ = x.shape
        # t = self.tem.expand(B, -1, -1)
        t = self.tem.expand(B, T, -1)

        B, D, L = x.shape
        device, dtype = x.device, x.dtype

        # 保证连续内存，避免 view 出问题
        x = x.contiguous()
        t = t.contiguous()

        # 1) 解析信号: z_x, z_t
        Zx = self._analytic_signal(x)  # (B, D, L), complex
        Zt = self._analytic_signal(t)  # (B, D, L), complex

        # 2) 包络 A_x, A_t
        Ax = torch.abs(Zx)  # (B, D, L), real
        At = torch.abs(Zt)  # (B, D, L), real

        Ax = Ax.clamp_min(self.eps)
        At = At.clamp_min(self.eps)

        logAx = torch.log(Ax)  # (B, D, L)
        logAt = torch.log(At)  # (B, D, L)

        # 3) 相位 φ_x, φ_t -> 用 cos φ, sin φ 表示，避免 2π 跳变
        phix = torch.angle(Zx)  # (B, D, L), in [-π, π]
        phit = torch.angle(Zt)  # (B, D, L)

        cos_phix = torch.cos(phix)
        sin_phix = torch.sin(phix)
        cos_phit = torch.cos(phit)
        sin_phit = torch.sin(phit)

        # 4) 组装 Hilbert 特征: [logAx, logAt, cosφx, sinφx, cosφt, sinφt]
        #    形状: (B, D, 6, L) -> (B*D, 6, L)
        feat = torch.stack(
            [logAx, logAt, cos_phix, sin_phix, cos_phit, sin_phit],
            dim=2,  # 在通道维上堆叠
        )  # (B, D, 6, L)

        feat = feat.view(B * D, 6, L)  # merge batch & channel -> (BD, 6, L)

        # 5) 一维卷积网络，沿时间轴生成门控 logits u
        u = self.hilbert_net(feat)  # (B*D, 1, L)
        u = u.view(B, D, L)         # (B, D, L)

        # 6) Sigmoid -> g ∈ (0,1)，作为 t 和 x 的融合权重
        g = torch.sigmoid(u)  # (B, D, L)

        # 7) 凸组合: f = (1 - g) * x + g * t
        f = (1.0 - g) * x + g * t   # (B, D, L)

        # 保持 dtype 一致
        f = f.to(dtype=dtype)

        return f


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)
        # self.SCDK01 = HilbertJointSCDK(hidden_channels=8, kernel_size=5, d_model=96)
        self.SCDK02 = HilbertJointSCDK(hidden_channels=8, kernel_size=5, d_model=96)

    def encoder(self, x):

        seasonal_init, trend_init = self.decompsition(x)

        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        # seasonal_init = self.SCDK01(seasonal_init)
        trend_init = self.SCDK02(trend_init)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None


