import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from layers.RevIN import RevIN


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

    # ------- 内部：输入检查 -------
    @staticmethod
    def _check_input(x: torch.Tensor, t: torch.Tensor):
        if not torch.is_tensor(x) or not torch.is_tensor(t):
            raise TypeError("x and t must be torch.Tensor.")

        if x.shape != t.shape:
            raise ValueError(f"x and t must have the same shape, got {x.shape} and {t.shape}.")

        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B, D, L), got {x.shape}.")
        if t.ndim != 3:
            raise ValueError(f"t must be 3D (B, D, L), got {t.shape}.")

        if not torch.is_floating_point(x) or not torch.is_floating_point(t):
            raise TypeError("x and t must be real-valued floating tensors.")

        if x.is_complex() or t.is_complex():
            raise TypeError("x and t must be real (not complex).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, L)
            t: (B, D, L)

        Returns:
            f: (B, D, L)
        """
        B, T, _ = x.shape
        # t = self.tem.expand(B, -1, -1)
        t = self.tem.expand(B, T, -1)

        self._check_input(x, t)

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


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_revin = True
        self.revin_layer = RevIN(configs.enc_in, affine=True)
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

        self.en_att01 = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True, dropout=0.5)
        self.en_att02 = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True, dropout=0.5)

        self.SCDK = HilbertJointSCDK(hidden_channels=8, kernel_size=5, enc_in=configs.enc_in, d_model=configs.d_model)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, mode='norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev


        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.SCDK(enc_out)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        # enc_out, attns = self.encoder(enc_out)
        enc_out = self.en_att01(enc_out, enc_out, enc_out)[0] + enc_out
        # enc_out = self.en_att02(enc_out, enc_out, enc_out)[0] + enc_out

        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        if self.use_revin:
            dec_out = self.revin_layer(dec_out, mode='denorm')
        else:
            dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
