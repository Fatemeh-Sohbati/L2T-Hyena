# dln_lm.py (نسخه نهایی کاملاً صحیح - GRU با خروجی 0-1 - اصلاح‌شده برای جلوگیری از اشباع با BatchNorm)

import torch
import torch.nn as nn
import torch.nn.functional as F
from meta_module_extended import MetaModule, MetaLinear, MetaLayerNorm

class DynamicLossNetworkLM(MetaModule):
    def __init__(self, layer_size: int, n_layers: int, dropout: float = 0.1, **factory_kwargs):
        super().__init__()
        self.input_feature_size = 5
        self.gru_hidden_size = 64  # اندازه حالت پنهان برای GRU
        
        # (اصلاح معماری GRU - صحیح و باقی می‌ماند)
        self.feature_summarizer_gru = nn.GRU(
            input_size=self.input_feature_size,
            hidden_size=self.gru_hidden_size,
            num_layers=1,
            batch_first=True,
            **factory_kwargs
        )
        
        # ✅ اصلاح: اضافه کردن BatchNorm برای نرمال کردن summary_vector و جلوگیری از مقادیر بزرگ
        self.batch_norm = nn.BatchNorm1d(self.gru_hidden_size, **factory_kwargs)
        
        # شبکه MLP اکنون ورودی GRU (hidden_size) را می‌گیرد (صحیح و باقی می‌ماند)
        layers = []
        in_size = self.gru_hidden_size 
        
        for i in range(n_layers):
            layers.append(MetaLinear(in_size, layer_size, **factory_kwargs))
            layers.append(nn.GELU())
            if i < n_layers - 1:
                layers.append(nn.Dropout(dropout))
            in_size = layer_size
        
        layers.append(MetaLinear(layer_size, 1, **factory_kwargs))
        self.net = nn.Sequential(*layers) 
        
        # Residual connection برای بهبود جریان گرادیان
        self.residual = MetaLinear(self.gru_hidden_size, 1, **factory_kwargs)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # ۱. محاسبه ویژگی‌ها (صحیح و باقی می‌ماند)
        log_probs = F.log_softmax(logits, dim=-1)
        
        logp_correct = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(log_probs).scatter_(-1, targets.unsqueeze(-1), 0.)
        logp_wrong = log_probs + mask * -1e9
        logp_wrong_max = logp_wrong.max(dim=-1)[0]
        margin = logp_correct - logp_wrong_max
        logp_wrong_mean = (logp_wrong.sum(dim=-1) - logp_wrong_max) / (log_probs.size(-1) - 1)
        logp_wrong_std = torch.std(logp_wrong, dim=-1)
        
        features = torch.stack([logp_correct, logp_wrong_max, margin, logp_wrong_mean, logp_wrong_std], dim=-1)

        # ۲. پایدارسازی ویژگی‌ها (اصلاح ریشه‌ای - حیاتی و باقی می‌ماند)
        f_std = features.std(dim=1, keepdim=True).clamp(min=1e-4)
        features = (features - features.mean(dim=1, keepdim=True)) / f_std
        
        # ۳. پردازش دنباله با GRU (اصلاح معماری - صحیح و باقی می‌ماند)
        gru_output, hidden_state = self.feature_summarizer_gru(features)
        summary_vector = hidden_state.squeeze(0)
        
        # ✅ اعمال BatchNorm برای جلوگیری از saturation
        summary_vector = self.batch_norm(summary_vector)

        # ۴. محاسبه وزن (اکنون بر اساس خلاصه GRU)
        raw_weight_reg = self.net(summary_vector) + self.residual(summary_vector)
        
        # === ✅ اصلاح: Sigmoid با clamp برای جلوگیری از اشباع ===
        weight_reg_per_sample = torch.sigmoid(raw_weight_reg).clamp(0.05, 0.95).squeeze(-1)

        # 'features' نرمال‌شده برای ذخیره در حافظه برگردانده می‌شوند
        return weight_reg_per_sample, features