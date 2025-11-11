# hyena_meta.py (نسخه نهایی و پایدار)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union
import dataclasses

from meta_module_extended import (
    MetaModule, MetaLinear, MetaConv1d, MetaLayerNorm, MetaEmbedding
)

@dataclasses.dataclass(kw_only=True)
class HyenaConfigBase:
    d_model: int
    n_layers: int
    vocab_size: int
    context_length: int
    pdrop_embed: float = 0.1

@dataclasses.dataclass(kw_only=True)
class HyenaConfig(HyenaConfigBase):
    d_embed: int
    d_filter_mlp: int
    n_filter_layers: int
    short_conv_size: int
    order: int
    pdrop_hyena: float
    omega: Optional[int]

class FFTConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h: torch.Tensor, x: torch.Tensor, B: Optional[torch.Tensor] = None) -> torch.Tensor:
        L = h.shape[-1]
        h_f = torch.fft.rfft(h, n=2 * L, norm="forward")
        x_f = torch.fft.rfft(x.to(dtype=h.dtype), n=2 * L)
        y_f = h_f * x_f
        y = torch.fft.irfft(y_f, n=2 * L, norm="forward")
        y = y[..., :L]
        if B is not None:
             y = y + x * B
        return y.to(dtype=x.dtype)


class MetaProjection(MetaModule):
    def __init__(self, d_model: int, N: int, conv_len: int,
                 device: Union[str, torch.device, None] = None, dtype: Union[torch.dtype, None] = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.d_model = d_model
        self.N = N
        total_dim = d_model * (N + 1)

        self.linear = MetaLinear(d_model, total_dim, bias=True, **factory_kwargs)
        self.conv = MetaConv1d(
            in_channels=total_dim, out_channels=total_dim,
            kernel_size=conv_len, groups=total_dim,
            padding=conv_len - 1, bias=True, **factory_kwargs
        )

    def forward(self, u: torch.Tensor) -> List[torch.Tensor]:
        z = self.linear(u)
        z = z.transpose(1, 2)
        L = z.shape[2]
        z_conv = self.conv(z)
        z_conv = z_conv[..., :L]
        x = torch.split(z_conv, self.d_model, dim=1)
        return list(x)

# ✅ اصلاح کلیدی: جایگزینی Window ناپایدار با گیت Sigmoid پایدار
class MetaSigmoidGate(MetaModule):
    """یک مکانیزم گیتینگ پایدار و قابل یادگیری با استفاده از Sigmoid."""
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.gate_proj = MetaLinear(d_model, d_model, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Order, D_Model, Seq_Len]
        # Reshape for linear layer: -> [Order, Seq_Len, D_Model]
        x_reshaped = x.transpose(1, 2)
        gate_values = torch.sigmoid(self.gate_proj(x_reshaped))
        # Apply gate and reshape back: -> [Order, D_Model, Seq_Len]
        return (x * gate_values.transpose(1, 2))


class MetaHyenaFilter(MetaModule):
    def __init__(
        self, d_model: int, order: int, context_length: int,
        device: Union[str, torch.device, None] = None, dtype: Union[torch.dtype, None] = None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.N = order
        self.context_length = context_length

        h_data = torch.randn((order, d_model, context_length), **factory_kwargs)
        self.h_basis = nn.Parameter(h_data)

        # ✅ استفاده از گیت پایدار جدید
        self.gate = MetaSigmoidGate(d_model, **factory_kwargs)

    def forward(self, L: int) -> torch.Tensor:
        if L > self.context_length:
            raise ValueError(f"Requested filter length L ({L}) exceeds context_length ({self.context_length})")
        
        h = self.h_basis[:, :, :L]
        h_gated = self.gate(h)
        
        # نرمال‌سازی L1 برای پایداری
        norm = torch.norm(h_gated, p=1, dim=-1, keepdim=True)
        h_normalized = h_gated / (norm + 1e-6)

        return h_normalized


class MetaHyenaBlock(MetaModule):
    def __init__(self, config: HyenaConfig,
                 device: Union[str, torch.device, None] = None, dtype: Union[torch.dtype, None] = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.config = config

        self.proj_input = MetaProjection(config.d_model, config.order, config.short_conv_size, **factory_kwargs)
        self.proj_output = MetaLinear(config.d_model, config.d_model, bias=True, **factory_kwargs)
        self.filter = MetaHyenaFilter(
            d_model=config.d_model, order=config.order,
            context_length=config.context_length, **factory_kwargs
        )
        self.dropout = nn.Dropout(config.pdrop_hyena)
        self.fft_conv = FFTConv()
        B_data = torch.randn((config.order, config.d_model, 1), **factory_kwargs)
        self.B = nn.Parameter(B_data)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        L = u.shape[1]
        batch_size = u.shape[0]
        x_list = self.proj_input(u)
        *xs, v = x_list
        h = self.filter(L)
        y = v
        for i in range(self.config.order):
            x_i = xs[i]
            h_i = h[i].unsqueeze(0).expand(batch_size, -1, -1)
            B_i = self.B[i].unsqueeze(0)
            conv_out = self.fft_conv(h_i, y, B_i)
            y = y + self.dropout(x_i * conv_out)
        y = y.transpose(1, 2)
        y_proj = self.proj_output(y)
        y_out = u + self.dropout(y_proj)
        return y_out


class MetaGPModel(MetaModule):
    def __init__(self, config: HyenaConfig,
                 device: Union[str, torch.device, None] = None, dtype: Union[torch.dtype, None] = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.config = config

        self.tok_emb = MetaEmbedding(config.vocab_size, config.d_model, **factory_kwargs)
        pos_emb_data = torch.randn(1, config.context_length, config.d_model, **factory_kwargs)
        nn.init.normal_(pos_emb_data, mean=0.0, std=0.02)
        self.pos_emb = nn.Parameter(pos_emb_data)
        self.drop_emb = nn.Dropout(config.pdrop_embed)
        self.layers = nn.ModuleList([MetaHyenaBlock(config, **factory_kwargs) for _ in range(config.n_layers)])
        self.ln_f = MetaLayerNorm(config.d_model, eps=1e-5, **factory_kwargs)
        self.head = MetaLinear(config.d_model, config.vocab_size, bias=False, **factory_kwargs)
        self.tie_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len = x.shape
        if seq_len > self.config.context_length:
            raise ValueError(f"Input seq_len ({seq_len}) exceeds model context_length ({self.config.context_length})")

        token_embeddings = self.tok_emb(x) 
        position_embeddings = self.pos_emb[:, :seq_len, :] 
        h = self.drop_emb(token_embeddings + position_embeddings) 
        for layer in self.layers:
            h = layer(h) 
        h = self.ln_f(h) 
        logits = self.head(h) 
        return logits

    def named_meta_parameters(self, prefix: str = '', recurse: bool = True):
        if hasattr(self, 'pos_emb'):
            yield f"{prefix}{'.' if prefix else ''}pos_emb", self.pos_emb
        if not recurse:
            return
        for name, module in self.named_children():
            if module is not None:
                submodule_prefix = f"{prefix}{'.' if prefix else ''}{name}"
                if hasattr(module, 'named_meta_parameters'):
                    param_iterator = module.named_meta_parameters(prefix=submodule_prefix, recurse=True)
                else:
                    param_iterator = module.named_parameters(prefix=submodule_prefix, recurse=True)
                for sub_name, param in param_iterator:
                    yield sub_name, param
    
    named_parameters = named_meta_parameters

    def tie_weights(self):
        if hasattr(self.head, 'weight') and self.head.weight is not None:
            self.head.weight = self.tok_emb.weight
            print("Successfully tied token embedding and output head weights.")
        else:
            print("Warning: Could not tie weights. Head or token embedding may be missing a 'weight' attribute.")