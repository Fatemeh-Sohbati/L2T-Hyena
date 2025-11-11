# teacher_gru.py (نسخه نهایی با قابلیت نمونه‌برداری از حافظه - اصلاح‌شده برای نمونه‌برداری متعادل)

import torch
import torch.nn as nn
from meta_module_extended import MetaModule, MetaLinear

class TeacherModel(MetaModule):
    def __init__(self, memory_size: int, memory_seq_len: int, hidden_size: int, **factory_kwargs):
        super().__init__()
        self.memory_size = memory_size
        self.memory_seq_len = memory_seq_len
        self.hidden_size = hidden_size
        self.input_feature_size = 5
        
        self.register_buffer('memory', torch.zeros((memory_size, memory_seq_len, self.input_feature_size), **factory_kwargs), persistent=False)
        self.register_buffer('memory_loss', torch.zeros((memory_size,), **factory_kwargs), persistent=False)
        self.register_buffer('memory_idx', torch.tensor(0, dtype=torch.long), persistent=False)
        
        input_dim_teacher = 1
        self.teacher_net = nn.Sequential(
            MetaLinear(input_dim_teacher, self.hidden_size, **factory_kwargs),
            nn.GELU(),
            MetaLinear(self.hidden_size, self.hidden_size, **factory_kwargs),
            nn.GELU(),
            MetaLinear(self.hidden_size, 1, **factory_kwargs)
        )

        with torch.no_grad():
            if hasattr(self.teacher_net[-1], 'bias') and self.teacher_net[-1].bias is not None:
                self.teacher_net[-1].bias.fill_(15.0) 

    def forward(self, weight_reg: torch.Tensor) -> torch.Tensor:
        if weight_reg.ndim == 1:
            weight_reg = weight_reg.unsqueeze(-1)
        predicted_losses = self.teacher_net(weight_reg).squeeze(-1)
        return predicted_losses

    def update_memory(self, features: torch.Tensor, student_loss_per_sample: torch.Tensor, batch_size: int):
        if features.ndim != 3 or features.shape[1] != self.memory_seq_len or features.shape[2] != self.input_feature_size:
             return 
        num_sequences_in_batch = features.shape[0]
        if student_loss_per_sample.shape[0] != num_sequences_in_batch:
            return 
        for i in range(num_sequences_in_batch):
            write_idx = self.memory_idx.item() % self.memory_size
            self.memory[write_idx] = features[i]
            self.memory_loss[write_idx] = student_loss_per_sample[i] 
            self.memory_idx += 1
            
    # ✅✅✅ متد جدید برای نمونه‌برداری از حافظه ✅✅✅
    def sample_memory(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """یک بچ تصادفی از ویژگی‌ها و زیان‌های تاریخی ذخیره شده را برمی‌گرداند."""
        
        # تعیین تعداد آیتم‌های معتبر در حافظه
        current_fill = min(self.memory_idx.item(), self.memory_size)
        
        if current_fill < batch_size:
            # اگر حافظه به اندازه کافی پر نیست، فعلاً داده‌های ساختگی برگردان
            return (torch.zeros((batch_size, self.memory_seq_len, self.input_feature_size), device=self.memory.device),
                    torch.zeros((batch_size,), device=self.memory_loss.device))

        # ✅ اصلاح: نمونه‌برداری متعادل‌تر با اضافه کردن noise به weights
        weights = torch.softmax(self.memory_loss[:current_fill] / 2.0, dim=0) * 0.5 + 0.5 / current_fill
        weights += torch.rand_like(weights) * 0.1  # اضافه کردن noise برای تنوع
        weights = weights / weights.sum()  # نرمال کردن دوباره
        sample_indices = torch.multinomial(weights, batch_size, replacement=True)
        
        # بازیابی داده‌های نمونه‌برداری شده
        sampled_features = self.memory[sample_indices]
        sampled_losses = self.memory_loss[sample_indices]
        
        return sampled_features, sampled_losses

    def get_memory(self) -> torch.Tensor:
        current_fill = min(self.memory_idx.item(), self.memory_size)
        if current_fill == 0:
             return torch.zeros((1, self.memory_seq_len, self.input_feature_size), device=self.memory.device)
        return self.memory[:current_fill]
