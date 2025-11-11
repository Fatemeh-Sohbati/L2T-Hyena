# train_hyena_l2t.py (نسخه اصلاح‌شده برای سرعت و مقابله با Overfitting - بدون Early Stopping)

import torch
import torch.optim as optim
import torch.nn as nn
import dataclasses
import json
import time
import math
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from contextlib import nullcontext 

from hyena_meta import MetaGPModel, HyenaConfig
from dln_lm import DynamicLossNetworkLM
from teacher_model import TeacherModel
from text_data_utils import process_text_dataset, collate_text, seed_torch, generate

@dataclasses.dataclass(kw_only=True)
class TrainConfig(HyenaConfig):
    d_model: int = 256
    n_layers: int = 6
    vocab_size: int = -1
    context_length: int = 64
    d_embed: int = 64
    d_filter_mlp: int = 256
    n_filter_layers: int = 4
    short_conv_size: int = 3
    order: int = 2
    pdrop_hyena: float = 0.15
    pdrop_embed: float = 0.15
    omega: float = 1.0
    
    learning_rate: float = 2e-4
    weight_decay: float = 0.15
    grad_clip_value: float = 1.0
    epochs: int = 10
    dataset_name: str = "penn_treebank"
    data_dir: str = "./data"
    train_batch_size: int = 128
    val_batch_size: int = 128
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    eval_interval: int = 1
    log_interval: int = 100
    dln_weight_reg: float = 1e-3
    reg_strength: float = 0.1
    memory_size: int = 500
    memory_seq_len: int = 64
    teacher_hidden_size: int = 128
    warmup_epochs: int = 2 
    # ❌ ۱. حذف پارامتر برای توقف زودهنگام
    # early_stopping_patience: int = 2

class L2TDLNTrainingSystem:
    def __init__(self, config: TrainConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        factory_kwargs = {'device': device}
        
        self.student = MetaGPModel(config, **factory_kwargs).to(device)
        self.dln = DynamicLossNetworkLM(config.d_filter_mlp, config.n_filter_layers, dropout=0.1, **factory_kwargs).to(device)
        self.teacher = TeacherModel(config.memory_size, config.memory_seq_len, config.teacher_hidden_size, **factory_kwargs).to(device)
        
        self.student_optimizer = optim.AdamW(self.student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.teacher_optimizer = optim.AdamW(self.teacher.parameters(), lr=2e-6, weight_decay=0.01)
        self.dln_optimizer = optim.AdamW(self.dln.parameters(), lr=5e-7, weight_decay=0.01)
        
        self.criterion_per_token = nn.CrossEntropyLoss(reduction='none') 
        self.criterion_mean = nn.CrossEntropyLoss() 
        self.teacher_criterion = nn.HuberLoss() 
        self.current_weights = 0.0

    def step(self, x: torch.Tensor, y: torch.Tensor, current_reg_strength: float):
        self.student_optimizer.zero_grad()
        self.teacher_optimizer.zero_grad()
        self.dln_optimizer.zero_grad()
        
        B, S = y.shape 
        ctx = nullcontext()
        with ctx:
            logits = self.student(x)
            ce_loss_per_sample = self.criterion_per_token(logits.view(-1, self.config.vocab_size), y.view(-1)).view(B, S).mean(dim=1)
            reg_loss_per_sample = current_reg_strength * (logits ** 2).mean(dim=(1, 2))
            live_weights_batch, live_features_batch = self.dln(logits, y)
            w_detached = live_weights_batch.detach()
            student_loss_per_sample = ((1.0 - w_detached) * ce_loss_per_sample) + (w_detached * reg_loss_per_sample)
            student_loss = student_loss_per_sample.mean()

            self.teacher.update_memory(live_features_batch.detach(), ce_loss_per_sample.detach(), B)
            mem_features, mem_ce_target = self.teacher.sample_memory(B)
            
            with torch.no_grad():
                target_mean = mem_ce_target.mean()
                target_std = mem_ce_target.std() + 1e-6
            stable_target = (mem_ce_target - target_mean) / target_std
            
            gru_output, hidden_state = self.dln.feature_summarizer_gru(mem_features)
            summary_vector = hidden_state.squeeze(0)
            mem_raw_weights = self.dln.net(summary_vector)

            mem_weights = torch.sigmoid(mem_raw_weights).squeeze(-1)
            mem_pred = self.teacher(mem_weights)
            
            teacher_reg_loss = 0.01 * sum(p.norm(2) for p in self.teacher.parameters())
            teacher_loss = self.teacher_criterion(mem_pred, stable_target.detach()) + teacher_reg_loss 

            logit_saturation_penalty = (torch.tanh(mem_raw_weights / 5.0) ** 2).mean() * 5e-1
            dln_objective_loss_per_sample = self.teacher_criterion(mem_pred, stable_target.detach())
            dln_weight_reg_loss = (mem_weights ** 2) * self.config.dln_weight_reg
            weight_diversity_loss = -torch.var(mem_weights) * 2.0
            raw_reg_penalty = torch.relu(mem_raw_weights.mean() - 1.0) ** 2 * 1.0
            
            dln_loss = (dln_objective_loss_per_sample + dln_weight_reg_loss + logit_saturation_penalty + weight_diversity_loss + raw_reg_penalty).mean()

            total_loss = student_loss + teacher_loss + dln_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"NaN/Inf detected. Skipping batch.")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        total_loss.backward()
        
        grad_norm_student = torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip_value)
        grad_norm_teacher = torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), 0.1)
        grad_norm_dln = torch.nn.utils.clip_grad_norm_(self.dln.parameters(), 1.0)
        
        self.student_optimizer.step()
        self.teacher_optimizer.step() 
        self.dln_optimizer.step()
        
        self.current_weights = live_weights_batch.mean().item()
        
        return student_loss.item(), teacher_loss.item(), dln_loss.item(), grad_norm_student, grad_norm_teacher, grad_norm_dln

def evaluate_and_generate(model, val_loader, criterion_mean, device, config, tok2id, id2tok, epoch):
    print(f"===== Evaluating Epoch {epoch+1} =====")
    model.eval()
    total_loss, total_steps = 0.0, 0
    print(f"Val Loader Size: {len(val_loader)}")
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion_mean(logits.view(-1, config.vocab_size), y.view(-1))
            total_loss += loss.item()
            total_steps += 1
    avg_val_loss = total_loss / total_steps if total_steps > 0 else float('inf')
    perplexity = math.exp(avg_val_loss) if avg_val_loss < 100 else float('inf')
    
    print(f"Validation Perplexity: {perplexity:.2f} | Avg Loss: {avg_val_loss:.4f}")
    prompt = "the meaning of life is"
    prompt_ids = torch.tensor([tok2id.get(word, 0) for word in prompt.split()]).to(device)
    generated_text = generate(model, tok2id, id2tok, prompt_ids, max_new_tokens=20, device=device)
    print(f"Generated Text: {generated_text}")
    
    with open("val_log.txt", "a") as f:
        f.write(f"Epoch {epoch+1}: Perplexity {perplexity:.2f}, Avg Loss: {avg_val_loss:.4f}\n")
    
    model.train()
    return perplexity

def train(config: TrainConfig):
    seed_torch(config.seed)
    device = torch.device(config.device)
    print(f"===== Config =====\n{json.dumps(dataclasses.asdict(config), indent=2)}\n")
    
    train_ds, val_ds, vocabulary, tok2id, id2tok = process_text_dataset(
        dataset_name=config.dataset_name, 
        context_length=config.context_length,
        data_dir=config.data_dir
    )
    config.vocab_size = len(vocabulary)
    
    train_loader = DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.val_batch_size, num_workers=0, drop_last=True, pin_memory=True)
    
    system = L2TDLNTrainingSystem(config, device)
    
    global global_step
    global_step = 0
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs 
    print(f"Total Steps: {total_steps} | Steps per Epoch: {steps_per_epoch} | Warmup Steps: {warmup_steps}")

    def create_scheduler_lambda(warmup, total):
        def scheduler_lambda(current_step):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup)) if warmup > 0 else 1.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup) / float(max(1, total - warmup)))))
        return scheduler_lambda
        
    student_scheduler = LambdaLR(system.student_optimizer, create_scheduler_lambda(warmup_steps, total_steps))
    teacher_scheduler = CosineAnnealingLR(system.teacher_optimizer, T_max=total_steps)
    dln_scheduler = CosineAnnealingLR(system.dln_optimizer, T_max=total_steps)

    best_val_perplexity = float('inf')
    # ❌ ۲. حذف شمارنده برای توقف زودهنگام
    # patience_counter = 0
    
    print(f"\n===== Starting L2T-DLN Training for {config.epochs} Epochs =====")
    
    for epoch in range(config.epochs):
        system.student.train(); system.teacher.train(); system.dln.train()
        total_student_loss = 0.0
        start_time = time.time()
        
        current_reg_strength = config.reg_strength
        
        for step, (t_x, t_y) in enumerate(train_loader):
            t_x = t_x.to(device)
            t_y = t_y.to(device)
            
            try:
                student_loss, teacher_loss, dln_loss, grad_s, grad_t, grad_d = system.step(t_x, t_y, current_reg_strength)
                total_student_loss += student_loss
            except Exception as e:
                print(f"Error at Step {global_step}: {e}")
                import traceback
                traceback.print_exc()
                system.student_optimizer.zero_grad()
                system.teacher_optimizer.zero_grad()
                system.dln_optimizer.zero_grad()
                continue
            
            student_scheduler.step()
            teacher_scheduler.step()
            dln_scheduler.step()
            
            global_step += 1
            
            if (global_step) % config.log_interval == 0: 
                print(f"Epoch {epoch+1} | Step {global_step} | "
                      f"Stu Loss: {student_loss:.3f} | Tea Loss: {teacher_loss:.3f} | DLN Loss: {dln_loss:.3f} | "
                      f"Avg W_REG: {system.current_weights:.3f} | Grads S/T/D: {grad_s:.2f}/{grad_t:.2f}/{grad_d:.2f}")
        
        avg_loss = total_student_loss / (step + 1)
        elapsed = time.time() - start_time
        print(f"\n===== Epoch {epoch+1} Summary =====")
        print(f"Time: {elapsed:.2f}s | Avg Student Loss: {avg_loss:.4f} | LR: {system.student_optimizer.param_groups[0]['lr']:.2e}")
        
        if (epoch + 1) % config.eval_interval == 0:
            val_perplexity = evaluate_and_generate(system.student, val_loader, system.criterion_mean, device, config, tok2id, id2tok, epoch) 
            print(f"Best Perplexity so far: {best_val_perplexity:.2f}")
            if val_perplexity < best_val_perplexity:
                best_val_perplexity = val_perplexity
                print(f"✔️ New best model found. PPL: {best_val_perplexity:.2f}. Saving...")
                torch.save(system.student.state_dict(), "best_L2T_HYENA_model.pth")
            # ❌ ۳. حذف کامل منطق توقف زودهنگام
            # else:
            #     patience_counter += 1
            #     print(f"Perplexity did not improve. Patience: {patience_counter}/{config.early_stopping_patience}")
            #     if patience_counter >= config.early_stopping_patience:
            #         print(f"Stopping early as validation perplexity has not improved for {config.early_stopping_patience} epochs.")
            #         break 
    
    print(f"\n===== Training Finished =====")
    print(f"Best Validation Perplexity: {best_val_perplexity:.2f}")

if __name__ == "__main__":
    config = TrainConfig()
    train(config)