# text_data_utils.py (نسخه نهایی با خواندن مستقیم فایل‌های متنی PTB)

import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import requests
from torch.utils.data import Dataset

def seed_torch(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class TextDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y

def download_ptb(data_dir):
    """Downloads the Penn Treebank dataset raw text files."""
    os.makedirs(data_dir, exist_ok=True)
    urls = {
        "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
        "valid": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
        "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
    }
    for split, url in urls.items():
        path = os.path.join(data_dir, f'ptb.{split}.txt')
        if not os.path.exists(path):
            print(f"Downloading {split} data to {path}...")
            r = requests.get(url)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(r.text)
        else:
            print(f"{split} data already exists at {path}.")
    print("Dataset download check complete.")

def process_text_dataset(dataset_name, context_length, data_dir="./data", **kwargs):
    if dataset_name == "penn_treebank":
        print("Processing Penn Treebank (PTB) dataset from local text files...")
        
        # ۱. دانلود فایل‌های دیتاست در صورت نیاز
        download_ptb(data_dir)
        
        # ۲. خواندن فایل‌های متنی
        splits = {}
        for split_name in ['train', 'valid', 'test']:
            path = os.path.join(data_dir, f'ptb.{split_name}.txt')
            with open(path, 'r', encoding='utf-8') as f:
                splits[split_name] = f.read().replace('\n', '<eos>').split()

        # ۳. ساخت دیکشنری از داده‌های آموزشی
        train_tokens = splits['train']
        vocabulary = sorted(list(set(train_tokens)))
        tok2id = {tok: i for i, tok in enumerate(vocabulary)}
        id2tok = {i: tok for i, tok in enumerate(vocabulary)}
        unk_token_id = tok2id.get('<unk>', 0)

        # ۴. تبدیل داده‌ها به ID
        train_data = torch.tensor([tok2id.get(token, unk_token_id) for token in splits['train']], dtype=torch.long)
        val_data = torch.tensor([tok2id.get(token, unk_token_id) for token in splits['valid']], dtype=torch.long)

        # ۵. ساخت دیتاست‌های پایتورچ
        train_ds = TextDataset(train_data, context_length)
        val_ds = TextDataset(val_data, context_length)
        
        print(f"PTB processing complete. Vocab size: {len(vocabulary)}")
        print(f"Train size: {len(train_ds)} sequences, Val size: {len(val_ds)} sequences")

        return train_ds, val_ds, vocabulary, tok2id, id2tok
    else:
        # کد مربوط به دیتاست قبلی شما (jsonl_custom)
        # در صورت نیاز می‌توانید آن را در اینجا قرار دهید
        raise ValueError(f"This script is now configured for 'penn_treebank'.")

def collate_text(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    return x, y

@torch.no_grad()
def generate(model, tok2id, id2tok, prompt_ids, max_new_tokens, device):
    model.eval()
    context = prompt_ids.unsqueeze(0).to(device)  # فرض: prompt_ids یک tensor است
    for _ in range(max_new_tokens):
        context_cond = context if context.size(1) <= model.config.context_length else context[:, -model.config.context_length:]
        logits = model(context_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token_id), dim=1)
    generated_ids = context[0].tolist()
    
    try:
        eos_idx = generated_ids.index(tok2id['<eos>'])
        generated_ids = generated_ids[:eos_idx]
    except (ValueError, KeyError):
        pass
    
    generated_text = ' '.join([id2tok.get(i, '<unk>') for i in generated_ids])
    return generated_text