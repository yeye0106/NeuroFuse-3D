import os
import torch
import random
import numpy as np

def set_seed(seed=42):
    """固定所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 不强制 cuDNN 确定性，释放计算速度
    print(f"✅ 全局随机种子已固定为: {seed}")

class EarlyStopping:
    """早停机制：监控指标，连续 patience 轮未提升则停止"""
    def __init__(self, patience=15, mode='max', save_path='best_model.pth'):
        self.patience = patience
        self.mode = mode # 'max' 表示越高越好 (用于准确率)，'min' 表示越低越好 (用于Loss)
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif (self.mode == 'max' and current_score <= self.best_score) or \
             (self.mode == 'min' and current_score >= self.best_score):
            self.counter += 1
            print(f"⚠️ EarlyStopping 计数: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        print(f"💾 核心指标破纪录！已拦截并保存当前最佳模型至: {self.save_path}")