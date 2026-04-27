import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from models import CrossModalAttentionNetwork
from utils import set_seed, EarlyStopping


def train():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config['data']['seed'])
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"🚀 启动 [最终基准定稿版] 训练引擎，使用设备: {device}")

    train_loader, val_loader, _ = get_dataloaders("config.yaml")

    model = CrossModalAttentionNetwork(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    # 稳扎稳打的学习率衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    save_dir = config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # 永远固定保存为 best_model.pth，不再改变！
    # ==========================================
    best_model_path = os.path.join(save_dir, "best_model.pth")

    early_stopping = EarlyStopping(patience=15, mode='max', save_path=best_model_path)

    epochs = config['train']['epochs']
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    for epoch in range(epochs):
        print(f"\n[{epoch + 1}/{epochs}] Epoch starting...")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar_train = tqdm(train_loader, desc=f"Training", leave=False)
        for imgs, clins, labels in pbar_train:
            imgs, clins, labels = imgs.to(device), clins.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs, clins)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs, clins)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar_train.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Validating", leave=False)
            for imgs, clins, labels in pbar_val:
                imgs, clins, labels = imgs.to(device), clins.to(device), labels.to(device)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(imgs, clins)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs, clins)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 总结 | Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4%}")
        print(f"         | Val Loss:   {avg_val_loss:.4f} - Val Acc:   {val_acc:.4%} | LR: {current_lr:.6f}")

        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f"🛑 触发早停机制！训练结束。")
            break

    print(f"\n🎉 最终定稿训练流程结束！权重已保存在: {best_model_path}")


if __name__ == "__main__":
    train()