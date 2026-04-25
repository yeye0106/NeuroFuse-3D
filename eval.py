import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

from dataset import get_dataloaders
from models import CrossModalAttentionNetwork


# 解决画图时中文或负号显示问题（可选，如果图表有乱码可以取消注释）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

def evaluate():
    print("🔍 启动测试集终极评估引擎...")

    # 1. 加载配置与测试数据
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders("config.yaml")  # 我们只取 test_loader

    # 2. 加载最强模型权重
    model = CrossModalAttentionNetwork(config).to(device)
    best_weight_path = os.path.join(config['train']['save_dir'], "best_multimodal_lightweight.pth")

    if not os.path.exists(best_weight_path):
        print(f"❌ 找不到权重文件: {best_weight_path}")
        return

    model.load_state_dict(torch.load(best_weight_path, map_location=device))
    model.eval()
    print("✅ 成功加载最优模型权重，开始推理...")

    # 3. 收集预测结果
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for imgs, clins, labels in test_loader:
            imgs, clins, labels = imgs.to(device), clins.to(device), labels.to(device)
            outputs = model(imgs, clins)

            # 使用 softmax 获取概率 (用于 ROC 曲线)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ==========================================
    # 4. 打印量化指标 (分类报告)
    # ==========================================
    class_names = ['CN', 'MCI', 'AD']
    acc = accuracy_score(all_labels, all_preds)
    print("\n" + "=" * 50)
    print(f"🏆 测试集最终准确率 (Test Accuracy): {acc:.4%}")
    print("=" * 50)
    print("\n📄 详细分类报告 (Classification Report):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # ==========================================
    # 5. 绘制论文图表 1: 混淆矩阵
    # ==========================================
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    # 使用 seaborn 画出漂亮的热力图
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)  # 300dpi 是 SCI 期刊的基本要求
    print(f"📊 混淆矩阵已保存至: {cm_path}")

    # ==========================================
    # 6. 绘制论文图表 2: 多分类 ROC 曲线与 AUC
    # ==========================================
    # 将标签二值化用于绘制 ROC
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均 (micro-average) ROC 曲线和 AUC 面积
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:0.4f})',
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:0.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) to Multi-Class', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    roc_path = "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    print(f"📈 ROC 曲线已保存至: {roc_path}")


if __name__ == "__main__":
    evaluate()