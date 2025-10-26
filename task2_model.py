import numpy as np
import torch
import torch.nn as nn
import random
import math
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader, TensorDataset
import joblib

# 设置环境和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

seed = 25   # 25 99
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class BasicRegressor(nn.Module):
    def __init__(self, input_dim=606):
        super(BasicRegressor, self).__init__()

        self.channel_dim = input_dim // 3
        assert self.channel_dim * 3 == input_dim, "输入维度应该能被3整除"

        print(f"输入维度: {input_dim}, 每个通道维度: {self.channel_dim}")

        # 共享权重处理每个通道的特征
        self.channel_processor = nn.Sequential(
            nn.Linear(self.channel_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 合并通道信息后的处理 - 写成单独的层
        self.fc1 = nn.Linear(32 * 3, 64)  # 3个通道的特征拼接
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # 分割成三个通道
        r_channel = x[:, :self.channel_dim]
        g_channel = x[:, self.channel_dim:2 * self.channel_dim]
        b_channel = x[:, 2 * self.channel_dim:]

        # 并行处理每个通道
        r_features = self.channel_processor(r_channel)
        g_features = self.channel_processor(g_channel)
        b_features = self.channel_processor(b_channel)

        # 合并通道特征
        combined = torch.cat([r_features, g_features, b_features], dim=1)

        # 最终预测 - 使用单独的层
        x = self.fc1(combined)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x.squeeze(-1)


def detect_and_remove_outliers(X, y, contamination=0.1, method='one_class_svm'):
    """
    检测并去除异常样本 - 优化版
    """
    print(f"原始训练集样本数: {X.shape[0]}")
    print(f"计划去除 {contamination * 100}% 的异常样本，约 {int(X.shape[0] * contamination)} 个样本")

    if method == 'isolation_forest':
        detector = IsolationForest(contamination=contamination, random_state=seed)
    elif method == 'elliptic_envelope':
        detector = EllipticEnvelope(contamination=contamination, random_state=seed)
    elif method == 'local_outlier':
        detector = LocalOutlierFactor(contamination=contamination, novelty=True)
    elif method == 'one_class_svm':
        # 优化One-Class SVM参数
        detector = OneClassSVM(
            nu=contamination,
            kernel='rbf',           # 使用RBF核，对复杂边界更有效
            gamma='scale',          # 自动调整gamma参数
            cache_size=500          # 提高计算效率
        )
    else:
        raise ValueError("不支持的异常检测方法")

    # 拟合异常检测器
    outlier_labels = detector.fit_predict(X)

    # 获取正常样本的索引 (label = 1)
    normal_indices = np.where(outlier_labels == 1)[0]
    outlier_indices = np.where(outlier_labels == -1)[0]

    X_clean = X[normal_indices]
    y_clean = y[normal_indices]

    print(f"清洗后训练集样本数: {X_clean.shape[0]}")
    print(f"去除的异常样本数: {len(outlier_indices)}")
    print(f"实际异常比例: {len(outlier_indices)/len(X):.2%}")

    return X_clean, y_clean, outlier_indices


# 数据加载
print("加载数据...")
XTrain = np.loadtxt(r'D:\Research work\Imaging\Data20250805\Datasets\SVI_train.csv', delimiter=',')
XValid = np.loadtxt(r'D:\Research work\Imaging\Data20250805\Datasets\SVI_val.csv', delimiter=',')
XTest = np.loadtxt(r'D:\Research work\Imaging\Data20250805\Datasets\SVI_test.csv', delimiter=',')

YTrain = np.loadtxt(r'D:\Research work\Imaging\Data20250805\Datasets\LIBS_train.csv', delimiter=',')
YValid = np.loadtxt(r'D:\Research work\Imaging\Data20250805\Datasets\LIBS_val.csv', delimiter=',')
YTest = np.loadtxt(r'D:\Research work\Imaging\Data20250805\Datasets\LIBS_test.csv', delimiter=',')

print(f"原始数据形状 - 训练集: {XTrain.shape}, 验证集: {XValid.shape}, 测试集: {XTest.shape}")

# 异常样本去除（在训练集上）
print("\n=== 开始异常样本检测与去除 ===")

XTrain_clean, YTrain_clean, outlier_indices = detect_and_remove_outliers(
    XTrain, YTrain, contamination=0.1, method='one_class_svm'
)

print("=== 异常样本处理完成 ===\n")

# 数据标准化（使用清洗后的训练集进行拟合）
scaler = StandardScaler()
XTrain_scaled = scaler.fit_transform(XTrain_clean)  # 使用清洗后的数据
XValid_scaled = scaler.transform(XValid)
XTest_scaled = scaler.transform(XTest)

scaler_path = r'D:\Research work\Imaging\Data20250805\Datasets\scaler.save'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# 创建数据集实例（使用清洗后的数据）
train_set = TensorDataset(torch.FloatTensor(XTrain_scaled), torch.FloatTensor(YTrain_clean))
valid_set = TensorDataset(torch.FloatTensor(XValid_scaled), torch.FloatTensor(YValid))
test_set = TensorDataset(torch.FloatTensor(XTest_scaled), torch.FloatTensor(YTest))

batch_size = 32
learning_rate = 0.001
num_epochs = 100

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 初始化模型
model = BasicRegressor().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024
print(f"参数内存: {param_mem:.2f} KB")

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 训练准备
best_val_loss = float('inf')
best_model_path = r'D:\Research work\Imaging\Data20250805\Datasets\best_model.pth'
train_losses = []
val_losses = []
best_epoch = 0

# 早停参数
patience = 10
early_stop_counter = 0
min_delta = 1e-4

# 训练循环
print("开始训练...")
start_time = time.time()

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)

    train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            running_val_loss += criterion(outputs, labels).item() * inputs.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    # 更新学习率
    scheduler.step()

    # 保存最佳模型
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f'↳ 保存最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})')
    else:
        early_stop_counter += 1

    # 打印进度
    print(f'Epoch [{epoch + 1}/{num_epochs}] | '
          f'Train Loss: {train_loss:.4f} | '
          f'Val Loss: {val_loss:.4f} | '
          f'Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}) | '
          f'EarlyStop: {early_stop_counter}/{patience}')

    # 检查早停条件
    if early_stop_counter >= patience:
        print(f"\n早停触发！在 Epoch {epoch + 1} 停止训练。")
        print(f"连续 {patience} 个 epoch 验证损失没有显著改善。")
        break

if early_stop_counter < patience:
    print(f"\n训练完成，所有 {num_epochs} 个 epoch 已执行完毕。")

end_time = time.time()
print(f"总训练耗时: {(end_time - start_time) / 60:.2f} 分钟")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress with Outlier Removal')
plt.legend()
plt.grid(True)
plt.show()

# 加载最佳模型进行评估
model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"\n加载最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

# 评估验证集
val_true, val_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        val_true.extend(labels.cpu().numpy())
        val_pred.extend(outputs.cpu().numpy())

val_true = np.array(val_true)
val_pred = np.array(val_pred)
val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
val_mae = mean_absolute_error(val_true, val_pred)
val_r2 = r2_score(val_true, val_pred)
file_path = r"D:\Research work\Imaging\Data20250805\Datasets\reconstructed_val.csv"
np.savetxt(file_path, val_pred, delimiter=",")

print(f"\n验证集结果 (最佳模型): MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")

# 评估测试集
test_true, test_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_true.extend(labels.cpu().numpy())
        test_pred.extend(outputs.cpu().numpy())

test_true = np.array(test_true)
test_pred = np.array(test_pred)
test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
test_mae = mean_absolute_error(test_true, test_pred)
test_r2 = r2_score(test_true, test_pred)
file_path = r"D:\Research work\Imaging\Data20250805\Datasets\reconstructed_test.csv"
np.savetxt(file_path, test_pred, delimiter=",")

print(f"测试集结果 (最佳模型): MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")