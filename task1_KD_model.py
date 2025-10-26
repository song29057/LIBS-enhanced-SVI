import numpy as np
import torch
import torch.nn as nn
import random
import math
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

seed = 90   # 90    27
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 教师模型定义（与原始代码一致）
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.ln1 = nn.LayerNorm(648)
        self.fc1 = nn.Linear(648, 32)
        self.relu1 = nn.ReLU()
        self.ln3 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.ln1(x)
        feature = self.fc1(x)
        x = self.relu1(feature)  # 保存特征用于蒸馏

        x = self.ln3(x)
        x = self.fc3(x)
        x = x.squeeze(-1)

        return x, feature


# 学生模型定义（修改后支持特征蒸馏）
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # 特征提取层
        self.ln1 = nn.LayerNorm(606)
        self.fc1 = nn.Linear(606, 32)
        self.relu1 = nn.ReLU()

        # 特征适配层（用于与教师特征对齐）
        self.adapter2 = nn.Linear(32, 32)
        self.adapter3 = nn.Tanh()

        # 输出层
        self.ln2 = nn.LayerNorm(32)  # 修正为32
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.ln1(x)
        student_feature = self.fc1(x)
        x = self.relu1(student_feature)

        # 特征适配
        adapted_feature = self.adapter2(student_feature)
        adapted_feature = self.adapter3(adapted_feature)

        # 输出预测
        x = self.ln2(x)
        x = self.fc2(x)
        x = x.squeeze(-1)

        return x, adapted_feature, student_feature


class SpecData(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        y = self.y[idx]
        return np.array(x), np.array(y)

    def __len__(self):
        return len(self.y)


# 加载学生数据（SVI）
XTrain_SVI = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/SVI_train.csv', delimiter=',')
XValid_SVI = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/SVI_val.csv', delimiter=',')
XTest_SVI = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/SVI_test.csv', delimiter=',')
YTrain = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/Y_train.csv', delimiter=',')
YValid = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/Y_val.csv', delimiter=',')
YTest = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/Y_test.csv', delimiter=',')

# 加载教师数据（LIBS）用于获取教师特征
XTrain_LIBS = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/LIBS_train.csv', delimiter=',')
XValid_LIBS = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/LIBS_val.csv', delimiter=',')
XTest_LIBS = np.loadtxt(r'D:\Research work\Imaging\Data20251020\Datasets/LIBS_test.csv', delimiter=',')

# 创建数据集实例
train_set = SpecData(XTrain_SVI, YTrain)
valid_set = SpecData(XValid_SVI, YValid)
test_set = SpecData(XTest_SVI, YTest)

# 为教师模型创建数据加载器（使用LIBS数据）
teacher_train_set = SpecData(XTrain_LIBS, YTrain)
teacher_val_set = SpecData(XValid_LIBS, YValid)
teacher_test_set = SpecData(XTest_LIBS, YTest)

batch_size = 16
learning_rate = 0.01
num_epochs = 100
distill_alpha = 0.1  # 蒸馏损失权重
task_alpha = 0.9  # 任务损失权重

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 教师模型的数据加载器
teacher_train_loader = DataLoader(teacher_train_set, batch_size=batch_size, shuffle=True)
teacher_val_loader = DataLoader(teacher_val_set, batch_size=batch_size, shuffle=False)
teacher_test_loader = DataLoader(teacher_test_set, batch_size=batch_size, shuffle=False)

# 加载预训练的教师模型
teacher_model = TeacherModel().to(device)
teacher_model_path = r'D:\Research work\Imaging\Data20251020\Datasets\LIBS_best_model.pth'
teacher_model.load_state_dict(torch.load(teacher_model_path))
teacher_model.eval()  # 设置为评估模式
# 冻结教师模型参数
for param in teacher_model.parameters():
    param.requires_grad = False

print("教师模型加载完成并冻结")

# 初始化学生模型
student_model = StudentModel().to(device)

total_params = sum(p.numel() for p in student_model.parameters())
print(f"学生模型总参数量: {total_params:,}")

# 损失函数和优化器
criterion_task = nn.L1Loss()  # 任务损失
criterion_distill = nn.CosineEmbeddingLoss()  # 蒸馏损失（余弦相似度）

optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 训练准备
best_val_loss = float('inf')
best_model_path = r'D:\Research work\Imaging\Data20251020\Datasets\SVI_distill_best_model.pth'
train_losses = []
val_losses = []
distill_losses = []
task_losses = []
best_epoch = 0

# 早停机制参数
patience = 10
no_improve_epochs = 0
early_stop = False

print("开始知识蒸馏训练...")
start_time = time.time()

for epoch in range(num_epochs):
    if early_stop:
        print(f"早停触发，停止训练。最佳 epoch: {best_epoch}")
        break

    # 训练阶段
    student_model.train()
    running_train_loss = 0.0
    running_distill_loss = 0.0
    running_task_loss = 0.0

    # 同时迭代学生和教师数据加载器
    for (student_inputs, labels), (teacher_inputs, _) in zip(train_loader, teacher_train_loader):
        student_inputs, labels = student_inputs.to(device), labels.to(device)
        teacher_inputs = teacher_inputs.to(device)

        optimizer.zero_grad()

        # 学生模型前向传播 - 修正返回值顺序
        student_outputs, adapted_features, _ = student_model(student_inputs)

        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            _, teacher_features = teacher_model(teacher_inputs)

        # 计算任务损失
        task_loss = criterion_task(student_outputs, labels)

        # 计算蒸馏损失（特征对齐）
        # 使用目标向量全为1，表示希望特征方向相同
        target = torch.ones(student_inputs.size(0)).to(device)
        distill_loss = criterion_distill(adapted_features, teacher_features, target)

        # 总损失
        total_loss = task_alpha * task_loss + distill_alpha * distill_loss

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        running_train_loss += total_loss.item() * student_inputs.size(0)
        running_distill_loss += distill_loss.item() * student_inputs.size(0)
        running_task_loss += task_loss.item() * student_inputs.size(0)

    # 计算平均损失
    train_loss = running_train_loss / len(train_loader.dataset)
    avg_distill_loss = running_distill_loss / len(train_loader.dataset)
    avg_task_loss = running_task_loss / len(train_loader.dataset)

    train_losses.append(train_loss)
    distill_losses.append(avg_distill_loss)
    task_losses.append(avg_task_loss)

    # 验证阶段
    student_model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _, _ = student_model(inputs)
            val_loss = criterion_task(outputs, labels)
            running_val_loss += val_loss.item() * inputs.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    # 更新学习率
    scheduler.step()

    # 检查是否是最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        no_improve_epochs = 0  # 重置未改善计数器
        torch.save(student_model.state_dict(), best_model_path)
        print(f"★ 发现新最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")
    else:
        no_improve_epochs += 1  # 增加未改善计数器
        print(f"ⓘ 验证损失未改善，连续 {no_improve_epochs}/{patience} 个 epoch")

    # 检查早停条件
    if no_improve_epochs >= patience:
        early_stop = True

    # 打印进度
    print(f'Epoch [{epoch + 1}/{num_epochs}] | '
          f'Train Total Loss: {train_loss:.4f} | '
          f'Task Loss: {avg_task_loss:.4f} | '
          f'Distill Loss: {avg_distill_loss:.4f} | '
          f'Val Loss: {val_loss:.4f} | '
          f'Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})')

end_time = time.time()
print(f"知识蒸馏训练完成，耗时: {(end_time - start_time) / 60:.2f} 分钟")

# 绘制损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Total Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.axvline(x=best_epoch - 1, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress with Early Stopping')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.subplot(1, 2, 2)
plt.plot(task_losses, label='Task Loss', color='green')
plt.plot(distill_losses, label='Distill Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Components')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# 加载最佳模型
student_model.load_state_dict(torch.load(best_model_path))
student_model.eval()
print(f"\n加载最佳蒸馏模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

# 评估验证集
val_true, val_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _, _ = student_model(inputs)
        val_true.extend(labels.cpu().numpy())
        val_pred.extend(outputs.cpu().numpy())

val_true = np.array(val_true)
val_pred = np.array(val_pred)
val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
val_mae = mean_absolute_error(val_true, val_pred)
val_r2 = r2_score(val_true, val_pred)
file_path = r"D:\Research work\Imaging\Data20251020\Datasets\value_SVI_distill_val.csv"
np.savetxt(file_path, val_pred, delimiter=",")

print(f"\n验证集结果 (蒸馏模型): MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")

# 评估测试集
test_true, test_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _, _ = student_model(inputs)
        test_true.extend(labels.cpu().numpy())
        test_pred.extend(outputs.cpu().numpy())

test_true = np.array(test_true)
test_pred = np.array(test_pred)
test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
test_mae = mean_absolute_error(test_true, test_pred)
test_r2 = r2_score(test_true, test_pred)
file_path = r"D:\Research work\Imaging\Data20251020\Datasets\value_SVI_distill_test.csv"
np.savetxt(file_path, test_pred, delimiter=",")

print(f"测试集结果 (蒸馏模型): MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

# 绘制结果散点图
plt.figure(figsize=(8, 6))

# 绘制散点图
val_scatter = plt.scatter(YValid, val_pred, alpha=0.6, color='royalblue', s=5, label='Validation')
test_scatter = plt.scatter(YTest, test_pred, alpha=0.6, color='crimson', s=5, label='Test')

# 绘制对角线
min_val = min(YTest.min(), YValid.min())
max_val = max(YTest.max(), YValid.max())
diagonal = plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')

# 设置坐标轴范围
margin = 0.1 * (max_val - min_val)
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

# 添加评估指标文本
metrics_text = (
    f"Validation Metrics:\n"
    f"MAE = {val_mae:.4f}\n"
    f"RMSE = {val_rmse:.4f}\n"
    f"R2 = {val_r2:.4f}\n\n"
    f"Test Metrics:\n"
    f"MAE = {test_mae:.4f}\n"
    f"RMSE = {test_rmse:.4f}\n"
    f"R2 = {test_r2:.4f}"
)

plt.text(
    x=max_val - 0.25 * (max_val - min_val),
    y=min_val + 0.05 * (max_val - min_val),
    s=metrics_text,
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round', pad=0.5),
    fontsize=11,
    ha='left',
    va='bottom'
)

# 添加标签和图例
plt.xlabel('Measured Values', fontsize=12, labelpad=10)
plt.ylabel('Predicted Values', fontsize=12, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

print("知识蒸馏过程完成！")