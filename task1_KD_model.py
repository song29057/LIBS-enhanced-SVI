import numpy as np
import torch
import torch.nn as nn
import random
import time
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. 环境配置与随机种子
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

seed = 27
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # 保证结果可复现

print(f"当前运行设备: {device}")


# ==========================================
# 2. 数据集类定义 (关键修改：合并数据)
# ==========================================
class PairedSpecData(Dataset):
    def __init__(self, x_student, x_teacher, y):
        # 转换为 Tensor，确保类型正确
        self.x_student = torch.FloatTensor(x_student)
        self.x_teacher = torch.FloatTensor(x_teacher)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, idx):
        # 返回对应的三元组
        return self.x_student[idx], self.x_teacher[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


# ==========================================
# 3. 模型定义
# ==========================================
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
        x = self.relu1(feature)
        x = self.ln3(x)
        x = self.fc3(x)
        return x.squeeze(-1), feature


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # 特征提取
        self.ln1 = nn.LayerNorm(606)
        self.fc1 = nn.Linear(606, 32)
        self.relu1 = nn.ReLU()

        # 特征适配层
        self.adapter2 = nn.Linear(32, 32)
        self.adapter3 = nn.Tanh()

        # 输出层
        self.ln2 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.ln1(x)
        student_feature = self.fc1(x)
        x = self.relu1(student_feature)

        # 适配特征用于蒸馏
        adapted_feature = self.adapter2(student_feature)
        adapted_feature = self.adapter3(adapted_feature)

        x = self.ln2(x)
        x = self.fc2(x)
        return x.squeeze(-1), adapted_feature, student_feature


# ==========================================
# 4. 数据加载与预处理
# ==========================================
# 路径配置
data_dir = r'D:\Research work\Imaging\Data20251020\Datasets'


def load_data(filename):
    return np.loadtxt(os.path.join(data_dir, filename), delimiter=',')


# 加载数据
print("正在加载数据...")
XTrain_SVI = load_data('SVI_train.csv')
XValid_SVI = load_data('SVI_val.csv')
XTest_SVI = load_data('SVI_test.csv')

XTrain_LIBS = load_data('LIBS_train.csv')
XValid_LIBS = load_data('LIBS_val.csv')
XTest_LIBS = load_data('LIBS_test.csv')

YTrain = load_data('Y_train.csv')
YValid = load_data('Y_val.csv')
YTest = load_data('Y_test.csv')

# 创建数据集
train_dataset = PairedSpecData(XTrain_SVI, XTrain_LIBS, YTrain)
val_dataset = PairedSpecData(XValid_SVI, XValid_LIBS, YValid)
test_dataset = PairedSpecData(XTest_SVI, XTest_LIBS, YTest)

# 创建 DataLoader
batch_size = 16
# shuffle=True 只对训练集开启，现在会同时打乱 SVI, LIBS 和 Y，保持它们之间的索引对应
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 5. 模型初始化与训练配置
# ==========================================
# 加载教师模型
teacher_model = TeacherModel().to(device)
teacher_path = os.path.join(data_dir, 'LIBS_best_model.pth')

if os.path.exists(teacher_path):
    teacher_model.load_state_dict(torch.load(teacher_path))
    print("教师模型加载成功")
else:
    print(f"警告: 未找到教师模型文件 {teacher_path}")

teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# 初始化学生模型
student_model = StudentModel().to(device)

# 优化器与损失
learning_rate = 0.01
num_epochs = 100
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

criterion_task = nn.L1Loss()
criterion_distill = nn.CosineEmbeddingLoss()

task_alpha = 0.9
distill_alpha = 0.1

# ==========================================
# 6. 训练循环
# ==========================================
best_val_loss = float('inf')
best_model_path = os.path.join(data_dir, 'SVI_distill_best_model.pth')

train_losses, val_losses = [], []
patience = 10
no_improve_epochs = 0
early_stop = False

print("\n开始知识蒸馏训练...")
start_time = time.time()

for epoch in range(num_epochs):
    if early_stop:
        break

    student_model.train()
    running_loss = 0.0
    running_task = 0.0
    running_distill = 0.0

    for student_input, teacher_input, labels in train_loader:
        student_input = student_input.to(device)
        teacher_input = teacher_input.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 1. 学生前向传播
        student_pred, adapted_feat, _ = student_model(student_input)

        # 2. 教师前向传播 (No Grad)
        with torch.no_grad():
            _, teacher_feat = teacher_model(teacher_input)

        # 3. 计算损失
        task_loss = criterion_task(student_pred, labels)

        target = torch.ones(student_input.size(0)).to(device)
        distill_loss = criterion_distill(adapted_feat, teacher_feat, target)

        total_loss = task_alpha * task_loss + distill_alpha * distill_loss

        total_loss.backward()
        optimizer.step()

        # 统计
        running_loss += total_loss.item() * student_input.size(0)
        running_task += task_loss.item() * student_input.size(0)
        running_distill += distill_loss.item() * student_input.size(0)

    # 计算 Epoch 平均损失
    train_loss = running_loss / len(train_dataset)
    avg_task = running_task / len(train_dataset)
    avg_distill = running_distill / len(train_dataset)

    train_losses.append(train_loss)

    # 验证阶段 (不需要教师数据，但 Loader 会返回，用 _ 忽略即可)
    student_model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for s_in, _, lbl in val_loader:  # 注意这里的解包
            s_in, lbl = s_in.to(device), lbl.to(device)
            pred, _, _ = student_model(s_in)
            loss = criterion_task(pred, lbl)
            val_running_loss += loss.item() * s_in.size(0)

    val_loss = val_running_loss / len(val_dataset)
    val_losses.append(val_loss)

    scheduler.step()

    # Checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(student_model.state_dict(), best_model_path)
        no_improve_epochs = 0
        print(
            f"Epoch {epoch + 1:03d} | Train: {train_loss:.4f} (Task:{avg_task:.4f}, Distill:{avg_distill:.4f}) | Val: {val_loss:.4f} ★")
    else:
        no_improve_epochs += 1
        print(
            f"Epoch {epoch + 1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | No improve: {no_improve_epochs}")

    if no_improve_epochs >= patience:
        print(f"早停触发于 Epoch {epoch + 1}")
        early_stop = True

print(f"训练结束，总耗时: {(time.time() - start_time) / 60:.2f} 分钟")

# ==========================================
# 7. 评估部分
# ==========================================
student_model.load_state_dict(torch.load(best_model_path))
student_model.eval()


def evaluate(loader, name):
    trues, preds = [], []
    with torch.no_grad():
        for s_in, _, lbl in loader:  # 忽略教师输入
            s_in = s_in.to(device)
            out, _, _ = student_model(s_in)
            trues.extend(lbl.cpu().numpy())
            preds.extend(out.cpu().numpy())

    trues = np.array(trues)
    preds = np.array(preds)

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    print(f"[{name}] MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    return trues, preds, mae, rmse, r2


print("\n--- 最终评估 ---")
val_true, val_pred, val_mae, val_rmse, val_r2 = evaluate(val_loader, "Validation")
test_true, test_pred, test_mae, test_rmse, test_r2 = evaluate(test_loader, "Test")

# 保存预测结果
np.savetxt(os.path.join(data_dir, 'value_SVI_distill_val.csv'), val_pred, delimiter=",")
np.savetxt(os.path.join(data_dir, 'value_SVI_distill_test.csv'), test_pred, delimiter=",")

# 绘图部分保持不变...
plt.figure(figsize=(8, 6))
plt.scatter(val_true, val_pred, alpha=0.6, color='royalblue', s=10, label='Validation')
plt.scatter(test_true, test_pred, alpha=0.6, color='crimson', s=10, label='Test')
min_val = min(np.min(test_true), np.min(test_pred))
max_val = max(np.max(test_true), np.max(test_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
plt.title(f'SVI Distillation Results (R2={test_r2:.3f})')
plt.legend()
plt.show()