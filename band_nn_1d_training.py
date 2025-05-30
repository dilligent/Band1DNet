import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class AtomicChainModel(nn.Module):
    """
    一维原子链能带结构预测模型
    具有排序不变性和平移不变性
    """
    def __init__(self, embedding_dim=64, hidden_dim=128, num_layers=3):
        super(AtomicChainModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 原子类型嵌入
        self.atom_embedding = nn.Embedding(2, embedding_dim)
        
        # 距离编码层
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 消息传递层（图神经网络）
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            ))
        
        # 节点更新层
        self.update_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.update_layers.append(nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU()
            ))
        
        # 输出处理层
        self.output_processor = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),  # +1 for unit cell length
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 50 * 26)  # 50个k点 x 26个能量本征值
        )
        
    def forward(self, unit_cell_length, atoms):
        """
        前向传播
        
        参数:
            unit_cell_length: 周期性单元长度 [batch_size]
            atoms: 原子类型和位置 [batch_size, max_atoms, 2]
                  第一列为原子类型(0或1)，第二列为位置
        
        返回:
            能带结构 [batch_size, 50, 26]
        """
        batch_size = unit_cell_length.size(0)
        max_atoms = atoms.size(1)
        
        # 创建掩码处理填充的原子
        mask = (atoms[:, :, 0] >= 0).float().unsqueeze(-1)  # [batch_size, max_atoms, 1]
        
        # 获取原子类型嵌入
        atom_types = torch.clamp(atoms[:, :, 0].long(), min=0)
        node_features = self.atom_embedding(atom_types) * mask
        
        # 计算原子间的相对距离（考虑周期性）
        positions = atoms[:, :, 1].unsqueeze(2)  # [batch_size, max_atoms, 1]
        positions_expanded1 = positions.expand(-1, -1, max_atoms)
        positions_expanded2 = positions.transpose(1, 2).expand(-1, max_atoms, -1)
        
        # 计算周期性距离
        direct_dist = torch.abs(positions_expanded1 - positions_expanded2)
        periodic_dist = unit_cell_length.unsqueeze(1).unsqueeze(2) - direct_dist
        distances = torch.min(direct_dist, periodic_dist)
        
        # 归一化距离
        normalized_distances = distances / unit_cell_length.unsqueeze(1).unsqueeze(2)
        
        # 编码距离
        edge_features = self.distance_encoder(normalized_distances.unsqueeze(-1))
        
        # 创建邻接矩阵（排除自循环）
        adj_matrix = torch.ones((batch_size, max_atoms, max_atoms), device=atoms.device)
        adj_matrix = adj_matrix - torch.eye(max_atoms, device=atoms.device).unsqueeze(0)
        adj_matrix = adj_matrix * mask.squeeze(-1).unsqueeze(1) * mask.squeeze(-1).unsqueeze(2)
        
        # 消息传递（图神经网络的核心）
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            # 准备消息输入
            senders = node_features.unsqueeze(2).expand(-1, -1, max_atoms, -1)
            receivers = node_features.unsqueeze(1).expand(-1, max_atoms, -1, -1)
            
            # 组合特征
            message_inputs = torch.cat([senders, receivers, edge_features], dim=3)
            
            # 批处理消息计算
            message_inputs_flat = message_inputs.view(batch_size * max_atoms * max_atoms, -1)
            messages_flat = message_layer(message_inputs_flat)
            messages = messages_flat.view(batch_size, max_atoms, max_atoms, -1)
            
            # 应用邻接矩阵
            messages = messages * adj_matrix.unsqueeze(-1)
            
            # 聚合消息（求和 - 保证排序不变性）
            aggregated_messages = messages.sum(dim=2)
            
            # 更新节点特征
            update_inputs = torch.cat([node_features, aggregated_messages], dim=2)
            node_features = update_layer(update_inputs) * mask
        
        # 全局池化（求和 - 进一步保证排序不变性）
        global_feature = (node_features * mask).sum(dim=1)
        
        # 结合单元长度
        combined_input = torch.cat([global_feature, unit_cell_length.unsqueeze(1)], dim=1)
        
        # 生成输出
        output = self.output_processor(combined_input)
        return output.view(batch_size, 50, 26)


class AtomicChainDataset(Dataset):
    """一维原子链数据集"""
    
    def __init__(self, data):
        """
        初始化数据集
        
        参数:
            data: 列表，每个元素为 (unit_cell_length, atoms, target)
                unit_cell_length: 浮点数，周期性单元长度
                atoms: 列表, 包含2~4个[atom_type, position]数组
                target: numpy数组, 形状为 [50, 26]，表示目标能带结构
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class EarlyStopping:
    """Early stopping to prevent overfitting
    
    Args:
        patience (int): How many epochs to wait after last improvement
        verbose (bool): If True, prints a message for each validation loss improvement
        delta (float): Minimum change in the monitored quantity to qualify as an improvement
        path (str): Path for the checkpoint to be saved to
    """
    def __init__(self, patience=200, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def collate_fn(batch):
    """
    将不同长度的样本打包成批量
    
    参数:
        batch: 列表，每个元素为 (unit_cell_length, atoms, target)
    
    返回:
        unit_cell_lengths: 张量 [batch_size]
        padded_atoms: 张量 [batch_size, max_atoms, 2]
        targets: 张量 [batch_size, 50, 26]
    """
    unit_cell_lengths = []
    max_atoms = max(len(atoms) for _, atoms, _ in batch)
    padded_atoms = []
    targets = []
    
    for unit_cell_length, atoms, target in batch:
        unit_cell_lengths.append(unit_cell_length)
        targets.append(target)
        
        # 填充原子列表到最大长度
        padded = np.zeros((max_atoms, 2))
        padded[:len(atoms)] = atoms
        # 将填充的原子类型设为-1（无效值）
        for i in range(len(atoms), max_atoms):
            padded[i, 0] = -1
        
        padded_atoms.append(padded)
    unit_cell_lengths = np.array(unit_cell_lengths)
    padded_atoms = np.array(padded_atoms)
    targets = np.array(targets)
    
    return (
        torch.tensor(unit_cell_lengths, dtype=torch.float32),
        torch.tensor(padded_atoms, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)
    )


def train(model, train_loader, optimizer, criterion, device, val_loader, epochs=100):
    """
    训练模型
    
    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备('cuda'或'cpu')
        val_loader: 验证数据加载器
        epochs: 训练轮数
    """
    model.train()
    loss_history = []
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=200, verbose=True, path='best_model.pth')
    
    for epoch in range(epochs):
        epoch_loss = 0
        for unit_cell_length, atoms, target in train_loader:
            unit_cell_length = unit_cell_length.to(device)
            atoms = atoms.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(unit_cell_length, atoms)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Check early stopping criteria
        early_stopping(val_loss, model)
        
        if (epoch+1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # If early stopping triggered, break the training loop
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    return loss_history


def evaluate(model, test_loader, criterion, device):
    """
    评估模型
    
    参数:
        model: 模型实例
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备
    
    返回:
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for unit_cell_length, atoms, target in test_loader:
            unit_cell_length = unit_cell_length.to(device)
            atoms = atoms.to(device)
            target = target.to(device)
            
            output = model(unit_cell_length, atoms)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def visualize_bands(pred_bands, true_bands=None, k_points=None):
    """可视化能带结构"""
    if k_points is None:
        k_points = np.linspace(-0.5, 0, pred_bands.shape[0])
    
    plt.figure(figsize=(10, 6))
    
    # 绘制预测能带
    for i in range(pred_bands.shape[1]):
        plt.plot(k_points, pred_bands[:, i], 'r-', alpha=0.7)
    
    # 如果有真实能带，也绘制它们
    if true_bands is not None:
        for i in range(true_bands.shape[1]):
            plt.plot(k_points, true_bands[:, i], 'b--', alpha=0.5)
    
    plt.xlabel('k-point')
    plt.ylabel('Energy')
    plt.axvline(-0.5, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title('Band Structure')
    plt.grid(True)
    plt.show()


def prepare_synthetic_data(n_total=201, n_val=41):
    """
    读取数据用于测试!
    """
    data = []

    for i in range(n_total):
        with open(f'./structures/{i+1}/one_dimension_{i+1}.inp', 'r') as f:
            content = f.readlines()
        unit_cell_length = float(content[23].split()[1]) # 读取周期性单元长度
        n_atoms = len(content) - 95 # 计算原子数量
        atoms = []
        for j in range(n_atoms):
            atom_type = 0 if 'Si' in content[28 + j] else 1
            position = float(content[28 + j].split()[1])
            atoms.append([atom_type, position])
        target = np.load(f'./structures/{i+1}/band_data.npz')['energies'][:, :26]
        data.append((unit_cell_length, atoms, target))

    train_data = data[:-n_val]
    val_data = data[-n_val:]
    
    return train_data, val_data


def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 准备数据
    train_data, val_data = prepare_synthetic_data()
    
    # 创建数据集和数据加载器
    train_dataset = AtomicChainDataset(train_data)
    val_dataset = AtomicChainDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 创建模型
    model = AtomicChainModel(
        embedding_dim=128,
        hidden_dim=512,
        num_layers=7
    ).to(device)
    
    # 设置优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练模型
    print("开始训练...")
    loss_history = train(model, train_loader, optimizer, criterion, device, val_loader, epochs=2000)
    
    # 评估模型
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"验证集损失: {val_loss:.6f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'atomic_chain_model.pth')
    print("模型已保存至 'atomic_chain_model.pth'")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    
    # 可视化一个样本的预测结果
    model.eval()
    with torch.no_grad():
        sample = val_data[2]
        unit_cell_length = torch.tensor([sample[0]], dtype=torch.float32).to(device)
        atoms = torch.tensor([sample[1]], dtype=torch.float32).to(device)
        true_bands = sample[2]
        
        pred_bands = model(unit_cell_length, atoms).cpu().numpy()[0]
        
        visualize_bands(pred_bands, true_bands)


if __name__ == "__main__":
    main()