import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AtomEmbedding(nn.Module):
    """原子嵌入层：将原子类型(0或1)映射到嵌入向量"""
    def __init__(self, embedding_dim=64):
        super(AtomEmbedding, self).__init__()
        self.embedding = nn.Embedding(2, embedding_dim)
        
    def forward(self, atom_types):
        return self.embedding(atom_types)

class PeriodicDistanceCalculation(nn.Module):
    """计算周期性边界条件下原子之间的距离"""
    def __init__(self):
        super(PeriodicDistanceCalculation, self).__init__()
        
    def forward(self, positions, cell_length):
        """
        计算周期性边界条件下的距离矩阵
        
        参数:
        - positions: [batch_size, num_atoms]，原子位置
        - cell_length: [batch_size]，周期性单元长度
        
        返回:
        - dist_matrix: [batch_size, num_atoms, num_atoms]，距离矩阵
        """
        batch_size, num_atoms = positions.shape
        
        # 扩展位置为矩阵形式
        pos_i = positions.unsqueeze(2)  # [B, N, 1]
        pos_j = positions.unsqueeze(1)  # [B, 1, N]
        
        # 计算直接距离
        dist = torch.abs(pos_i - pos_j)  # [B, N, N]
        
        # 考虑周期性边界条件，取最小距离
        cell_length_expanded = cell_length.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        periodic_dist = torch.min(dist, cell_length_expanded - dist)  # [B, N, N]
        
        # 归一化距离
        normalized_dist = periodic_dist / cell_length_expanded  # [B, N, N]
        
        return normalized_dist

class MessagePassingLayer(nn.Module):
    """消息传递层：实现排序不变性的节点特征更新"""
    def __init__(self, feature_dim=64):
        super(MessagePassingLayer, self).__init__()
        
        # 边特征网络
        self.edge_nn = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 消息生成网络
        self.message_nn = nn.Sequential(
            nn.Linear(feature_dim * 2 + feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 节点更新网络
        self.update_nn = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, node_features, positions, cell_length, mask=None):
        batch_size, num_atoms, feature_dim = node_features.shape
        
        # 计算周期性距离
        dist_calculator = PeriodicDistanceCalculation()
        dist_matrix = dist_calculator(positions, cell_length)  # [B, N, N]
        
        # 生成边特征
        edge_features = self.edge_nn(dist_matrix.unsqueeze(-1))  # [B, N, N, F]
        
        # 准备消息传递
        node_features_i = node_features.unsqueeze(2).expand(-1, -1, num_atoms, -1)  # [B, N, N, F]
        node_features_j = node_features.unsqueeze(1).expand(-1, num_atoms, -1, -1)  # [B, N, N, F]
        
        # 生成消息
        message_inputs = torch.cat([node_features_i, node_features_j, edge_features], dim=-1)  # [B, N, N, 3F]
        messages = self.message_nn(message_inputs)  # [B, N, N, F]
        
        # 排除自环
        self_mask = 1.0 - torch.eye(num_atoms, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        messages = messages * self_mask.unsqueeze(-1)  # [B, N, N, F]
        
        # 应用节点掩码（如果提供）
        if mask is not None:
            # 扩展mask以涵盖所有节点对
            mask_i = mask.unsqueeze(2).expand(-1, -1, num_atoms)  # [B, N, N]
            mask_j = mask.unsqueeze(1).expand(-1, num_atoms, -1)  # [B, N, N]
            pair_mask = mask_i * mask_j * self_mask  # [B, N, N]
            messages = messages * pair_mask.unsqueeze(-1)  # [B, N, N, F]
        
        # 聚合消息（求和，保证排序不变性）
        aggregated_messages = messages.sum(dim=2)  # [B, N, F]
        
        # 更新节点特征
        update_inputs = torch.cat([node_features, aggregated_messages], dim=-1)  # [B, N, 2F]
        updated_features = self.update_nn(update_inputs)  # [B, N, F]
        
        # 残差连接和层归一化
        updated_features = self.norm(node_features + updated_features)
        
        # 应用节点掩码
        if mask is not None:
            updated_features = updated_features * mask.unsqueeze(-1)
        
        return updated_features

class GlobalPooling(nn.Module):
    """全局池化层：将节点特征聚合为图级特征（保证排序不变性）"""
    def __init__(self, feature_dim=64):
        super(GlobalPooling, self).__init__()
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, 1)
        )
    def forward(self, node_features, mask=None):
        # 获取输入张量的形状
        batch_size, num_nodes, feature_dim = node_features.shape
        
        # 计算注意力权重
        attn_weights = self.attention(node_features)  # [B, N, 1]
        
        if mask is not None:
            # 对非原子位置施加掩码
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        # Softmax确保权重和为1
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 注意力加权求和
        weighted_sum = (node_features * attn_weights).sum(dim=1)  # [B, F]
        
        # 最大池化（另一种排序不变的操作）
        if mask is not None:
            masked_features = node_features.clone()
            masked_features = masked_features.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            max_features, _ = masked_features.max(dim=1)  # [B, F]
        else:
            max_features, _ = node_features.max(dim=1)  # [B, F]
        
        # 平均池化（也是排序不变的）
        if mask is not None:
            # 考虑mask的平均值
            mask_sum = mask.sum(dim=1, keepdim=True)  # [B, 1]
            # 修改这一行，移除多余的unsqueeze操作
            mean_features = node_features.sum(dim=1) / torch.clamp(mask_sum, min=1)  # [B, F]
        else:
            mean_features = node_features.mean(dim=1)  # [B, F]
        
        # 确保所有张量都是2维的
        if weighted_sum.dim() > 2:
            weighted_sum = weighted_sum.view(batch_size, feature_dim)
        if max_features.dim() > 2:
            max_features = max_features.view(batch_size, feature_dim)
        if mean_features.dim() > 2:
            mean_features = mean_features.view(batch_size, feature_dim)
        
        # 连接不同的池化结果 - 注意这里使用dim=1而不是dim=-1
        return torch.cat([weighted_sum, max_features, mean_features], dim=1)  # [B, 3F]
    

class BandStructureModel(nn.Module):
    """一维原子链能带结构预测模型，支持可变数量的能带"""
    def __init__(self, embedding_dim=64, num_layers=3, max_atoms=4, max_bands=30):
        super(BandStructureModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_atoms = max_atoms
        self.max_bands = max_bands  # 使用最大能带数量
        
        # 原子类型嵌入
        self.atom_embedding = AtomEmbedding(embedding_dim)
        
        # 位置编码（使用正弦余弦编码确保平移不变性）
        self.position_encoding = nn.Sequential(
            nn.Linear(2, embedding_dim),  # 使用sin和cos编码
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 消息传递层
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(embedding_dim) for _ in range(num_layers)
        ])
        
        # 全局池化
        self.global_pooling = GlobalPooling(embedding_dim)
        
        # 单元长度编码
        self.cell_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # k点编码
        self.k_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim),  # 使用sin和cos编码
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 全局特征处理
        self.global_processor = nn.Sequential(
            nn.Linear(embedding_dim * 3 + embedding_dim, embedding_dim * 2),  # 池化特征 + 单元长度特征
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2)
        )
        
        # 能带预测层 - 预测最大可能的能带数
        self.band_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2 + embedding_dim, embedding_dim * 2),  # 全局特征 + k点特征
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, max_bands)
        )
        
    def forward(self, atoms, cell_length, k_points, num_bands=None):
        """
        前向传播函数
        
        参数:
        - atoms: [batch_size, max_atoms, 2] 原子数据, [:,:,0]是类型, [:,:,1]是位置
        - cell_length: [batch_size] 周期性单元长度
        - k_points: [batch_size, num_k] k点值
        - num_bands: 可选，指定每个样本的能带数量
        
        返回:
        - band_energies: [batch_size, num_k, num_bands] 预测的能带
        """
        batch_size = atoms.shape[0]
        num_k = k_points.shape[1]
        
        # 提取原子类型和位置
        atom_types = atoms[:, :, 0].long()  # [B, N]
        atom_positions = atoms[:, :, 1]  # [B, N]
        
        # 创建有效原子掩码（非填充位置）
        mask = (atom_types >= 0).float()  # [B, N]
        
        # 原子类型嵌入
        type_features = self.atom_embedding(atom_types)  # [B, N, F]
        
        # 位置编码（使用正弦余弦确保平移不变性）
        pos_normalized = atom_positions / cell_length.unsqueeze(1)  # [B, N]
        sin_pos = torch.sin(2 * np.pi * pos_normalized).unsqueeze(-1)  # [B, N, 1]
        cos_pos = torch.cos(2 * np.pi * pos_normalized).unsqueeze(-1)  # [B, N, 1]
        pos_encoding_input = torch.cat([sin_pos, cos_pos], dim=-1)  # [B, N, 2]
        pos_features = self.position_encoding(pos_encoding_input)  # [B, N, F]
        
        # 初始节点特征
        node_features = type_features + pos_features  # [B, N, F]
        
        # 消息传递更新
        for layer in self.message_layers:
            node_features = layer(node_features, atom_positions, cell_length, mask)
        
        # 全局池化
        global_features = self.global_pooling(node_features, mask)  # [B, 3F]
        
        # 编码周期性单元长度
        cell_features = self.cell_encoder(cell_length.unsqueeze(-1))  # [B, F]
        
        # 合并全局特征和单元长度特征
        combined_features = torch.cat([global_features, cell_features], dim=1)  # [B, 4F]
        processed_global = self.global_processor(combined_features)  # [B, 2F]
        
        # 预测每个k点的能带
        bands = []
        for i in range(num_k):
            k_val = k_points[:, i].unsqueeze(-1)  # [B, 1]
            sin_k = torch.sin(2 * np.pi * k_val)  # [B, 1]
            cos_k = torch.cos(2 * np.pi * k_val)  # [B, 1]
            k_encoding_input = torch.cat([sin_k, cos_k], dim=-1)  # [B, 2]
            k_features = self.k_encoder(k_encoding_input)  # [B, F]
            
            # 确保k_features是2维的 [B, F]
            if k_features.dim() > 2:
                k_features = k_features.squeeze(1)  # 移除多余的维度
            
            # 确保processed_global是2维的 [B, 2F]
            if processed_global.dim() > 2:
                processed_global = processed_global.view(batch_size, -1)
                
            # 合并特征并预测能带
            predictor_input = torch.cat([processed_global, k_features], dim=1)  # [B, 3F]
            band_energies = self.band_predictor(predictor_input)  # [B, max_bands]
            
            # 如果指定了每个样本的能带数量，截取相应的输出
            if num_bands is not None:
                # 找出这个批次中的最大能带数
                max_bands_in_batch = num_bands.max().item()
                
                # 对每个样本截取并填充到批次内的最大能带数
                batch_bands = []
                for b in range(batch_size):
                    # 截取该样本的实际能带数
                    curr_bands = band_energies[b, :num_bands[b]]
                    
                    # 如果当前样本的能带数小于批次中的最大能带数，需要填充
                    if num_bands[b] < max_bands_in_batch:
                        padding = torch.zeros(max_bands_in_batch - num_bands[b], device=curr_bands.device, dtype=curr_bands.dtype)
                        curr_bands = torch.cat([curr_bands, padding], dim=0)
                    
                    batch_bands.append(curr_bands.unsqueeze(0))
                
                batch_bands = torch.cat(batch_bands, dim=0)
                bands.append(batch_bands.unsqueeze(1))
            else:
                bands.append(band_energies.unsqueeze(1))
        
        # 组合所有k点的能带
        bands = torch.cat(bands, dim=1)  # [B, num_k, num_bands]
        
        return bands

class AtomChainDataset(Dataset):
    """一维原子链数据集"""
    def __init__(self, data_list, max_atoms=4):
        self.data_list = data_list
        self.max_atoms = max_atoms
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # 解包数据
        cell_length = sample[0]  # 周期性单元长度
        atoms = sample[1]  # 原子类型和位置的列表
        band_energies = sample[2]  # 能带矩阵 [50, num_bands]
        occupation = sample[3]  # 占据数矩阵 [50, num_bands]
        
        # 确保原子位置在[0, cell_length)范围内（平移不变性）
        atoms_array = np.array(atoms)
        atoms_array[:, 1] = atoms_array[:, 1] % cell_length
        
        # 填充原子数组到最大长度
        num_atoms = len(atoms)
        padded_atoms = np.zeros((self.max_atoms, 2))
        padded_atoms[:num_atoms] = atoms_array
        
        # 创建k点网格
        num_k_points = band_energies.shape[0]  # 通常为50
        k_points = np.linspace(0, 1, num_k_points, endpoint=False)
        
        # 记录能带数量
        num_bands = band_energies.shape[1]
        
        # 转换为PyTorch张量
        cell_length_tensor = torch.tensor(cell_length, dtype=torch.float32)
        atoms_tensor = torch.tensor(padded_atoms, dtype=torch.float32)
        k_points_tensor = torch.tensor(k_points, dtype=torch.float32)
        band_energies_tensor = torch.tensor(band_energies, dtype=torch.float32)
        occupation_tensor = torch.tensor(occupation, dtype=torch.int64)
        
        return {
            'cell_length': cell_length_tensor,
            'atoms': atoms_tensor,
            'k_points': k_points_tensor,
            'band_energies': band_energies_tensor,
            'occupation': occupation_tensor,
            'num_atoms': num_atoms,
            'num_bands': num_bands  # 添加能带数量信息
        }

# 自定义的批次整理函数，处理不同样本中能带数量不同的情况
def custom_collate_fn(batch):
    # 提取所有样本的键
    keys = batch[0].keys()
    
    # 存储每个样本的能带数量
    num_bands_list = [item['num_bands'] for item in batch]
    max_num_bands = max(num_bands_list)  # 批次中的最大能带数
    
    # 创建结果字典
    result = {}
    
    # 处理特殊的'band_energies'和'occupation'字段
    band_energies_list = []
    occupation_list = []
    
    for item in batch:
        # 获取当前样本的能带和占据数
        curr_bands = item['band_energies']  # [50, curr_num_bands]
        curr_occ = item['occupation']       # [50, curr_num_bands]
        
        # 如果当前样本的能带数小于最大能带数，需要填充
        if curr_bands.shape[1] < max_num_bands:
            # 为能带填充零
            pad_bands = torch.zeros((curr_bands.shape[0], max_num_bands - curr_bands.shape[1]), dtype=curr_bands.dtype)
            curr_bands = torch.cat([curr_bands, pad_bands], dim=1)  # [50, max_num_bands]
            
            # 为占据数填充零
            pad_occ = torch.zeros((curr_occ.shape[0], max_num_bands - curr_occ.shape[1]), dtype=curr_occ.dtype)
            curr_occ = torch.cat([curr_occ, pad_occ], dim=1)  # [50, max_num_bands]
        
        band_energies_list.append(curr_bands)
        occupation_list.append(curr_occ)
    
    # 将填充后的能带和占据数堆叠为批次
    result['band_energies'] = torch.stack(band_energies_list)
    result['occupation'] = torch.stack(occupation_list)
    
    # 处理其他字段 - 修改这部分代码来处理非张量类型
    for key in keys:
        if key not in ['band_energies', 'occupation', 'num_bands']:
            # 检查是否为张量类型
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in batch])
            else:
                # 对于非张量类型（如整数），先转换为张量
                result[key] = torch.tensor([item[key] for item in batch])
    
    # 保存每个样本的原始能带数量
    result['num_bands'] = torch.tensor(num_bands_list, dtype=torch.long)
    
    return result


def train_model(model, train_loader, val_loader, num_epochs=1000, lr=0.001):
    """训练模型"""
    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            # 移动数据到设备
            cell_length = batch['cell_length'].to(device)
            atoms = batch['atoms'].to(device)
            k_points = batch['k_points'].unsqueeze(1).to(device)  # [B, 1, K] -> [B, K, 1]
            band_energies = batch['band_energies'].to(device)
            num_bands = batch['num_bands'].to(device)  # 每个样本的实际能带数
            
            # 前向传播（带有每个样本的能带数信息）
            outputs = model(atoms, cell_length, k_points.transpose(1, 2), num_bands)
            
            # 创建掩码以忽略填充的能带
            max_bands = band_energies.shape[2]
            band_mask = torch.zeros_like(band_energies, dtype=torch.bool)
            for b in range(band_energies.shape[0]):
                band_mask[b, :, :num_bands[b]] = True
            
            # 计算损失（仅考虑非填充部分）
            masked_outputs = outputs.masked_select(band_mask)
            masked_targets = band_energies.masked_select(band_mask)
            loss = criterion(masked_outputs, masked_targets)
             
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                cell_length = batch['cell_length'].to(device)
                atoms = batch['atoms'].to(device)
                k_points = batch['k_points'].unsqueeze(1).to(device)
                band_energies = batch['band_energies'].to(device)
                num_bands = batch['num_bands'].to(device)
                
                outputs = model(atoms, cell_length, k_points.transpose(1, 2), num_bands)
                
                # 创建掩码以忽略填充的能带
                max_bands = band_energies.shape[2]
                band_mask = torch.zeros_like(band_energies, dtype=torch.bool)
                for b in range(band_energies.shape[0]):
                    band_mask[b, :, :num_bands[b]] = True
                
                # 计算损失（仅考虑非填充部分）
                masked_outputs = outputs.masked_select(band_mask)
                masked_targets = band_energies.masked_select(band_mask)
                loss = criterion(masked_outputs, masked_targets)

                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_band_structure_model.pth')
            print(f'Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.6f}')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()
    
    return model, train_losses, val_losses

def predict_bands(model, cell_length, atoms, num_k_points=50, num_bands=30):
    """使用训练好的模型预测能带结构"""
    model.eval()
    
    # 准备输入数据
    max_atoms = model.max_atoms
    padded_atoms = np.zeros((max_atoms, 2))
    padded_atoms[:len(atoms)] = atoms
    
    cell_length_tensor = torch.tensor(cell_length, dtype=torch.float32).unsqueeze(0).to(device)
    atoms_tensor = torch.tensor(padded_atoms, dtype=torch.float32).unsqueeze(0).to(device)
    k_points = torch.linspace(0, 1, num_k_points, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 预测能带（不传递num_bands参数，使用模型的最大能带数）
    with torch.no_grad():
        predicted_bands = model(atoms_tensor, cell_length_tensor, k_points.unsqueeze(-1))
    
    # 截取所需的能带数量
    predicted_bands = predicted_bands[0, :, :num_bands].cpu().numpy()
    
    return predicted_bands  # [num_k, num_bands]
  
def prepare_data(num_samples=201):

    data = []

    for i in range(num_samples):
        with open(f'./structures/{i+1}/one_dimension_{i+1}.inp', 'r') as f:
            content = f.readlines()
        
        cell_length = float(content[23].split()[1])  # 读取周期性单元长度
        num_atoms = len(content) - 95
        atoms = []
        for j in range(num_atoms):
            atom_type = 0 if 'Si' in content[28 + j] else 1
            position = float(content[28 + j].split()[1])
            atoms.append([atom_type, position])
        
        stored_data = np.load(f'./structures/{i+1}/band_data.npz')
        band_energies = stored_data['energies']
        occupation = stored_data['occupations']
        
        data.append((cell_length, atoms, band_energies, occupation))
    
    return data


def main():
    # 模型参数
    embedding_dim = 64
    num_layers = 3
    max_atoms = 4
    max_bands = 30  # 最大可能的能带数
    batch_size = 128
    num_epochs = 1000
    
    # 加载数据
    all_data = prepare_data(201)  # 根据您的实际数据集大小调整
    print(f"First sample structure: {len(all_data[0])}")
    for i, item in enumerate(all_data[0]):
        print(f"Item {i}: {type(item)}, Shape/Length: {getattr(item, 'shape', len(item) if hasattr(item, '__len__') else 'scalar')}")
    
    # 检查第一个样本的能带数量
    print(f"第一个样本的能带数量: {all_data[0][2].shape[1]}")
    
    # 统计数据集中的能带数量分布
    band_counts = {}
    for sample in all_data:
        num_bands = sample[2].shape[1]
        band_counts[num_bands] = band_counts.get(num_bands, 0) + 1
    
    print("能带数量分布:")
    for num_bands, count in sorted(band_counts.items()):
        print(f"  {num_bands} 个能带: {count} 个样本 ({count/len(all_data)*100:.1f}%)")

    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    
    # 创建数据集和数据加载器，使用自定义的collate_fn
    train_dataset = AtomChainDataset(train_data, max_atoms=max_atoms)
    val_dataset = AtomChainDataset(val_data, max_atoms=max_atoms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=custom_collate_fn)
    
    # 初始化模型
    model = BandStructureModel(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        max_atoms=max_atoms,
        max_bands=max_bands
    ).to(device)
    
    # 训练模型
    trained_model, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs
    )
    
    print("模型训练完成！")
    
    # 测试预测
    test_cell_length = 5.0
    test_atoms = np.array([[0, 1.2], [1, 2.5], [0, 3.8]])
    predicted_bands = predict_bands(trained_model, test_cell_length, test_atoms, num_bands=26)
    
    print(f"预测能带形状: {predicted_bands.shape}")
    
    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    for i in range(min(10, predicted_bands.shape[1])):  # 仅显示前10个轨道
        plt.plot(np.linspace(0, 1, 50), predicted_bands[:, i], label=f'Band {i+1}')
    
    plt.xlabel('k-point')
    plt.ylabel('Energy')
    plt.title('Predicted Band Structure')
    plt.legend()
    plt.grid(True)
    plt.savefig('predicted_bands.png')
    plt.close()

if __name__ == "__main__":
    main()