import numpy as np

def extract_band_data(filename):
    """
    从VASP输出文件中提取能带数据
    
    参数:
        filename: 包含能带数据的文本文件名
        
    返回:
        energies: 形状为(50, 26)的numpy数组, 包含所有点的能量值
        occupations: 形状为(50, 26)的numpy数组, 包含所有点的占据值
    """
    # 初始化存储数据的数组
    num_points = 50
    num_bands = 26
    energies = np.zeros((num_points, num_bands))
    occupations = np.zeros((num_points, num_bands))
    
    current_point = -1  # 当前处理的点索引
    reading_bands = False  # 是否正在读取能带数据
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 检查是否是新的点
            if line.startswith('#  Point'):
                parts = line.split()
                try:
                    # 提取点的索引（从0开始）
                    current_point = int(parts[2]) - 1
                    reading_bands = False
                except (ValueError, IndexError):
                    continue
                
                # 如果点的索引超出范围，标记为无效
                if current_point < 0 or current_point >= num_points:
                    current_point = -1
                continue
            
            # 检查是否是表头行
            if line.startswith('#   Band'):
                reading_bands = True
                continue
            
            # 处理能带数据行
            if current_point >= 0 and reading_bands and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # 获取能带索引（从0开始）
                        band_index = int(parts[0]) - 1
                        
                        # 确保能带索引在有效范围内
                        if 0 <= band_index < num_bands:
                            # 提取能量和占据值
                            energies[current_point, band_index] = float(parts[1])
                            occupations[current_point, band_index] = float(parts[2])
                    except (ValueError, IndexError):
                        continue
    
    return energies, occupations

def main():
    # 获取输入文件名
    input_file = "band_structure.out"
    
    # 提取数据
    print("正在提取数据...")
    energies, occupations = extract_band_data(input_file)
    
    # 获取输出文件名
    output_file = "band_data.npz"
    
    # 保存数据到NPZ文件
    np.savez(output_file, energies=energies, occupations=occupations)
    # print(f"数据已成功保存到 {output_file}")
    # print(f"能量数据形状: {energies.shape}")
    # print(f"占据数据形状: {occupations.shape}")

if __name__ == "__main__":
    main()