import numpy as np
import matplotlib.pyplot as plt

def plot_bands(energies, occupations, title='Band Structure', xlabel='k-point', ylabel='Energy (eV)', output_file='band_structure.png'):
    """
    绘制能带结构图。

    参数:
        energies: 形状为(50, 26)的numpy数组, 包含所有点的能量值
        occupations: 形状为(50, 26)的numpy数组, 包含所有点的占据值
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        output_file: 输出图像文件名
    """
    num_points, num_bands = energies.shape

    # 创建k点索引
    k_points = np.arange(num_points)

    plt.figure(figsize=(10, 6))

    # 绘制每个能带
    for band in range(num_bands):
        plt.plot(k_points, energies[:, band], marker='o', markersize=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # 添加零能量线
    plt.grid()
    
    # # 保存图像
    # plt.savefig(output_file)
    plt.show()

def main():
    # 获取输入文件名
    input_file = 'band_data.npz'  # 假设输入文件为numpy格式的文件
    
    # 读取能带数据
    energies = np.load(input_file)['energies']
    occupations = np.load(input_file)['occupations']
    
    # 绘制能带结构图
    plot_bands(energies, occupations)

if __name__ == "__main__":
    main()