import os
import copy
import numpy as np

def main():
    # 原始输入文件路径
    input_file = "one_dimension.inp"
    input_file_name = "one_dimension"
    
    number_of_structures = 200  # 生成的结构数量 

    # 读取CP2K输入文件
    with open(input_file, 'r') as f:
        oringin_content = f.readlines()

    start = 28

    # 生成100个随机的输入文件
    for i in range(number_of_structures):
        
        content = copy.deepcopy(oringin_content)

        # 创建带编号的输出文件名
        output_file = f"./structures/{i+1}/{input_file_name}_{i+1}.inp"
        
        # 随机生成2或4个原子
        number_of_atoms = np.random.randint(2, 5)

        if number_of_atoms == 2:
            lenth_of_structure = np.random.rand() * 3 + 4  # 随机生成结构长度，范围在10到20之间
        elif number_of_atoms == 3:
            lenth_of_structure = np.random.rand() * 2 + 6
        else:
            lenth_of_structure = np.random.rand() * 2 + 7

        # 生成一个长度为number_of_atoms的只含有随机0或1的数组, 决定原子类型
        random_atoms = np.random.randint(0, 2, size=number_of_atoms)

        coordinates_of_atoms = np.random.rand(number_of_atoms) * (lenth_of_structure - 1) + 0.5  # 初始化原子坐标数组
        coordinates_of_atoms.sort()  # 确保坐标是有序的
        flag = False
        while not all(1 < min(abs(coordinates_of_atoms[i] - coordinates_of_atoms[i-1]), lenth_of_structure - abs(coordinates_of_atoms[i] - coordinates_of_atoms[i-1])) < 3 for i in range(number_of_atoms)):
            coordinates_of_atoms = np.random.rand(number_of_atoms) * (lenth_of_structure - 1) + 0.5  # 重新生成原子坐标
            coordinates_of_atoms.sort()

        del(content[start-5])
        content.insert(start-5, f"      ABC    {lenth_of_structure:.9f}      10.00000000     10.00000000\n")

        del(content[start:start+3])  # 删除原有的原子数目行

        for j in range(number_of_atoms):
            if random_atoms[j] == 0:
                content.insert(start + j, f"      Si     {coordinates_of_atoms[j]:.9f}      0.000000000      0.000000000\n")
            else:
                content.insert(start + j, f"      C      {coordinates_of_atoms[j]:.9f}      0.000000000      0.000000000\n")
        

        # 将修改后的内容写入输出文件
        with open(output_file, 'w') as f:
            f.writelines(content)

        # 打印进度信息
        print(f"已生成第 {i+1}/{number_of_structures} 个输入文件: {output_file}")


    print("完成!")


if __name__ == "__main__":
    main()