import os
import shutil
import random

def split_dataset_2_layer(source_folder, train_folder, test_folder, train_ratio=0.8):
    
    """
    随机划分数据集为训练集和测试集。
    保持原有的文件夹结构，并将图片随机分配到训练集和测试集中。
    """
    # 确保目标文件夹存在
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 遍历父文件夹中的所有子文件夹
    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # 创建训练集和测试集的目标文件夹
        train_subfolder = os.path.join(train_folder, subfolder)
        test_subfolder = os.path.join(test_folder, subfolder)
        os.makedirs(train_subfolder, exist_ok=True)
        os.makedirs(test_subfolder, exist_ok=True)

        # 获取子文件夹中的所有图片文件
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        random.shuffle(image_files)  # 随机打乱文件顺序

        # 划分训练集和测试集
        split_index = int(len(image_files) * train_ratio)
        train_files = image_files[:split_index]
        test_files = image_files[split_index:]

        # 复制图片到训练集和测试集
        for file in train_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(train_subfolder, file))
        for file in test_files:
            shutil.copy(os.path.join(subfolder_path, file), os.path.join(test_subfolder, file))

        print(f"Processed folder: {subfolder_path}")
        print(f"  Train set: {len(train_files)} images")
        print(f"  Test set: {len(test_files)} images")

def split_dataset_1_layer(source_folder, train_folder, test_folder, train_ratio=0.8):
    """
    随机划分数据集为训练集和测试集。
    将图片随机分配到训练集和测试集中。
    """
    # 确保目标文件夹存在
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 获取源文件夹中的所有图片文件
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    random.shuffle(image_files)  # 随机打乱文件顺序

    # 划分训练集和测试集
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # 复制图片到训练集和测试集
    for file in train_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(train_folder, file))
    for file in test_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(test_folder, file))

    print(f"Processed folder: {source_folder}")
    print(f"  Train set: {len(train_files)} images")
    print(f"  Test set: {len(test_files)} images")
    
    
    
# 使用示例
if __name__ == "__main__":
    # 处理单层文件夹包含
    source_folder = "/home/sata_one/mwp/Military_dataset/NonMilitaryReal/"  # 替换为你的源文件夹路径
    train_folder = "/home/sata_one/mwp/Military_dataset/train/NonMilitaryReal/"   # 替换为你的训练集目标文件夹路径
    test_folder = "/home/sata_one/mwp/Military_dataset/test/NonMilitaryReal/"     # 替换为你的测试集目标文件夹路径
    split_dataset_1_layer(source_folder, train_folder, test_folder)
    
    # 处理双层文件夹包含
    # split_dataset_2_layer