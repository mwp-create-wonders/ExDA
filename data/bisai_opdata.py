import pandas as pd
import os
import shutil

# --- 配置区 (请根据您的实际情况修改以下变量) ---

# 1. 包含所有图片的根目录 (例如 'D:/my_project/images/')
#    这是CSV第一列路径的起点
BASE_IMAGE_DIR = '/home/sata_one/mwp/ComImageDataset/train0/'  # 请务必修改为您的主文件夹地址

# 2. 您的CSV文件的完整路径
CSV_FILE_PATH = '/home/sata_one/mwp/ComImageDataset/train.csv'  # 请修改为您的CSV文件路径

# 3. 您希望存放“真实”图片的目标文件夹
REAL_OUTPUT_DIR = '/home/sata_one/mwp/ComImageDataset/train/real/' # 请修改为您想要存放真实图片的文件夹

# 4. 您希望存放“伪造”图片的目标文件夹
FAKE_OUTPUT_DIR = '/home/sata_one/mwp/ComImageDataset/train/fake/' # 请修改为您想要存放伪造图片的文件夹

# 5. CSV文件中列的索引 (注意：第一列是0，第三列是2)
PATH_COLUMN_INDEX = 0  # 文件相对路径所在的列 (第一列)
LABEL_COLUMN_INDEX = 2 # 真伪标签所在的列 (第三列)

# 6. CSV文件中标签的具体值是什么
#    例如，如果CSV里用 0 代表真实, 1 代表伪造，就设置为:
REAL_LABEL_VALUE = '0'
FAKE_LABEL_VALUE = '1'


def process_images_from_csv():
    """
    根据CSV文件中的信息，将图片分类、重命名并复制到指定文件夹。
    """
    print("--- 任务开始 ---")

    # 步骤1: 检查并创建目标文件夹
    try:
        os.makedirs(REAL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(FAKE_OUTPUT_DIR, exist_ok=True)
        print(f"真实图片将存入: {REAL_OUTPUT_DIR}")
        print(f"伪造图片将存入: {FAKE_OUTPUT_DIR}")
    except OSError as e:
        print(f"错误：无法创建目标文件夹，请检查路径权限。错误信息: {e}")
        return

    # 步骤2: 读取CSV文件
    try:
        df = pd.read_csv(CSV_FILE_PATH, header=None)
        print(f"成功读取CSV文件，共找到 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误：CSV文件未找到，请检查路径: '{CSV_FILE_PATH}'")
        return
    except Exception as e:
        print(f"错误：读取CSV文件时发生问题。错误信息: {e}")
        return
        
    # 步骤3: 遍历CSV的每一行并处理文件
    real_count = 0
    fake_count = 0
    error_count = 0
    
    print("\n--- 开始处理图片 ---")
    for index, row in df.iterrows():
        try:
            relative_path = str(row[PATH_COLUMN_INDEX])
            label = row[LABEL_COLUMN_INDEX]

            # 3.1. 构建源文件完整路径
            source_path = os.path.join(BASE_IMAGE_DIR, relative_path)
            
            # 3.2. 检查源文件是否存在
            if not os.path.exists(source_path):
                print(f"警告 (行 {index + 1}): 源文件不存在，已跳过 -> {source_path}")
                error_count += 1
                continue

            # 3.3. 根据标签确定目标文件夹和新文件名后缀
            if label == REAL_LABEL_VALUE:
                destination_dir = REAL_OUTPUT_DIR
                suffix = '_0_real'
                real_count += 1
            elif label == FAKE_LABEL_VALUE:
                destination_dir = FAKE_OUTPUT_DIR
                suffix = '_1_fake'
                fake_count += 1
            else:
                print(f"警告 (行 {index + 1}): 未知的标签 '{label}'，已跳过。")
                error_count += 1
                continue

            # 3.4. 生成新的文件名
            # 分离出原始文件名和扩展名
            original_filename = os.path.basename(relative_path)
            name_part, extension = os.path.splitext(original_filename)
            # 拼接成新文件名
            new_filename = f"{name_part}{suffix}{extension}"
            
            # 3.5. 构建完整的目标路径
            destination_path = os.path.join(destination_dir, new_filename)
            
            # 3.6. 复制文件
            shutil.copy2(source_path, destination_path)
            
            print(f"处理成功 (行 {index + 1}): {original_filename} -> {new_filename}")

        except Exception as e:
            print(f"错误 (行 {index + 1}): 处理时发生未知异常，已跳过。错误信息: {e}")
            error_count += 1
            
    # 步骤4: 打印最终总结
    print("\n--- 任务完成 ---")
    print("处理结果总结:")
    print(f"  成功分类的真实图片: {real_count}")
    print(f"  成功分类的伪造图片: {fake_count}")
    print(f"  跳过或出错的记录数: {error_count}")
    print("--------------------")


if __name__ == '__main__':
    process_images_from_csv()