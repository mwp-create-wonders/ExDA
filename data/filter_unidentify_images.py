import os
import shutil
from PIL import Image, UnidentifiedImageError

def filter_invalid_images(source_folder, invalid_folder):
    """
    过滤并删除一个文件夹中所有无法识别或损坏的图像文件。
    :param source_folder: 包含图像文件的源文件夹路径。
    :param invalid_folder: 用于存放损坏文件的目标文件夹路径。
    """
    # 确保目标文件夹存在
    os.makedirs(invalid_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            try:
                # 尝试打开并验证图像文件
                with Image.open(file_path) as img:
                    img.verify()  # 验证文件是否损坏
            except (IOError, UnidentifiedImageError) as e:
                # 如果文件无法识别或损坏，移动到损坏文件文件夹
                invalid_path = os.path.join(invalid_folder, filename)
                shutil.move(file_path, invalid_path)
                print(f"Moved invalid file to {invalid_path}: {e}")

    print("Filtering complete.")

# 使用示例
if __name__ == "__main__":
    source_folder = "/home/sata_one/mwp/Military_dataset/test/GenImage/ADM_1_fake/"  # 替换为你的源文件夹路径
    invalid_folder = "/home/sata_one/mwp/Military_dataset/test/invalid_images/"  # 替换为你的损坏文件目标文件夹路径
    filter_invalid_images(source_folder, invalid_folder)