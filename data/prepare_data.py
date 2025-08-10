import os

def rename_images(root_folder):
    """
    递归遍历文件夹及其子文件夹中的图片文件，并重命名这些文件。
    文件名格式：数字_一级文件夹名_二级文件夹名_1_fake/0_real
    """
    for root, dirs, files in os.walk(root_folder):
        # 获取当前路径的文件夹名称
        folder_names = root.split(os.sep)
        if len(folder_names) < 2:
            # 如果当前路径不是至少包含两个文件夹，则跳过
            continue

        # 获取一级和二级文件夹名称
        first_level_folder = folder_names[-2]
        second_level_folder = folder_names[-1]

        # 初始化文件编号
        file_counter = 1

        for file in sorted(files):
            # 检查文件是否是图片文件（通过扩展名）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 获取文件扩展名
                file_extension = os.path.splitext(file)[1]

                # 构造新的文件名（数字 + 后缀）
                # 自己修改后缀为真假
                new_name = f"{file_counter}_{first_level_folder}_{second_level_folder}_0_real{file_extension}"

                # 构造完整的旧文件路径和新文件路径
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_name)

                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")

                # 更新文件编号
                file_counter += 1


def rename_files_with_suffix_and_number(folder_path, suffix):
    """
    为指定文件夹内的所有文件添加后缀和编号。
    :param folder_path: 文件夹路径。
    :param suffix: 要添加的后缀（不包括下划线）。
    """
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # 对文件进行编号并重命名
    for index, filename in enumerate(files, start=1):
        # 获取文件的完整路径
        old_file_path = os.path.join(folder_path, filename)
        
        # 分离文件名和扩展名
        file_name, file_extension = os.path.splitext(filename)
        
        # 构造新的文件名（原文件名_编号_后缀.原扩展名）
        new_file_name = f"{index}_{suffix}{file_extension}"
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} -> {new_file_path}")


# 使用示例
if __name__ == "__main__":
    # root_folder = "/home/sata_one/mwp/Military_dataset/NonMilitaryReal/"  # 替换为你的根文件夹路径
    # rename_images(root_folder)
    folder_path = "/home/sata_one/mwp/Military_dataset/train/ChameleonTest_0_real/0_real/"
    suffix = "0_real"
    rename_files_with_suffix_and_number(folder_path, suffix)