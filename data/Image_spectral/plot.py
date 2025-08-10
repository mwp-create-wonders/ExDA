import os
import numpy as np
import matplotlib.pyplot as plt

def load_npz_files(folder_path):
    """批量加载 .npz 文件"""
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    data_dict = {}
    for file in npz_files:
        label = os.path.splitext(file)[0]
        data = np.load(os.path.join(folder_path, file))
        data_dict[label] = {key: data[key] for key in data.keys()}
    return data_dict

def compute_fdr_sqrt(data_dict):
    """
    计算每个角度方向的费雪判别率（FDR）并取平方根。
    假设 data_dict 中包含两类数据：真实图像和生成图像。
    """
    # 初始化存储 FDR 的字典
    fdr_sqrt_dict = {}

    for label, features in data_dict.items():
        angular_data = features.get("angular", None)
        if angular_data is not None:
            # 假设 angular_data 已经分为两类数据
            # 例如：前半部分是真实图像，后半部分是生成图像
            half_len = len(angular_data) // 2
            real_data = angular_data[:half_len]  # 真实图像数据
            fake_data = angular_data[half_len:]  # 生成图像数据

            # 计算均值和方差
            mu_real = np.mean(real_data)
            mu_fake = np.mean(fake_data)
            var_real = np.var(real_data)
            var_fake = np.var(fake_data)

            # 计算 FDR 并取平方根
            fdr = ((mu_real - mu_fake) ** 2) / (var_real + var_fake)
            fdr_sqrt = np.sqrt(fdr)

            # 存储结果
            fdr_sqrt_dict[label] = fdr_sqrt * angular_data  # 将 FDR 应用到原始数据上
        else:
            print(f"Warning: No angular data found for {label}")

    return fdr_sqrt_dict


def plot_circular_angular(data_dict, save_path):
    """
    绘制角度谱的圆形图（极坐标图）。
    每个 .npz 文件的角度谱用不同的颜色表示，线条有颜色但无填充色。
    图例放在图的最下方，避免遮挡圆形。
    """
    # 定义颜色映射
    colors = plt.cm.tab10.colors  # 使用 tab10 调色板，最多支持 10 种颜色

    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # 绘制每个 .npz 文件的角度谱
    for idx, (label, angular_data) in enumerate(data_dict.items()):
        if angular_data is not None:
            theta = np.linspace(0, 2 * np.pi, len(angular_data), endpoint=True)  # 角度范围：0 到 360°
            angular_data[0] = angular_data[-1]
            label = label.split('_')[0]
            ax.plot(theta, angular_data, color=colors[idx % len(colors)], linewidth=2, label=label)
        else:
            print(f"Warning: No angular data found for {label}")

    # 设置图形样式
    ax.set_title("")
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))  # 设置角度刻度：0°, 45°, ..., 360°
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=10)
    
    # 图例放在图的最下方
    ax.legend(
        fontsize=16,
        loc="lower center", 
        bbox_to_anchor=(0.5, -0.25),  # 将图例放置在图的正下方
        ncol=3,  # 图例分为多列，避免过长
        frameon=True  # 去掉图例边框
    )

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)  # 增加 pad_inches 避免裁剪图例
    plt.close()
    
    
def plot_radial_spectrum(data_dict, save_path):
    """
    绘制径向功率谱密度图。
    横坐标是频率范围 [0, 0.5]，划分为与 radial_data 数据点数量相同的间隔。
    每个 .npz 文件的径向谱用不同的颜色表示，并添加图例。
    """
    # 定义颜色映射
    colors = plt.cm.tab10.colors  # 使用 tab10 调色板，最多支持 10 种颜色

    # 创建绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制每个 .npz 文件的径向谱
    for idx, (label, features) in enumerate(data_dict.items()):
        radial_data = features.get("radial", None)
        if radial_data is not None:
            # 提取图例名称：只保留第一个下划线之前的内容
            legend_label = label.split('_')[0]
            
            # 计算频率范围 [0, 0.5]，划分为与 radial_data 数据点数量相同的间隔
            num_points = len(radial_data)
            frequencies = np.linspace(0, 0.5, num_points)  # 频率范围 [0, 0.5]
            
            # 截取频率范围从 0.1 开始
            valid_indices = frequencies >= 0.2
            frequencies = frequencies[valid_indices]
            radial_data = radial_data[valid_indices]
            
            # 绘制径向功率谱密度
            ax.plot(frequencies, radial_data, color=colors[idx % len(colors)], linewidth=1.8, label=legend_label)
        else:
            print(f"Warning: No radial data found for {label}")

    # 设置图形样式
    # ax.set_title("Radial Spectrum Power Density Comparison", fontsize=14, pad=10)
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Power Density", fontsize=12)
    ax.set_xlim(0.2, 0.5)  # 设置横坐标范围为 [0.1, 0.5]
    ax.set_ylim(0, 2)      # 设置纵坐标范围为 [0, 5]
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")  # 图例位置调整到右上角

    # 保存图像
    plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    

def plot_zh():
    # Define the data arrays
    MID_qf = [98.98, 98.7, 98.01, 97.28]
    Gen_qf = [94.8, 96.64, 91.82, 89.39]
    MID_gau = [97.74, 96.82, 95.52, 93.52]
    Gen_gau = [89.49, 86.03, 83.37, 79.39]

    # Group data by element position
    data = np.array([MID_qf, Gen_qf, MID_gau, Gen_gau])
    
    

    # Define colors for each array
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Define labels for each group
    labels = ['Element 1', 'Element 2', 'Element 3', 'Element 4']

    # Define x positions for the bars
    x = np.arange(len(labels))

    # Define width of a bar
    width = 0.2

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each group of bars
    for i, (group_data, color) in enumerate(zip(data, colors)):
        ax.bar(x + i * width, group_data, width, label=f'Group {i+1}', color=color)

    # Add labels, title, and legend
    ax.set_xlabel('Elements')
    ax.set_ylabel('Values (%)')
    ax.set_title('Comparison of Elements Across Different Groups')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(labels)
    ax.legend(title='Groups')

    # Improve layout
    plt.tight_layout()

    # Save the figure
    plt.savefig('bar_chart.png', dpi=300)

    # Show the plot
    plt.show()
    
# 主程序
if __name__ == "__main__":
    # 定义文件夹路径和保存路径
    npz_folder = "/home/mwp/UniversalFakeDetect-main/data/Image_spectral/result/NPZ_data"  # 替换为你的 .npz 文件所在文件夹路径
    output_image_path = "/home/mwp/UniversalFakeDetect-main/data/Image_spectral/result/NPZ_data/freq.svg"  # 替换为你希望保存的图片路径
    # npz_folder = "/home/mwp/UniversalFakeDetect-main/data/Image_spectral/result/NPZ_data"  # 替换为你的 .npz 文件所在文件夹路径
    # output_image_path = "/home/mwp/UniversalFakeDetect-main/data/Image_spectral/result/NPZ_data/radial.svg"  # 替换为你希望保存的图片路径

    
    # 加载数据
    data_dict = load_npz_files(npz_folder)
    
    # 绘制并保存径向功率谱密度图
    # plot_radial_spectrum(data_dict, output_image_path)

    # 计算 FDR 并取平方根
    fdr_sqrt_dict = compute_fdr_sqrt(data_dict)

    # # 绘制并保存圆形角度谱图
    plot_circular_angular(fdr_sqrt_dict, output_image_path)
    
    # plot_zh()
    