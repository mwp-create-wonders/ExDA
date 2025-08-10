import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from Enhanced import EnhancedImageAnalyzer, get_image_paths

class UseForTrain(EnhancedImageAnalyzer):
    def __init__(self, image_size=(256,256), model="model"):
        super().__init__(image_size, output_dir=f"{model}_result")
        self.model = model
        # 提前创建输出目录
        os.makedirs(os.path.join(self.output_dir, "spectrum"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "autocorr"), exist_ok=True)

    def getFolder(self, image_paths, label="Unknown", selected_features=None):
        """多进程并行处理并直接保存结果"""
        selected_features = selected_features or list(self.feature_handlers.keys())
        self._validate_features(selected_features)
        
        # 多进程处理
        with Pool(4) as pool:
            results = list(tqdm(
                pool.imap(self._process_single, image_paths),
                total=len(image_paths), desc=f"Processing {label}"
            ))
        return results  # 直接返回原始结果（已实时保存图像）

    def _process_single(self, path):
        """覆盖父类方法：处理单张图像并直接保存结果图"""
        try:
            # 调用父类处理流程
            results = super()._process_single(path)
            if results is None:
                return None

            # 保存频谱图（无边框无刻度）
            if "spectrum" in results:
                filename = os.path.splitext(os.path.basename(path))[0]
                save_path = os.path.join(
                    self.output_dir,
                    "spectrum",
                    f"{self.model}_{filename}_spectrum.png"
                )
                self._save_clean_plot(
                    results["spectrum"],
                    save_path,
                    cmap='jet',
                    log_scale=True
                )

            return results
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return None

    def _save_clean_plot(self, data, save_path, cmap='jet', log_scale=False):
        """通用保存函数：无边框、无刻度、无白边"""
        plt.figure(figsize=(5,5), frameon=False)
        ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)
        
        if log_scale:
            data=np.log(data)

        ax.imshow(data, cmap=cmap, aspect='auto')
        plt.savefig(
            save_path,
            bbox_inches='tight',
            pad_inches=0,
            dpi=300,
            transparent=True
        )
        plt.close()

if __name__=="__main__":
    model="CycleGAN"

    analyzer=UseForTrain(model=model)

    image_path=get_image_paths("CycleGAN\\warship")
    image_path=image_path[:1250]

    analyzer.getFolder(image_paths=image_path,label=model,selected_features=["spectrum"])
