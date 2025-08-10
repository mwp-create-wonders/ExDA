import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class EnhancedImageAnalyzer:
    def __init__(self, image_size=(256, 256), output_dir=None, device="cuda"):
        # 规定大小和输出路径
        self.image_size = image_size
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 创建输出和临时文件夹
        os.makedirs(output_dir, exist_ok=True)
        self.feature_handlers = {
            "autocorr": self._handle_autocorr,
            "spectrum": self._handle_spectrum,
            "radial": self._handle_radial,
            "angular": self._handle_angular
        }
        

    def analyze_images(self, image_paths, label=None, selected_features=None):
        """多进程并行处理"""
        selected_features = selected_features or list(self.feature_handlers.keys())
        self._validate_features(selected_features)
        
        
        # 初始化结果缓冲区
        analysis_buffers = {ftr: [] for ftr in selected_features}
        # print(result)
        
        # 单进程处理每个图像路径
        for image_path in tqdm(image_paths, desc=f"Processing {label}"):
            res = self._process_single(image_path)
            if res is not None:
                for ftr in selected_features:
                    analysis_buffers[ftr].append(res[ftr])
        
        # 计算平均结果
        avg_results = {ftr: np.mean(analysis_buffers[ftr], axis=0) for ftr in selected_features}
        
        # 保存结果
        self._save_results(avg_results, label, selected_features)
        
        return avg_results
    


    def _process_single(self, path):
        """单张图像处理流程（返回PyTorch Tensor）"""
        try:
            img = self._load_image(path)
            residual = self._extract_residual(img)
            results = {}
            for ftr in self.feature_handlers:
                results[ftr] = self.feature_handlers[ftr](residual).cpu().numpy()  # 统一在此转换numpy
            return results
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return None
        
        
    def _validate_features(self, features):
        """验证特征选择有效性"""
        valid_features = set(self.feature_handlers.keys())
        input_features = set(features)
        
        if not input_features.issubset(valid_features):
            invalid = input_features - valid_features
            raise ValueError(f"Invalid features selected: {invalid}")


    def _load_image(self, path):
        """GPU加速的图像加载"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.resize(img, self.image_size)
        return torch.tensor(img, device=self.device).float() / 255.0


    def _extract_residual(self, image, denoiser_sigma=1):
        """GPU噪声残差提取（保持张量在设备上）"""
        image_cpu = (image.cpu().numpy() * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(image_cpu, h=denoiser_sigma)
        return image - torch.tensor(denoised/255.0, device=self.device).float()


    def _handle_autocorr(self, residual):
        """优化后的自相关计算（避免过大padding）"""
        # 使用FFT加速计算
        x = residual.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        fft_x = torch.fft.rfft2(x, s=(x.shape[-2]*2-1, x.shape[-1]*2-1))
        corr = torch.fft.irfft2(fft_x * fft_x.conj()).real
        # 裁剪中心65x65区域
        h, w = corr.shape[-2], corr.shape[-1]
        center = corr[..., h//2-32:h//2+33, w//2-32:w//2+33]
        return center.squeeze()


    def _handle_spectrum(self, residual):
        """GPU功率谱计算"""
        fft = torch.fft.fft2(residual)
        fft_shift = torch.fft.fftshift(fft)
        return torch.abs(fft_shift).pow(2)


    def _radial_angular_analysis(self, spectrum):
        """极坐标分析（完全GPU实现）"""
        rows, cols = spectrum.shape
        cy, cx = rows//2, cols//2
        
        y, x = torch.meshgrid(
            torch.arange(-cy, rows-cy, device=self.device),
            torch.arange(-cx, cols-cx, device=self.device),
            indexing='ij'
        )
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        
        # 径向谱
        radial_bins = torch.linspace(0, 0.5, 128, device=self.device)
        bin_indices = torch.bucketize(r/rows, radial_bins)
        radial_profile = torch.zeros_like(radial_bins)
        radial_counts = torch.zeros_like(radial_bins)
        for i in range(len(radial_bins)):
            mask = (bin_indices == i)
            radial_profile[i] = torch.sum(spectrum[mask])
            radial_counts[i] = torch.sum(mask)
        radial = radial_profile / (radial_counts + 1e-6)
        
        # 角度谱
        theta_bins = torch.linspace(-torch.pi, torch.pi, 17, device=self.device)
        bin_indices = torch.bucketize(theta, theta_bins)
        angular_profile = torch.zeros(16, device=self.device)
        angular_counts = torch.zeros(16, device=self.device)
        for i in range(16):
            mask = (bin_indices == i)
            angular_profile[i] = torch.sum(spectrum[mask])
            angular_counts[i] = torch.sum(mask)
        angular = angular_profile / (angular_counts + 1e-6)
        
        return radial, angular
    
    
    def _handle_radial(self, residual):
        spectrum = self._handle_spectrum(residual)
        radial, _ = self._radial_angular_analysis(spectrum)
        return radial


    def _handle_angular(self, residual):
        spectrum = self._handle_spectrum(residual)
        _, angular = self._radial_angular_analysis(spectrum)
        return angular


    def _save_results(self, results, label, selected_features):
        """保存结果"""
        np.savez(os.path.join(self.output_dir, f"{label}_features.npz"), **results)
        self._plot_features(results, label, selected_features)


    def _plot_features(self, results, label, selected_features):
        """动态生成可视化图表"""
        plot_count = len([ftr for ftr in selected_features if ftr in ["autocorr", "spectrum"]])
        if plot_count == 0:
            return

        # 绘制自相关图
        if "autocorr" in selected_features:
            plt.figure(figsize=(6, 6))
            plt.imshow(results["autocorr"], cmap='jet')
            plt.axis('off')  # 关闭坐标轴
            # plt.title(f"{label} - Autocorrelation", fontsize=16)  # 添加标题（可选）
            plt.savefig(os.path.join(self.output_dir, f"{label}_autocorr.png"), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # 绘制功率谱图
        if "spectrum" in selected_features:
            plt.figure(figsize=(6, 6))
            plt.imshow(np.log(results["spectrum"] + 1e-6), cmap='jet')
            plt.axis('off')  # 关闭坐标轴
            # plt.title(f"{label} - Power Spectrum (log)", fontsize=16)  # 添加标题（可选）
            plt.savefig(os.path.join(self.output_dir, f"{label}_spectrum.png"), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()


# --------------------------------------------
# 新增实用函数
# --------------------------------------------
def get_image_paths(folder_path, extensions=('jpg', 'jpeg', 'png','webp')):
    """遍历文件夹获取图像路径列表"""
    valid_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                valid_paths.append(os.path.join(root, file))
    return valid_paths

# --------------------------------------------
# 主函数
# --------------------------------------------
if __name__=="__main__":
    model="SD15_2img"

    analyzer=EnhancedImageAnalyzer(image_size=(512,512),output_dir="/home/mwp/UniversalFakeDetect-main/data/Image_spectral/result/"+model)

    image_folder="/home/sata_one/mwp/Military_dataset/train/SD15_1_fake/"
    image_path=get_image_paths(image_folder)
    
    # print(image_path)
    cutsome=image_path[:20]

    analyzer.analyze_images(image_paths=cutsome,label=model+"_SD15",selected_features=["spectrum", "radial", "autocorr", "angular"])

    #analyzer.getFolder(image_paths=cutsome,label=model,selected_features=["spectrum"],model=model)


        # self.feature_handlers = {
        #     "autocorr": self._handle_autocorr,
        #     "spectrum": self._handle_spectrum,
        #     "radial": self._handle_radial,
        #     "angular": self._handle_angular
        # }
        
