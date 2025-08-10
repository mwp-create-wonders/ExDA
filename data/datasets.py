import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from random import random, choice, shuffle, sample
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import pickle
import os 
from skimage.io import imread
from copy import deepcopy

# 即使图片文件被截断，也不会报错
ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}



def recursively_read(rootdir, must_contain, exts=["png", "jpg", "jpeg"], counts=1000):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            ext = os.path.splitext(file)[1].lower()[1:]  # 获取扩展名并去掉点号
            if ext in exts and must_contain in os.path.join(r, file):
                out.append(os.path.join(r, file))
    return out


def few_recursively_read(rootdir, must_contain, count, exts=["png", "jpg", "jpeg"]):
    out = []
    # 获取根目录下的所有子文件夹
    subfolders = [os.path.join(rootdir, folder) for folder in os.listdir(rootdir)
                  if os.path.isdir(os.path.join(rootdir, folder))]
    
    for subfolder in subfolders:
        # 检查子文件夹名称是否包含 must_contain 关键词
        if must_contain not in os.path.basename(subfolder):
            continue
        
        # 遍历当前子文件夹，收集所有符合条件的图片
        valid_files = []
        for r, d, f in os.walk(subfolder):
            for file in f:
                ext = os.path.splitext(file)[1].lower()[1:]  # 获取扩展名并去掉点号
                if ext in exts:
                    valid_files.append(os.path.join(r, file))
        
        # 如果该子文件夹中有符合条件的文件，随机选择指定数量的文件
        if valid_files:
            selected_files = sample(valid_files, min(count, len(valid_files)))
            out.extend(selected_files)
    
    return out


def get_list(path, must_contain=''):
    # 判断是否是.pickle文件
    if ".pickle" in path:
        try:
            with open(path, 'rb') as f:
                image_list = pickle.load(f)
            image_list = [item for item in image_list if must_contain in item]
        except (pickle.PickleError, FileNotFoundError) as e:
            print(f"Error loading pickle file: {e}")
            return []
    else:
        image_list = recursively_read(path, must_contain)
    return image_list

def data_augment(img, opt):
    img = np.array(img)
    # 通道扩展
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    # 高斯模糊
    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    # jpeg压缩
    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)

# **************高斯模糊*****************
def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
    
# **************************************** #

# **************JPEG压缩******************
def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

# **************************************** #

# ***************裁剪*********************
rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])

# *************************************** #

class RealFakeDataset(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        #assert opt.data_mode in ["ours", "wang2020", "ours_wang2020"]
        self.data_label  = opt.data_label
        
        
        # 选择数据位置
        # ******************************MM代码********************************************************
        # ******************************************************************************************************************************
        # if opt.data_mode == 'ours':
        #     temp = 'train' if opt.data_label == 'train' else 'test'
        #     if temp=='train':
        #         print("begin create data")
        #         # real_list = few_recursively_read( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real', count=4500)
        #         # fake_list = few_recursively_read( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake', count=1000)
        #         # real_list = few_recursively_read( "/home/sata_one/mwp/ComImageDataset/train", must_contain='0_real', count=10000)
        #         # fake_list = few_recursively_read( "/home/sata_one/mwp/ComImageDataset/train", must_contain='1_fake', count=10000)               
        #     else:
        #         real_list = get_list( os.path.join(opt.wang2020_data_path,temp,'MilitaryReal'), must_contain='0_real' )
        #         fake_list = get_list( os.path.join(opt.wang2020_data_path,temp,'LD'), must_contain='1_fake' )
        
        # elif opt.data_mode == 'wang2020':
        #     # 只先用一种验证一下
        #     temp = 'train' if opt.data_label == 'train' else 'test'
        #     # real_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real' )
        #     # fake_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake' )
            
        #     # 首先采用同等的正负样本
        #     if temp=='train':
        #         real_list = get_list( os.path.join(opt.wang2020_data_path,temp,'MilitaryReal'), must_contain='0_real' )
        #         fake_list = get_list( os.path.join(opt.wang2020_data_path,temp,'SD14'), must_contain='1_fake' )   
        #     else:
        #         real_list = get_list( os.path.join(opt.wang2020_data_path,temp,'MilitaryReal'), must_contain='0_real' )
        #         fake_list = get_list( os.path.join(opt.wang2020_data_path,temp,'LD'), must_contain='1_fake' )      
            
        # elif opt.data_mode == 'ours_wang2020':
        #     pickle_name = "train.pickle" if opt.data_label=="train" else "val.pickle"
        #     real_list = get_list( os.path.join(opt.real_list_path, pickle_name) )
        #     fake_list = get_list( os.path.join(opt.fake_list_path, pickle_name) )
        #     temp = 'train/progan' if opt.data_label == 'train' else 'test/progan'
        #     real_list += get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real' )
        #     fake_list += get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake' )
        
        # # ******************************************************************************************************************************
        # # ******************************************************************************************************************************

        # 竞赛训练
        real_list = few_recursively_read( "/home/sata_one/mwp/ComImageDataset/train", must_contain='0_real', count=10000)
        fake_list = few_recursively_read( "/home/sata_one/mwp/ComImageDataset/train", must_contain='1_fake', count=10000)    

        # setting the labels for the dataset
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        # 做数据增强
        # TODO 希望在数据增强中采用一些优化技术
        # **********************************************************************#
        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)
        if not opt.isTrain and opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        
        
        # ************************************************************************#
        # 选择统计信息
        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        print("mean and std stats are from: ", stat_from)
        # 判断是否使用2b架构
        if '2b' not in opt.arch:
            print ("using Official CLIP's normalization")
            self.transform = transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        else:
            print ("Using CLIP 2B transform")
            self.transform = None # will be initialized in trainer.py


    def __len__(self):
        return len(self.total_list)


    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label



