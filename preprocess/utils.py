import datetime
import os
import pickle
import numpy as np
import torch
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from astropy.io import fits
import torchvision.transforms as transforms
import random
from torch.backends import cudnn
import math
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.gaia import Gaia
import warnings
warnings.filterwarnings("ignore")
Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"  # Select early Data Release 3
SAVE_PATH = "/data/renhaoye/decals_2022/"  # the head of the directory to save
PIXEL = 5
DEGREE = 30


def get_weight(num_samples: list):
    sum = 0
    for i in num_samples:
        sum += i
    for i in range(len(num_samples)):
        num_samples[i] = 1 - num_samples[i] / sum
    return num_samples


def changeAxis(data: np.ndarray):
    """
    input: ndarray, C×H×W, should do before ToTensor()
    """
    img = np.swapaxes(data, 0, 2)
    img = np.swapaxes(img, 0, 1)
    return img


def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    # torch.backends.cudnn.deterministic = True # 这个可能会导致训练过程非常慢


def normalization(img: np.ndarray, shape: str, lock: bool):
    """
    :param img: CHW和HWC均可
    :param shape: 传入CHW或者HWC，输出同一形式
    :param lock: True时，三通道一起归一化，否则分通道归一化
    :return: 返回CHW或HWC
    """

    def compute(data: np.ndarray):
        h, w = data.shape
        data = data.reshape(-1)
        max = np.max(data)
        min = np.min(data)
        if max == min:
            return 65535
        else:
            norm_data = (data - min) / (max - min)
            return norm_data.reshape((h, w))

    if lock:
        original_shape = img.shape
        flatten = img.reshape(-1)
        normalized = (flatten - flatten.min()) / (flatten.max() - flatten.min())
        return normalized.reshape(original_shape)
    else:
        if shape == "CHW" or shape == "chw":
            g, r, z = img[0], img[1], img[2]
            norm_g, norm_r, norm_z = compute(g), compute(r), compute(z)
            if type(norm_g) == int or type(norm_r) == int or type(norm_z) == int:
                return 65535
            else:
                return np.array((norm_g, norm_r, norm_z))
        elif shape == "HWC" or shape == "hwc":
            g, r, z = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            norm_g, norm_r, norm_z = compute(g), compute(r), compute(z)
            if type(norm_g) == int or type(norm_r) == int or type(norm_z) == int:
                return 65535
            else:
                return np.concatenate(
                    (norm_g.reshape(256, 256, 1), norm_r.reshape(256, 256, 1), norm_z.reshape(256, 256, 1)), axis=2)
        else:
            raise RuntimeError


def mtf(data: np.ndarray, m: float):
    """
    非线性变换
    :param data: 输入数据
    :param m: midtones值
    :return: 结果
    """
    return ((m - 1) * data) / ((2 * m - 1) * data - m)

def get_midtones(median):
    return (7*median)/(6*median + 1)
def channel_cut(gray):
    highlight = 1.
    hist, bar = np.histogram(gray.reshape(-1), bins=65536)
    cdf = hist.cumsum()
    shadow_index = np.argwhere(cdf > 0.001 * gray.reshape(-1).shape[0])[0]
    shadow = bar[shadow_index]
    midtones = np.median(gray) - shadow
    gray[gray < shadow] = shadow
    gray[gray > highlight] = 1.
    gray = gray.reshape(-1)
    norm_data = (gray - gray.min()) / (gray.max() - gray.min())
    gray = norm_data.reshape((256, 256))
    right_midtones = get_midtones(np.median(gray))
    return mtf(gray, right_midtones)


def auto_scale(data: np.ndarray):
    g, r, z = channel_cut(data[0]), channel_cut(data[1]), channel_cut(data[2])
    img = np.array((g, r, z))
    img[img < 0] = 0
    img[img > 1] = 1.
    return img


def chw2hwc(img):
    ch1, ch2, ch3 = img[0], img[1], img[2]
    h, w = ch1.shape
    return np.concatenate((ch1.reshape(h, w, 1), ch2.reshape(h, w, 1), ch3.reshape(h, w, 1)), axis=2)


def hwc2chw(img):
    ch1, ch2, ch3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return np.array((ch1, ch2, ch3))


def load_img(file):
    """
    加载图像，dat和fits均支持，不过仅支持CxHxW
    :param filename: 传入文件名，应当为CHW
    :return: 返回CHW的ndarray
    """
    if ".fits" in file:
        with fits.open(file) as hdul:
            return hdul[0].data.astype(np.float32)
    elif ".dat" in file:
        with open(file, "rb") as f:
            return pickle.load(f)
    else:
        raise TypeError


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def mk_model_dir(info):
    """
    返回文件夹名：model_info_datetime
    """
    model_package = 'model_%s' % info
    if not model_package[-1] == '_':
        model_package += '_'
    model_package += str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    return model_package


def save_fits(data: np.ndarray, filename: str):
    """
    将ndarray保存成fits文件
    :param data: 待保存数据
    :param filename: 保存文件名
    :return:
    """
    if len(data.shape) == 2:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    elif data.shape[-1] == 3:
        g, r, z = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        data = np.array((g, r, z))
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    elif data.shape[0] == 3:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    else:
        raise RuntimeError



# 读取模型
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    #     print(model)
    #     for name in model.state_dict():
    #         print(name)
    return model


class Img:
    def __init__(self, image, rows, cols, center=None):
        self.g_dst = None
        self.r_dst = None
        self.z_dst = None
        if center is None:
            center = [0, 0]
        self.dst = None
        self.g_src = image[0]
        self.r_src = image[1]
        self.z_src = image[2]
        self.transform = None
        self.rows = rows
        self.cols = cols
        self.center = center  # rotate center

    def Shift(self, delta_x, delta_y):  # 平移
        # delta_x>0 shift left  delta_y>0 shift top
        self.transform = np.array([[1, 0, delta_x],
                                   [0, 1, delta_y],
                                   [0, 0, 1]])

    def Flip(self):  # vertically flip
        self.transform = np.array([[-1, 0, self.rows - 1],
                                   [0, 1, 0],
                                   [0, 0, 1]])

    def Rotate(self, beta):  # rotate
        # beta<0 rotate clockwise
        self.transform = np.array([[math.cos(beta), -math.sin(beta), 0],
                                   [math.sin(beta), math.cos(beta), 0],
                                   [0, 0, 1]])

    def Process(self):
        self.g_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.r_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.z_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([i - self.center[0], j - self.center[1], 1])
                [x, y, z] = np.dot(self.transform, src_pos)
                x = int(x) + self.center[0]
                y = int(y) + self.center[1]
                if x >= self.rows or y >= self.cols:
                    self.g_dst[i][j] = 0.
                    self.r_dst[i][j] = 0.
                    self.z_dst[i][j] = 0.
                else:
                    self.g_dst[i][j] = self.g_src[int(x)][int(y)]
                    self.r_dst[i][j] = self.r_src[int(x)][int(y)]
                    self.z_dst[i][j] = self.z_src[int(x)][int(y)]
        self.dst = np.array((self.g_dst, self.r_dst, self.z_dst))


def flip(img, save_dir):
    height, width = img.shape[1:3]
    output = Img(img, height, width, [0, 0])
    output.Flip()
    output.Process()
    save_fits(output.dst, save_dir + "_flipped.fits")


def rotate(img, save_dir):
    seed = random.randint(0, 90)
    height, width = img.shape[1:3]
    output = Img(img, height, width, [height / 2, width / 2])
    output.Rotate(seed)
    output.Process()
    save_fits(output.dst, save_dir + "_rotated.fits")


def shift(img, save_dir, pixel):
    height, width = img.shape[1:3]
    output = Img(img, height, width, [0, 0])
    output.Shift(pixel, 0)
    output.Process()
    save_fits(output.dst, save_dir + "_shifted.fits")
