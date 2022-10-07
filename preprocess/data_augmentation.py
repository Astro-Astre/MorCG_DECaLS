import csv
from utils import *
from astropy.io import fits
import random
from functools import partial
import math
from tqdm import tqdm
import numpy as np
import multiprocessing

random.seed(1926)

SAVE_PATH = "/data/renhaoye/MorCG/dataset/"  # the head of the directory to save


def augmentation(i, files):
    dst_dir = "/data/renhaoye/MorCG/dataset/in_decals/agmtn/"  # 目标存放文件夹
    src_dir = "/data/renhaoye/MorCG/dataset/in_decals/raw_fits/"  # 原始路径
    src_file = src_dir + files[i]  # 原始图片绝对路径
    ra_dec = files[i].split(".fits")[0]
    dst_file = dst_dir + ra_dec  # 保存绝对路径 不带扩展名
    # if not (os.path.exists(dst_file + ".fits") or os.path.exists(dst_file + "_flipped.fits") or os.path.exists(dst_file + "_rotated.fits") or os.path.exists(dst_file + "_shifted.fits")):
    normalized_unlock = normalization(load_img(src_file), shape="CHW", lock=False)  # 先做一次分通道归一化
    if not type(normalized_unlock) == int:
        normalized_lock = normalization(normalized_unlock, shape="CHW", lock=True)  # 再做一次全通道归一化
        if not type(normalized_lock) == int:
            scaled = auto_scale(normalized_lock)  # 再做stf
            save_fits(scaled, dst_file + '.fits')
            flip(scaled, dst_file)
            shift(scaled, dst_file, random.randint(1, 10))
            rotate(scaled, dst_file)
        else:
            print(src_file)
            # os.system("mv %s /data/renhaoye/MorCG/dataset/in_decals/raw_fits_deleted/" % src_file)
    else:
        print(src_file)
            # os.system("mv %s /data/renhaoye/MorCG/dataset/in_decals/raw_fits_deleted/" % src_file)


if __name__ == "__main__":
    # print("start")
    src = os.listdir("/data/renhaoye/MorCG/dataset/in_decals/raw_fits/")
    # src = [
    #     "176.92935835560405_31.288642439785587.fits",
    #     "217.84099821957136_11.994322267034859.fits",
    #     "332.75792104030495_-0.10110663218382575.fits",
    #     "140.21107123021102_3.9801104515071883.fits"]
    index = []
    for i in range(len(src)):
    # for i in range(10):
        index.append(i)
    p = multiprocessing.Pool(4)
    p.map(partial(augmentation, files=src), index)
    p.close()
    p.join()
