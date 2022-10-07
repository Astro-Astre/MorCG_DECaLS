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

def scale_out_decals(i, files):
    dst_dir = "/data/renhaoye/MorCG/dataset/out_decals/scaled/"  # 目标存放文件夹
    src_dir = "/data/renhaoye/MorCG/dataset/out_decals/raw_fits/"  # 原始路径
    src_file = src_dir + files[i]  # 原始图片绝对路径
    ra_dec = files[i].split(".fits")[0]
    dst_file = dst_dir + ra_dec  # 保存绝对路径 不带扩展名
    # if not (os.path.exists(dst_file + ".fits"):
    normalized_unlock = normalization(load_img(src_file), shape="CHW", lock=False)  # 先做一次分通道归一化
    if not type(normalized_unlock) == int:
        normalized_lock = normalization(normalized_unlock, shape="CHW", lock=True)  # 再做一次全通道归一化
        if not type(normalized_lock) == int:
            scaled = auto_scale(normalized_lock)  # 再做stf
            save_fits(scaled, dst_file + '.fits')
        else:
            os.system("mv %s /data/renhaoye/MorCG/dataset/out_decals/raw_fits_deleted/" % src_file)
    else:
        os.system("mv %s /data/renhaoye/MorCG/dataset/out_decals/raw_fits_deleted/" % src_file)


if __name__ == "__main__":
    src = os.listdir("/data/renhaoye/MorCG/dataset/out_decals/raw_fits/")
    index = []
    for i in range(len(src)):
    # for i in range(10):
        index.append(i)
    p = multiprocessing.Pool(10)
    p.map(partial(scale_out_decals, files=src), index)
    p.close()
    p.join()
