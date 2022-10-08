"""
after downloading the raw fits using urls created by MGS_urls.ipynb,
we need use this script to rename files
"""

from functools import partial
import os
import multiprocessing


def mv(i, row):
    src_dir = "/data/renhaoye/MorCG/dataset/out_decals/raw_fits/"  # 原始路径
    src_name = src_dir + row[i]  # 原始图片绝对路径
    if "fits-cutout" in row[i]:
        if os.path.getsize(src_name) == 792000:
            ra = row[i].split("ra=")[1].split("&dec=")[0]
            dec = row[i].split("&dec=")[1].split("&layer=")[0]
            dst_name = src_dir + ra + "_" + dec + ".fits"
            # print("mv \"%s\" \"%s\"" % (src_name, dst_name))
            os.system("mv \"%s\" \"%s\"" % (src_name, dst_name))
        else:
            os.system("rm -rf %s" % src_name)
    elif "fits" not in row[i]:
        # print("rm -rf %s" % src_name)
        os.system("rm -rf %s" % src_name)


if __name__ == "__main__":
    src_dir = "/data/renhaoye/MorCG/dataset/out_decals/raw_fits/"  # 原始路径
    src_files = os.listdir(src_dir)
    print("start")
    index = []
    # for i in range(2):
    for i in range(len(src_files)):
        index.append(i)
    p = multiprocessing.Pool(30)
    p.map(partial(mv, row=src_files), index)
    p.close()
    p.join()
