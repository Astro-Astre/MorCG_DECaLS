import numpy as np
from astropy.io import fits


def chw2hwc(img):
    ch1, ch2, ch3 = img[0], img[1], img[2]
    h, w = ch1.shape
    return np.concatenate((ch1.reshape(h, w, 1), ch2.reshape(h, w, 1), ch3.reshape(h, w, 1)), axis=2)


def hwc2chw(img):
    ch1, ch2, ch3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return np.array((ch1, ch2, ch3))


def save_fits(data: np.ndarray, filename: str):
    """
    save np.ndarray as fits
    :param data: img to be saved
    :param filename: save path
    :return: None
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


class AstroImg:
    def __init__(self, path):
        self.path = path
        self.img = self.load()

    def load(self):
        assert ".fits" or ".dat" in self.path, "only support .fits or .dat"
        if ".fits" in self.path:
            with fits.open(self.path) as hdul:
                return hdul[0].data


class OpticalImg:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.scaled = None
        self.c, self.h, self.w, self.i = self.get_shape()

    def get_shape(self):
        """
        get img's shape, only support image in CxHxW or HxWxC. HxW cannot reverse
        :return: c: channels, h: height, w: width
        """
        assert len(self.img.shape) in [2, 3], "only support image in 2 or 3 channels"
        if len(self.img.shape) == 3:
            assert (self.img.shape[0] < self.img.shape[1] and self.img.shape[0] < self.img.shape[2]) or \
                   (self.img.shape[2] < self.img.shape[1] and self.img.shape[2] < self.img.shape[
                       0]), "Pixels should > Channels, whatever CxHxW or HxWxC"
        shape = np.array(self.img.shape)
        temp = [0, 1, 2]
        c_poi = int(np.argmin(shape))
        assert c_poi == 0 or 2, "only support image in CxHxW or HxWxC"
        temp.remove(c_poi)
        if len(self.img.shape) == 3:
            if c_poi == 0:
                return self.img.shape[c_poi], self.img.shape[temp[0]], self.img.shape[temp[1]], "CHW"
            if c_poi == 2:
                return self.img.shape[c_poi], self.img.shape[temp[0]], self.img.shape[temp[1]], "HWC"
        if len(self.img.shape) == 2:
            return 1, self.img.shape[0], self.img.shape[1], "G"

    def compute(self, data: np.ndarray):
        data = data.reshape(-1)
        max, min = np.max(data), np.min(data)
        # noinspection PyBroadException
        try:
            temp = (data - min) / (max - min)
            return temp.reshape((self.h, self.w))
        except:
            return 65535

    def normalization(self, lock: bool):
        if lock:
            original_shape = self.img.shape
            flatten = self.img.reshape(-1)
            normalized = (flatten - flatten.min()) / (flatten.max() - flatten.min())
            return normalized.reshape(original_shape)
        else:
            if self.i == "CHW":
                c1, c2, c3 = self.img[0], self.img[1], self.img[2]
                norm_g, norm_r, norm_z = self.compute(c1), self.compute(c2), self.compute(c3)
                if type(norm_g) == int or type(norm_r) == int or type(norm_z) == int:
                    return 65535
                else:
                    return np.array((norm_g, norm_r, norm_z))
            elif self.i == "HWC":
                c1, c2, c3 = self.img[:, :, 0], self.img[:, :, 1], self.img[:, :, 2]
                norm_g, norm_r, norm_z = self.compute(c1), self.compute(c2), self.compute(c3)
                if type(norm_g) == int or type(norm_r) == int or type(norm_z) == int:
                    return 65535
                else:
                    return np.concatenate(
                        (norm_g.reshape(256, 256, 1), norm_r.reshape(256, 256, 1), norm_z.reshape(256, 256, 1)), axis=2)
            elif self.i == "G":
                return self.compute(self.img)
