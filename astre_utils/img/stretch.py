from prepare import *


def get_midtones(median):
    return (7 * median) / (6 * median + 1)


def mtf(data: np.ndarray, m: float):
    """
    no-linear transformation method
    :param data: input data
    :param m: midtones value
    :return: output
    """
    return ((m - 1) * data) / ((2 * m - 1) * data - m)


def stretch(optical_img: OpticalImg):
    optical_img.img = optical_img.normalization(lock=False)  # 先做一次分通道归一化
    if not type(optical_img.img) == int:
        optical_img.img = optical_img.normalization(lock=True)  # 再做一次全通道归一化
        if not type(optical_img.img) == int:
            return auto_scale(optical_img)  # 再做拉伸


def auto_scale(optical_img: OpticalImg):
    if optical_img.c == 1:
        optical_img.scaled = c_cut(optical_img.img, optical_img.h, optical_img.w)
        optical_img.scaled[optical_img.scaled < 0] = 0
        optical_img.scaled[optical_img.scaled > 1] = 1.
        return optical_img
    if optical_img.c == 3:
        c1, c2, c3 = c_cut(optical_img.img[0], optical_img.h, optical_img.w), \
                     c_cut(optical_img.img[1], optical_img.h, optical_img.w), \
                     c_cut(optical_img.img[2], optical_img.h, optical_img.w)
        optical_img.scaled = np.array((c1, c2, c3))
        print(optical_img.scaled)
        optical_img.scaled[optical_img.scaled < 0] = 0
        optical_img.scaled[optical_img.scaled > 1] = 1.
        return optical_img


def c_cut(gray, h, w):
    highlight = 1.
    hist, bar = np.histogram(gray.reshape(-1), bins=65536)
    cdf = hist.cumsum()
    shadow_index = np.argwhere(cdf > 0.001 * gray.reshape(-1).shape[0])[0]
    shadow = bar[shadow_index]
    gray[gray < shadow] = shadow
    gray[gray > highlight] = 1.
    gray = gray.reshape(-1)
    temp = (gray - gray.min()) / (gray.max() - gray.min())
    gray = temp.reshape((h, w))
    right_midtones = get_midtones(np.median(gray))
    return mtf(gray, right_midtones)