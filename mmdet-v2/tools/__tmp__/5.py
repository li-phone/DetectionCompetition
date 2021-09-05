# coding=utf-8
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import math


def guidedfilter(I, p, r, eps):
    '''
    引导滤波
    '''
    height, width = I.shape
    m_I = cv.boxFilter(I, -1, (r, r))
    m_p = cv.boxFilter(p, -1, (r, r))
    m_Ip = cv.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv.boxFilter(a, -1, (r, r))
    m_b = cv.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def get_dc_A(I, r, eps, w, maxGray):
    '''
    计算暗通道图dc和光照值A：V1 = 1-t/A
    INPUT -> 归一化图像数组, 滤波器半径, 噪声, 灰度范围, maxGray
    '''
    # 分离三通道
    B, G, R = cv.split(I)
    # 求出每个像素RGB分量中的最小值, 得到暗通道图
    dc = cv.min(cv.min(R, G), B)
    # 使用引导滤波优化
    dc = guidedfilter(dc, cv.erode(dc, np.ones((2 * r + 1, 2 * r + 1))), r, eps)
    bins = 2000
    # 计算大气光照A
    ht = np.histogram(dc, bins)
    d = np.cumsum(ht[0]) / float(dc.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(I, 2)[dc >= ht[1][lmax]].max()
    # 对值范围进行限制
    dc = np.minimum(dc * w, maxGray)
    return dc, A


if __name__ == '__main__':
    fname = '../c000111.jpg'
    src = np.array(Image.open(fname))
    I = src.astype('float64') / 255

    # 得到遮罩图像和大气光照
    dc, A = get_dc_A(I, 111, 0.001, 1.0, 0.80)
    # 调用雾图模型
    J = np.zeros(I.shape)
    for k in range(3):
        J[:, :, k] = (I[:, :, k] - dc) / (1 - dc / A)
    J = np.clip(J, 0, 1)
    # 伽马校正
    J = J ** (np.log(0.5) / np.log(J.mean()))
    # 拼接结果
    # output = np.hstack((I, J))
    output = J
    output = np.uint8(output * 255)
    plt.imshow(Image.fromarray(output))
    cv.imwrite(fname + '.jpg', output[:, :, ::-1])
    plt.show()
