import numpy as np
# from skimage.measure import compare_psnr
# from leo_utils import get_image_by_cv, save_image_by_cv2

def add_gaussian_noise(img_np, sigma):
    """ Add gaussian noise to clean image.
    :param img_np: numpy:float
    :param sigma:  if img_np belong to [0, 1]: sigma belong to [0, 1]
    :return:
    """
    noise = np.random.normal(loc=0, scale=sigma, size=img_np.shape)   # generate gaussian noise
    nim = img_np+noise

    # clip to normal range
    if np.max(img_np) <= 1:     # img belong to  [0, 1]
        return np.clip(nim, 0, 1)
    else:                       # img belong to [0, 255]
        return np.clip(nim, 0, 255)

def add_impulse_noise(oriImg, ND, NT=1):
    """  Add salt&pepper noise(NT=0) or Add random impulse noise(NT=1)
    :param oriImg: numpy float (belong to [0, 255])
    :param ND:   density of impulse noise  0~1
    :param NT:  type of impulse noise: 0 for salt&pepper and 1 for random impulse
    :return:   noisy image ,  mask(1 means noisy pixel 0 for free)
    """
    if np.max(oriImg) <= 1:
        oriImg *= 255.

    Nim = oriImg.copy()
    Narr = np.random.rand(Nim.shape[0], Nim.shape[1], Nim.shape[2])  # 生成0～1随机矩阵

    if NT == 0:   # for salt&peppers
        Nim[Narr < ND/2] = 0   # 一半0
        Nim[(Narr >= ND/2) & (Narr < ND)] = 255  # 一半255
    elif NT == 1:   # for random impulse
        N = Narr.copy()
        N[N >= ND] = 0  # 不是噪声的位置标记0

        N1 = N.copy()
        N1 = N1[N1 > 0]   # 所有噪声位置的值
        Imn = min(N1[:])
        Imx = max(N1[:])
        N = (((N - Imn) * (255 - 0)) / (Imx - Imn))
        Nim[Narr < ND] = N[Narr < ND]  # 加噪声
    else:
        raise TypeError('parameter 3  is wrong ')
    mask = Narr.copy()
    mask[Narr < ND] = 1
    mask[Narr >= ND] = 0
    # chen change ,default neednt clip and astype
    Nim /= 255.
    # mask = np.clip(mask, 0, 1)
    return Nim.astype('float32'), mask.astype('float32')

def add_poisson_noise(oriImg, PEAK):
    '''

    :param oriImg:  0~1
    :param PEAK:   0~255   lower means more noisy
    :return:
    '''
    image = oriImg.copy()
    noisy = np.random.poisson(image * PEAK) / PEAK    # noisy image
    return np.clip(noisy, 0., 1.)

def add_poisson_gauss_noise(oriImg, PEAK, sigma):
    '''

    :param oriImg: 0~1
    :param PEAK: 0~255
    :param sigma: 0~255
    :return:
    '''
    image = oriImg.copy()
    p_noisy = np.random.poisson(image * PEAK) / PEAK    # poisson noisy image
    noisy = p_noisy + np.random.normal(loc=0, scale=sigma/255., size=image.shape)

    return np.clip(noisy, 0., 1.)
# if __name__ == '__main__':
#     ori = get_image_by_cv(r'D:\pytorch_network\TrainData\PoissonNoise\BSD\peak=30\target\1.jpg', grayOrColor='color')
#     noisy = add_poisson_noise(ori, PEAK=30)
#     psnr = compare_psnr(im_true=ori, im_test=noisy)
#     save_image_by_cv2(image=noisy, path=r'C:\Users\ThinkStationP410\Desktop\ss\1.jpg')
#     noised = get_image_by_cv(r'C:\Users\ThinkStationP410\Desktop\ss\1.jpg', grayOrColor='color')
#     psnr_true = compare_psnr(im_true=ori, im_test=noised)
#
#     print('PYTHON---psnr=%.4f' % psnr)
#     print('MATLAB---psnr=%.4f' % psnr_true)