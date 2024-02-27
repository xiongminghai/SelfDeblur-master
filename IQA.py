import numpy as np
import pywt 
from scipy.ndimage import convolve

def getCDF97(weight = 1):
    # https://github.com/PyWavelets/pywt/issues/176
    analysis_LP = np.array([0, 0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,\
                            0.602949018236, 0.266864118443, -0.078223266529, -0.016864118443, 0.026748757411])
    analysis_LP *= weight
    
    analysis_HP = np.array([0, 0.091271763114, -0.057543526229, -0.591271763114, 1.11508705,\
                           -0.591271763114, -0.057543526229, 0.091271763114, 0, 0])
    analysis_HP *= weight
    
    synthesis_LP = np.array([0, -0.091271763114, -0.057543526229, 0.591271763114, 1.11508705,\
                             0.591271763114, -0.057543526229, -0.091271763114, 0, 0])
    synthesis_LP *= weight

    synthesis_HP = np.array([0, 0.026748757411, 0.016864118443, -0.078223266529, -0.266864118443,\
                             0.602949018236, -0.266864118443, -0.078223266529, 0.016864118443, 0.026748757411])
    synthesis_HP *= weight
    return pywt.Wavelet('CDF97', [analysis_LP, analysis_HP, synthesis_LP, synthesis_HP])


def sbEn(coeffs: np.array) -> float:
    '''
    Description:
        This function calculates the energy in the sub-band corresponding
        to the provided wavelet coefficients
    Args:
        coeffs, wavelet coefficients of sub-band
    '''
    I,J = coeffs.shape
    return np.log(1 + np.sum(np.square(coeffs))/(I*J))

def En(lvlCoeffs: tuple, alpha = 0.8) -> float:
    '''
    Description:
        This function implements the weighted energy calculation for
        wavelet decomposition level n
    Args:
        lvlCoeffs, Tuple of wavelet coefficients for decomposition level of form (LHn, HLn, HHn)
        alpha, Relative significance parameter, recommended value of 0.8 prioritizes HH sub-band
               as recommended in ISBN: 0139353224, cited in OMQDI paper
    '''
    LHn, HLn, HHn = lvlCoeffs
    return (1 - alpha)*(sbEn(LHn) + sbEn(HLn)) / 2 + alpha*sbEn(HHn)

def S(decompCoeffs: list, alpha = 0.8) -> float:
    '''
    Description:
        This function calculates the cumulative energy of each level of decomposition
    Args:
        decompCoeffs, List of wavelet coefficients for all decomposition levels
                      takes the form: [(LHn, HLn, HHn), ...(LH1, HL1, HH1)]
        alpha, Relative significance parameter, recommended value of 0.8 prioritizes HH sub-band
               as recommended in ISBN: 0139353224, cited in OMQDI paper
    '''
    energy = 0
    for i, lvlCoeffs in enumerate(decompCoeffs):
        n = i+1
        energy += 2**(3-n)*En(lvlCoeffs, alpha)
    return energy

def local_mean(img: np.array, window = 3, pad_mode = 'reflect') -> np.array:
    '''
    Description:
        This function calculates the local mean of grey levels in image
    Args:
        img, Input image (M, N)
        window, Size of window for local mean calculation
        pad_mode, Padding mode for mean convolution
    '''
    return convolve(img, np.full((window, window), 1/window**2), mode = pad_mode)

def local_variance(img: np.array) -> np.array:
    '''
    Description:
        This function calculates the local varaince of an image as 
        formulated in the original paper
    Args:
        img, Input image (M, N)
    '''
    mu_sq = np.square(local_mean(img))
    return local_mean(np.square(img)) - mu_sq

def noise_power(img: np.array) -> float:
    '''
    Description:
        This function estimates the noise power (sigma hat) of an image
        Noise power is the mean of the local variance in the image
    Args:
        img, input image of dims (M, N)
    '''
    return np.mean(local_variance(img))
    

def OMQDI(X: np.array, Y: np.array, C = 1e-10) -> tuple:
    '''
    Description:
        This function implements the Objective Measure of Quality of
        Denoised Images | DOI: 10.1016/j.bspc.2021.102962
    Args:
        X, noisy input image, single channel of shape (M, N)
        Y, denoised output image, single channel of shape (M, N)
        C, arbitrary constant with negligibly small value
    Returns: 
        (OMQDI: float [1,2], Q1: float [0,1], Q2: float [0,1]) 
    Note:
        OMQDI, Combined metric Q1+Q2, ideal value is 2
        Q1, Edge-Preservation Factor, ideal value is 1
        Q2, Noise-Suppression Factor, ideal value is 1 
    '''
    
    # get CDF97 wavelet
    CDF97 = getCDF97()

    # compute 2D wavelet decomposition of X,Y with decomp level of 3
    # coeffs of form: LL, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1)
    coeffX = pywt.wavedec2(X, CDF97, level = 3)
    coeffY = pywt.wavedec2(Y, CDF97, level = 3)
    
    # compute edge sharpness of noisy and denoised images: S(X), S(Y)
    SX = S(coeffX[1:])
    SY = S(coeffY[1:])

    # compute noise power of noisy and denoised images: sigma^hat_n(X), sigma^hat_n(Y)
    npX = noise_power(X)
    npY = noise_power(Y)

    # compute Q1 (Edge-Preservation Factor) and Q2 (Noise Suppression Factor)
    Q1 = (2 * SX * SY + C) / (SX**2 + SY**2 + C)
    Q2 = (npX - npY)**2 / (npX**2 + npY**2 + C)
    return (Q1 + Q2, Q1, Q2)


def omqdi_q1(X: np.array, Y: np.array, C=1e-10) -> tuple:
    '''
    Description:
        This function implements the Objective Measure of Quality of
        Denoised Images | DOI: 10.1016/j.bspc.2021.102962
    Args:
        X, noisy input image, single channel of shape (M, N)
        Y, denoised output image, single channel of shape (M, N)
        C, arbitrary constant with negligibly small value
    Returns:
        (OMQDI: float [1,2], Q1: float [0,1], Q2: float [0,1])
    Note:
        OMQDI, Combined metric Q1+Q2, ideal value is 2
        Q1, Edge-Preservation Factor, ideal value is 1
        Q2, Noise-Suppression Factor, ideal value is 1
    '''

    # get CDF97 wavelet
    CDF97 = getCDF97()

    # compute 2D wavelet decomposition of X,Y with decomp level of 3
    # coeffs of form: LL, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1)
    coeffX = pywt.wavedec2(X, CDF97, level=3)
    coeffY = pywt.wavedec2(Y, CDF97, level=3)

    # compute edge sharpness of noisy and denoised images: S(X), S(Y)
    SX = S(coeffX[1:])
    SY = S(coeffY[1:])

    # compute Q1 (Edge-Preservation Factor)
    Q1 = (2 * SX * SY + C) / (SX ** 2 + SY ** 2 + C)
    return Q1,SX,SY


if __name__ == '__main__':
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import scipy.ndimage as scim
    im = mpimg.imread('Head-MRI.jpg')/255
    im_noise = im + 0.5*np.random.normal(loc=0, scale=.2, size=im.shape)
    im_denoise = scim.gaussian_filter(im_noise, 3)
    im_med = scim.median_filter(im_noise, 5)
    BO, BQ1, BQ2 = OMQDI(im_noise,im_denoise)
    OO, OQ1, OQ2 = OMQDI(im_noise,im_med)
    UO, UQ1, UQ2 = OMQDI(im_noise,im)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(im_noise)
    ax1.set_title("Noisy")
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(im_med)
    ax2.set_title("5x5 Median Filter")
    ax2.set_xlabel(f'OMQDI: {round(OO,3)}, EPF: {round(OQ1,3)}, NSF: {round(OQ2,3)}')
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(im_denoise)
    ax3.set_title("Gaussian Blur std = 3")
    ax3.set_xlabel(f'OMQDI: {round(BO,3)}, EPF: {round(BQ1,3)}, NSF: {round(BQ2,3)}')
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(im)
    ax4.set_title("Original Reference")
    ax4.set_xlabel(f'OMQDI: {round(UO,3)}, EPF: {round(UQ1,3)}, NSF: {round(UQ2,3)}')
    plt.show()
