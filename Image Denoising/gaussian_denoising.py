import numpy as np
from PIL import Image


# Gaussian Blur for denoising

def gkern(l=5, sig=1.):
    kernel = np.zeros((l, l))
    r = l // 2
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            kernel[i + r, j + r] = (1 / (2 * np.pi * sig ** 2)) * np.exp(
                -(i ** 2 + j ** 2) / (2 * sig ** 2))
    return kernel


def denoise_gauss(image):
    # Convolve with Gaussian Kernel
    l = 5
    kernel = np.expand_dims(gkern(l=l, sig=1), axis=2)
    # pad image with 0s to not reduce width and height
    image = np.pad(
        image,
        pad_width=[(l // 2, l // 2), (l // 2, l // 2), (0, 0)],
        mode='constant'
    )
    sub_shape = tuple(np.subtract(image.shape, kernel.shape) + 1)
    # make an array of submatrices
    submatrices = np.lib.stride_tricks.as_strided(
        image, kernel.shape + sub_shape, image.strides * 2)
    # sum the submatrices and kernel
    denoised_image = np.einsum(
        'hij,hijklm->klm', kernel, submatrices).squeeze()
    if denoised_image.mean() > 1:
        denoised_image = np.uint8(denoised_image)
    return denoised_image


if __name__ == '__main__':
    im = np.asarray(Image.open('image_path').convert('RGB'))
    output = denoise_gauss(im)
