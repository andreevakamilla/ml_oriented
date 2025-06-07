import numpy as np
from scipy.signal import convolve2d


def get_bayer_masks(n_rows, n_cols):
    red = np.zeros((n_rows, n_cols), dtype=bool)
    green = np.zeros((n_rows, n_cols), dtype=bool)
    blue = np.zeros((n_rows, n_cols), dtype=bool)

    for i in range(n_rows):
        for j in range(n_cols):
            if i % 2 == 0 and j % 2 == 1:
                red[i, j] = True
            if i % 2 == j % 2:
                green[i, j] = True
            if i % 2 == 1 and j % 2 == 0:
                blue[i, j] = True
    
    return np.dstack((red, green , blue))


def get_colored_img(raw_img):
    return raw_img[..., np.newaxis] * get_bayer_masks(*raw_img.shape)


def bilinear_interpolation(colored_img):
    mask = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    inter_img = np.zeros_like(colored_img, dtype=float)
    
    kernel_g = np.array([[0.0, 0.25, 0.0],
                         [0.25, 0.0, 0.25],
                         [0.0, 0.25, 0.0]])
    
    kernel = np.array([[0.25, 0.5, 0.25],
                       [0.5, 0.0, 0.5],
                       [0.25, 0.5, 0.25]])
    for i, kernel in zip(range(3), [kernel, kernel_g, kernel]):
        orig = colored_img[..., i]
        mask_i = mask[..., i]
        interpol = convolve2d(orig, kernel, mode='same', boundary='symm')
        inter_img[..., i] = np.where(mask_i, orig, interpol)
    
    return np.clip(inter_img, 0, 255).astype(np.uint8)


def improved_interpolation(raw_img):
    raw_img = raw_img.astype(np.uint64)
    rows, cols = raw_img.shape
    result = np.zeros((rows, cols, 3), dtype=np.uint64)

    mask_r = np.zeros_like(raw_img);      mask_r[0::2, 1::2] = 1
    mask_gr = np.zeros_like(raw_img);     mask_gr[1::2, 1::2] = 1
    mask_gb = np.zeros_like(raw_img);     mask_gb[0::2, 0::2] = 1
    mask_b = np.zeros_like(raw_img);      mask_b[1::2, 0::2] = 1

    kr_r = np.array([[0,0,-3,0,0],[0,4,0,4,0],[-3,0,12,0,-3],[0,4,0,4,0],[0,0,-3,0,0]])
    kr_gb = np.array([[0,0,1,0,0],[0,-2,0,-2,0],[-2,8,10,8,-2],[0,-2,0,-2,0],[0,0,1,0,0]])
    kr_gr = np.array([[0,0,-2,0,0],[0,-2,8,-2,0],[1,0,10,0,1],[0,-2,8,-2,0],[0,0,-2,0,0]])
    kg = np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]])

    def interp(base_mask, other_masks, kernels, scales):
        img = raw_img.astype(np.float64)
        out = img * base_mask
        for mask, k, s in zip(other_masks, kernels, scales):
            if k is not None: 
                conv = convolve2d(img / s, k, mode='same', boundary='symm')
                out[mask == 1] = conv[mask == 1]
            else:
                out[mask == 1] = img[mask == 1]  
        return np.clip(out, 0, 255)


    result[..., 0] = interp(mask_r, [mask_gb, mask_gr, mask_b], [kr_gb, kr_gr, kr_r], [16]*3)
    result[..., 1] = interp(mask_gr, [mask_gb, mask_r, mask_b], [None, kg, kg], [1, 8, 8])
    result[..., 2] = interp(mask_b, [mask_gr, mask_gb, mask_r], [kr_gb, kr_gr, kr_r], [16]*3)


    return result.astype(np.uint8)


def compute_psnr(img_pred, img_gt):
    mse = np.sum((np.float64(img_pred) - np.float64(img_gt))**2) / np.prod(img_pred.shape)
    if mse == 0:
        raise ValueError("MSE Cringe...")
    return 10 * np.log10((np.max(np.float64(img_gt))**2) / mse)