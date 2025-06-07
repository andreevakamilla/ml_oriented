import numpy as np
from scipy import fftpack
import cv2

def get_parameters(img):
    height = img.shape[0] // 3
    delta_width = int(img.shape[1] * 0.1)
    delta_height = int(height * 0.1)
    return height, delta_width, delta_height

def get_channels(img):
    height, delta_width, delta_height = get_parameters(img)
    
    blue = img[delta_height:height-delta_height, delta_width:-delta_width]
    green = img[height+delta_height:2*height-delta_height, delta_width:-delta_width]
    red = img[2*height+delta_height:3*height-delta_height, delta_width:-delta_width]
    
    return blue, green, red

def fourier_matching(base_img, img_to_align):
    f_base = fftpack.fft2(base_img)
    f_to_align = fftpack.fft2(img_to_align)
    
    cross_corr = fftpack.ifft2(f_base * np.conj(f_to_align)).real
    
    max_loc = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    shifts = np.array(max_loc)
    dims = np.array(base_img.shape)
    shifts = (shifts + dims // 2) % dims - dims // 2
    
    return shifts[0], shifts[1]

def roll_channels(img, blue_shift, red_shift):
    rolled = img.copy()
    
    rolled[:, :, 0] = np.roll(rolled[:, :, 0], blue_shift[0], axis=0)
    rolled[:, :, 0] = np.roll(rolled[:, :, 0], blue_shift[1], axis=1)
    
    rolled[:, :, 2] = np.roll(rolled[:, :, 2], red_shift[0], axis=0)
    rolled[:, :, 2] = np.roll(rolled[:, :, 2], red_shift[1], axis=1)
    
    return rolled

def align(img, green_coord):
    blue, green, red = get_channels(img)
    
    y_b, x_b = fourier_matching(green, blue)  
    y_r, x_r = fourier_matching(green, red)   
    
    img_stack = np.dstack((blue, green, red))
    
    aligned_img = roll_channels(img_stack, (y_b, x_b), (y_r, x_r))
    
    height, _, _ = get_parameters(img)
    b_row = green_coord[0] - height - y_b
    b_col = green_coord[1] - x_b
    r_row = green_coord[0] + height - y_r
    r_col = green_coord[1] - x_r
    
    return aligned_img[:, :, ::-1], (b_row, b_col), (r_row, r_col)