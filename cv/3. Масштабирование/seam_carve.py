import numpy as np

def compute_energy(img):
    img = img.astype(np.float64)
    Y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    I_part = np.zeros((Y.shape[0], Y.shape[1], 2))
    I_part[:, 1:-1, 0] = (Y[:, 2:] - Y[:, :-2]) * 0.5
    I_part[:, 0, 0] = Y[:, 1] - Y[:, 0]
    I_part[:, -1, 0] = Y[:, -1] - Y[:, -2]
    I_part[1:-1, :, 1] = (Y[2:, :] - Y[:-2, :]) * 0.5
    I_part[0, :, 1] = Y[1, :] - Y[0, :]
    I_part[-1, :, 1] = Y[-1, :] - Y[-2, :]
    return np.round(np.sqrt(I_part[:,:,0]**2 + I_part[:,:,1]**2), 6)


def compute_seam_matrix(energy, mode):
    height, width = energy.shape
    seam_matrix = np.zeros_like(energy, dtype=np.float64)
    if mode == 'horizontal':
        seam_matrix[0, :] = energy[0, :]

        for i in range(1, height):
            for j in range(width):
                if j == 0:
                    min_prev = min(seam_matrix[i-1, j], seam_matrix[i-1, j+1])
                elif j == width - 1:
                    min_prev = min(seam_matrix[i-1, j-1], seam_matrix[i-1, j])
                else:
                    min_prev = min(seam_matrix[i-1, j-1], seam_matrix[i-1, j], seam_matrix[i-1, j+1])

                seam_matrix[i, j] = energy[i, j] + min_prev

    elif mode == 'vertical':
        seam_matrix = compute_seam_matrix(energy.T, 'horizontal').T
    
    return seam_matrix

def remove_minimal_seam(image, seam_matrix, mode):
    height, width = image.shape
    if mode == 'horizontal_shrift':
        



