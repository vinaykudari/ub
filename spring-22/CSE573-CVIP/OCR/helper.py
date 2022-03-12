import cv2
import numpy as np
import heapq
from scipy.spatial.distance import correlation
from scipy.stats import *


def extract_features(image, extractor):
    keypoints, desc = extractor.detectAndCompute(image, None)
    return keypoints, desc


def n_cross_corr(a, b):
    return correlation(a, b)


def norm_l2(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))


def norm_l1(a, b):
    return np.linalg.norm((np.asarray(a) - np.asarray(b)), ord=1)


def earth_mover_distance(a, b):
    return wasserstein_distance(a, b)


def measure(a, b):
    pass


def threshold(image, val=0.0, reverse=False):
    if reverse:
        background = image[:, :] > val
    else:
        background = image[:, :] < val

    image[~background] = 255
    image[background] = 0

    return image


def otsu(image, *args, **kwargs):
    pixel_number = image.shape[0] * image.shape[1]
    mean_weight = 1.0 / pixel_number
    his, bins = np.histogram(image, np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)

    for t in bins[1:-1]:
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)

        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value

    output = image.copy()
    output[image > final_thresh] = 255
    output[image < final_thresh] = 0

    return output


def gaussian_pyramid(image, n, kernel_len=5, sigma=1):
    image = image.copy()
    res = []
    for i in range(n):
        image = convolve(
            image=image,
            kernel=gaussian_kernel(
                size=kernel_len,
                sigma=sigma,
            ),
            stride=2,
        )
        res.append(image)

    return res


def convolve(image, kernel, stride, padding=0):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    h = ((img_h - k_h + (2 * padding)) // stride) + 1
    w = ((img_w - k_w + (2 * padding)) // stride) + 1

    kernel = np.flipud(np.fliplr(kernel))
    output = []
    image = np.pad(image, pad_width=padding)

    for i in range(0, img_h - k_h + 1, stride):
        for j in range(0, img_w - k_w + 1, stride):
            region = image[i:i + k_h, j:j + k_w]
            output.append(np.multiply(region, kernel).sum())

    output = np.asarray(output).reshape(h, w)
    return output


def gaussian_kernel(size, sigma):
    kernel_1d = np.linspace(- (size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(kernel_1d) / np.square(sigma))
    kernel_2d = np.outer(gauss, gauss)
    return kernel_2d / np.sum(kernel_2d)


def matcher(s_desc, t_desc, measure, thresh=0.8, reverse=False):
    arr = []

    for s_idx, s_d in enumerate(s_desc):
        heap = []
        for t_idx, t_d in enumerate(t_desc):
            score = measure(s_d, t_d)
            heapq.heappush(heap, score)

        if len(heap) >= 2:
            min_1 = heapq.heappop(heap)
            min_2 = heapq.heappop(heap)

            print(round(min_1 / min_2, 2), end=' | ')

            if (min_1 / min_2) <= thresh:
                arr.append(round(min_1, 2))

    return sorted(arr, reverse=reverse)


def expand(image, factor=2):
    """Nearest neighbour interpolation"""
    img_h, img_w = image.shape
    h, w = img_h * factor, img_w * factor
    output = []

    for i in range(img_h):
        for j in range(img_w):
            for _ in range(factor):
                output.append(image[i][j])
        for _ in range(factor - 1):
            output += output[-img_w * factor:]

    output = np.asarray(output).reshape(h, w)
    return output


def get_neighbours(k):
    neighbours = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
    ]
    if k == 8:
        diag_neighbours = [
            [-1, -1],
            [1, 1],
            [-1, 1],
            [1, -1]
        ]
        neighbours += diag_neighbours

    return neighbours


def connected_components(arr, p, components, h, foreground, n_dir=8):
    arr = arr.copy()
    height, width = len(arr), len(arr[0])
    visited = set()
    neighbour_idx = get_neighbours(n_dir)

    def dfs(i, j, p):
        nonlocal height, width, arr, visited, components
        if i < 0 or i >= height or j < 0 or j >= width or arr[i][j] == -1 or (i, j) in visited:
            return

        visited.add((i, j))
        arr[i][j] = p
        components[p]['left'] = min(components[p].get('left', float('inf')), i + h)
        components[p]['right'] = max(components[p].get('right', float('-inf')), i + h)
        components[p]['top'] = min(components[p].get('top', float('inf')), j)
        components[p]['bottom'] = max(components[p].get('bottom', float('-inf')), j)

        for x, y in neighbour_idx:
            dfs(i + x, j + y, p)

    for j in range(width):
        for i in range(height):
            if arr[i][j] == foreground and (i, j) not in visited:
                dfs(i, j, p)
                p += 1

    return p
