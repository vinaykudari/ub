import heapq

import cv2
import numpy as np


def extract_features(image, extractor):
    keypoints, desc = extractor.detectAndCompute(image, None)
    return keypoints, desc


def norm(a, b, n):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)

    if not isinstance(b, np.ndarray):
        b = np.asarray(b)

    res = np.sum(np.abs(a - b)**n, axis=-1)**(1./n)
    return res


def crop(image):
    h, w = image.shape
    top, bottom, left, right = 0, h, 0, w

    for left in range(w):
        if (image[:, left] == 0).any():
            break

    for right in range(w - 1, -1, -1):
        if (image[:, right] == 0).any():
            break

    for top in range(h):
        if (image[top, :] == 0).any():
            break

    for bottom in range(h - 1, -1, -1):
        if (image[bottom, :] == 0).any():
            break

    return image[top:bottom + 1, left:right + 1]


def otsu(image, *args, **kwargs):
    total_pixels = image.shape[0] * image.shape[1]
    mean_weight = 1.0 / total_pixels
    his, bins = np.histogram(image, np.arange(0, 257))
    # his, bins = hist(image=image)
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)

    for thresh in bins[1:-1]:
        left = np.sum(his[:thresh])
        right = np.sum(his[thresh:])
        w_left = left * mean_weight
        w_right = right * mean_weight

        mub = np.sum(intensity_arr[:thresh] * his[:thresh]) / float(left)
        muf = np.sum(intensity_arr[thresh:] * his[thresh:]) / float(right)

        value = w_left * w_right * (mub - muf) ** 2

        if value > final_value:
            final_thresh = thresh
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
            else:
                if heap:
                    arr.append(round(sum(heap)/len(heap), 2))

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


def resize(image, scale):
    h, w = image.shape
    output = cv2.resize(image, (int(w*scale), int(h*scale)))
    return output


def resize_to(image, new_w, new_h):
    output = cv2.resize(image, (new_w, new_h))
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
        neighbour_idx = get_neighbours(n_dir)

        for x, y in neighbour_idx:
            dfs(i + x, j + y, p)

    for j in range(width):
        for i in range(height):
            if arr[i][j] == foreground and (i, j) not in visited:
                dfs(i, j, p)
                p += 1

    return p
