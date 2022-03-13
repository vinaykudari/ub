import cv2
import numpy as np
import heapq


def extract_features(image, extractor):
    keypoints, desc = extractor.detectAndCompute(image, None)
    return keypoints, desc


def norm(a, b, n):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)

    if not isinstance(b, np.ndarray):
        b = np.asarray(b)

    res = np.sum(np.abs(a - b) ** n, axis=-1) ** (1. / n)
    return res


def hist(image):
    res = {i: 0 for i in range(256)}
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            res[image[i][j]] += 1

    return list(res.values()), list(res.keys())


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


def convolve_2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    h, w = image.shape

    padded = np.zeros((h + 2, w + 2))
    padded[1:-1, 1:-1] = image

    for i in range(w):
        for j in range(h):
            output[j, i] = (kernel * padded[j: j + 3, i: i + 3]).sum()

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

            if (min_1 / min_2) < thresh:
                arr.append(round(min_1, 2))
            else:
                if heap:
                    arr.append(round(sum(heap)/len(heap), 2))

    return sorted(arr, reverse=reverse)


def resize(image, factor=2):
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
