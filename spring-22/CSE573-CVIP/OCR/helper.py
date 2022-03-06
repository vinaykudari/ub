import sys
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.setrecursionlimit(1500000000)


def threshold(image, val=0.0, reverse=False):
    h, w, ch = image.shape
    if ch > 1:
        image = np.mean(image, axis=2)

    if reverse:
        background = image[:, :] > val
    else:
        background = image[:, :] < val

    image[~background] = 255
    image[background] = 0

    return image

def gaussian_kernel(size, sigma):
    kernel_1d = np.linspace(- (size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(kernel_1d) / np.square(sigma))
    kernel_2d = np.outer(gauss, gauss)
    return kernel_2d / np.sum(kernel_2d)

def convolve(image, kernel, stride, padding=0):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    h = ((img_h - k_h + (2 * padding)) // stride) + 1
    w = ((img_w - k_w + (2 * padding)) // stride) + 1

    kernel = np.flipud(np.fliplr(kernel))
    output = []
    image = np.pad(image, pad_width=padding)

    for i in range(0, img_h-k_h+1, stride):
        for j in range(0, img_w-k_w+1, stride):
            region = image[i:i+k_h, j:j+k_w]
            output.append(np.multiply(region, kernel).sum())

    output = np.asarray(output).reshape(h, w)
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

def expand(image, factor=2):
    "Nearest neighbour interpolation"
    img_h, img_w = image.shape
    h, w = img_h * factor, img_w * factor
    output = []

    for i in range(img_h):
        for j in range(img_w):
            for _ in range(factor):
                output.append(image[i][j])
        for _ in range(factor-1):
            output += output[-img_w*factor:]

    output = np.asarray(output).reshape(h, w)
    return output


def extract_features(image, detector):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, desc = detector.detectAndCompute(image, None)

    return keypoints, desc

def norm_l2(a, b):
    return np.linalg.norm(np.asarray(a)-np.asarray(b))


def get_component(components, idx, img, show=False):
    component = components[idx]
    left, right = component['left'], component['right']
    top, bottom = component['top'], component['bottom']

    if show:
        _ = plt.imshow(img[left:right + 1, top:bottom + 1], cmap='gray')

    return img[left:right + 1, top:bottom + 1]

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

def connected_components(arr, foreground, background, n_dir=8):
    arr = arr.copy()
    height, width = len(arr), len(arr[0])
    components = defaultdict(dict)
    visited = set()

    def dfs(i, j, p):
        nonlocal height, width, arr, visited
        if i < 0 or i >= height or j < 0 or j >= width or arr[i][j] == -1 or (i, j) in visited:
            return

        visited.add((i, j))
        arr[i][j] = p
        neighbour_idx = get_neighbours(n_dir)
        for x, y in neighbour_idx:
            dfs(i + x, j + y, p)

    p = 1

    for i in range(height):
        for j in range(width):
            if arr[i][j] == background:
                arr[i][j] = -1

    for i in range(height):
        for j in range(width):
            if arr[i][j] == foreground:
                dfs(i, j, p)
                p += 1

    for i in range(height):
        for j in range(width):
            if arr[i][j] > 0:
                if i < components[arr[i][j]].get('left', float('inf')):
                    components[arr[i][j]]['left'] = i
                if i > components[arr[i][j]].get('right', float('-inf')):
                    components[arr[i][j]]['right'] = i

                if j < components[arr[i][j]].get('top', float('inf')):
                    components[arr[i][j]]['top'] = j
                if j > components[arr[i][j]].get('bottom', float('-inf')):
                    components[arr[i][j]]['bottom'] = j

    return components


def matcher(s_desc, t_desc, measure, reverse=False):
    arr = []

    for s_idx, s_d in enumerate(s_desc):
        min_score = float('inf')
        min_idx = 0
        for t_idx, t_d in enumerate(t_desc):
            score = measure(s_d, t_d)
            if score < min_score:
                min_score = score
                min_idx = t_idx

        arr.append(Matcher(s_idx, min_idx, round(min_score, 2)))

    return sorted(arr, reverse=reverse, key=lambda x: x.score)