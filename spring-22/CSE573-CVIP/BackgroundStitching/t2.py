# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades
from collections import defaultdict

import cv2
import numpy as np
import json
import math


def resize(img, factor):
    h, w, a = shape(img)
    op = cv2.resize(img, (int(w*factor), int(h*factor)))
    return op


def shape(img):
    s = img.shape
    a = 0
    if len(s) == 3:
        h, w, a = img.shape
    else:
        h, w = img.shape

    return h, w, a


def coordinates(img):
    size = img.shape
    return 0, 0, size[1], size[0]


def translate_homography(dx, dy, h):
    dx, dy = abs(dx) if dx < 0 else 0, abs(dy) if dy < 0 else 0
    t = np.float32(
        [
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1],
        ]
    )
    new_h = t.dot(h)
    return new_h, (dx, dy)


def extract_features(image, extractor):
    kp, desc = extractor.detectAndCompute(image, None)
    return kp, desc


def pairwise(x, y):
    dists = -2 * np.dot(x, y.T) + np.sum(
        np.square(y), axis=1, keepdims=True,
    ).T + np.sum(np.square(x), axis=1, keepdims=True)
    return np.sqrt(dists)


def transform(c, homography):
    x, y, w, h = c
    src = np.float32([[x, y], [x, y + h - 1], [x + w - 1, y], [x + w - 1, y + h - 1]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(src, homography)
    l, _, _ = dst.shape
    dst = dst.reshape((l, 2))

    min_x, max_x = np.min(dst[:, 0]), np.max(dst[:, 0])
    min_y, max_y = np.min(dst[:, 1]), np.max(dst[:, 1])

    new_x = int(np.floor(min_x))
    new_y = int(np.floor(min_y))
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    return new_x, new_y, new_w, new_h


def match_desc(desc1, desc2, lowes_ratio=0.8):
    pair_dist = pairwise(desc1, desc2)
    min_pairs = np.partition(pair_dist, 1)[:, :2]
    idx = (min_pairs[:, 0] / min_pairs[:, 1]) < lowes_ratio
    n_matches = len(min_pairs[idx])
    matches = []

    for q_idx, t_idx in zip(np.nonzero(idx)[0], np.argmin(pair_dist[idx], 1)):
        matches.append(
            cv2.DMatch(
                _queryIdx=q_idx,
                _trainIdx=t_idx,
                _distance=pair_dist[q_idx, t_idx],
            )
        )

    return matches, n_matches


def get_matching_points(matches, kp1, kp2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    return src_pts, dst_pts


def homography(match_pts, cv=True):
    src_pts, dst_pts = match_pts
    k = cv2.RANSAC if cv is True else 0
    h, mask = cv2.findHomography(src_pts, dst_pts, k)

    return h, mask


def ransac(match_pts, thresh, factor, p=4, n=10000, prob=0.99):
    src_pts, dst_pts = match_pts
    k = len(src_pts)

    best_inliers = float('-inf')
    mask = []
    src_pts_f = []
    dest_pts_f = []

    for i in range(n):
        rand_idx = np.random.choice(k, p)
        src_pts_s = src_pts[rand_idx]
        dst_pts_s = dst_pts[rand_idx]

        h, _ = cv2.findHomography(src_pts_s.reshape(-1, 1, 2), dst_pts_s.reshape(-1, 1, 2))
        dst_pts_p = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), h).reshape(-1, 2)

        errors = np.linalg.norm(dst_pts_p - dst_pts, axis=1)
        errors = errors / factor
        inliers = errors < thresh
        n_inliers = inliers.sum()

        if n_inliers > best_inliers:
            min_error = errors.mean()
            print(i, min_error, n_inliers, n_inliers / k)
            best_inliers = n_inliers
            src_pts_f = src_pts[inliers]
            dest_pts_f = dst_pts[inliers]
            mask = inliers

        if n_inliers / k > prob:
            break

    return src_pts_f, dest_pts_f, mask.astype(np.uint8)


def get_H(img1_c, img2_c, thresh=0.015, cv=True):
    sift = cv2.SIFT_create()
    img1 = cv2.cvtColor(img1_c, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_c, cv2.COLOR_BGR2GRAY)

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    kp1, desc1 = extract_features(img1, sift)
    kp2, desc2 = extract_features(img2, sift)

    matches, n_matches = match_desc(desc1, desc2, lowes_ratio=0.8)
    src_pts, dst_pts = get_matching_points(matches, kp1, kp2)

    if cv is False:
        k = math.sqrt((h1 + h2) ** 2 + (w1 + w2) ** 2)
        src_pts, dst_pts, mask = ransac((src_pts, dst_pts), thresh=thresh, factor=k)

    h, _ = homography((src_pts, dst_pts), cv=cv)

    return h


def warp(img_c, H):
    _, _, a = shape(img_c)
    dx, dy, _, _ = transform(coordinates(img_c), H)
    new_H, (tx, ty) = translate_homography(dx, dy, H)
    x, y, w, h = transform(coordinates(img_c), new_H)
    warped = cv2.warpPerspective(img_c, new_H, (x + w, y + h))
    return warped, (tx, ty)


def pad_img(img, c):
    top, bottom, left, right = 0, 0, 0, 0
    x, y, w, h = c

    if y < 0:
        top = -y
    if x < 0:
        left = -x

    if y + h > img.shape[0]:
        bottom = y + h - img.shape[0]
    if x + w > img.shape[1]:
        right = x + w - img.shape[1]

    print(top, bottom, left, right)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )
    return img, (left, top)


def blend(img1, img2):
    assert (img1.shape == img2.shape)
    locs1 = np.where(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) != 0)
    blended1 = np.copy(img2)
    blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]
    locs2 = np.where(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) != 0)
    blended2 = np.copy(img1)
    blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
    blended = cv2.addWeighted(blended1, 0.5, blended2, 0.5, 0)
    return blended


def merge(img1_c, img2_c, h):
    result, pos = warp(img1_c, h)
    x_pos, y_pos = pos
    rect = x_pos, y_pos, img2_c.shape[1], img2_c.shape[0]
    result, _ = pad_img(result, rect)
    idx = np.s_[y_pos: y_pos + img2_c.shape[0], x_pos: x_pos + img2_c.shape[1]]
    result[idx] = blend(result[idx], img2_c)
    x, y, w, h = cv2.boundingRect(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY))
    result = result[y: y + h, x: x + w]

    return result, (x_pos - x, y_pos - y)


def extract_all_features(imgs):
    sift = cv2.SIFT_create()
    feature_desc = np.array(
        list(map(lambda x: extract_features(x, sift), imgs)),
        dtype=object,
    )[:, 1]

    return feature_desc


def get_all_matches(imgs, feature_desc, thresh=200):
    n = len(imgs)
    all_matches = defaultdict(lambda: np.zeros(n))
    best_n = float('-inf')

    for idx, img1 in enumerate(imgs):
        desc1 = feature_desc[idx]

        for jdx, img2 in enumerate(imgs):
            if idx == jdx:
                continue

            desc2 = feature_desc[jdx]
            _, n_matches = match_desc(desc1, desc2)
            all_matches[idx][jdx] = n_matches

        temp = all_matches[idx]
        if temp[temp > thresh].shape[0] > best_n:
            best_n = temp[temp > thresh].shape[0]
            best_match = idx

    return dict(all_matches), best_match


def get_overlap(imgs, fd, thresh):
    match_score, _ = get_all_matches(imgs, fd)
    x = np.array(list(match_score.values()))
    temp = x == 0
    x[x < thresh] = 0
    x[x > thresh] = 1
    x[temp] = 1

    return x


def stitch_imgs(img1, img2, thresh, cv):
    h = get_H(img1, img2, thresh=thresh, cv=cv)
    img, _ = merge(img1, img2, h)
    return img


def pano(imgs, descriptors, extractor, r_thresh, thresh, cv):
    nxt_imgs = []
    nxt_desc = []
    n = len(imgs)
    helper = np.ones(n)
    flag = True

    while np.where(helper == 1)[0].shape[0] > 0:
        curr = np.where(helper == 1)[0][0]
        helper[curr] = 0
        best_n = float('-inf')
        best_nxt = None

        for nxt in np.where(helper == 1)[0]:
            matches, n_matches = match_desc(
                descriptors[curr],
                descriptors[nxt],
            )
            if n_matches > best_n and n_matches > thresh:
                best_n = n_matches
                best_nxt = nxt

        if best_nxt is not None:
            helper[best_nxt] = 0
            nxt_imgs.append(stitch_imgs(imgs[curr], imgs[best_nxt], thresh=r_thresh, cv=cv))
            _, fd = extract_features(nxt_imgs[-1], extractor)
            nxt_desc.append(fd)
            print(f'Stitched {best_nxt}, {curr} | shape: {nxt_imgs[-1].shape}')

        elif flag is False:
            nxt_imgs.append(imgs[curr])
            nxt_desc.append(descriptors[curr])

        flag = False

    return nxt_imgs, nxt_desc


def stitch_pano(imgs, r_thresh=0.02, thresh=200, cv=True):
    sift = cv2.SIFT_create()
    a = imgs
    b = [extract_features(img, sift)[1] for img in a]
    c = 0
    while a:
        print(f'iter: {c}, imgs: {len(a)}')
        a, b = pano(a, b, sift, r_thresh, thresh, cv)
        if len(a) != 0:
            res = a
        c += 1
    return res


def stitch(imgmark, N=5, savepath=''):
    # For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    """The output image should be saved in the save path."""
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."

    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N + 1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        img = resize(img, 0.2)
        imgs.append(img)
    "Start you code here"
    fd = extract_all_features(imgs)
    overlap = get_overlap(imgs, fd, 200)
    panorama = stitch_pano(list(imgs), thresh=150, r_thresh=0.018, cv=True)
    cv2.imwrite(savepath, panorama[0])

    return overlap


if __name__ == "__main__":
    # task2
    # overlap_arr = stitch('t2', N=4, savepath='task2.png')
    # with open('t2_overlap.txt', 'w') as outfile:
    #     json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t4', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
