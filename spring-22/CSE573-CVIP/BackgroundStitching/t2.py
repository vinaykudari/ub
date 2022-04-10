# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades
from collections import defaultdict

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


def resize(img, factor):
    h, w, a = shape(img)
    op = cv2.resize(img, (int(w * factor), int(h * factor)))
    return op


def show(img, dpi=100, disable_axis=True, color=True):
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(dpi=dpi)
    plt.imshow(img, cmap='gray')

    if disable_axis:
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


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


def match_desc(desc1, desc2, lowes_ratio):
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
    if len(src_pts) > 4:
        h, mask = cv2.findHomography(src_pts, dst_pts, k, 5)
    else:
        h, mask = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), None

    return h, mask


def ransac(match_pts, thresh, factor, p=10, n=10000, prob=0.99):
    src_pts, dst_pts = match_pts
    k = len(src_pts)

    best_inliers = float('-inf')
    mask = []
    src_pts_f = []
    dest_pts_f = []
    print('\n')
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
            mean_error = round(errors.mean(), 2)
            print(f'idx:{i}, mean_error:{mean_error}, n_inliers:{n_inliers}, prob:{round(n_inliers / k, 2)}')
            best_inliers = n_inliers
            src_pts_f = src_pts[inliers]
            dest_pts_f = dst_pts[inliers]
            mask = inliers

        if n_inliers / k > prob:
            break

    return src_pts_f, dest_pts_f, mask.astype(np.uint8)


def get_homography(
        ransac_thresh, matches,
        kp1, kp2, cv,
):
    src_pts, dst_pts = get_matching_points(matches, kp1, kp2)

    if cv is False and len(src_pts) > 4:
        src_pts, dst_pts, mask = ransac(
            (src_pts, dst_pts), thresh=ransac_thresh, factor=1,
        )
        print(f'after ransac n_matches: {len(src_pts)}')
    h, mask = homography((src_pts, dst_pts), cv=cv)

    return h, mask


def warp(img_c, H):
    _, _, a = shape(img_c)
    dx, dy, _, _ = transform(coordinates(img_c), H)
    new_h, (tx, ty) = translate_homography(dx, dy, H)
    x, y, w, h = transform(coordinates(img_c), new_h)
    warped = cv2.warpPerspective(img_c, new_h, (x + w, y + h))
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

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )
    return img, (left, top)


def blend(img1, img2):
    idx1 = np.where(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) != 0)
    b1 = np.copy(img2)
    b1[idx1[0], idx1[1]] = img1[idx1[0], idx1[1]]
    idx2 = np.where(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) != 0)
    b2 = np.copy(img1)
    b2[idx2[0], idx2[1]] = img2[idx2[0], idx2[1]]
    blended = cv2.addWeighted(b1, 0.5, b2, 0.5, 0)
    return blended


def merge(img1_c, img2_c, H):
    result, pos = warp(img1_c, H)
    x_pos, y_pos = pos
    rect = x_pos, y_pos, img2_c.shape[1], img2_c.shape[0]
    result, _ = pad_img(result, rect)
    idx = np.s_[y_pos: y_pos + img2_c.shape[0], x_pos: x_pos + img2_c.shape[1]]
    result[idx] = blend(result[idx], img2_c)
    x, y, w, h = cv2.boundingRect(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY))
    result = result[y: y + h, x: x + w]

    return result, (x_pos - x, y_pos - y)


def get_all_matches(descriptors, lowes_ratio):
    n = len(descriptors)
    all_matches = defaultdict(lambda: np.zeros(n))
    for idx in range(n):
        desc1 = descriptors[idx]

        for jdx in range(n):
            if idx == jdx:
                continue

            desc2 = descriptors[jdx]
            _, n_matches = match_desc(desc1, desc2, lowes_ratio=lowes_ratio)
            all_matches[idx][jdx] = n_matches

    return dict(all_matches)


def get_overlap(fd, thresh, lowes_ratio=0.7):
    match_score = get_all_matches(fd, lowes_ratio)
    x = np.array(list(match_score.values()))
    h, w = x.shape
    for i in range(h):
        for j in range(w):
            if i == j:
                x[i][j] = 1
                continue
            if x[i][j] > thresh:
                x[i][j] = 1
            else:
                x[i][j] = 0

    return x


def stitch_images(img1, img2, ransac_thresh, matches, kp1, kp2, cv):
    h, _ = get_homography(
        ransac_thresh=ransac_thresh,
        matches=matches, kp1=kp1, kp2=kp2, cv=cv,
    )
    img, _ = merge(img1, img2, h)
    return img


def pano(
        imgs, descriptors, keypoints,
        extractor, match_thresh, ransac_thresh,
        lowes_ratio, show_inter, cv,
):
    nxt_imgs = []
    nxt_desc = []
    nxt_kp = []

    n = len(imgs)
    helper = np.ones(n)
    flag = True
    stitched_img = None

    while True:
        curr = np.where(helper == 1)[0][0]
        helper[curr] = 0

        best_n = float('-inf')
        best_nxt = None
        best_matches = []
        temp = None

        for nxt in np.where(helper == 1)[0]:
            matches, n_matches = match_desc(
                descriptors[nxt],
                descriptors[curr],
                lowes_ratio=lowes_ratio,
            )

            try:
                temp = n_matches
                _, mask = get_homography(
                    ransac_thresh=ransac_thresh, matches=matches,
                    kp1=keypoints[nxt], kp2=keypoints[curr], cv=cv,
                )
                if mask is not None:
                    n_matches = mask.sum()
            except Exception as e:
                print(e)
                print('No.of Inlier less than required')

            print(f'curr:{curr}, nxt:{nxt}, n_matches_b:{temp}, n_matches_a:{n_matches}')

            if n_matches > best_n and n_matches > match_thresh:
                best_n = n_matches
                best_nxt = nxt
                best_matches = matches

        if best_nxt is not None:
            print(f'curr:{curr}, best_nxt:{best_nxt}, n_matches:{best_n}')
            helper[best_nxt] = 0
            stitched_img = stitch_images(
                img1=imgs[best_nxt], img2=imgs[curr], cv=cv,
                kp1=keypoints[best_nxt], kp2=keypoints[curr],
                ransac_thresh=ransac_thresh, matches=best_matches,
            )

            nxt_imgs.append(stitched_img)

            kps, fds = extract_features(nxt_imgs[-1], extractor)
            nxt_desc.append(fds)
            nxt_kp.append(kps)

            print(f'\nStitched {curr}, {best_nxt} | shape: {nxt_imgs[-1].shape}\n')

            if show_inter:
                show(nxt_imgs[-1])
        elif flag is False:
            nxt_imgs.append(imgs[curr])
            nxt_desc.append(descriptors[curr])
            nxt_kp.append(keypoints[curr])

        flag = False
        if np.where(helper == 1)[0].shape[0] == 0:
            break

    return nxt_imgs, nxt_desc, nxt_kp, stitched_img


def stitch_pano(
        imgs, ransac_thresh=10, match_thresh=100,
        lowes_ratio=0.7, show_inter=False, cv=True,
):
    sift = cv2.SIFT_create()
    descriptors = []
    keypoints = []

    for idx, img in enumerate(imgs):
        kp, desc = extract_features(img, sift)
        keypoints.append(kp)
        descriptors.append(desc)

    desc_list = descriptors.copy()
    count = 0
    res = None
    while imgs:
        print('------------------------------')
        print(f'iter: {count}, imgs: {len(imgs)}\n')
        imgs, descriptors, keypoints, stitched_img = pano(
            imgs=imgs, descriptors=descriptors, keypoints=keypoints,
            extractor=sift, match_thresh=match_thresh,
            ransac_thresh=ransac_thresh, lowes_ratio=lowes_ratio,
            show_inter=show_inter, cv=cv,
        )
        if stitched_img is not None:
            res = stitched_img
        count += 1

    return res, desc_list


def stitch(imgmark, N=4, savepath=''):
    # For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    """The output image should be saved in the save path."""
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."

    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N + 1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    panorama, descriptors = stitch_pano(
        imgs, ransac_thresh=10, match_thresh=50, lowes_ratio=0.7,
    )
    overlap = get_overlap(descriptors, thresh=80)
    cv2.imwrite(savepath, panorama)

    return overlap


if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
