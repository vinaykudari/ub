# Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades
import math

import cv2
import numpy as np


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


def transform(coordinates, homography):
    x, y, w, h = coordinates
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


def homography(match_pts, cv=True):
    src_pts, dst_pts = match_pts
    k = cv2.RANSAC if cv is True else 0
    if len(src_pts) > 4:
        h, mask = cv2.findHomography(src_pts, dst_pts, k, 5)
    else:
        h, mask = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), None

    return h, mask


def warp(img1, img2, H):
    _, _, a = shape(img1)
    dx, dy, _, _ = transform(coordinates(img1), H)
    new_h, (tx, ty) = translate_homography(dx, dy, H)
    x, y, w, h = transform(coordinates(img1), new_h)
    width, height = max(tx + img2.shape[1], x + w), max(ty + img2.shape[0], y + h)

    warped_img1 = cv2.warpPerspective(img1, new_h, (width, height))
    result = warped_img1.copy()
    result[ty:ty + img2.shape[0], tx:tx + img2.shape[1]] = img2
    if a:
        warped_img2 = np.zeros((height, width, a), dtype=np.uint8)
        warped_img2[ty:ty + img2.shape[0], tx:tx + img2.shape[1], :] = img2
    else:
        warped_img2 = np.zeros((height, width), dtype=np.uint8)
        warped_img2[ty:ty + img2.shape[0], tx:tx + img2.shape[1]] = img2

    return warped_img1, warped_img2, result


def overlap_region(w_img1, w_img2):
    mask = np.logical_and(
        w_img1,
        w_img2,
    ).astype('int8')
    intersection = np.argwhere(mask)

    max_dims = np.max(intersection, axis=0)
    min_dims = np.min(intersection, axis=0)

    y_max, x_max = max_dims[0], max_dims[1]
    y_min, x_min = min_dims[0], min_dims[1]

    return mask, (x_max, y_max, x_min, y_min)


def apply_mask(mask, img1, img2, p=0.60):
    mask1 = mask.copy()
    mask2 = mask.copy()

    mask1[mask == 0] = p
    mask1[mask == 1] = 0

    mask2[mask == 1] = 1 - p
    mask2[mask == 0] = 0

    img1[img1 == 0] = img2[img1 == 0]
    img2[img2 == 0] = img1[img2 == 0]

    img = (img1 * mask1) + (img2 * mask2)

    return img


def select(img1, img2, mask, thresh=5):
    h, w, a = shape(img1)
    if a:
        img_mask = np.ones((h, w, a))
    else:
        img_mask = np.ones((h, w))

    lis = []

    # ignore empty image space
    for p in range(w):
        n1 = (img1[:, p] == 0.0).sum()
        n2 = (img2[:, p] == 0.0).sum()

        if n1 > 0 or n2 > 0:
            break

    for i in range(p, w):
        n1 = (img1[:, i] == 0).sum()
        n2 = (img2[:, i] == 0).sum()
        img1_col = img1[:, i]
        img2_col = img2[:, i]

        if i == p:
            if n1 < n2:
                ref_col = img1_col
                last = 1
                flag = False
            else:
                ref_col = img2_col
                last = 2
                flag = True
        else:
            # select pixels in the overlap region and find difference
            d1 = int((img1_col * mask[:, i] - ref_col * mask[:, i - 1]).sum())
            d2 = int((img2_col * mask[:, i] - ref_col * mask[:, i - 1]).sum())

            if d1 < d2:
                if last == 2:
                    if lis and i - lis[-1] < thresh:
                        lis.pop()
                    else:
                        lis.append(i)
                    last = 1
                ref_col = img1_col
            else:
                if last == 1:
                    if lis and i - lis[-1] < thresh:
                        lis.pop()
                    else:
                        lis.append(i)
                    last = 2
                ref_col = img2_col

    lis.append(i)
    k = 0

    for x in lis:
        img_mask[:, k:x + 1] = flag
        flag = not flag
        k = x

    return lis, np.logical_not(img_mask)


def merge(img1, img2, x_min, y_min, x_max, y_max, mask, thresh=3, pad=5):
    h, w, a = shape(img1)
    mask_r = np.zeros_like(img1, dtype=bool)

    x_min_n = max(x_min - pad, 0)
    y_min_n = max(y_min - pad, 0)
    x_max_n = min(x_max + pad, w)
    y_max_n = min(y_max + pad, h)

    img1 = img1[y_min_n:y_max_n + 1, x_min_n:x_max_n + 1]
    img2 = img2[y_min_n:y_max_n + 1, x_min_n:x_max_n + 1]
    mask = mask[y_min_n:y_max_n + 1, x_min_n:x_max_n + 1]

    l, main_mask = select(img1, img2, mask=mask, thresh=thresh)
    k = 0

    for x in l[1:]:
        sub_img1 = img1[:, k:x + 1]
        sub_img2 = img2[:, k:x + 1]
        new_mask = mask[:, k:x + 1]

        if a:
            sub_img1 = sub_img1.transpose((1, 0, 2))
            sub_img2 = sub_img2.transpose((1, 0, 2))
            new_mask = new_mask.transpose((1, 0, 2))
        else:
            sub_img1 = sub_img1.T
            sub_img2 = sub_img2.T
            new_mask = new_mask.T

        _, sub_mask = select(
            sub_img1,
            sub_img2,
            new_mask,
            thresh=thresh,
        )

        if a:
            sub_mask = sub_mask.transpose((1, 0, 2))
        else:
            sub_mask = sub_mask.T

        main_mask[:, k:x + 1] = sub_mask
        k = x

    mask_r[y_min_n:y_max_n + 1, x_min_n:x_max_n + 1] = main_mask

    return mask_r[y_min:y_max + 1, x_min:x_max + 1]


def stitch(img1_c, img2_c, r_thresh=20, cv=True):
    sift = cv2.SIFT_create()
    img1 = cv2.cvtColor(img1_c, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_c, cv2.COLOR_BGR2GRAY)

    kp1, desc1 = extract_features(img1, sift)
    kp2, desc2 = extract_features(img2, sift)

    matches, score = match_desc(desc1, desc2, lowes_ratio=0.7)
    src_pts, dst_pts = get_matching_points(matches, kp1, kp2)

    if cv is False and len(src_pts) > 4:
        src_pts, dst_pts, mask = ransac((src_pts, dst_pts), thresh=r_thresh, factor=1)
    h, _ = homography((src_pts, dst_pts), cv=cv)

    w_img1, w_img2, warped_img = warp(img1_c, img2_c, h)
    overlap_mask, (x_max, y_max, x_min, y_min) = overlap_region(w_img1, w_img2)
    main_mask = merge(
        w_img1,
        w_img2,
        x_min, y_min, x_max, y_max,
        overlap_mask,
        thresh=5,
    )
    result = apply_mask(
        main_mask,
        w_img1[y_min:y_max + 1, x_min:x_max + 1],
        w_img2[y_min:y_max + 1, x_min:x_max + 1],
    )
    warped_img[y_min:y_max + 1, x_min:x_max + 1] = result

    return warped_img


def stitch_background(img1, img2, savepath=''):
    """The output image should be saved in the savepath."""
    "Do NOT modify the code provided."
    result = stitch(img1, img2, cv=True)
    cv2.imwrite(savepath, result)
    return result


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
