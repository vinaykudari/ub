"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import glob
import sys
from collections import defaultdict

import cv2
import numpy as np

from helper import extract_features, connected_components, expand, matcher, norm_l2, norm_l1, \
    gaussian_kernel, convolve, otsu, gaussian_pyramid, n_cross_corr, measure, norm_l4

sys.setrecursionlimit(10 ** 6)


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    # feature extractor
    sift = cv2.SIFT_create()
    n_padding = 3
    n_scale = 2
    
    # test_img = gaussian_pyramid(test_img, 2)[1]
    # test_img = expand(test_img, 3)

    enrollment(
        characters=characters,
        extractor=sift,
        n_padding=1,
    )

    detection(
        test_img=test_img,
        threshold_func=otsu,
        extractor=sift,
        n_padding=2,
        n_scale=n_scale,
    )

    res = recognition(
        dist_measure=norm_l4,
        measure_type='distance',
        min_thresh=float('-inf'),
        max_thresh=210,
    )
    return res


def enrollment(characters, extractor, n_padding=4):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    char_features = {}
    for name, img in characters:
        # pad image
        img_p = np.pad(img, n_padding, constant_values=255.)
        _, descriptor = extract_features(img_p, extractor)
        char_features[name] = descriptor.tolist() if descriptor.any() else []

    with open('char_features.json', 'w') as f:
        json.dump(char_features, f)


def detection(test_img, threshold_func, extractor, n_scale=2, n_padding=1):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    flag = True
    n_component = 0
    heights = []

    # sharpen test image
    # g_kernel = gaussian_kernel(size=5, sigma=0.5)
    # test_img = convolve(test_img, g_kernel, stride=1)
    # kernel = np.array(
    #     [
    #         [0, -1, 0],
    #         [-1, 5, -1],
    #         [0, -1, 0],
    #     ]
    # )
    # test_img = convolve(test_img, kernel, stride=1)
    components = defaultdict(dict)
    h, w = test_img.shape
    bin_img = threshold_func(test_img, 100).astype(np.int16)
    component_features = {}

    # get heights of each line in the image
    for i in range(h):
        row = (bin_img[i, :] == 0.).any()
        if row == flag:
            if flag:
                heights.append([i])
            else:
                heights[-1].append(i)
            flag = not flag

    # set background pixel values to -1 to facilitate DFS
    for i in range(h):
        for j in range(w):
            if bin_img[i][j] == 255:
                bin_img[i][j] = -1

    for h1, h2 in heights:
        # break image line by line and find connected components
        n_component = connected_components(
            bin_img[h1:h2 + 1, :],
            n_component,
            components,
            h1,
            foreground=0,
        )
    print(len(list(components.keys())))

    for n_component in components:
        component = components[n_component]
        left, right = component['left'], component['right']
        top, bottom = component['top'], component['bottom']
        img = bin_img[left:right + 2, top:bottom + 2]
        # pad image
        img_p = np.pad(img, n_padding, constant_values=255.).astype(np.uint8)
        if min(img_p.shape) < 16:
            img_p = expand(img_p, n_scale)

        _, descriptor = extract_features(image=img_p, extractor=extractor)
        component_features[n_component] = {
            'coordinates': {
                'x': top,
                'y': left,
                'w': bottom - top + 1,
                'h': right - left + 1,
            },
            'descriptor': descriptor.tolist() if descriptor is not None else [],
        }

    with open('test_char_features.json', 'w') as f:
        json.dump(component_features, f)


def recognition(
        dist_measure,
        min_thresh,
        max_thresh,
        measure_type='similarity',
        char_path='char_features.json',
        test_char_path='test_char_features.json',
):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    res = []

    with open(char_path) as char_f:
        matching_chars = json.loads(char_f.read())

    with open(test_char_path) as test_char_f:
        test_chars = json.loads(test_char_f.read())

    for n_component in test_chars:
        print(f'For {n_component}')
        tgt_desc = test_chars[n_component]['descriptor']
        if measure_type == 'similarity':
            score = float('-inf')
        else:
            score = float('inf')

        match_ch = 'UNKNOWN'
        for ch in matching_chars:
            match_desc = matching_chars[ch]
            # if n_component == 0:
            print(f'ch: {ch} | match_desc: {len(match_desc)} | tgt_desc: {len(tgt_desc)}')
            # tgt_desc = tgt_desc / np.linalg.norm(tgt_desc)
            # match_desc = match_desc / np.linalg.norm(match_desc)
            scores = matcher(tgt_desc, match_desc, dist_measure)
            mean_score = sum(scores) / len(scores) if scores else float('inf')
            print(ch, mean_score)
            if min_thresh <= mean_score <= max_thresh:
                if measure_type != 'similarity':
                    if mean_score < score:
                        score = mean_score
                        match_ch = ch
                else:
                    if mean_score > score:
                        score = mean_score
                        match_ch = ch
        print(f'match_ch: {match_ch}')
        print("\n")
        res.append(
            {
                'bbox': list(test_chars[n_component]['coordinates'].values()),
                'name': match_ch,
            },
        )

    return res


def save_results(results, rs_directory):
    """
    Donot modify this code
    """
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")

    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
