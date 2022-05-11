import os

'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from face_recognition import *
from sklearn.cluster import *

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    for folder, _, files in os.walk(input_path):
        for file in files:
            if '.jpg' in file:
                img = cv2.imread(f'{folder}/{file}', 0)
                # img = cv2.equalizeHist(img)
                _, face_boxes = get_faces(img)
                boxes = [bbox(file, box) for box in face_boxes]
                result_list.extend(boxes)
    return result_list


'''
K: number of clusters
'''


def cluster_faces(input_path: str, K: int) -> dict:
    res, imgs = cluster_helper(input_path, K)
    return res


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""


def bbox(fname, box):
    return {
        'iname': fname,
        'bbox': box,
    }


def dnn_faces(img, typ):
    h, w = img.shape
    model_path = 'yunet.onxx'

    if typ == 'yunet_quantized':
        model_path = 'yunet_quantized.onnx'

    model = open(model_path)

    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),
        0.9,
        0.3,
        100,
    )
    detector.setInputSize((h, w))
    faces = detector.detect(img)

    return faces

def get_faces(img):
    boxes = []
    faces = []
    face_boxes = face_cascade.detectMultiScale(
        img, scaleFactor=1.03, minNeighbors=30,
    )
    k = 20 / 100
    for box in face_boxes:
        x, y, w, h = box.tolist()
        boxes.append(
            [
                int(max(x - (k * w), 0)),
                int(max(y - (k * h), 0)),
                int(w + (k * w)),
                int(h + (k * h)),
            ],
        )
        faces.append(img[y:y + h, x:x + w])

    return faces, boxes


def cluster(encodings, k=None, typ=None):
    if typ is None:
        print('Using DBSCAN')
        model = DBSCAN(metric='euclidean', n_jobs=-1)
    elif typ == 'kmeans':
        print('Using KMeans')
        model = KMeans(n_clusters=k or 5, random_state=0, n_jobs=-1)
    elif typ == 'mean_shift':
        print('Using MeanShift')
        model = MeanShift(n_jobs=-1)
    elif typ == 'optics':
        print('Using OPTICS')
        model = OPTICS(n_jobs=-1)
    elif typ == 'spectral':
        model = SpectralClustering(n_clusters=k, n_jobs=-1)
    model.fit(encodings)

    return model


def batch_cluster(faces):
    if len(faces) == 0:
        return []

    res = []
    size = 0
    for face in faces:
        size += sum(face.shape[:2])

    size = size // (len(faces) * 2)

    for face in faces:
        h, w = face.shape[0], face.shape[1]
        if h > w:
            x = w // 2
            x1 = h // 2 - x
            x2 = h // 2 + x
            face = face[x1:x2 + 1, :, :]
        elif w > h:
            x = h // 2
            x1 = w // 2 - x
            x2 = w // 2 + x
            face = face[:, x1:x2 + 1, :]

        face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
        res.append(face)

    return res


def cluster_helper(input_path: str, K: int):
    res = []
    clusters = {}
    cluster_faces = []
    count = 0
    cluster_map = {}
    encodings = []

    for folder, _, files in os.walk(input_path):
        for file in files:
            if '.jpg' in file:
                img = cv2.imread(f'{folder}/{file}')
                faces, _ = get_faces(img)
                for face in faces:
                    cluster_faces.append(face)
                    cluster_map[count] = file
                    count += 1

    cluster_faces = np.array(cluster_faces, dtype=object)
    for im in cluster_faces:
        enc = face_encodings(im)
        if len(enc) > 0:
            encodings.append(enc[0])
    encodings = np.array(encodings, dtype=object)

    model = cluster(encodings, k=K, typ='spectral')
    labels = np.unique(model.labels_)
    for label in labels:
        idxs = np.where(model.labels_ == label)[0]
        cropped = batch_cluster(cluster_faces[idxs])
        elements = [cluster_map[idx] for idx in idxs.tolist()]
        info = {
            'cluster_no': int(label),
            'elements': elements,
        }
        clusters[label] = cropped
        res.append(info)

    return model, res, clusters


path = '/'.join(cv2.__file__.split('/')[:-1]) + '/data/haarcascade_frontalface_default.xml'

cascade_file = cv2.samples.findFile(path)
face_cascade = cv2.CascadeClassifier(cascade_file)


def show(img, dpi=100, disable_axis=True, color=True):
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(dpi=dpi)
    plt.imshow(img, cmap='gray')

    if disable_axis:
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


def show_batch(imgs, size=10, dpi=100, axes_pad=0.4, col=4, disable_axis=True, show_title=False):
    n = len(imgs)
    row = (n // col) + 1
    fig = plt.figure(figsize=(size, size), dpi=dpi)
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(row, col),
        axes_pad=axes_pad,
    )

    for idx, (ax, im) in enumerate(zip(grid, imgs)):
        if len(im.shape) > 2:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(im, cmap='gray')

        if show_title:
            ax.set_title(f'Image {idx}')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def draw_boxes(img, boxes):
    img = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img
