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
                img = cv2.imread(f'{folder}/{file}')
                _, face_boxes = get_faces(img, p=0.0, typ='cv_faces')
                boxes = [bbox(file, box) for box in face_boxes]
                result_list.extend(boxes)
    return result_list


'''
K: number of clusters
'''


def cluster_faces(input_path: str, K: int) -> dict:
    _, res, _ = cluster_helper(input_path, K)
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


def cv_faces(img):
    h, w = img.shape[:2]
    model_path = './yunet.onnx'
    boxes = []
    faces = []

    # img = cv2.resize(img, (300, 300))
    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (w, h),
        0.90,
        0.2,
        5000,
    )
    detector.setInputSize((w, h))
    _, imgs = detector.detect(img)
    if imgs is not None:
        imgs = imgs.astype(np.int32).tolist()
        for face in imgs:
            x1, y1, w, h = face[:4]
            x1, y1 = max(x1, 0), max(y1, 0)
            faces.append(img[y1:y1 + h + 1, x1:x1 + w + 1])
            boxes.append([x1, y1, w, h])

    return faces, boxes


def dnn_faces(img, prototxt, model, thresh):
    boxes = []
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0, (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype('int')
            boxes.append([x1, y1, x2-x1, y2-y1])

    return boxes


def resize_box(img, box, k, max_h, max_w):
    x, y, w, h = box
    x = int(max(x - (k * w), 0))
    y = int(max(y - (k * h), 0))
    w = min(int(w + (k * w)), max_w-1)
    h = min(int(h + (k * h)), max_h-1)
    box = [x, y, w, h]
    face = img[y:y + h, x:x + w]
    return face, box


def get_faces(img, typ='dnn', p=0.2):
    boxes = []
    faces = []
    h, w = img.shape[:2]
    
    if typ == 'cascade':
        face_boxes = face_cascade.detectMultiScale(
            img, scaleFactor=1.03, minNeighbors=30,
        )
    elif typ == 'dnn':
        face_boxes = dnn_faces(
            img, thresh=0.96,
            model='res_300.caffemodel',
            prototxt='res_300.prototxt.txt',
        )
    else:
        _, face_boxes = cv_faces(img)

    for box in face_boxes:
        x, y, _, _ = box
        if x > w or y > h:
            continue
        face, new_box = resize_box(
            img, box, p,
            max_h=h, max_w=w,
        )
        boxes.append(new_box)
        faces.append(face)

    return faces, boxes


def cluster(encodings, k=None, typ=None):
    if typ == 'kmeans':
        print('Using KMeans')
        model = KMeans(n_clusters=k or 5, random_state=0)
    elif typ == 'mean_shift':
        print('Using MeanShift')
        model = MeanShift(n_jobs=-1)
    elif typ == 'optics':
        print('Using OPTICS')
        model = OPTICS(n_jobs=-1)
    elif typ == 'spectral':
        print('Using Spectral')
        model = SpectralClustering(n_clusters=k, n_jobs=-1)
    else:
        print('Using DBSCAN')
        model = DBSCAN(metric='euclidean', n_jobs=-1)
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


def cluster_helper(input_path: str, K: int, face_detector='cascade', cluster_method='kmeans', p=0.2):
    K = int(K)
    input_path = input_path + '/'
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
                faces, _ = get_faces(img, p=p, typ=face_detector)
                for face in faces:
                    cluster_faces.append(face)
                    cluster_map[count] = file
                    count += 1
    cluster_faces = np.array(cluster_faces, dtype=object)
    print(cluster_faces.shape)
    for im in cluster_faces:
        enc = face_encodings(im)
        if len(enc) > 0:
            encodings.append(enc[0])
    encodings = np.array(encodings, dtype=object)

    model = cluster(encodings, k=K, typ=cluster_method)
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
