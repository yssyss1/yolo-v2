import sys
sys.path.append("..")

import random
import baker
import numpy as np
from utils.data_utils import load_npy
import json
import os


class PriorAnchors:
    def __init__(self, config):
        self.num_anchors = config['box_num']
        self.train_annotation_path = config['train_annotation_path']
        self.labels = config['labels']
        self.grid_w = config['grid_w']
        self.grid_h = config['grid_h']
        self.prior_anchors_result = './anchors.txt'

    def run_kmeans(self):
        if not os.path.exists(self.train_annotation_path):
            raise FileNotFoundError('{} is not exists'.format(self.train_annotation_path))

        print("Load train annotation...")
        train_annotation = load_npy(self.train_annotation_path)

        annotations = []

        for annotation in train_annotation:
            cell_w = annotation['width'] / self.grid_w
            cell_h = annotation['height'] / self.grid_h

            for obj in annotation['object']:
                relative_w = (float(obj['xmax']) - obj['xmin']) / cell_w
                relatice_h = (float(obj["ymax"]) - obj['ymin']) / cell_h
                annotations.append((relative_w, relatice_h))

        annotations = np.array(annotations)
        print("Start K means Dimension Priors...")
        prior_anchors = self._kmeans(annotations)

        print('\naverage IOU for', self.num_anchors, 'anchors:', '%.2f' % self._avg_iou(annotations, prior_anchors))
        print('\nprior anchors are saved in {}'.format(self.prior_anchors_result))
        np.savetxt(self.prior_anchors_result, prior_anchors, fmt='%.5f')
        print("End K means Dimension Priors!")

    def _kmeans(self, annotations):
        annotation_num = annotations.shape[0]
        prev_assignments = np.ones(annotation_num) * (-1)
        iteration = 0

        initial_idx = [random.randrange(annotations.shape[0]) for i in range(self.num_anchors)]
        centroids = annotations[initial_idx]
        anchor_dim = annotations.shape[1]

        while True:
            distances = []
            iteration += 1

            for i in range(annotation_num):
                d = 1 - self._iou(annotations[i], centroids)
                distances.append(d)

            assignments = np.argmin(distances, axis=1)

            if np.sum(assignments == prev_assignments) == annotation_num:
                return centroids

            centroid_sums = np.zeros((self.num_anchors, anchor_dim), np.float)

            for i in range(annotation_num):
                centroid_sums[assignments[i]] += annotations[i]

            for j in range(self.num_anchors):
                centroids[j] = centroid_sums[j] / (np.sum(assignments == j) + 1e-6)

            print("iteration {}: average iou = {}".format(iteration, self._avg_iou(annotations, centroids)))

            prev_assignments = assignments.copy()

    def _iou(self, annotation, centroids):
        w, h = annotation
        ious = []

        for centroid in centroids:
            c_w, c_h = centroid

            if c_w >= w and c_h >= h:
                iou = w * h / (c_w * c_h)
            elif c_w >= w and c_h <= h:
                iou = w * c_h / (w * h + (c_w - w) * c_h)
            elif c_w <= w and c_h >= h:
                iou = c_w * h / (w * h + c_w * (c_h - h))
            else:
                iou = (c_w * c_h) / (w * h)
            ious.append(iou)

        return np.array(ious)

    def _avg_iou(self, annotations, centroids):
        n, d = annotations.shape
        sum = 0.

        for i in range(annotations.shape[0]):
            sum += max(self._iou(annotations[i], centroids))

        return sum / n


@baker.command(
    params={
        "config_path": "configuration file path",
    }
)
def k_means_prior_anchors(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
        prior_anchors = PriorAnchors(config)
        prior_anchors.run_kmeans()


if __name__ == '__main__':
    baker.run()
