import numpy as np
from skimage.measure import label
from skimage.morphology import erosion
from random_walker import random_walker
from skimage.segmentation import relabel_sequential
from optimize import Optimize
class Ensemble:
    def __init__(self, segmentations, erosions=6, beta=30):
        self.segmentations = segmentations
        self.segmentation_list = [self.segmentations[segmentation]['orig'] for segmentation in self.segmentations]
        self.erosions = erosions
        self.beta = beta
        self.options = {
            'walker_binary': self.walker_binary,
            'walker_label': self.walker_label,
            'union': self.union,
            'opt': self.opt,
        }

    def ensemble(self, method):
        return self.options[method]()


    def walker_binary(self, erosions=None):
        if erosions is None:
            erosions = self.erosions
        seeds = self._erode_images()
        seed_union = np.zeros_like(seeds[0])
        orig_intersection = np.zeros_like(self.segmentation_list[0])
        for i in range(len(self.segmentation_list)):
            seed_union = seed_union + seeds[i]
            orig_intersection = orig_intersection + self.segmentation_list[i]
        labels = np.where(seed_union > 0, 1, 0)
        labels[orig_intersection == 0] = 2
        data = np.dstack([segmentation for segmentation in self.segmentation_list])
        final = random_walker(data, labels, beta=self.beta, mode='bf')
        final[final == 2] = 0
        return label(final)

    def walker_label(self):
        seeds = self._erode_images()
        seed_union = np.zeros_like(seeds[0])
        orig_intersection = np.zeros_like(self.segmentation_list[0])
        for i in range(len(self.segmentations)):
            seed_union = seed_union + seeds[i]
            orig_intersection = orig_intersection + self.segmentation_list[i]
        labels = np.where(seed_union > 0, 1, 0)
        labels = label(labels)
        labels, _, _ = relabel_sequential(labels, offset=2)
        labels[orig_intersection == 0] = 1
        data = np.dstack([segmentation for segmentation in self.segmentation_list])
        final = random_walker(data, labels, beta=self.beta, mode='bf')
        final[final == 1] = 0
        return label(final)

    def union(self):
        union = np.zeros_like(self.segmentations[0])
        for segmentation in self.segmentations:
            union = union + segmentation
        union = np.where(union > 0, 1, 0)
        return label(union)

    def opt(self):
        erosions = {}
        for i in range(2, 9):
            erosions[i] = {}
        for erosion in erosions.keys():
            erosions[erosion]['orig'] = self.walker_binary(erosions=erosion)
            erosions[erosion]['mask'] = np.where(erosions[erosion]['orig'] > 0, 255, 0)
        opt = Optimize(self.segmentations, erosions).optimize()
        return erosions[opt]['orig']


    def _erode_images(self):
        seeds = [x.copy() for x in self.segmentation_list]
        for i in range(self.erosions):
            for j in range(len(self.segmentation_list)):
                seeds[j] = erosion(seeds[j])
        return seeds
