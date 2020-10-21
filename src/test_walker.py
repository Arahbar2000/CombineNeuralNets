from skimage.io import imread
from skimage.segmentation import random_walker, relabel_sequential
from skimage.measure import label
from cv2 import imwrite
from predictor import predict
from random_walker import random_walker
import skimage.morphology
import numpy as np

def get_images():
    unet = imread('/Users/arianrahbar/Dropbox/Unet/ContourAdjusted/1073_crop.png', as_gray=True)
    mrcnn = imread('/Users/arianrahbar/Dropbox/Mrcnn/OutLabels/1073_crop.png', as_gray=True)
    annot = imread('/Users/arianrahbar/Dropbox/raw_annotations/1073_crop.png', as_gray=True)
    return unet, mrcnn, annot

def test_walker_binary():
    unet, mrcnn, annot = get_images()
    annot_mask = np.where(annot > 0, 1, 0)
    unet_mask = np.where(unet > 0, 1, 0)
    mrcnn_mask = np.where(mrcnn > 0, 1, 0)
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, 6, 6)
    labels = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    labels[unet_mask + mrcnn_mask == 0] = 2
    data = np.dstack((unet, mrcnn))
    final = random_walker(data, labels, beta=30, mode='bf')
    final[final == 2] = 0
    imwrite('unet.png', unet_mask * 255)
    imwrite('mrcnn.png', mrcnn_mask * 255)
    imwrite('final_.png', final * 255)
    imwrite('annot.png', annot_mask * 255)

def test_walker_label():
    unet, mrcnn, annot = get_images()
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, 6, 6)
    annot_mask = np.where(annot > 0, 1, 0)
    unet_mask = np.where(unet > 0, 1, 0)
    mrcnn_mask = np.where(mrcnn > 0, 1, 0)
    labels = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    labels = label(labels)
    labels, _, _ = relabel_sequential(labels, offset=2)
    labels[unet_mask + mrcnn_mask == 0] = 1
    data = np.dstack((unet, mrcnn))
    final = random_walker(data, labels, beta=50, mode='bf')
    final[final == 1] = 0
    imwrite('unet.png', unet_mask*255)
    imwrite('mrcnn.png', mrcnn_mask*255)
    imwrite('final_.png', final*255)
    imwrite('annot.png', annot_mask * 255)
    return final



def test():
    a = np.array([[[0,1,2], [3,4,5], [6,7,8]], [[9,10,11], [12,13,14], [15,16,17]]])
    i = np.unravel_index(5, a.shape)
    i += (1,)
    print(a[0, :, :])

def erode_images(img1, img2, img1_e, img2_e):
    img1_seed = img1.copy()
    img2_seed = img2.copy()
    for i in range(img1_e):
        img1_seed = skimage.morphology.erosion(img1_seed)
    for i in range(img2_e):
        img2_seed = skimage.morphology.erosion(img2_seed)

    return img1_seed, img2_seed

if __name__ == '__main__':
    test_walker_1()