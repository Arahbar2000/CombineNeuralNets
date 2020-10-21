import cv2
import numpy as np
import skimage.io
import skimage.morphology
import skimage.segmentation
import skimage.measure
import os
import predictor as pred

def get_images():
    unet = skimage.io.imread('/Users/arianrahbar/Dropbox/Unet/OutProbs/1_crop.png')
    mrcnn = cv2.imread('/Users/arianrahbar/Dropbox/Mrcnn/OutLabels/1_crop.png', cv2.IMREAD_GRAYSCALE)
    unet = pred.predict(unet)
    return unet, mrcnn

def ensemble_segmentations():
    unet, mrcnn = get_images()
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn, 6, 6)
    ensemble = np.where(unet + mrcnn > 0, 1, 0)
    # ensemble = skimage.measure.label(ensemble)
    ensemble_seed = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    # ensemble_seed = skimage.measure.label(ensemble_seed)
    # ensemble_seed, _,  _ = skimage.segmentation.relabel_sequential(ensemble_seed, offset=2);

    # labels = np.where(ensemble > 0, 0, 1)
    # labels = np.where(ensemble_seed > 0, ensemble_seed, labels)
    labels = np.where(ensemble > 0, 0, 2)
    labels[ensemble_seed > 0] = 1
    final = skimage.segmentation.random_walker(ensemble, labels)
    # final[final == 1] = 0
    final[final == 2] = 0
    final = skimage.measure.label(final)
    unet = np.where(unet > 0, 255, 0)
    mrcnn = np.where(mrcnn  > 0, 255, 0)
    final = np.where(final > 0, 255, 0)
    cv2.imwrite('unet.png', unet)
    cv2.imwrite('mrcnn.png', mrcnn)
    cv2.imwrite('final.png', final)

def ensemble_segmentations_two():
    unet, mrcnn = get_images()
    final = unet + mrcnn
    final = np.where(final > 0, 255, 0)
    cv2.imwrite('unet.png', unet*255)
    cv2.imwrite('mrcnn.png', mrcnn*255)
    cv2.imwrite('final.png', final)


def process_images():
    unet, mrcnn = get_images()
    unet, unet_num = skimage.measure.label(unet, return_num=True, background=0)
    mrcnn, mrcnn_num = skimage.measure.label(mrcnn, return_num=True, background=0)
    final = np.zeros((256, 256))
    mrcnn_labels = set(range(1, mrcnn_num + 1))
    for i in range(1, unet_num + 1):
        max_overlap = 0
        mrcnn_label = 0
        unet_test = np.where(unet == i, 1, 0)
        for j in range(1, mrcnn_num + 1):
            mrcnn_test = np.where(mrcnn == j, 1, 0)
            overlap_img = unet_test * mrcnn_test
            overlap = np.count_nonzero(overlap_img)
            if overlap > max_overlap:
                mrcnn_label = j
                max_overlap = overlap
        if mrcnn_label in mrcnn_labels:
            mrcnn_labels.remove(mrcnn_label)
        if mrcnn_label != 0:
            unet_instance = np.where(unet == i, 1, 0)
            mrcnn_instance = np.where(mrcnn == mrcnn_label, 1, 0)
            instance = process_instances(unet_instance, mrcnn_instance)
            final[instance > 0] = i

    for label, m_label in enumerate(mrcnn_labels, unet_num + 1):
        final[mrcnn == m_label] = label

    cv2.imwrite('unet.png', unet * 255)
    cv2.imwrite('mrcnn.png', mrcnn * 255)
    cv2.imwrite('final.png', final*255)

def process_one():
    unet, mrcnn = get_images()
    unet, unet_num = skimage.measure.label(unet, return_num=True, background=0)
    mrcnn, mrcnn_num = skimage.measure.label(mrcnn, return_num=True, background=0)
    final = np.zeros((256, 256))
    mrcnn_labels = set(range(1, mrcnn_num + 1))
    i = 3
    max_overlap = 0
    mrcnn_label = 0
    unet_test = np.where(unet == i, 1, 0)
    for j in range(1, mrcnn_num + 1):
        mrcnn_test = np.where(mrcnn == j, 1, 0)
        overlap_img = unet_test * mrcnn_test
        overlap = np.count_nonzero(overlap_img)
        if overlap > max_overlap:
            mrcnn_label = j
            max_overlap = overlap
    if mrcnn_label in mrcnn_labels:
        mrcnn_labels.remove(mrcnn_label)
    if mrcnn_label != 0:
        unet_instance = np.where(unet == i, 1, 0)
        mrcnn_instance = np.where(mrcnn == mrcnn_label, 1, 0)
        instance = process_instances(unet_instance, mrcnn_instance)
        final[instance > 0] = i

    # for label, m_label in enumerate(mrcnn_labels, unet_num + 1):
    #     final[mrcnn == m_label] = label
    #
    # cv2.imwrite('unet.png', unet * 255)
    # cv2.imwrite('mrcnn.png', mrcnn * 255)
    # cv2.imwrite('final.png', final * 255)


def process_instances(unet, mrcnn):
    # obtaining seeds
    unet_seed = unet.copy()
    mrcnn_seed = mrcnn.copy()
    unique, counts = np.unique(mrcnn_seed, return_counts=True)
    for i in range(6):
        unet_seed = skimage.morphology.erosion(unet_seed)
        mrcnn_seed = skimage.morphology.erosion(mrcnn_seed)

    ensemble = np.where(unet + mrcnn > 0, 1, 0)
    ensemble_seed = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    # ensemble_seed = skimage.measure.label(ensemble_seed)
    ensemble_seed, _, _ = skimage.segmentation.relabel_sequential(ensemble_seed, offset=2);
    print(np.unique(ensemble_seed))
    labels = np.where(ensemble > 0, 0, 1)
    labels = np.where(ensemble_seed > 0, ensemble_seed, labels)
    final = skimage.segmentation.random_walker(ensemble_seed, labels)
    final[final == 1] = 0
    final = skimage.measure.label(final)
    # labels = np.where(ensemble > 0, 0, 2)
    # labels[ensemble_seed > 0] = 1
    # # labels = np.where(ensemble_seed > 0, 1, 0)
    # final = skimage.segmentation.random_walker(ensemble_seed, labels)
    # final[final == 1] = 1
    # final[final == 2] = 0
    labels[labels == 0] = 100
    labels[labels == 1] = 0
    labels[labels == 2] = 255
    # cv2.imwrite('unet.png', unet*255)
    # cv2.imwrite('mrcnn.png', mrcnn*255)
    # cv2.imwrite('final.png', final*255)
    # cv2.imwrite('labels.png', labels)
    return final


def test():
    np.random.seed(0)
    a = np.zeros((10, 10)) + 0.2 * np.random.rand(10, 10)
    a[5:8, 5:8] += 1
    b = np.zeros_like(a, dtype=np.int32)
    print(a)

def erode_images(img1, img2, img1_e, img2_e):
    img1_seed = img1.copy()
    img2_seed = img2.copy()
    for i in range(img1_e):
        img1_seed = skimage.morphology.erosion(img1_seed)
    for i in range(img2_e):
        img2_seed = skimage.morphology.erosion(img2_seed)

    return img1_seed, img2_seed



def test_2():
    unet, mrcnn = get_images()
    unet_seed, mrcnn_seed = erode_images(unet, mrcnn)
    ensemble = np.where(unet + mrcnn > 0, 1, 0)
    ensemble_seed = np.where(unet_seed + mrcnn_seed > 0, 1, 0)
    labels = np.where(ensemble > 0, 0, 2)
    labels[ensemble_seed > 0] = 1
    final = skimage.segmentation.random_walker(ensemble_seed, labels)
    final[final == 1] = 255
    final[final == 2] = 0
    cv2.imwrite('unet.png', unet * 255)
    cv2.imwrite('mrcnn.png', mrcnn * 255)
    cv2.imwrite('unet_seed.png', unet_seed * 255)
    cv2.imwrite('mrcnn_seed.png', mrcnn_seed * 255)
    cv2.imwrite('ensemble.png', ensemble * 255)
    cv2.imwrite('ensemble_seed.png', ensemble_seed * 255)
    cv2.imwrite('final.png', final)

def test_3():
    s = '102_crop.png'
    unet = '/Users/arianrahbar/Dropbox/Unet/OutLabels/'
    files = open('/Users/arianrahbar/Dropbox/test.txt').readlines()
    files = sorted(files)
    print(s)
    print(files[0])
    print(s == files[0].strip())
    path = os.path.join(unet, s)
    print(os.path.isfile(path))



if __name__ == '__main__':
    ensemble_segmentations_two()