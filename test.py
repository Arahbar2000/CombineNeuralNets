import cv2
import numpy as np
import skimage.io
import skimage.morphology
import skimage.segmentation
import skimage.measure

def get_images():
    unet = cv2.imread('/Users/arianrahbar/Dropbox/Unet/OutLabels/1_crop.png', cv2.IMREAD_GRAYSCALE)
    mrcnn = cv2.imread('/Users/arianrahbar/Dropbox/Mrcnn/OutLabels/1_crop.png', cv2.IMREAD_GRAYSCALE)
    return unet, mrcnn

def get_interior(img):
    unet_class = skimage.io.imread('/Users/arianrahbar/Dropbox/Unet/OutProbs/1_crop.png')
    unet_mask = np.argmax(unet_class, -1)
    unet_interior = (unet_mask == 1)
    unet_interior = skimage.morphology.remove_small_holes(unet_interior, area_threshold=25)
    unet_interior = skimage.morphology.remove_small_objects(unet_interior, min_size=25)
    [label, _] = skimage.morphology.label(unet_interior, return_num=True)
    label[np.where(label != 0)] += 1
    return label


def process_images():
    unet, mrcnn= get_images()
    unet, unet_num = skimage.measure.label(unet, return_num=True)
    mrcnn, mrcnn_num = skimage.measure.label(mrcnn, return_num=True)
    max_overlap = 0
    mrcnn_label = 0
    for j in range(1, mrcnn_num+1):
        unet_test = np.where(unet == 1, 1, 0)
        mrcnn_test = np.where(mrcnn == j, 1, 0)
        overlap_img = unet_test * mrcnn_test
        overlap = np.count_nonzero(overlap_img)
        if overlap > max_overlap:
            mrcnn_label = j
            max_overlap = overlap
    unet_instance = np.where(unet == 1, 1, 0)
    mrcnn_instance = np.where(mrcnn == mrcnn_label, 1, 0)
    process_instances(unet_instance, mrcnn_instance)


def process_instances(unet, mrcnn):
    # obtaining seeds
    unet_seed = unet.copy()
    mrcnn_seed = mrcnn.copy()
    for i in range(4):
        unet_seed = skimage.morphology.erosion(unet_seed)
        mrcnn_seed = skimage.morphology.erosion(mrcnn_seed)

    cv2.imwrite('unet_seed.png', unet_seed)
    cv2.imwrite('unet.png', unet)
    cv2.imwrite('mrcnn_seed.png', mrcnn_seed)
    cv2.imwrite('mrcnn.png', mrcnn)







# def test():
#     unet, unet_seed, mrcnn_seed = get_images()
#     combined = unet + mrcnn
#     combined_mask = np.where(combined > 0, 1, 0)
#     combined_seed = unet_seed + mrcnn_seed
#     combined_seed_mask = np.where(combined_seed > 0, 1, 0)
#     labels = np.where(combined_mask > 0, 0, -1)
#     labels[combined_seed_mask > 0] = 1
#     final = skimage.segmentation.random_walker(combined_mask, labels)
#     cv2.imwrite('unet.png', unet)
#     cv2.imwrite('mrcnn.png', mrcnn)
#     cv2.imwrite('final.png', final*255)




if __name__ == '__main__':
    process_images()
