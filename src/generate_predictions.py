from predictor import predict
import os
from skimage.io import imread
from cv2 import imwrite
def process_images():
    unet_dir = '/Users/arianrahbar/Dropbox/Unet/OutProbs/'
    out_dir = '/Users/arianrahbar/Dropbox/Unet/ContourAdjusted'
    os.makedirs(out_dir, exist_ok=True)
    for filename in os.listdir(unet_dir):
        print(filename)
        img = imread(os.path.join(unet_dir, filename))
        prediction = predict(img)
        imwrite(os.path.join(out_dir, filename), prediction)

if __name__ == '__main__':
    process_images()