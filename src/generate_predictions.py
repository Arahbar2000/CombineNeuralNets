from predictor import predict
import os
from skimage.io import imread
import argparse
from skimage.io import imwrite
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")

    required.add_argument('-i', '--input', type=str, required=True)
    required.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()
    return {
        'input': args.input,
        'output': args.output
    }
def process_images(input, output):
    if os.path.isdir(output):
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)
    for segmentation in os.listdir(input):
        os.makedirs(os.path.join(output, segmentation), exist_ok=True)
        for filename in os.listdir(os.path.join(input, segmentation)):
            img = imread(os.path.join(input, segmentation, filename))
            prediction = predict(img)
            imwrite(os.path.join(output, segmentation, filename), prediction)

if __name__ == '__main__':
    options = parse_arguments()
    process_images(options['input'], options['output'])