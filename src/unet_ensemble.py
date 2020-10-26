import os
import argparse
import numpy as np
import pandas as pd
import skimage.io
import skimage.segmentation
import skimage.measure
import skimage.morphology
import cv2
import compute_metrics as comp
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
from random_walker import random_walker


class Config:
    def __init__(self, output, root, annot, file, counter=None, erosions=5, beta=30, instance=False):
        self.root = root
        self.annot = annot
        self.segmentations = [x for x in os.listdir(root) if not x.startswith('.')]
        # f = open('/mnt/resources/work/CombineNeuralNets/' + file)
        f = open('/Users/arianrahbar/ResearchEllison/CombineNeuralNets/' + file)
        self.files = f.readlines()
        self.instance = instance
        self.filename = file.split('.')[0]
        self.erosions = erosions
        self.beta = beta
        self.counter = counter
        self.output = output
        f.close()


def parse_arguments():
    description = 'Gets the root directory'
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group("required arguments")

    required.add_argument('-r', '--root', type=str, required=True)
    required.add_argument('-a', '--annot', type=str, required=True)
    required.add_argument('-f', '--file', type=str, required=True)
    required.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    return {
        'root': args.root,
        'annot': args.annot,
        'file': args.file,
        'output': args.output
    }


def process_all_images(config):
    filenames = sorted(config.files)
    tables = {}
    segmentations = {}
    ensembles = {}
    for segmentation in config.segmentations:
        segmentations[segmentation] = {}
    ensembles['ensemble'] = {}
    for key in ['jac', 'af1', 'merge_rate', 'split_rate']:
        tables[key] = pd.DataFrame(columns=list(segmentations.keys()), copy=True)
    os.makedirs(config.output, exist_ok=True)
    root_dir = os.path.join(config.output, config.filename)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    counter = 0
    for file in filenames:
        if counter % 50 == 0:
            print(counter)
        if counter == config.counter:
            break
        annot_path = os.path.join(config.annot, file.strip())
        annot = skimage.io.imread(annot_path, as_gray=True)
        for segmentation in segmentations.keys():
            path = os.path.join(config.root, segmentation, file.strip())
            segmentations[segmentation]['orig'] = skimage.io.imread(path ,as_gray=True)
            segmentations[segmentation]['results'] = comp.get_per_image_metrics(
                annot, segmentations[segmentation]['orig'], False)
            segmentations[segmentation]['mask'] = np.where(segmentations[segmentation]['orig'] > 0, 255, 0)
        for ensemble in ensembles.keys():
            ensembles[ensemble]['orig'] = ensemble_segmentations_binary(segmentations, config.erosions)
            ensembles[ensemble]['results'] = comp.get_per_image_metrics(annot, ensembles[ensemble]['orig'], False)
            ensembles[ensemble]['mask'] = np.where(ensembles[ensemble]['orig'] > 0, 255, 0)
        for key in tables.keys():
            results = {}
            for segmentation in segmentations.keys():
                results[segmentation] = segmentations[segmentation]['results'][key]
            for ensemble in ensembles.keys():
                results[ensemble] = ensembles[ensemble]['results'][key]
            tables[key] = tables[key].append(results, ignore_index=True)
        counter += 1

    os.makedirs(os.path.join(root_dir, 'stats'), exist_ok=True)
    methods = list(segmentations.keys())
    methods.extend(list(ensembles.keys()))
    output_charts(tables, methods,
                  os.path.join(root_dir, 'stats'), config)


def ensemble_segmentations_label(segmentations, erosions=5, beta=30):
    erode_images(segmentations, erosions)
    seed_union = np.zeros_like(segmentations[list(segmentations.keys())[0]]['orig'])
    orig_intersection = np.zeros_like(seed_union)
    for segmentation in segmentations.keys():
        seed_union = seed_union + segmentations[segmentation]['seed']
        orig_intersection = orig_intersection + segmentations[segmentation]['orig']
    labels = np.where(seed_union > 0, 1, 0)
    labels = skimage.measure.label(labels)
    labels, _, _ = skimage.segmentation.relabel_sequential(labels, offset=2)
    labels[orig_intersection == 0] = 1
    data = np.dstack([segmentations[segmentation]['orig'] for segmentation in segmentations.keys()])
    final = random_walker(data, labels, beta=beta, mode='bf')
    final[final == 1] = 0
    return skimage.segmentation.label(final)


def ensemble_segmentations_binary(segmentations, erosions=5, beta=30):
    erode_images(segmentations, erosions)
    seed_union = np.zeros_like(segmentations[list(segmentations.keys())[0]]['orig'])
    orig_intersection = np.zeros_like(seed_union)
    for segmentation in segmentations.keys():
        seed_union = seed_union + segmentations[segmentation]['seed']
        orig_intersection = orig_intersection + segmentations[segmentation]['orig']
    labels = np.where(seed_union > 0, 1, 0)
    labels[orig_intersection == 0] = 2
    data = np.dstack([segmentations[segmentation]['orig'] for segmentation in segmentations.keys()])
    final = random_walker(data, labels, beta=beta, mode='bf')
    final[final == 2] = 0
    return skimage.measure.label(final)


def erode_images(segmentations, num):
    for segmentation in segmentations.keys():
        segmentations[segmentation]['seed'] = segmentations[segmentation]['orig'].copy()
    for i in range(num):
        for segmentation in segmentations.keys():
            segmentations[segmentation]['seed'] = skimage.morphology.erosion(segmentations[segmentation]['seed'])


def output_charts(tables, methods, root_dir, config):
    bins = np.linspace(0, 1, 100)
    reference = {
        'jac': 'Jaccard Score',
        'af1': 'F1 Score',
        'merge_rate': 'Merge Rate',
        'split_rate': 'Split Rate'
    }
    for key in tables.keys():
        for method in methods:
            sns.distplot(tables[key][method].tolist(), hist=False, kde=True,
                         kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label=method)
        plt.legend(loc='upper left') if (key == 'jac' or key == 'f1') else plt.legend()
        plt.title(f"{reference[key]} {config.filename.capitalize()} (erosions={config.erosions}, beta={config.beta})")
        plt.xlabel(reference[key])
        plt.savefig(os.path.join(root_dir, f'{reference[key].split()[0]}.png'))
        plt.clf()
        avg = tables[key].mean()
        std = tables[key].std()
        avg.plot.bar(yerr=std.tolist())
        plt.ylim(bottom=0)
        plt.xticks(rotation='horizontal')
        plt.title(
            f'{reference[key]} Averages {config.filename.capitalize()} (erosions={config.erosions}, beta={config.beta})')
        plt.xlabel('Method')
        plt.ylabel(f'{reference[key]}')
        plt.savefig(os.path.join(root_dir, f'{reference[key].split()[0]}_averages.png'))
        plt.clf()


def output_multiple_tables(tables, root_dir):
    reference = {
        'jac': 'Jaccard Score',
        'af1': 'F1 Score',
        'merge_rate': 'Merge Rate',
        'split_rate': 'Split Rate'
    }
    f = open(os.path.join(root_dir, 'tables.txt'), 'w')
    for key in tables.keys():
        f.write(f'{reference[key]}')
        f.write(tables[key].to_string(index=True))
        f.write('\n\n')
    f.close()


if __name__ == '__main__':
    options = parse_arguments()

    CONFIG = Config(output=options['output'],
                    root=options['root'],
                    annot=options['annot'],
                    file=options['file'],
                    instance=False,
                    erosions=6,
                    beta=30,
                    counter=None)
    process_all_images(CONFIG)
